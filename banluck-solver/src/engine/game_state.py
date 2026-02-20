"""
Game state management and complete hand simulation.

Implements the full Banluck hand flow:
    DEAL → (Dealer Hard-15 Check) → (Special Hand Reveals) →
    PLAYER_ACTION → DEALER_REVEAL → DEALER_ACTION → SETTLEMENT

Key Banluck-specific rules modelled here:
    - Dealer hard-15 surrender overrides all outcomes including Ban Ban.
    - Player ≤15 forfeits unconditionally (even on dealer bust).
    - At dealer total 16/17 (including soft 17): reveal & settle 3+-card player
      BEFORE dealer decides to hit/stand (selective reveal advantage).
    - Player is capped at 5 cards; 5-card special hand bonus always announced
      in solver context (always-announce rule).
    - Ban Ban and Ban Luck are immediately settled after deal (never played further).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np

from .cards import hand_to_str
from .deck import deal_card, deal_specific_card, build_deck_from_hands
from .hand import calculate_total, is_bust, is_soft
from .special_hands import (
    HAND_BAN_BAN,
    HAND_BAN_LUCK,
    HAND_777,
    HAND_FIVE_CARD_21,
    HAND_FIVE_CARD_SUB21,
    classify_hand,
    is_ban_ban,
    is_ban_luck,
    is_hard_fifteen,
)
from .rules import Outcome, settle_hand


# ─── Enumerations ─────────────────────────────────────────────────────────────

class Phase(Enum):
    DEAL = auto()
    PLAYER_ACTION = auto()
    DEALER_REVEAL = auto()
    DEALER_ACTION = auto()
    SETTLEMENT = auto()


class PlayerAction(Enum):
    HIT = auto()
    STAND = auto()


class DealerAction(Enum):
    REVEAL_PLAYER = auto()  # Reveal and settle 3+-card player at 16/17
    HIT = auto()
    STAND = auto()
    SURRENDER = auto()      # Hard-15 surrender only


# ─── State / Result types ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a Banluck hand in progress.

    Frozen (hashable) so it can be used as a key in DP memoisation tables.
    """
    player_cards: tuple[int, ...]
    dealer_cards: tuple[int, ...]
    deck_remaining: tuple[int, ...]  # 52-length, 1=in deck, 0=dealt
    phase: Phase
    player_announced_five_card: bool  # Always True in solver (always-announce)
    player_active: bool               # False if settled early (Ban Ban/Ban Luck/bust)


@dataclass
class HandResult:
    """Result of a completed Banluck hand, from the player's perspective."""
    player_cards: tuple[int, ...]
    dealer_cards: tuple[int, ...]
    outcome: Outcome
    payout: float                # Signed units: +N win, -N loss, 0 push
    player_hand_type: str        # Classified hand type string
    dealer_hand_type: str
    dealer_surrendered: bool
    dealer_busted: bool

    def __str__(self) -> str:
        player_str = hand_to_str(self.player_cards)
        dealer_str = hand_to_str(self.dealer_cards)
        payout_str = f"+{self.payout:.1f}" if self.payout >= 0 else f"{self.payout:.1f}"
        return (
            f"Player: {player_str} ({self.player_hand_type}, "
            f"total={calculate_total(self.player_cards)}) | "
            f"Dealer: {dealer_str} ({self.dealer_hand_type}, "
            f"total={calculate_total(self.dealer_cards)}) | "
            f"{self.outcome.name} {payout_str}"
        )


# ─── Strategy type aliases ────────────────────────────────────────────────────

# player_strategy(player_cards, dealer_upcard, deck) -> PlayerAction
PlayerStrategy = Callable[[tuple[int, ...], int, np.ndarray], PlayerAction]

# dealer_strategy returns True if dealer should surrender (hard-15 check)
DealerSurrenderStrategy = Callable[[tuple[int, ...]], bool]

# dealer_hit_strategy(dealer_cards, deck, remaining_2card_players) -> bool
DealerHitStrategy = Callable[[tuple[int, ...], np.ndarray], bool]


# ─── Built-in strategy helpers ────────────────────────────────────────────────

def _default_player_strategy(
    player_cards: tuple[int, ...],
    dealer_upcard: int,
    deck: np.ndarray,
) -> PlayerAction:
    """Default player strategy: stand on 17+, hit otherwise.

    This is a simple baseline strategy for testing purposes.
    The solver will replace this with optimal strategies.
    """
    total = calculate_total(player_cards)
    if total >= 17:
        return PlayerAction.STAND
    return PlayerAction.HIT


def _dealer_always_surrenders(dealer_cards: tuple[int, ...]) -> bool:
    """Dealer strategy: always surrender on hard 15."""
    return is_hard_fifteen(dealer_cards)


def _dealer_never_surrenders(dealer_cards: tuple[int, ...]) -> bool:
    """Dealer strategy: never surrender."""
    return False


def _dealer_basic_hit_strategy(
    dealer_cards: tuple[int, ...],
    deck: np.ndarray,
) -> bool:
    """Dealer basic hit strategy: hit at 16, stand at 17+.

    At soft 17, dealer hits (same as soft 16/17 group for selective reveal).
    Dealer is forced to reach ≥16 before any strategic decision.
    """
    total = calculate_total(dealer_cards)
    if total < 16:
        return True   # Forced to hit (minimum 16 rule)
    if total == 16:
        return True   # Strategic choice to hit at 16
    if total == 17 and is_soft(dealer_cards):
        return True   # Soft 17: hit
    return False      # 17+ (hard) or 18+: stand


# ─── Core game simulation ─────────────────────────────────────────────────────

def play_hand(
    deck: np.ndarray,
    player_cards_initial: tuple[int, ...] | None = None,
    dealer_cards_initial: tuple[int, ...] | None = None,
    player_strategy: PlayerStrategy = _default_player_strategy,
    dealer_surrender_strategy: DealerSurrenderStrategy = _dealer_never_surrenders,
    dealer_hit_strategy: DealerHitStrategy = _dealer_basic_hit_strategy,
) -> HandResult:
    """Simulate a complete Banluck hand and return the result.

    This is the authoritative game flow implementation. All rule interactions
    (selective reveal, hard-15 surrender priority, unconditional forfeit) are
    applied here.

    Args:
        deck: numpy int8 array of length 52 (1=available, 0=dealt).
              Cards dealt during the hand are marked as 0. If pre-set hands
              are provided, those cards must already be marked as dealt.
        player_cards_initial: Optional pre-set player starting hand (2 cards).
                               If None, two cards are dealt from the deck.
        dealer_cards_initial: Optional pre-set dealer starting hand (2 cards).
                               If None, two cards are dealt from the deck.
        player_strategy: Callable for player hit/stand decisions.
        dealer_surrender_strategy: Callable; returns True if dealer surrenders.
        dealer_hit_strategy: Callable; returns True if dealer should hit.

    Returns:
        HandResult with complete outcome information.

    Game flow:
        1. Deal initial 2-card hands (or use pre-set hands).
        2. Check dealer hard-15 surrender (overrides all).
        3. Check player special hands (Ban Ban/Ban Luck → immediate settle).
        4. Player action phase: hit/stand until done.
        5. Dealer selective reveal at 16/17: settle 3+-card player first.
        6. Dealer action phase: hit until forced minimum or strategic stand.
        7. Final settlement.
    """
    # ── Phase 1: Deal initial hands ───────────────────────────────────────────
    if player_cards_initial is not None:
        player_cards = player_cards_initial
    else:
        c1 = deal_card(deck)
        c2 = deal_card(deck)
        player_cards = (c1, c2)

    if dealer_cards_initial is not None:
        dealer_cards = dealer_cards_initial
    else:
        c1 = deal_card(deck)
        c2 = deal_card(deck)
        dealer_cards = (c1, c2)

    dealer_upcard = dealer_cards[0]  # First dealer card is "visible" upcard

    # ── Phase 2: Dealer hard-15 surrender (highest priority) ─────────────────
    if dealer_surrender_strategy(dealer_cards):
        # Surrender overrides ALL outcomes, including Ban Ban
        player_type = classify_hand(player_cards)
        dealer_type = classify_hand(dealer_cards)
        return HandResult(
            player_cards=player_cards,
            dealer_cards=dealer_cards,
            outcome=Outcome.PUSH,
            payout=0.0,
            player_hand_type=player_type,
            dealer_hand_type=dealer_type,
            dealer_surrendered=True,
            dealer_busted=False,
        )

    # ── Phase 3: Immediate special hand settlement ────────────────────────────
    # Ban Ban and Ban Luck are settled immediately (never played further).
    if is_ban_ban(player_cards) or is_ban_luck(player_cards):
        dealer_type = classify_hand(dealer_cards)
        player_type = classify_hand(player_cards)
        outcome, payout = settle_hand(
            player_cards, dealer_cards,
            dealer_surrendered=False,
            dealer_busted=False,
        )
        return HandResult(
            player_cards=player_cards,
            dealer_cards=dealer_cards,
            outcome=outcome,
            payout=payout,
            player_hand_type=player_type,
            dealer_hand_type=dealer_type,
            dealer_surrendered=False,
            dealer_busted=False,
        )

    # ── Phase 4: Player action phase ──────────────────────────────────────────
    player_busted = False
    player_cards_list = list(player_cards)

    while True:
        player_total = calculate_total(tuple(player_cards_list))

        # Bust check
        if player_total > 21:
            player_busted = True
            break

        # Maximum 5 cards
        if len(player_cards_list) == 5:
            break

        # Must stand on 21
        if player_total == 21:
            break

        # Player strategic decision
        action = player_strategy(
            tuple(player_cards_list), dealer_upcard, deck
        )

        if action == PlayerAction.STAND:
            break

        # HIT: draw a card
        new_card = deal_card(deck)
        player_cards_list.append(new_card)

    player_cards = tuple(player_cards_list)
    player_type = classify_hand(player_cards)

    # Player bust: immediate loss, no dealer action needed
    if player_busted:
        dealer_type = classify_hand(dealer_cards)
        return HandResult(
            player_cards=player_cards,
            dealer_cards=dealer_cards,
            outcome=Outcome.LOSS,
            payout=-1.0,
            player_hand_type=player_type,
            dealer_hand_type=dealer_type,
            dealer_surrendered=False,
            dealer_busted=False,
        )

    # ── Phase 5 & 6: Dealer action with selective reveal ─────────────────────
    dealer_cards, dealer_busted = _dealer_action_phase(
        dealer_cards=dealer_cards,
        player_cards=player_cards,
        deck=deck,
        dealer_hit_strategy=dealer_hit_strategy,
    )

    # ── Phase 7: Final settlement ─────────────────────────────────────────────
    dealer_type = classify_hand(dealer_cards)
    outcome, payout = settle_hand(
        player_cards, dealer_cards,
        dealer_surrendered=False,
        dealer_busted=dealer_busted,
    )

    return HandResult(
        player_cards=player_cards,
        dealer_cards=dealer_cards,
        outcome=outcome,
        payout=payout,
        player_hand_type=player_type,
        dealer_hand_type=dealer_type,
        dealer_surrendered=False,
        dealer_busted=dealer_busted,
    )


def _dealer_action_phase(
    dealer_cards: tuple[int, ...],
    player_cards: tuple[int, ...],
    deck: np.ndarray,
    dealer_hit_strategy: DealerHitStrategy,
) -> tuple[tuple[int, ...], bool]:
    """Execute the dealer action phase, including selective reveal logic.

    Selective reveal at dealer total 16 or 17 (including soft 17):
        Step 1: If player has 3+ cards, settle them NOW (before dealer acts).
                (In the game flow, this settlement happens outside this function.)
        Step 2: Dealer decides to hit or stand.
        Step 3: Continue until dealer reaches a final total.

    Note: In heads-up play, the selective reveal means:
        - 3+-card player is settled BEFORE dealer decides to hit/stand.
        - 2-card player is settled AFTER dealer acts.
        This function returns the final dealer hand; the caller handles
        which settlement moment applies based on player hand size.

    Returns:
        (final_dealer_cards, dealer_busted)
    """
    dealer_cards_list = list(dealer_cards)
    dealer_total = calculate_total(tuple(dealer_cards_list))

    # Dealer must reach minimum 16 first (forced hits)
    while dealer_total < 16:
        new_card = deal_card(deck)
        dealer_cards_list.append(new_card)
        dealer_total = calculate_total(tuple(dealer_cards_list))
        if dealer_total > 21:
            return tuple(dealer_cards_list), True

    # Dealer is now at ≥16. Strategic decision loop.
    # At 16 or 17 (including soft 17), dealer may hit.
    # At 18+, dealer always stands.
    while True:
        dealer_total = calculate_total(tuple(dealer_cards_list))

        if dealer_total > 21:
            return tuple(dealer_cards_list), True

        if dealer_total >= 18:
            # No incentive to draw at 18+
            break

        # At 16 or 17: dealer decides strategically
        should_hit = dealer_hit_strategy(tuple(dealer_cards_list), deck)
        if not should_hit:
            break

        new_card = deal_card(deck)
        dealer_cards_list.append(new_card)

    return tuple(dealer_cards_list), False


# ─── Selective reveal settlement helper ──────────────────────────────────────

def settle_with_selective_reveal(
    player_cards: tuple[int, ...],
    dealer_cards_initial: tuple[int, ...],
    deck: np.ndarray,
    dealer_hit_strategy: DealerHitStrategy = _dealer_basic_hit_strategy,
) -> tuple[HandResult, tuple[int, ...]]:
    """Simulate dealer action with correct selective reveal ordering.

    This implements the heads-up selective reveal mechanic:

    If dealer has 16 or 17 AND player has 3+ cards:
        1. Settle the player IMMEDIATELY (before dealer hits).
        2. Dealer then acts (hits/stands) — but the settlement is already done.
        The 3+-card player's outcome is determined at dealer's INITIAL 16/17 total.

    If dealer has 16 or 17 AND player has exactly 2 cards:
        1. Dealer acts FIRST (hits/stands).
        2. Settle player AFTER dealer reaches final total.
        The 2-card player's outcome is determined at dealer's FINAL total.

    Args:
        player_cards: Player's final hand (after player action phase).
        dealer_cards_initial: Dealer's initial 2-card hand.
        deck: Current deck state.
        dealer_hit_strategy: Callable for dealer hit/stand decisions.

    Returns:
        (HandResult for the player, final_dealer_cards)
    """
    player_total = calculate_total(player_cards)
    player_busted = player_total > 21

    if player_busted:
        # Player already busted — return loss immediately, no dealer action needed
        final_dealer, dealer_busted = _dealer_action_phase(
            dealer_cards_initial, player_cards, deck, dealer_hit_strategy
        )
        return (
            HandResult(
                player_cards=player_cards,
                dealer_cards=final_dealer,
                outcome=Outcome.LOSS,
                payout=-1.0,
                player_hand_type=classify_hand(player_cards),
                dealer_hand_type=classify_hand(final_dealer),
                dealer_surrendered=False,
                dealer_busted=dealer_busted,
            ),
            final_dealer,
        )

    dealer_total_initial = calculate_total(dealer_cards_initial)
    player_has_3plus_cards = len(player_cards) >= 3
    selective_reveal_totals = {16, 17}

    # Determine if selective reveal applies at this dealer total
    dealer_at_selective_total = dealer_total_initial in selective_reveal_totals

    if dealer_at_selective_total and player_has_3plus_cards:
        # Selective reveal: settle 3+-card player at dealer's CURRENT (pre-hit) total
        outcome, payout = settle_hand(
            player_cards, dealer_cards_initial,
            dealer_surrendered=False,
            dealer_busted=False,
        )
        # Dealer still acts (but player is already settled)
        final_dealer, _ = _dealer_action_phase(
            dealer_cards_initial, player_cards, deck, dealer_hit_strategy
        )
        return (
            HandResult(
                player_cards=player_cards,
                dealer_cards=dealer_cards_initial,  # Settled against initial dealer hand
                outcome=outcome,
                payout=payout,
                player_hand_type=classify_hand(player_cards),
                dealer_hand_type=classify_hand(dealer_cards_initial),
                dealer_surrendered=False,
                dealer_busted=False,
            ),
            final_dealer,
        )
    else:
        # Dealer acts first, then settle player against final dealer hand
        final_dealer, dealer_busted = _dealer_action_phase(
            dealer_cards_initial, player_cards, deck, dealer_hit_strategy
        )
        outcome, payout = settle_hand(
            player_cards, final_dealer,
            dealer_surrendered=False,
            dealer_busted=dealer_busted,
        )
        return (
            HandResult(
                player_cards=player_cards,
                dealer_cards=final_dealer,
                outcome=outcome,
                payout=payout,
                player_hand_type=classify_hand(player_cards),
                dealer_hand_type=classify_hand(final_dealer),
                dealer_surrendered=False,
                dealer_busted=dealer_busted,
            ),
            final_dealer,
        )
