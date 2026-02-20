"""
Settlement, hand comparison, and payout calculation.

Settlement priority (highest to lowest):
    1. Dealer hard 15 surrender  → push (overrides ALL including Ban Ban)
    2. Player bust (>21)         → player loses 1 unit
    3. Player ≤15 forfeit        → player loses 1 unit (unconditional, even on dealer bust)
    4. Hand hierarchy comparison → winner's multiplier applied
    5. Total comparison          → regular hands compared by total

Payout convention (from player's perspective):
    +N  = player wins N units
    -N  = player loses N units
     0  = push (bet returned)
"""

from __future__ import annotations

from enum import Enum, auto

from .hand import calculate_total, is_bust
from .special_hands import (
    HAND_BAN_BAN,
    HAND_BAN_LUCK,
    HAND_777,
    HAND_FIVE_CARD_21,
    HAND_FIVE_CARD_SUB21,
    HAND_REGULAR,
    HAND_BUST,
    PAYOUT_MULTIPLIERS,
    classify_hand,
    hand_hierarchy_rank,
)


class Outcome(Enum):
    WIN = auto()
    LOSS = auto()
    PUSH = auto()


# ─── Core settlement function ─────────────────────────────────────────────────

def settle_hand(
    player_cards: tuple[int, ...],
    dealer_cards: tuple[int, ...],
    dealer_surrendered: bool = False,
    dealer_busted: bool = False,
) -> tuple[Outcome, float]:
    """Determine the outcome and payout for a completed hand.

    Args:
        player_cards: Player's final hand as a tuple of card integers.
        dealer_cards: Dealer's final hand as a tuple of card integers.
        dealer_surrendered: True if the dealer invoked the hard-15 surrender.
                            Overrides all other outcomes — all bets returned.
        dealer_busted: True if the dealer's final total exceeds 21.

    Returns:
        (Outcome, payout) where payout is from the player's perspective.
        Positive payout = player profit in units; negative = player loss.

    Settlement rules applied in order:
        1. Dealer surrender → PUSH, 0
        2. Player bust      → LOSS, -1
        3. Player ≤15       → LOSS, -1  (unconditional forfeit)
        4. Special hand / total comparison → multiplier-based payout
    """
    # ── Rule 1: Dealer surrender overrides everything ─────────────────────────
    if dealer_surrendered:
        return Outcome.PUSH, 0.0

    player_total = calculate_total(player_cards)

    # ── Rule 2: Player bust ───────────────────────────────────────────────────
    if is_bust(player_total):
        return Outcome.LOSS, -1.0

    # ── Rule 3: Player ≤15 unconditional forfeit ──────────────────────────────
    # This applies even if dealer busted — no exceptions.
    if player_total <= 15:
        return Outcome.LOSS, -1.0

    # ── Rule 4: Hand comparison ───────────────────────────────────────────────
    player_type = classify_hand(player_cards)
    # player_type cannot be 'bust' at this point (already handled above)

    if dealer_busted:
        # Dealer bust: player wins at their special hand multiplier (or 1:1)
        multiplier = PAYOUT_MULTIPLIERS.get(player_type, 1)
        return Outcome.WIN, float(multiplier)

    dealer_total = calculate_total(dealer_cards)
    dealer_type = classify_hand(dealer_cards)

    return _compare_hands(player_type, player_total, dealer_type, dealer_total)


def _compare_hands(
    player_type: str,
    player_total: int,
    dealer_type: str,
    dealer_total: int,
) -> tuple[Outcome, float]:
    """Compare two non-bust hands and determine outcome + payout.

    Args:
        player_type: Player's hand classification string.
        player_total: Player's best total.
        dealer_type: Dealer's hand classification string.
        dealer_total: Dealer's best total.

    Returns:
        (Outcome, payout) from the player's perspective.
    """
    player_rank = hand_hierarchy_rank(player_type)
    dealer_rank = hand_hierarchy_rank(dealer_type)

    if player_rank < dealer_rank:
        # Player has a strictly stronger special hand
        return Outcome.WIN, float(PAYOUT_MULTIPLIERS[player_type])

    if dealer_rank < player_rank:
        # Dealer has a strictly stronger special hand — player pays dealer's multiplier
        return Outcome.LOSS, -float(PAYOUT_MULTIPLIERS[dealer_type])

    # Same hierarchy level — handle each tier
    return _resolve_same_tier(player_type, player_total, dealer_type, dealer_total)


def _resolve_same_tier(
    player_type: str,
    player_total: int,
    dealer_type: str,
    dealer_total: int,
) -> tuple[Outcome, float]:
    """Resolve outcome when player and dealer have the same hierarchy tier.

    Mirror special hands always push (Ban Ban vs Ban Ban, Ban Luck vs Ban Luck,
    777 vs 777). Five-card vs five-card compares totals at 1:1. Regular hands
    compare totals at 1:1.
    """
    # Mirror-push hands: identical type → push regardless of total
    if player_type in (HAND_BAN_BAN, HAND_BAN_LUCK, HAND_777):
        return Outcome.PUSH, 0.0

    # Five-card vs five-card: compare totals at regular 1:1 payout
    if player_type in (HAND_FIVE_CARD_21, HAND_FIVE_CARD_SUB21):
        # Both are five-card hands — winner determined by total at 1:1
        if player_total > dealer_total:
            return Outcome.WIN, 1.0
        if dealer_total > player_total:
            return Outcome.LOSS, -1.0
        return Outcome.PUSH, 0.0

    # Regular vs regular (includes regular 21): compare totals at 1:1
    if player_total > dealer_total:
        return Outcome.WIN, 1.0
    if dealer_total > player_total:
        return Outcome.LOSS, -1.0
    return Outcome.PUSH, 0.0


# ─── Convenience helpers ──────────────────────────────────────────────────────

def calculate_payout(outcome: Outcome, payout_units: float, bet: float = 1.0) -> float:
    """Convert a payout in units to a dollar amount given the bet size.

    Args:
        outcome: The hand outcome (unused — kept for clarity in call sites).
        payout_units: Signed payout in units (from settle_hand).
        bet: Bet size in dollars (default 1 unit).

    Returns:
        Net profit/loss in dollars. Negative means the player loses money.

    Examples:
        >>> calculate_payout(Outcome.WIN, 3.0, 10.0)   # Ban Ban win, $10 bet
        30.0
        >>> calculate_payout(Outcome.LOSS, -1.0, 10.0)  # Regular loss, $10 bet
        -10.0
        >>> calculate_payout(Outcome.PUSH, 0.0, 10.0)   # Push
        0.0
    """
    return payout_units * bet
