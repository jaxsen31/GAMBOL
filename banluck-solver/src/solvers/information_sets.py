"""
Information set types for the Banluck CFR+ solver (Phase 2).

An information set (infoset) encodes exactly what a player observes at a given
decision point. In Banluck, all dealer cards are face-down — the player never
sees the dealer's hand. The dealer sees their own cards and can observe each
player's card count (2-card vs 3+-card), which determines whether REVEAL_PLAYER
is a legal action at 16/17.

Three infoset types cover the three distinct decision points in the game:

    PlayerHitStandInfoSet   — player chooses HIT or STAND
    DealerSurrenderInfoSet  — dealer checks for hard-15 surrender (before play)
    DealerActionInfoSet     — dealer chooses REVEAL_PLAYER / HIT / STAND at 16/17+

All types are NamedTuples: they subclass tuple and are therefore hashable and
usable as keys in CFR regret tables with no extra overhead.

Enums (PlayerAction, DealerAction) are imported from game_state — do not
redefine them here.
"""

from __future__ import annotations

from typing import NamedTuple

from src.engine.game_state import DealerAction, PlayerAction
from src.engine.hand import calculate_total, is_soft
from src.engine.special_hands import is_hard_fifteen


# ─── Information set types ────────────────────────────────────────────────────

class PlayerHitStandInfoSet(NamedTuple):
    """Player's information when deciding to hit or stand.

    In Banluck the dealer is fully face-down, so the player observes only
    their own hand. The state is abstracted to (total, num_cards, is_soft)
    matching the Phase 1.1 DP strategy key, which is sufficient because:
      - deck is reshuffled every hand (no card-counting state)
      - player cannot see dealer cards
      - ace valuation is deterministic given (total, num_cards, is_soft)

    Attributes:
        total:     Best hand total after greedy ace resolution (16–21 range for
                   decision states; ≤15 and >21 are terminal or near-terminal).
        num_cards: Number of cards in hand (2–5).
        is_soft:   True if an ace is contributing 10 (2-card: ace=10 or 11;
                   3+-card: ace=10 not 1) to the current total.

    Example:
        >>> PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=True)
        PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=True)
    """
    total: int
    num_cards: int
    is_soft: bool


class DealerSurrenderInfoSet(NamedTuple):
    """Dealer's information at the hard-15 surrender decision point.

    Checked immediately after the deal, before any player actions.
    The dealer sees only their own 2-card hand and decides whether to surrender.

    Attributes:
        total:          Dealer's 2-card total.
        is_hard_fifteen: True iff exactly 2 cards, total == 15, no ace present.
                         Only hard 15 (e.g. 6+9, 7+8, 5+10) qualifies — soft 15
                         (A+4) does NOT.

    Example:
        >>> DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
        DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
    """
    total: int
    is_hard_fifteen: bool


class DealerActionInfoSet(NamedTuple):
    """Dealer's information at the 16/17 strategic decision point.

    After reaching ≥16, the dealer sees their own cards and can observe the
    player's card count (2-card vs 3+-card). The player's card count determines
    whether REVEAL_PLAYER is a legal action: the selective-reveal mechanic only
    applies when there IS a multi-card player to settle early.

    Attributes:
        dealer_total: Dealer's current total.
        dealer_nc:    Number of dealer cards in hand.
        is_soft:      True if dealer total includes an ace counted as 10.
        player_nc:    Player's card count. REVEAL_PLAYER is only legal when
                      player_nc >= 3 (a 3+-card player can be settled early).

    Example:
        >>> DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=3)
        DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=3)
    """
    dealer_total: int
    dealer_nc: int
    is_soft: bool
    player_nc: int


# ─── Factory / extractor functions ───────────────────────────────────────────

def make_player_info_set(player_cards: tuple[int, ...]) -> PlayerHitStandInfoSet:
    """Extract PlayerHitStandInfoSet from a raw card integer tuple.

    Args:
        player_cards: Tuple of card integers (each 0–51).

    Returns:
        PlayerHitStandInfoSet with total, num_cards, is_soft filled in.

    Example:
        >>> from src.engine.cards import str_to_card
        >>> cards = (str_to_card('AS'), str_to_card('6C'))
        >>> make_player_info_set(cards)
        PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=True)
    """
    return PlayerHitStandInfoSet(
        total=calculate_total(player_cards),
        num_cards=len(player_cards),
        is_soft=is_soft(player_cards),
    )


def make_dealer_surrender_info_set(
    dealer_cards: tuple[int, ...]
) -> DealerSurrenderInfoSet:
    """Extract DealerSurrenderInfoSet from the dealer's initial 2-card hand.

    Args:
        dealer_cards: Dealer's 2-card hand as card integers.

    Returns:
        DealerSurrenderInfoSet with total and is_hard_fifteen filled in.

    Example:
        >>> from src.engine.cards import str_to_card
        >>> cards = (str_to_card('7C'), str_to_card('8D'))
        >>> make_dealer_surrender_info_set(cards)
        DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
    """
    return DealerSurrenderInfoSet(
        total=calculate_total(dealer_cards),
        is_hard_fifteen=is_hard_fifteen(dealer_cards),
    )


def make_dealer_action_info_set(
    dealer_cards: tuple[int, ...],
    player_nc: int,
) -> DealerActionInfoSet:
    """Extract DealerActionInfoSet at the 16/17 strategic decision point.

    Args:
        dealer_cards: Dealer's current hand as card integers.
        player_nc:    Player's card count at this moment.

    Returns:
        DealerActionInfoSet with dealer state and player_nc filled in.

    Example:
        >>> from src.engine.cards import str_to_card
        >>> cards = (str_to_card('9C'), str_to_card('7D'))
        >>> make_dealer_action_info_set(cards, player_nc=3)
        DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=3)
    """
    return DealerActionInfoSet(
        dealer_total=calculate_total(dealer_cards),
        dealer_nc=len(dealer_cards),
        is_soft=is_soft(dealer_cards),
        player_nc=player_nc,
    )


# ─── Legal action queries ─────────────────────────────────────────────────────

def get_legal_player_actions(
    info_set: PlayerHitStandInfoSet,
) -> list[PlayerAction]:
    """Return legal player actions for a given infoset.

    Rules:
    - total == 21 or num_cards == 5: STAND only (forced terminal — must stand).
    - total > 21: already busted — no actions (terminal node; this function
      should not be called in that case, but returns [] defensively).
    - total ≤ 15: [HIT, STAND] — STAND is legal but unconditionally loses at
      showdown (forfeit rule). CFR must learn that standing ≤15 = −1.0 EV.
    - 16–20: [HIT, STAND]

    Args:
        info_set: PlayerHitStandInfoSet describing the current hand.

    Returns:
        List of legal PlayerAction values.
    """
    total = info_set.total
    num_cards = info_set.num_cards

    if total > 21:
        return []  # busted — terminal

    if total == 21 or num_cards == 5:
        return [PlayerAction.STAND]  # forced stand

    return [PlayerAction.HIT, PlayerAction.STAND]


def get_legal_dealer_surrender_actions(
    info_set: DealerSurrenderInfoSet,
) -> list[DealerAction]:
    """Return legal dealer actions at the surrender decision point.

    DealerAction.HIT is used as the "continue without surrendering" sentinel
    here, since the dealer must eventually hit to reach ≥16 anyway.

    Rules:
    - is_hard_fifteen == True:  [DealerAction.SURRENDER, DealerAction.HIT]
    - otherwise:                [DealerAction.HIT]

    Args:
        info_set: DealerSurrenderInfoSet for the dealer's initial 2-card hand.

    Returns:
        List of legal DealerAction values.
    """
    if info_set.is_hard_fifteen:
        return [DealerAction.SURRENDER, DealerAction.HIT]
    return [DealerAction.HIT]


def get_legal_dealer_actions(
    info_set: DealerActionInfoSet,
) -> list[DealerAction]:
    """Return legal dealer actions at the strategic 16/17+ decision point.

    Rules (in order of precedence):
    - dealer_total < 16:   [HIT] only — must reach the minimum 16 (forced).
    - dealer_total >= 18:  [STAND] only — no incentive to draw at 18+.
    - dealer_total in {16, 17} (including soft 17):
        [REVEAL_PLAYER, HIT, STAND]  if player_nc >= 3
        [HIT, STAND]                 if player_nc == 2
      REVEAL_PLAYER is only legal when there is a multi-card player to settle.

    Note: for CFR, the dealer has free strategic choice at 16/17 (including
    soft 17). The baseline DP used a fixed "hit soft 17" rule, but CFR
    determines the optimal mixed strategy.

    Args:
        info_set: DealerActionInfoSet at the current decision point.

    Returns:
        List of legal DealerAction values.
    """
    total = info_set.dealer_total
    player_nc = info_set.player_nc

    if total < 16:
        return [DealerAction.HIT]  # forced — hasn't reached minimum 16

    if total >= 18:
        return [DealerAction.STAND]  # no incentive to draw

    # total is 16 or 17 (including soft 17)
    actions: list[DealerAction] = [DealerAction.HIT, DealerAction.STAND]
    if player_nc >= 3:
        actions.append(DealerAction.REVEAL_PLAYER)
    return actions
