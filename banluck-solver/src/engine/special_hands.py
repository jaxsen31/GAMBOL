"""
Special hand detection and classification for Banluck.

Special hand hierarchy (strongest to weakest):
    1. ban_ban      — Two aces (initial 2-card deal), total 21, pays 3:1
    2. ban_luck     — Ace + {10,J,Q,K} (initial 2-card deal), total 21, pays 2:1
    3. 777          — Dealt (7,7) initially, drew third 7 (suit-independent), pays 7:1
    4. five_card_21 — Exactly 5 cards totaling 21, pays 3:1
    5. five_card_sub21 — Exactly 5 cards totaling <21, pays 2:1
    6. regular      — Any other non-bust hand (including regular 21)

Note: 'bust' is not a special hand type — bust hands are handled separately.

All functions operate on tuples of card integers for Numba compatibility.
"""

from __future__ import annotations

from .cards import RANK_ACE, RANK_SEVEN, RANK_VALUES, TEN_VALUE_RANKS
from .hand import calculate_total

# ─── Hand type string constants ───────────────────────────────────────────────

HAND_BAN_BAN: str = "ban_ban"
HAND_BAN_LUCK: str = "ban_luck"
HAND_777: str = "777"
HAND_FIVE_CARD_21: str = "five_card_21"
HAND_FIVE_CARD_SUB21: str = "five_card_sub21"
HAND_REGULAR: str = "regular"
HAND_BUST: str = "bust"

# Hierarchy rank: lower number = stronger hand
_HIERARCHY: dict[str, int] = {
    HAND_BAN_BAN: 1,
    HAND_BAN_LUCK: 2,
    HAND_777: 3,
    HAND_FIVE_CARD_21: 4,
    HAND_FIVE_CARD_SUB21: 5,
    HAND_REGULAR: 6,
    HAND_BUST: 7,
}

# Payout multipliers for special hands (winner's multiplier applied to bet)
PAYOUT_MULTIPLIERS: dict[str, int] = {
    HAND_BAN_BAN: 3,
    HAND_BAN_LUCK: 2,
    HAND_777: 7,
    HAND_FIVE_CARD_21: 3,
    HAND_FIVE_CARD_SUB21: 2,
    HAND_REGULAR: 1,
}


# ─── Detection functions ───────────────────────────────────────────────────────


def is_ban_ban(cards: tuple[int, ...]) -> bool:
    """Return True if the hand is a Ban Ban (two aces, exactly 2 cards).

    Ban Ban total = 21 (one ace=11, one ace=10 in 2-card rules).
    Immediately settled — this hand is never played further.

    Examples:
        >>> is_ban_ban((48, 49))   # AC, AD — two aces
        True
        >>> is_ban_ban((48, 51))   # AC, AS — two aces
        True
        >>> is_ban_ban((48, 36))   # AC, JC — not Ban Ban
        False
        >>> is_ban_ban((48, 49, 50))  # three aces — not Ban Ban (not 2-card)
        False
    """
    if len(cards) != 2:
        return False
    return all(c // 4 == RANK_ACE for c in cards)


def is_ban_luck(cards: tuple[int, ...]) -> bool:
    """Return True if the hand is a Ban Luck (ace + 10-value card, exactly 2 cards).

    Ban Luck total = 21 (ace=11 + 10-value card).
    Immediately settled — this hand is never played further.

    Note: Ban Ban takes priority — a two-ace hand is Ban Ban, not Ban Luck.

    Examples:
        >>> is_ban_luck((48, 36))  # AC, JC — ace + jack
        True
        >>> is_ban_luck((51, 32))  # AS, 10C — ace + ten
        True
        >>> is_ban_luck((48, 49))  # AC, AD — two aces, this is Ban Ban not Ban Luck
        False
        >>> is_ban_luck((48, 4))   # AC, 3C — ace + 3, not Ban Luck
        False
    """
    if len(cards) != 2:
        return False
    ranks = (cards[0] // 4, cards[1] // 4)
    has_ace = RANK_ACE in ranks
    if not has_ace:
        return False
    # The non-ace card must be a ten-value rank
    other_rank = ranks[1] if ranks[0] == RANK_ACE else ranks[0]
    return other_rank in TEN_VALUE_RANKS


def is_777(cards: tuple[int, ...]) -> bool:
    """Return True if the hand is a 777 (exactly three 7s, suit-independent).

    The 777 requires that the first two cards dealt (indices 0 and 1) are both
    7s, and the third card drawn is also a 7. This ensures 777 can only be
    achieved by starting with a (7,7) initial deal and drawing a third 7.

    In practice, if cards is a 3-tuple and all three are 7s, this is a valid 777
    (since the game flow always puts the initial deal at indices 0 and 1).

    Examples:
        >>> is_777((20, 21, 22))   # 7C, 7D, 7H — three sevens
        True
        >>> is_777((20, 21, 23))   # 7C, 7D, 7S — three sevens (suit-independent)
        True
        >>> is_777((20, 21, 0))    # 7C, 7D, 2C — not three sevens
        False
        >>> is_777((20, 21))       # only two cards
        False
    """
    if len(cards) != 3:
        return False
    return all(c // 4 == RANK_SEVEN for c in cards)


def is_five_card_hand(cards: tuple[int, ...]) -> bool:
    """Return True if the hand is a valid 5-card special hand (5 cards, total ≤21).

    A 5-card hand beats ANY non-5-card hand regardless of totals.
    Only another 5-card hand can push or beat it.

    Examples:
        >>> is_five_card_hand((0, 4, 8, 12, 16))  # 2C,3C,4C,5C,6C = 20
        True
        >>> is_five_card_hand((0, 4, 8, 12, 20))  # 2C,3C,4C,5C,7C = 21
        True
        >>> is_five_card_hand((32, 36, 40, 44, 48))  # 10+J+Q+K+A = bust (41) — not valid
        False
    """
    if len(cards) != 5:
        return False
    return calculate_total(cards) <= 21


def is_hard_fifteen(cards: tuple[int, ...]) -> bool:
    """Return True if the hand is a hard 15 (exactly 2 cards, total=15, no ace).

    Hard 15 definition: any 2-card hand totaling 15 with no ace present.
    Examples: 6-9, 7-8, 5-10.
    A soft 15 (A-4) does NOT qualify — the ace disqualifies it.

    This is relevant for the dealer surrender rule: dealer may surrender on
    hard 15, overriding ALL settlements.

    Examples:
        >>> is_hard_fifteen((str_to_card('6C'), str_to_card('9H')))  # 6-9 = 15, no ace
        True
        >>> is_hard_fifteen((str_to_card('7H'), str_to_card('8D')))  # 7-8 = 15, no ace
        True
        >>> is_hard_fifteen((str_to_card('AS'), str_to_card('4H')))  # A-4 = soft 15, not hard
        False
        >>> is_hard_fifteen((str_to_card('5C'), str_to_card('9H'), str_to_card('AC')))  # 3 cards
        False
    """
    if len(cards) != 2:
        return False
    ranks = (cards[0] // 4, cards[1] // 4)
    if RANK_ACE in ranks:
        return False
    total = sum(RANK_VALUES[r] for r in ranks)  # type: ignore[misc]
    return total == 15


# ─── Classification ───────────────────────────────────────────────────────────


def classify_hand(cards: tuple[int, ...]) -> str:
    """Classify a hand into its type string.

    Checks in hierarchy order — first match wins. Bust hands return 'bust'.
    All other non-special hands (including regular 21) return 'regular'.

    Args:
        cards: Tuple of card integers.

    Returns:
        One of: 'ban_ban', 'ban_luck', '777', 'five_card_21',
                'five_card_sub21', 'regular', 'bust'.

    Examples:
        >>> classify_hand((48, 49))            # AC, AD
        'ban_ban'
        >>> classify_hand((48, 36))            # AC, JC
        'ban_luck'
        >>> classify_hand((20, 21, 22))        # 7C, 7D, 7H
        '777'
        >>> classify_hand((36, 40, 44, 8, 4))  # J, Q, K, 10, 3 = 33 — bust
        'bust'
    """
    total = calculate_total(cards)
    if total > 21:
        return HAND_BUST
    if is_ban_ban(cards):
        return HAND_BAN_BAN
    if is_ban_luck(cards):
        return HAND_BAN_LUCK
    if is_777(cards):
        return HAND_777
    if len(cards) == 5:
        # is_five_card_hand already confirmed total <= 21 above
        return HAND_FIVE_CARD_21 if total == 21 else HAND_FIVE_CARD_SUB21
    return HAND_REGULAR


def hand_hierarchy_rank(hand_type: str) -> int:
    """Return the numeric hierarchy rank for a hand type (lower = stronger).

    Used for hand comparison. Ban Ban = 1 (strongest), bust = 7 (weakest).

    Examples:
        >>> hand_hierarchy_rank('ban_ban')
        1
        >>> hand_hierarchy_rank('regular')
        6
        >>> hand_hierarchy_rank('bust')
        7
    """
    return _HIERARCHY[hand_type]
