"""
Card constants, encoding, and human-readable I/O helpers.

Card encoding (integer 0–51):
    rank_index = card // 4  ->  0=2, 1=3, ..., 7=9, 8=10, 9=J, 10=Q, 11=K, 12=A
    suit_index = card % 4   ->  0=C, 1=D, 2=H, 3=S

This integer-only encoding is Numba-compatible for hot loops.
String representations are used exclusively at I/O boundaries.
"""

from __future__ import annotations

# Rank value lookup: index matches rank_index.
# Ace (index 12) is None — always resolved contextually by resolve_ace().
RANK_VALUES: list[int | None] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, None]

RANK_NAMES: list[str] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUIT_NAMES: list[str] = ['C', 'D', 'H', 'S']

# Rank index for special ranks
RANK_ACE: int = 12   # index of Ace in RANK_VALUES / RANK_NAMES
RANK_TEN: int = 8    # index of 10 in RANK_VALUES
RANK_JACK: int = 9
RANK_QUEEN: int = 10
RANK_KING: int = 11
RANK_SEVEN: int = 5  # index of 7 (2=0, 3=1, 4=2, 5=3, 6=4, 7=5)

# Set of rank indices that count as 10 points (non-ace)
TEN_VALUE_RANKS: frozenset[int] = frozenset({RANK_TEN, RANK_JACK, RANK_QUEEN, RANK_KING})


def card_rank(card: int) -> int:
    """Return the rank index (0–12) of a card.

    Examples:
        >>> card_rank(0)   # 2 of Clubs
        0
        >>> card_rank(51)  # Ace of Spades
        12
    """
    return card // 4


def card_suit(card: int) -> int:
    """Return the suit index (0–3) of a card.

    Examples:
        >>> card_suit(0)   # 2 of Clubs
        0
        >>> card_suit(51)  # Ace of Spades
        3
    """
    return card % 4


def card_value(card: int) -> int | None:
    """Return the point value of a card, or None for Ace (context-dependent).

    Examples:
        >>> card_value(0)    # 2 of Clubs -> 2
        2
        >>> card_value(36)   # Jack of Clubs -> 10
        10
        >>> card_value(48)   # Ace of Clubs -> None
    """
    return RANK_VALUES[card // 4]


def card_to_str(card: int) -> str:
    """Convert a card integer to its human-readable string representation.

    Examples:
        >>> card_to_str(0)   # 2 of Clubs
        '2C'
        >>> card_to_str(51)  # Ace of Spades
        'AS'
        >>> card_to_str(32)  # 10 of Clubs
        '10C'
    """
    return RANK_NAMES[card // 4] + SUIT_NAMES[card % 4]


def str_to_card(s: str) -> int:
    """Parse a human-readable card string to its integer encoding.

    The format is <rank><suit> where suit is the last character.
    Rank can be '2'-'9', '10', 'J', 'Q', 'K', or 'A'.
    Suit can be 'C', 'D', 'H', or 'S'.

    Examples:
        >>> str_to_card('2C')
        0
        >>> str_to_card('AS')
        51
        >>> str_to_card('10C')
        32
        >>> str_to_card('7H')
        22
    """
    suit_char = s[-1]
    rank_str = s[:-1]
    rank = RANK_NAMES.index(rank_str)
    suit = SUIT_NAMES.index(suit_char)
    return rank * 4 + suit


def hand_to_str(cards: tuple[int, ...]) -> str:
    """Convert a hand (tuple of card ints) to a human-readable string.

    Examples:
        >>> hand_to_str((48, 51))
        'AC AS'
    """
    return ' '.join(card_to_str(c) for c in cards)
