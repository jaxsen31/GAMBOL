"""
Hand evaluation: total calculation and ace resolution.

Banluck ace valuation is hand-size dependent:
    2-card hand:  Ace = 10 or 11 (choose highest that doesn't bust)
    3+ card hand: Ace = 1  or 10 (choose highest that doesn't bust)

This is the critical non-standard rule that must be applied consistently
everywhere totals are computed.

All functions operate on tuples of card integers (Numba-compatible).
"""

from __future__ import annotations

from .cards import RANK_VALUES, RANK_ACE


def resolve_ace(hand_size: int, current_total: int) -> int:
    """Choose the optimal ace value given the current hand size and running total.

    Args:
        hand_size: Total number of cards in the hand (including the ace being resolved).
        current_total: Sum of all non-ace cards and previously resolved aces,
                       before adding this ace.

    Returns:
        The ace value to use: 11 or 10 for 2-card hands, 10 or 1 for 3+ card hands.

    Examples:
        >>> resolve_ace(2, 7)    # A-7 two-card: 7+11=18 <= 21, use 11
        11
        >>> resolve_ace(2, 11)   # A-A two-card, second ace: 11+11=22 > 21, use 10
        10
        >>> resolve_ace(3, 7)    # A-x-7 three-card: 7+10=17 <= 21, use 10
        10
        >>> resolve_ace(3, 15)   # A-x-x three-card: 15+10=25 > 21, use 1
        1
    """
    if hand_size == 2:
        return 11 if current_total + 11 <= 21 else 10
    else:
        return 10 if current_total + 10 <= 21 else 1


def calculate_total(cards: tuple[int, ...]) -> int:
    """Calculate the best achievable total for a hand without busting, if possible.

    Aces are resolved greedily (highest non-busting value first).
    If even the minimum-value assignment busts, returns the minimum possible total
    (all aces at their lowest value), which will be > 21.

    Args:
        cards: Tuple of card integers (0–51).

    Returns:
        Best total <= 21, or minimum bust total if hand is unavoidably bust.

    Examples:
        >>> calculate_total((str_to_card('AS'), str_to_card('7H')))  # A-7, 2-card
        18
        >>> calculate_total((str_to_card('AC'), str_to_card('AS')))  # A-A, Ban Ban
        21
        >>> calculate_total((str_to_card('AS'), str_to_card('7H'), str_to_card('3D')))  # A-7-3, 3-card
        21
        >>> calculate_total((str_to_card('KH'), str_to_card('QD'), str_to_card('5C')))  # K-Q-5 bust
        25
    """
    hand_size = len(cards)

    # Separate aces from non-aces
    non_ace_total = 0
    num_aces = 0
    for card in cards:
        rank = card // 4
        if rank == RANK_ACE:
            num_aces += 1
        else:
            non_ace_total += RANK_VALUES[rank]  # type: ignore[operator]

    # Greedily add each ace at the highest non-busting value
    total = non_ace_total
    for _ in range(num_aces):
        total += resolve_ace(hand_size, total)

    return total


def is_bust(total: int) -> bool:
    """Return True if a total exceeds 21 (bust).

    Examples:
        >>> is_bust(21)
        False
        >>> is_bust(22)
        True
    """
    return total > 21


def is_soft(cards: tuple[int, ...]) -> bool:
    """Return True if the hand has an ace counted at its high value (soft hand).

    For 2-card hands: soft means ace is counted as 11.
    For 3+ card hands: soft means ace is counted as 10.

    A hand is soft if removing the high-value ace adjustment still gives ≤21.
    Equivalently: if there's an ace AND the total with ace at high value is ≤21.

    Examples:
        >>> is_soft((str_to_card('AS'), str_to_card('6H')))  # A-6 two-card: total 17, soft
        True
        >>> is_soft((str_to_card('AS'), str_to_card('KH')))  # A-K two-card (Ban Luck): total 21, soft
        True
        >>> is_soft((str_to_card('AS'), str_to_card('7H'), str_to_card('8D')))  # A-7-8 three-card: ace must be 1, hard
        False
    """
    hand_size = len(cards)
    num_aces = sum(1 for c in cards if c // 4 == RANK_ACE)
    if num_aces == 0:
        return False

    non_ace_total = sum(
        RANK_VALUES[c // 4]  # type: ignore[operator]
        for c in cards
        if c // 4 != RANK_ACE
    )

    # Check if at least one ace can be counted at the high value
    high_value = 11 if hand_size == 2 else 10
    return non_ace_total + high_value <= 21
