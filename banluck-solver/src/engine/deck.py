"""
Deck creation and card dealing operations.

The deck is a numpy int8 array of length 52.
    1 = card is available in the deck
    0 = card has been dealt

Integer encoding: card // 4 = rank index, card % 4 = suit index.
This representation is Numba-compatible for hot simulation loops.
"""

from __future__ import annotations

import numpy as np

from .cards import str_to_card


def create_deck() -> np.ndarray:
    """Create a fresh, full 52-card deck.

    Returns:
        np.ndarray: int8 array of shape (52,), all 1s (all cards available).

    Examples:
        >>> deck = create_deck()
        >>> deck.sum()
        52
        >>> deck.dtype
        dtype('int8')
    """
    return np.ones(52, dtype=np.int8)


def available_cards(deck: np.ndarray) -> np.ndarray:
    """Return the indices of cards still available in the deck.

    Examples:
        >>> deck = create_deck()
        >>> len(available_cards(deck))
        52
    """
    return np.where(deck == 1)[0]


def cards_remaining(deck: np.ndarray) -> int:
    """Return the count of cards still available in the deck.

    Examples:
        >>> deck = create_deck()
        >>> cards_remaining(deck)
        52
    """
    return int(deck.sum())


def deal_card(deck: np.ndarray) -> int:
    """Draw one random available card from the deck and mark it as dealt.

    Args:
        deck: Mutable deck array — modified in place.

    Returns:
        The integer index of the dealt card.

    Raises:
        ValueError: If the deck is empty.

    Examples:
        >>> deck = create_deck()
        >>> card = deal_card(deck)
        >>> 0 <= card <= 51
        True
        >>> cards_remaining(deck)
        51
    """
    avail = available_cards(deck)
    if len(avail) == 0:
        raise ValueError("Cannot deal from an empty deck.")
    card = int(np.random.choice(avail))
    deck[card] = 0
    return card


def deal_specific_card(deck: np.ndarray, card: int) -> None:
    """Mark a specific card as dealt without randomness.

    Used for deterministic test setups and solver initialisation.

    Args:
        deck: Mutable deck array — modified in place.
        card: Integer index (0–51) of the card to deal.

    Raises:
        ValueError: If the card is not available in the deck.

    Examples:
        >>> deck = create_deck()
        >>> deal_specific_card(deck, 0)  # deal 2C
        >>> deck[0]
        0
    """
    if deck[card] == 0:
        raise ValueError(f"Card {card} has already been dealt.")
    deck[card] = 0


def deal_specific_card_by_str(deck: np.ndarray, card_str: str) -> int:
    """Mark a specific card (given as string) as dealt.

    Convenience wrapper for test setup using human-readable card names.

    Args:
        deck: Mutable deck array — modified in place.
        card_str: Human-readable card string, e.g. 'AS', '7H', '10C'.

    Returns:
        The integer index of the dealt card.

    Examples:
        >>> deck = create_deck()
        >>> card = deal_specific_card_by_str(deck, 'AS')
        >>> card
        51
        >>> deck[51]
        0
    """
    card = str_to_card(card_str)
    deal_specific_card(deck, card)
    return card


def build_deck_from_hands(
    *hands: tuple[int, ...],
) -> np.ndarray:
    """Create a deck with all cards from the given hands already marked as dealt.

    Useful for building test states where specific hands have been dealt.

    Args:
        *hands: Any number of card tuples (player hand, dealer hand, etc.)

    Returns:
        np.ndarray: deck with those cards marked as dealt.

    Examples:
        >>> player = (51, 47)  # AS, KS
        >>> dealer = (48, 32)  # AC, 10C
        >>> deck = build_deck_from_hands(player, dealer)
        >>> cards_remaining(deck)
        48
    """
    deck = create_deck()
    for hand in hands:
        for card in hand:
            deal_specific_card(deck, card)
    return deck
