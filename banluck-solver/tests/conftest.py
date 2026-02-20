"""
Shared pytest fixtures for Banluck solver tests.

Provides convenience wrappers around str_to_card for building known hands.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.cards import str_to_card
from src.engine.deck import build_deck_from_hands, create_deck


def hand(*card_strs: str) -> tuple[int, ...]:
    """Build a hand tuple from human-readable card strings.

    Examples:
        >>> hand('AS', 'AC')   # Ace of Spades, Ace of Clubs
        (51, 48)
        >>> hand('7C', '7D', '7H')
        (20, 21, 22)
    """
    return tuple(str_to_card(s) for s in card_strs)


@pytest.fixture
def fresh_deck() -> np.ndarray:
    """Return a full 52-card deck."""
    return create_deck()


@pytest.fixture
def h():
    """Expose the hand() helper as a fixture for convenience."""
    return hand
