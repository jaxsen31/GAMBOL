"""Tests for src/engine/deck.py — deck creation and dealing operations."""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.deck import (
    available_cards,
    build_deck_from_hands,
    cards_remaining,
    create_deck,
    deal_card,
    deal_specific_card,
    deal_specific_card_by_str,
)
from src.engine.cards import str_to_card


class TestCreateDeck:
    def test_length(self):
        deck = create_deck()
        assert len(deck) == 52

    def test_all_ones(self):
        deck = create_deck()
        assert deck.sum() == 52

    def test_dtype_int8(self):
        deck = create_deck()
        assert deck.dtype == np.int8

    def test_immutable_between_calls(self):
        deck1 = create_deck()
        deck2 = create_deck()
        deck1[0] = 0
        assert deck2[0] == 1  # Modifying deck1 does not affect deck2


class TestAvailableCards:
    def test_full_deck(self):
        deck = create_deck()
        avail = available_cards(deck)
        assert len(avail) == 52
        assert set(avail) == set(range(52))

    def test_after_deal(self):
        deck = create_deck()
        deck[5] = 0
        avail = available_cards(deck)
        assert 5 not in avail
        assert len(avail) == 51


class TestCardsRemaining:
    def test_full_deck(self):
        assert cards_remaining(create_deck()) == 52

    def test_after_deal(self):
        deck = create_deck()
        deck[0] = 0
        deck[1] = 0
        assert cards_remaining(deck) == 50

    def test_empty_deck(self):
        deck = np.zeros(52, dtype=np.int8)
        assert cards_remaining(deck) == 0


class TestDealCard:
    def test_valid_card_index(self):
        deck = create_deck()
        card = deal_card(deck)
        assert 0 <= card <= 51

    def test_deck_decrements(self):
        deck = create_deck()
        deal_card(deck)
        assert cards_remaining(deck) == 51

    def test_card_marked_as_dealt(self):
        deck = create_deck()
        card = deal_card(deck)
        assert deck[card] == 0

    def test_deals_unique_cards(self):
        deck = create_deck()
        dealt = [deal_card(deck) for _ in range(52)]
        assert len(set(dealt)) == 52

    def test_empty_deck_raises(self):
        deck = np.zeros(52, dtype=np.int8)
        with pytest.raises(ValueError, match="empty"):
            deal_card(deck)

    def test_only_available_card_is_dealt(self):
        """When only one card remains, deal_card always returns that card."""
        deck = np.zeros(52, dtype=np.int8)
        deck[17] = 1
        card = deal_card(deck)
        assert card == 17


class TestDealSpecificCard:
    def test_marks_card_as_dealt(self):
        deck = create_deck()
        deal_specific_card(deck, 0)
        assert deck[0] == 0
        assert cards_remaining(deck) == 51

    def test_already_dealt_raises(self):
        deck = create_deck()
        deal_specific_card(deck, 5)
        with pytest.raises(ValueError):
            deal_specific_card(deck, 5)

    def test_deal_ace_of_spades(self):
        deck = create_deck()
        deal_specific_card(deck, 51)  # AS
        assert deck[51] == 0


class TestDealSpecificCardByStr:
    def test_basic(self):
        deck = create_deck()
        card = deal_specific_card_by_str(deck, 'AS')
        assert card == 51
        assert deck[51] == 0

    def test_ten_card(self):
        deck = create_deck()
        card = deal_specific_card_by_str(deck, '10C')
        assert card == 32
        assert deck[32] == 0


class TestBuildDeckFromHands:
    def test_two_card_hands(self):
        player = (str_to_card('AS'), str_to_card('KH'))
        dealer = (str_to_card('AC'), str_to_card('QD'))
        deck = build_deck_from_hands(player, dealer)
        assert cards_remaining(deck) == 48
        assert deck[str_to_card('AS')] == 0
        assert deck[str_to_card('KH')] == 0
        assert deck[str_to_card('AC')] == 0
        assert deck[str_to_card('QD')] == 0

    def test_single_hand(self):
        player = (str_to_card('7C'), str_to_card('7D'), str_to_card('7H'))
        deck = build_deck_from_hands(player)
        assert cards_remaining(deck) == 49

    def test_empty(self):
        deck = build_deck_from_hands()
        assert cards_remaining(deck) == 52

    def test_duplicate_card_raises(self):
        hand_a = (str_to_card('AS'),)
        hand_b = (str_to_card('AS'),)
        with pytest.raises(ValueError):
            build_deck_from_hands(hand_a, hand_b)
