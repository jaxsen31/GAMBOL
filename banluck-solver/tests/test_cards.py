"""Tests for src/engine/cards.py — card constants, encoding, and string I/O."""

from __future__ import annotations

import pytest

from src.engine.cards import (
    RANK_ACE,
    RANK_NAMES,
    RANK_SEVEN,
    RANK_VALUES,
    SUIT_NAMES,
    TEN_VALUE_RANKS,
    card_rank,
    card_suit,
    card_to_str,
    card_value,
    hand_to_str,
    str_to_card,
)


class TestCardEncoding:
    def test_total_cards(self):
        """Cards 0–51 represent all 52 cards."""
        assert len(range(52)) == 52

    def test_rank_range(self):
        for card in range(52):
            assert 0 <= card_rank(card) <= 12

    def test_suit_range(self):
        for card in range(52):
            assert 0 <= card_suit(card) <= 3

    def test_two_of_clubs_is_card_zero(self):
        assert card_rank(0) == 0  # rank 0 = '2'
        assert card_suit(0) == 0  # suit 0 = 'C'

    def test_ace_of_spades_is_card_51(self):
        assert card_rank(51) == RANK_ACE  # 12
        assert card_suit(51) == 3  # 'S'

    def test_ace_of_clubs_is_card_48(self):
        assert card_rank(48) == RANK_ACE
        assert card_suit(48) == 0  # 'C'

    def test_seven_rank_index(self):
        """7 is rank index 5 (0=2, 1=3, 2=4, 3=5, 4=6, 5=7)."""
        assert RANK_SEVEN == 5
        assert RANK_VALUES[RANK_SEVEN] == 7

    def test_unique_cards(self):
        """Each card integer encodes a unique rank+suit combination."""
        pairs = {(card_rank(c), card_suit(c)) for c in range(52)}
        assert len(pairs) == 52

    def test_four_cards_per_rank(self):
        for rank in range(13):
            rank_cards = [c for c in range(52) if card_rank(c) == rank]
            assert len(rank_cards) == 4


class TestCardValue:
    def test_two_to_nine_face_value(self):
        expected = [2, 3, 4, 5, 6, 7, 8, 9]
        for i, exp in enumerate(expected):
            card = i * 4  # rank i, suit 0 (clubs)
            assert card_value(card) == exp, f"Rank {i} should have value {exp}"

    def test_ten_jack_queen_king_are_ten(self):
        for rank_idx in [8, 9, 10, 11]:  # 10, J, Q, K
            card = rank_idx * 4
            assert card_value(card) == 10, f"Rank {rank_idx} should be 10"

    def test_ace_value_is_none(self):
        for suit in range(4):
            ace = 12 * 4 + suit
            assert card_value(ace) is None

    def test_rank_values_length(self):
        assert len(RANK_VALUES) == 13

    def test_ten_value_ranks(self):
        assert TEN_VALUE_RANKS == {8, 9, 10, 11}
        for rank in TEN_VALUE_RANKS:
            assert RANK_VALUES[rank] == 10


class TestCardToStr:
    def test_two_of_clubs(self):
        assert card_to_str(0) == "2C"

    def test_ace_of_spades(self):
        assert card_to_str(51) == "AS"

    def test_ace_of_clubs(self):
        assert card_to_str(48) == "AC"

    def test_ten_of_clubs(self):
        assert card_to_str(32) == "10C"

    def test_jack_of_diamonds(self):
        assert card_to_str(37) == "JD"

    def test_seven_of_hearts(self):
        # 7 is rank index 5, heart is suit 2 -> 5*4+2=22
        assert card_to_str(22) == "7H"

    def test_roundtrip_all_cards(self):
        for card in range(52):
            assert str_to_card(card_to_str(card)) == card


class TestStrToCard:
    def test_two_of_clubs(self):
        assert str_to_card("2C") == 0

    def test_ace_of_spades(self):
        assert str_to_card("AS") == 51

    def test_ten_of_clubs(self):
        assert str_to_card("10C") == 32

    def test_seven_of_hearts(self):
        assert str_to_card("7H") == 22

    def test_king_of_spades(self):
        # K is rank 11, S is suit 3 -> 11*4+3=47
        assert str_to_card("KS") == 47

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError):
            str_to_card("1C")  # '1' is not a valid rank

    def test_invalid_suit_raises(self):
        with pytest.raises(ValueError):
            str_to_card("AX")  # 'X' is not a valid suit

    def test_all_suits(self):
        for suit_idx, suit_char in enumerate(SUIT_NAMES):
            card = str_to_card(f"A{suit_char}")
            assert card_suit(card) == suit_idx

    def test_all_ranks(self):
        for rank_idx, rank_char in enumerate(RANK_NAMES):
            card = str_to_card(f"{rank_char}C")
            assert card_rank(card) == rank_idx


class TestHandToStr:
    def test_two_card_hand(self):
        result = hand_to_str((48, 51))  # AC AS
        assert result == "AC AS"

    def test_single_card(self):
        assert hand_to_str((0,)) == "2C"

    def test_empty_hand(self):
        assert hand_to_str(()) == ""
