"""Tests for src/engine/hand.py — total calculation and ace resolution."""

from __future__ import annotations

import pytest

from src.engine.cards import str_to_card
from src.engine.hand import calculate_total, is_bust, is_soft, resolve_ace
from tests.conftest import hand


# ─── resolve_ace tests ────────────────────────────────────────────────────────

class TestResolveAce:
    def test_two_card_hand_uses_11_if_no_bust(self):
        assert resolve_ace(2, 7) == 11    # 7+11=18, no bust

    def test_two_card_hand_uses_10_if_11_busts(self):
        assert resolve_ace(2, 11) == 10   # 11+11=22 busts, use 10

    def test_two_card_hand_uses_11_on_zero_total(self):
        assert resolve_ace(2, 0) == 11    # 0+11=11, no bust

    def test_two_card_hand_uses_10_on_total_11(self):
        # Ace+Ace: first ace takes 11, second ace: current=11, 11+11=22>21, use 10
        assert resolve_ace(2, 11) == 10

    def test_three_plus_hand_uses_10_if_no_bust(self):
        assert resolve_ace(3, 7) == 10    # 7+10=17, no bust

    def test_three_plus_hand_uses_1_if_10_busts(self):
        assert resolve_ace(3, 15) == 1    # 15+10=25>21, use 1

    def test_three_plus_hand_uses_10_on_zero_total(self):
        assert resolve_ace(3, 0) == 10

    def test_three_plus_hand_edge_at_11(self):
        # 11+10=21, exactly 21, no bust
        assert resolve_ace(3, 11) == 10

    def test_three_plus_hand_edge_at_12(self):
        # 12+10=22>21, use 1
        assert resolve_ace(3, 12) == 1

    def test_five_card_hand_uses_3plus_rules(self):
        # 5-card hand is still 3+ card
        assert resolve_ace(5, 7) == 10
        assert resolve_ace(5, 15) == 1


# ─── calculate_total tests ────────────────────────────────────────────────────

class TestCalculateTotal:
    # Two-card non-ace hands
    def test_two_card_ten_plus_nine(self):
        assert calculate_total(hand('10C', '9H')) == 19

    def test_two_card_king_plus_queen(self):
        assert calculate_total(hand('KH', 'QD')) == 20

    def test_two_card_seven_plus_eight(self):
        assert calculate_total(hand('7C', '8D')) == 15

    # Two-card ace hands
    def test_ace_plus_seven_two_card(self):
        # A-7 with 2-card rule: 11+7=18
        assert calculate_total(hand('AS', '7H')) == 18

    def test_ace_plus_ten_two_card(self):
        # Ban Luck: A-10 -> 11+10=21
        assert calculate_total(hand('AS', '10C')) == 21

    def test_ace_plus_jack_two_card(self):
        # A-J -> 11+10=21
        assert calculate_total(hand('AC', 'JD')) == 21

    def test_ace_plus_king_two_card(self):
        assert calculate_total(hand('AC', 'KH')) == 21

    def test_ace_plus_ace_two_card(self):
        # Ban Ban: A-A -> 11+10=21
        assert calculate_total(hand('AC', 'AS')) == 21

    def test_ace_plus_four_two_card(self):
        # A-4 (soft 15): 11+4=15
        assert calculate_total(hand('AC', '4H')) == 15

    def test_ace_plus_nine_two_card(self):
        # A-9: 11+9=20
        assert calculate_total(hand('AS', '9D')) == 20

    # Three+ card ace hands
    def test_ace_seven_three_three_card(self):
        # A-7-3: non-aces=10, ace: 10+10=20<=21, add 10 -> 20
        assert calculate_total(hand('AS', '7H', '3D')) == 20

    def test_ace_seven_eight_three_card(self):
        # A-7-8: non-aces=15, ace: 15+10=25>21, add 1 -> 16
        assert calculate_total(hand('AS', '7H', '8D')) == 16

    def test_ace_five_five_three_card(self):
        # A-5-5: non-aces=10, ace: 10+10=20<=21, add 10 -> 20
        assert calculate_total(hand('AS', '5C', '5D')) == 20

    def test_ace_ten_ten_three_card(self):
        # A-10-10: non-aces=20, ace: 20+10=30>21, add 1 -> 21
        assert calculate_total(hand('AC', '10C', '10D')) == 21

    # Bust hands
    def test_ten_plus_ten_plus_five_bust(self):
        total = calculate_total(hand('10C', '10D', '5H'))
        assert total == 25
        assert is_bust(total)

    def test_king_queen_jack_bust(self):
        total = calculate_total(hand('KH', 'QD', 'JC'))
        assert total == 30
        assert is_bust(total)

    # Multiple aces in 3+ card hand
    def test_two_aces_plus_two_three_card(self):
        # A-A-2: non-aces=2, ace1: 2+10=12 -> 10, ace2: 12+10=22>21 -> 1. Total=13
        assert calculate_total(hand('AC', 'AS', '2D')) == 13

    def test_two_aces_plus_seven_three_card(self):
        # A-A-7: non-aces=7, ace1: 7+10=17 -> 10, ace2: 17+10=27>21 -> 1. Total=18
        assert calculate_total(hand('AC', 'AS', '7D')) == 18

    def test_three_aces_three_card(self):
        # A-A-A: non-aces=0, ace1: 0+10=10, ace2: 10+10=20, ace3: 20+10=30>21 -> 1. Total=21
        assert calculate_total(hand('AC', 'AD', 'AS')) == 21

    # Five-card hands
    def test_five_card_no_bust(self):
        # 2+3+4+5+6=20
        assert calculate_total(hand('2C', '3D', '4H', '5S', '6C')) == 20

    def test_five_card_exactly_21(self):
        # 2+3+4+5+7=21
        assert calculate_total(hand('2C', '3D', '4H', '5S', '7C')) == 21

    def test_five_card_bust(self):
        total = calculate_total(hand('10C', 'JD', 'QH', 'KS', '2C'))
        assert total > 21


# ─── is_bust tests ────────────────────────────────────────────────────────────

class TestIsBust:
    def test_21_not_bust(self):
        assert not is_bust(21)

    def test_22_is_bust(self):
        assert is_bust(22)

    def test_20_not_bust(self):
        assert not is_bust(20)

    def test_zero_not_bust(self):
        assert not is_bust(0)


# ─── is_soft tests ────────────────────────────────────────────────────────────

class TestIsSoft:
    def test_ace_six_two_card_is_soft(self):
        # A-6 two-card: 11+6=17, ace counted as 11 -> soft
        assert is_soft(hand('AS', '6H'))

    def test_ace_four_two_card_is_soft(self):
        # A-4 two-card: 11+4=15 -> soft
        assert is_soft(hand('AC', '4H'))

    def test_ace_king_two_card_is_soft(self):
        # Ban Luck: A-K two-card, 11+10=21 -> soft
        assert is_soft(hand('AC', 'KH'))

    def test_ace_ace_two_card_is_soft(self):
        # Ban Ban: A-A two-card (first ace=11, which doesn't bust) -> soft
        assert is_soft(hand('AC', 'AS'))

    def test_no_ace_not_soft(self):
        assert not is_soft(hand('7C', '8D'))
        assert not is_soft(hand('10C', '9H', '2D'))

    def test_ace_seven_eight_not_soft(self):
        # A-7-8: ace must be 1 (1+7+8=16), not soft
        assert not is_soft(hand('AS', '7H', '8D'))

    def test_ace_five_three_card_is_soft(self):
        # A-5-3: non_aces=8, 8+10=18<=21, ace counted as 10 -> soft
        assert is_soft(hand('AS', '5H', '3D'))
