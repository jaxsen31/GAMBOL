"""Tests for src/engine/special_hands.py — special hand detection and classification."""

from __future__ import annotations

from src.engine.special_hands import (
    HAND_777,
    HAND_BAN_BAN,
    HAND_BAN_LUCK,
    HAND_BUST,
    HAND_FIVE_CARD_21,
    HAND_FIVE_CARD_SUB21,
    HAND_REGULAR,
    PAYOUT_MULTIPLIERS,
    classify_hand,
    hand_hierarchy_rank,
    is_777,
    is_ban_ban,
    is_ban_luck,
    is_five_card_hand,
    is_hard_fifteen,
)
from tests.conftest import hand

# ─── is_ban_ban ───────────────────────────────────────────────────────────────


class TestIsBanBan:
    def test_two_aces_same_suit_is_ban_ban(self):
        # All four aces: any two aces = Ban Ban
        assert is_ban_ban(hand("AC", "AD"))

    def test_two_aces_different_suits_is_ban_ban(self):
        assert is_ban_ban(hand("AC", "AS"))
        assert is_ban_ban(hand("AH", "AD"))

    def test_ace_plus_face_is_not_ban_ban(self):
        assert not is_ban_ban(hand("AC", "KH"))

    def test_two_cards_no_ace_is_not_ban_ban(self):
        assert not is_ban_ban(hand("7C", "7D"))

    def test_three_aces_not_ban_ban(self):
        # Three aces is not a 2-card hand
        assert not is_ban_ban(hand("AC", "AD", "AH"))

    def test_single_ace_not_ban_ban(self):
        assert not is_ban_ban(
            hand(
                "AC",
            )
        )

    def test_empty_hand_not_ban_ban(self):
        assert not is_ban_ban(())


# ─── is_ban_luck ──────────────────────────────────────────────────────────────


class TestIsBanLuck:
    def test_ace_plus_ten_is_ban_luck(self):
        assert is_ban_luck(hand("AC", "10C"))

    def test_ace_plus_jack_is_ban_luck(self):
        assert is_ban_luck(hand("AS", "JH"))

    def test_ace_plus_queen_is_ban_luck(self):
        assert is_ban_luck(hand("AC", "QD"))

    def test_ace_plus_king_is_ban_luck(self):
        assert is_ban_luck(hand("AH", "KC"))

    def test_reversed_order_is_ban_luck(self):
        # Order should not matter
        assert is_ban_luck(hand("KH", "AC"))

    def test_two_aces_is_not_ban_luck(self):
        # Two aces is Ban Ban, not Ban Luck
        assert not is_ban_luck(hand("AC", "AD"))

    def test_ace_plus_nine_is_not_ban_luck(self):
        assert not is_ban_luck(hand("AC", "9H"))

    def test_ace_plus_two_is_not_ban_luck(self):
        assert not is_ban_luck(hand("AC", "2H"))

    def test_no_ace_not_ban_luck(self):
        assert not is_ban_luck(hand("KH", "QD"))

    def test_three_card_not_ban_luck(self):
        assert not is_ban_luck(hand("AC", "KH", "2D"))


# ─── is_777 ──────────────────────────────────────────────────────────────────


class TestIs777:
    def test_three_sevens_same_suits(self):
        # 7C, 7D, 7H
        assert is_777(hand("7C", "7D", "7H"))

    def test_three_sevens_all_suits(self):
        # 7C, 7D, 7S
        assert is_777(hand("7C", "7D", "7S"))

    def test_three_sevens_any_suit_combination(self):
        # Suit-independent
        assert is_777(hand("7S", "7C", "7D"))
        assert is_777(hand("7H", "7S", "7C"))

    def test_two_sevens_not_777(self):
        assert not is_777(hand("7C", "7D"))

    def test_three_cards_not_all_sevens(self):
        assert not is_777(hand("7C", "7D", "8H"))

    def test_four_sevens_not_777(self):
        # Impossible in one deck, but 4 cards != 3 cards
        assert not is_777(hand("7C", "7D", "7H", "7S"))

    def test_seven_seven_six_not_777(self):
        assert not is_777(hand("7C", "7D", "6H"))

    def test_empty_not_777(self):
        assert not is_777(())


# ─── is_five_card_hand ────────────────────────────────────────────────────────


class TestIsFiveCardHand:
    def test_five_cards_under_21(self):
        # 2+3+4+5+6=20
        assert is_five_card_hand(hand("2C", "3D", "4H", "5S", "6C"))

    def test_five_cards_exactly_21(self):
        # 2+3+4+5+7=21
        assert is_five_card_hand(hand("2C", "3D", "4H", "5S", "7C"))

    def test_five_cards_bust_not_five_card_hand(self):
        # 10+10+10+10+2=42 (bust)
        assert not is_five_card_hand(hand("10C", "10D", "10H", "10S", "2C"))

    def test_four_cards_not_five_card_hand(self):
        assert not is_five_card_hand(hand("2C", "3D", "4H", "5S"))

    def test_six_cards_not_five_card_hand(self):
        # 6 cards is illegal in the game, but the function checks len==5
        assert not is_five_card_hand(hand("2C", "3D", "4H", "5S", "6C", "2D"))


# ─── is_hard_fifteen ─────────────────────────────────────────────────────────


class TestIsHardFifteen:
    def test_six_nine_is_hard_15(self):
        assert is_hard_fifteen(hand("6C", "9H"))

    def test_seven_eight_is_hard_15(self):
        assert is_hard_fifteen(hand("7H", "8D"))

    def test_five_ten_is_hard_15(self):
        assert is_hard_fifteen(hand("5C", "10H"))

    def test_five_jack_is_hard_15(self):
        assert is_hard_fifteen(hand("5C", "JH"))

    def test_ace_four_is_soft_15_not_hard(self):
        # A-4 is soft 15, not hard
        assert not is_hard_fifteen(hand("AC", "4H"))

    def test_ace_four_reversed_not_hard(self):
        assert not is_hard_fifteen(hand("4H", "AC"))

    def test_seven_seven_is_14_not_15(self):
        assert not is_hard_fifteen(hand("7C", "7D"))

    def test_three_card_15_not_hard_15(self):
        # Hard 15 requires exactly 2 cards
        assert not is_hard_fifteen(hand("5C", "5H", "5D"))

    def test_two_card_16_not_hard_15(self):
        assert not is_hard_fifteen(hand("7C", "9H"))


# ─── classify_hand ────────────────────────────────────────────────────────────


class TestClassifyHand:
    def test_ban_ban(self):
        assert classify_hand(hand("AC", "AS")) == HAND_BAN_BAN

    def test_ban_luck_ace_king(self):
        assert classify_hand(hand("AC", "KH")) == HAND_BAN_LUCK

    def test_ban_luck_ace_ten(self):
        assert classify_hand(hand("AS", "10C")) == HAND_BAN_LUCK

    def test_777(self):
        assert classify_hand(hand("7C", "7D", "7H")) == HAND_777

    def test_five_card_21(self):
        assert classify_hand(hand("2C", "3D", "4H", "5S", "7C")) == HAND_FIVE_CARD_21

    def test_five_card_sub21(self):
        assert classify_hand(hand("2C", "3D", "4H", "5S", "6C")) == HAND_FIVE_CARD_SUB21

    def test_regular_21(self):
        # K+J+A (3-card) = 10+10+1=21 (regular, not special)
        assert classify_hand(hand("KH", "JD", "AC")) == HAND_REGULAR

    def test_regular_18(self):
        assert classify_hand(hand("10C", "8H")) == HAND_REGULAR

    def test_bust(self):
        assert classify_hand(hand("KH", "QD", "JC")) == HAND_BUST

    def test_five_card_bust(self):
        # 10+10+10+10+2 = 42, bust
        assert classify_hand(hand("10C", "10D", "10H", "10S", "2C")) == HAND_BUST


# ─── hand_hierarchy_rank ─────────────────────────────────────────────────────


class TestHandHierarchyRank:
    def test_ban_ban_is_strongest(self):
        ban_ban_rank = hand_hierarchy_rank(HAND_BAN_BAN)
        for other in [
            HAND_BAN_LUCK,
            HAND_777,
            HAND_FIVE_CARD_21,
            HAND_FIVE_CARD_SUB21,
            HAND_REGULAR,
            HAND_BUST,
        ]:
            assert ban_ban_rank < hand_hierarchy_rank(other)

    def test_hierarchy_order(self):
        order = [
            HAND_BAN_BAN,
            HAND_BAN_LUCK,
            HAND_777,
            HAND_FIVE_CARD_21,
            HAND_FIVE_CARD_SUB21,
            HAND_REGULAR,
            HAND_BUST,
        ]
        ranks = [hand_hierarchy_rank(h) for h in order]
        assert ranks == sorted(ranks), "Hierarchy should be strictly increasing"

    def test_bust_is_weakest(self):
        assert hand_hierarchy_rank(HAND_BUST) > hand_hierarchy_rank(HAND_REGULAR)


# ─── PAYOUT_MULTIPLIERS ──────────────────────────────────────────────────────


class TestPayoutMultipliers:
    def test_ban_ban_is_3(self):
        assert PAYOUT_MULTIPLIERS[HAND_BAN_BAN] == 3

    def test_ban_luck_is_2(self):
        assert PAYOUT_MULTIPLIERS[HAND_BAN_LUCK] == 2

    def test_777_is_7(self):
        assert PAYOUT_MULTIPLIERS[HAND_777] == 7

    def test_five_card_21_is_3(self):
        assert PAYOUT_MULTIPLIERS[HAND_FIVE_CARD_21] == 3

    def test_five_card_sub21_is_2(self):
        assert PAYOUT_MULTIPLIERS[HAND_FIVE_CARD_SUB21] == 2

    def test_regular_is_1(self):
        assert PAYOUT_MULTIPLIERS[HAND_REGULAR] == 1
