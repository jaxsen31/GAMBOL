"""Tests for src/engine/rules.py — settlement and payout logic."""

from __future__ import annotations

import pytest

from src.engine.rules import Outcome, calculate_payout, settle_hand
from tests.conftest import hand


class TestSettleHandDealerSurrender:
    """Rule 1: Dealer hard-15 surrender overrides all outcomes."""

    def test_surrender_gives_push(self):
        player = hand('AC', 'AS')   # Ban Ban
        dealer = hand('6C', '9H')   # Hard 15
        outcome, payout = settle_hand(player, dealer, dealer_surrendered=True)
        assert outcome == Outcome.PUSH
        assert payout == 0.0

    def test_surrender_overrides_player_ban_ban(self):
        player = hand('AC', 'AS')
        dealer = hand('7H', '8D')
        outcome, payout = settle_hand(player, dealer, dealer_surrendered=True)
        assert outcome == Outcome.PUSH
        assert payout == 0.0

    def test_surrender_overrides_player_high_hand(self):
        player = hand('7C', '7D', '7H')   # 777
        dealer = hand('6C', '9H')
        outcome, payout = settle_hand(player, dealer, dealer_surrendered=True)
        assert outcome == Outcome.PUSH
        assert payout == 0.0

    def test_surrender_overrides_dealer_bust(self):
        player = hand('10C', '8H')  # 18
        dealer = hand('6C', '9H')
        outcome, payout = settle_hand(
            player, dealer, dealer_surrendered=True, dealer_busted=True
        )
        assert outcome == Outcome.PUSH
        assert payout == 0.0


class TestSettleHandPlayerBust:
    """Rule 2: Player bust loses 1 unit normally; 5-card bust loses 2 units."""

    def test_player_bust_loses_one_unit(self):
        player = hand('10C', 'KH', '5D')   # 25, bust
        dealer = hand('6C', '9H')          # 15
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_player_bust_loses_even_on_dealer_bust(self):
        player = hand('10C', 'KH', '5D')   # 25
        dealer = hand('10D', 'KS', '5H')   # also 25
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_player_five_card_bust_loses_two_units(self):
        """Five-card bust: symmetric penalty — player loses 2 units."""
        player = hand('10C', 'JD', 'QH', 'KS', '2C')  # 10+10+10+10+2 = 42, 5-card bust
        dealer = hand('9C', '8H')
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -2.0

    def test_player_five_card_bust_loses_two_on_dealer_bust(self):
        """5-card bust still costs 2 even when dealer busted."""
        player = hand('10C', 'JD', 'QH', 'KS', '2C')  # 5-card bust
        dealer = hand('10D', 'KH', '5S')               # also bust
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.LOSS
        assert payout == -2.0


class TestSettleHandPlayerForfeit:
    """Rule 3: Player ≤15 is an unconditional forfeit — no exceptions."""

    def test_player_fifteen_loses(self):
        player = hand('7C', '8H')   # 15
        dealer = hand('10C', '8D')  # 18
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_player_fourteen_loses(self):
        player = hand('6C', '8H')   # 14
        dealer = hand('7D', '8S')   # 15
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_player_fifteen_loses_even_on_dealer_bust(self):
        """Edge Case 8: Player ≤15 forfeits even when dealer busts."""
        player = hand('7C', '8H')   # 15
        dealer = hand('10C', 'KH')  # doesn't matter
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_player_fifteen_forfeits_vs_dealer_fifteen(self):
        """Edge Case 9: Player ≤15 forfeits regardless — not a push."""
        player = hand('7C', '8H')   # 15
        dealer = hand('7D', '8S')   # 15 (equal total)
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_player_sixteen_does_not_forfeit(self):
        player = hand('7C', '9H')   # 16
        dealer = hand('7D', '8S')   # 15
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 1.0


class TestSettleHandDealerBust:
    """Dealer bust: active players win at their hand type multiplier."""

    def test_regular_hand_wins_1_to_1_on_dealer_bust(self):
        player = hand('10C', '8H')  # 18
        dealer = hand('10D', '8S')  # doesn't matter, dealer busted
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.WIN
        assert payout == 1.0

    def test_ban_luck_wins_2_to_1_on_dealer_bust(self):
        player = hand('AC', 'KH')  # Ban Luck
        dealer = hand('10D', '8S')
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.WIN
        assert payout == 2.0

    def test_777_wins_7_to_1_on_dealer_bust(self):
        player = hand('7C', '7D', '7H')  # 777
        dealer = hand('10D', '8S')
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.WIN
        assert payout == 7.0

    def test_five_card_sub21_wins_2_to_1_on_dealer_bust(self):
        player = hand('2C', '3D', '4H', '5S', '6C')  # five-card 20
        dealer = hand('10D', '8S')
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.WIN
        assert payout == 2.0

    def test_dealer_five_card_bust_player_regular_wins_two(self):
        """Dealer 5-card bust: symmetric penalty — regular player wins 2 units."""
        player = hand('10C', '9H')                         # regular 19
        dealer = hand('10D', 'JC', 'QH', 'KS', '2C')      # 5-card bust
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.WIN
        assert payout == 2.0

    def test_dealer_five_card_bust_player_ban_ban_wins_three(self):
        """Player Ban Ban vs dealer 5-card bust: Ban Ban multiplier (3) takes priority."""
        player = hand('AC', 'AS')                          # Ban Ban
        dealer = hand('10D', 'JC', 'QH', 'KS', '2C')      # 5-card bust
        outcome, payout = settle_hand(player, dealer, dealer_busted=True)
        assert outcome == Outcome.WIN
        assert payout == 3.0


class TestSettleHandBanBanComparisons:
    def test_ban_ban_vs_regular_wins_3_to_1(self):
        player = hand('AC', 'AS')   # Ban Ban
        dealer = hand('10C', '9H')  # 19
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 3.0

    def test_ban_ban_vs_regular_21_wins(self):
        player = hand('AC', 'AS')   # Ban Ban (21)
        dealer = hand('10C', '8H', '3D')  # regular 21
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 3.0

    def test_ban_ban_vs_ban_ban_pushes(self):
        """Edge Case 10: Ban Ban vs Ban Ban = push."""
        player = hand('AC', 'AD')
        dealer = hand('AH', 'AS')
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.PUSH
        assert payout == 0.0

    def test_dealer_ban_ban_beats_player_regular(self):
        player = hand('10C', '8H')  # 18
        dealer = hand('AC', 'AS')   # Ban Ban
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -3.0

    def test_dealer_ban_ban_beats_player_ban_luck(self):
        player = hand('AH', 'KH')   # Ban Luck
        dealer = hand('AC', 'AS')   # Ban Ban
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -3.0


class TestSettleHandBanLuckComparisons:
    def test_ban_luck_vs_regular_wins_2_to_1(self):
        player = hand('AC', 'KH')   # Ban Luck
        dealer = hand('10C', '8D')  # 18
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 2.0

    def test_ban_luck_vs_ban_luck_pushes(self):
        """Edge Case 11: Ban Luck vs Ban Luck = push."""
        player = hand('AC', 'KH')   # Ban Luck
        dealer = hand('AS', 'QD')   # Ban Luck
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.PUSH
        assert payout == 0.0

    def test_dealer_ban_luck_beats_player_regular(self):
        player = hand('10C', '8H')  # 18
        dealer = hand('AS', 'QD')   # Ban Luck
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -2.0


class TestSettleHand777:
    def test_777_vs_regular_21_wins_7_to_1(self):
        """Edge Case 6: 777 beats dealer 21."""
        player = hand('7C', '7D', '7H')   # 777 (total=21)
        dealer = hand('JC', 'JD', 'AC')   # regular 21
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 7.0

    def test_777_vs_regular_20_wins(self):
        player = hand('7C', '7D', '7H')
        dealer = hand('10C', '10D')
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 7.0

    def test_dealer_ban_ban_beats_777(self):
        player = hand('7C', '7D', '7H')
        dealer = hand('AC', 'AS')
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -3.0


class TestSettleHandFiveCard:
    def test_five_card_sub21_beats_non_five_card_at_2_to_1(self):
        # 5-card 20 beats dealer 21 (non-five-card)
        player = hand('2C', '3D', '4H', '5S', '6C')   # five-card 20
        dealer = hand('10C', '8D', '3H')               # regular 21 = 21
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 2.0

    def test_five_card_21_beats_regular_21_at_3_to_1(self):
        player = hand('2C', '3D', '4H', '5S', '7C')   # five-card 21
        dealer = hand('10C', '8D', '3H')               # regular 21
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 3.0

    def test_five_card_vs_five_card_same_total_push(self):
        """Edge Case 4: Both player and dealer have 5-card hands with same total."""
        player = hand('2C', '3D', '4H', '5S', '6C')   # five-card 20
        dealer = hand('2D', '3H', '4S', '5C', '6D')   # five-card 20
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.PUSH
        assert payout == 0.0

    def test_five_card_vs_five_card_higher_wins_at_1_to_1(self):
        """Edge Case 5: Player five-card 20 vs dealer five-card 19 — player wins 1:1."""
        player = hand('2C', '3D', '4H', '5S', '6C')   # five-card 20
        dealer = hand('2D', '3H', '4S', '5C', '4D')   # five-card 18
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 1.0

    def test_five_card_vs_five_card_lower_loses_at_1_to_1(self):
        player = hand('2C', '3D', '4H', '5S', '4D')   # five-card 18
        dealer = hand('2D', '3H', '4S', '5C', '6D')   # five-card 20
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_dealer_five_card_beats_player_regular(self):
        """Dealer five-card bonus is symmetric — player pays 2:1."""
        player = hand('10C', '9H')                     # regular 19
        dealer = hand('2D', '3H', '4S', '5C', '6D')   # five-card 20
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -2.0

    def test_five_card_21_dealer_beats_player_regular(self):
        player = hand('10C', '9H')                     # regular 19
        dealer = hand('2D', '3H', '4S', '5C', '7D')   # five-card 21
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -3.0


class TestSettleHandRegular:
    def test_higher_total_wins_1_to_1(self):
        player = hand('10C', '9H')  # 19
        dealer = hand('10D', '7S')  # 17
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 1.0

    def test_lower_total_loses_1_to_1(self):
        player = hand('10C', '7H')  # 17
        dealer = hand('10D', '9S')  # 19
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.LOSS
        assert payout == -1.0

    def test_equal_totals_push(self):
        player = hand('10C', '8H')  # 18
        dealer = hand('10D', '8S')  # 18
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.PUSH
        assert payout == 0.0

    def test_player_21_vs_dealer_20_wins(self):
        player = hand('10C', '8H', '3D')  # 21
        dealer = hand('10D', '10S')       # 20
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 1.0

    def test_player_16_vs_dealer_15_wins(self):
        """Edge Case D: Dealer 16 vs Player 16 — standard win for higher total."""
        player = hand('7C', '9H')   # 16
        dealer = hand('7D', '8S')   # 15
        outcome, payout = settle_hand(player, dealer)
        assert outcome == Outcome.WIN
        assert payout == 1.0


class TestCalculatePayout:
    def test_win_1_unit_bet(self):
        assert calculate_payout(Outcome.WIN, 1.0, 10.0) == 10.0

    def test_loss_1_unit_bet(self):
        assert calculate_payout(Outcome.LOSS, -1.0, 10.0) == -10.0

    def test_push(self):
        assert calculate_payout(Outcome.PUSH, 0.0, 10.0) == 0.0

    def test_ban_ban_payout(self):
        assert calculate_payout(Outcome.WIN, 3.0, 5.0) == 15.0

    def test_777_payout(self):
        assert calculate_payout(Outcome.WIN, 7.0, 1.0) == 7.0
