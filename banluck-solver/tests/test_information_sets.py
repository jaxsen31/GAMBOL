"""
Tests for src/solvers/information_sets.py

Covers:
    - PlayerHitStandInfoSet: construction, legal actions, edge cases
    - DealerSurrenderInfoSet: construction, hard-15 detection, legal actions
    - DealerActionInfoSet: construction, REVEAL_PLAYER legality, totals
    - make_* factory functions: round-trips from card integers
"""

from __future__ import annotations

import pytest

from src.engine.cards import str_to_card
from src.engine.game_state import DealerAction, PlayerAction
from src.solvers.information_sets import (
    DealerActionInfoSet,
    DealerSurrenderInfoSet,
    PlayerHitStandInfoSet,
    get_legal_dealer_actions,
    get_legal_dealer_surrender_actions,
    get_legal_player_actions,
    make_dealer_action_info_set,
    make_dealer_surrender_info_set,
    make_player_info_set,
)


def hand(*card_strs: str) -> tuple[int, ...]:
    return tuple(str_to_card(s) for s in card_strs)


# ─── TestPlayerHitStandInfoSet ────────────────────────────────────────────────

class TestPlayerHitStandInfoSet:

    def test_2card_hard_hand(self):
        """9+8 = hard 17, nc=2, not soft."""
        info = PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=False)
        assert info.total == 17
        assert info.num_cards == 2
        assert info.is_soft is False

    def test_2card_soft_hand(self):
        """A+6 = soft 17, nc=2, is_soft=True."""
        info = PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=True)
        assert info.total == 17
        assert info.num_cards == 2
        assert info.is_soft is True

    def test_3card_hand_ace_as_1(self):
        """A+9+5 = 15 (ace = 1 in 3-card hand), not soft."""
        info = PlayerHitStandInfoSet(total=15, num_cards=3, is_soft=False)
        assert info.total == 15
        assert info.num_cards == 3
        assert info.is_soft is False

    def test_5card_hand(self):
        """5-card hand has nc=5."""
        info = PlayerHitStandInfoSet(total=18, num_cards=5, is_soft=False)
        assert info.num_cards == 5

    def test_legal_actions_total_21(self):
        """Stand forced on 21."""
        info = PlayerHitStandInfoSet(total=21, num_cards=2, is_soft=False)
        assert get_legal_player_actions(info) == [PlayerAction.STAND]

    def test_legal_actions_5cards(self):
        """Stand forced at 5 cards (max hand)."""
        info = PlayerHitStandInfoSet(total=18, num_cards=5, is_soft=False)
        assert get_legal_player_actions(info) == [PlayerAction.STAND]

    def test_legal_actions_total_16_nc2(self):
        """16 with 2 cards: HIT and STAND both legal."""
        info = PlayerHitStandInfoSet(total=16, num_cards=2, is_soft=False)
        actions = get_legal_player_actions(info)
        assert PlayerAction.HIT in actions
        assert PlayerAction.STAND in actions

    def test_legal_actions_total_15(self):
        """Total ≤15: HIT and STAND both legal (forfeit happens at showdown)."""
        info = PlayerHitStandInfoSet(total=15, num_cards=2, is_soft=False)
        actions = get_legal_player_actions(info)
        assert PlayerAction.HIT in actions
        assert PlayerAction.STAND in actions

    def test_legal_actions_bust(self):
        """Bust (>21) returns empty list — terminal node."""
        info = PlayerHitStandInfoSet(total=22, num_cards=3, is_soft=False)
        assert get_legal_player_actions(info) == []

    def test_hashable_as_dict_key(self):
        """NamedTuple must be usable as a dict key for CFR regret tables."""
        info = PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=True)
        d = {info: 0.5}
        assert d[info] == 0.5

    def test_equality(self):
        """Two infosets with same values should be equal."""
        a = PlayerHitStandInfoSet(total=18, num_cards=3, is_soft=False)
        b = PlayerHitStandInfoSet(total=18, num_cards=3, is_soft=False)
        assert a == b

    def test_legal_actions_total_11(self):
        """Total ≤ 11: HIT and STAND both legal (can't bust by hitting)."""
        info = PlayerHitStandInfoSet(total=11, num_cards=2, is_soft=False)
        actions = get_legal_player_actions(info)
        assert PlayerAction.HIT in actions
        assert PlayerAction.STAND in actions


# ─── TestDealerSurrenderInfoSet ───────────────────────────────────────────────

class TestDealerSurrenderInfoSet:

    def test_hard_15_seven_eight(self):
        """7+8 = hard 15, is_hard_fifteen=True."""
        info = DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
        assert info.total == 15
        assert info.is_hard_fifteen is True

    def test_hard_15_six_nine(self):
        """6+9 = hard 15, is_hard_fifteen=True."""
        info = DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
        assert info.is_hard_fifteen is True

    def test_soft_15_ace_four(self):
        """A+4 = soft 15, NOT a hard 15 (ace disqualifies)."""
        info = DealerSurrenderInfoSet(total=15, is_hard_fifteen=False)
        assert info.is_hard_fifteen is False

    def test_hard_16_not_eligible(self):
        """7+9 = 16, is_hard_fifteen=False."""
        info = DealerSurrenderInfoSet(total=16, is_hard_fifteen=False)
        assert info.is_hard_fifteen is False

    def test_total_14_not_eligible(self):
        """Total 14 is not hard 15."""
        info = DealerSurrenderInfoSet(total=14, is_hard_fifteen=False)
        assert info.is_hard_fifteen is False

    def test_hashable(self):
        """DealerSurrenderInfoSet is hashable."""
        info = DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
        d = {info: "surrender"}
        assert d[info] == "surrender"

    def test_surrender_in_legal_actions_for_hard_15(self):
        """Hard 15: SURRENDER must be a legal action."""
        info = DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
        actions = get_legal_dealer_surrender_actions(info)
        assert DealerAction.SURRENDER in actions

    def test_surrender_not_in_legal_actions_for_non_hard_15(self):
        """Non-hard-15: SURRENDER must NOT be a legal action."""
        info = DealerSurrenderInfoSet(total=16, is_hard_fifteen=False)
        actions = get_legal_dealer_surrender_actions(info)
        assert DealerAction.SURRENDER not in actions


# ─── TestDealerActionInfoSet ──────────────────────────────────────────────────

class TestDealerActionInfoSet:

    def test_16_player_nc3_has_reveal(self):
        """At dealer 16 with 3+-card player, REVEAL_PLAYER is legal."""
        info = DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=3)
        assert DealerAction.REVEAL_PLAYER in get_legal_dealer_actions(info)

    def test_16_player_nc2_no_reveal(self):
        """At dealer 16 with 2-card player, REVEAL_PLAYER is NOT legal."""
        info = DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=2)
        assert DealerAction.REVEAL_PLAYER not in get_legal_dealer_actions(info)

    def test_17_player_nc3_has_reveal(self):
        """At dealer 17 with 3+-card player, REVEAL_PLAYER is legal."""
        info = DealerActionInfoSet(dealer_total=17, dealer_nc=2, is_soft=False, player_nc=3)
        assert DealerAction.REVEAL_PLAYER in get_legal_dealer_actions(info)

    def test_17_player_nc2_no_reveal(self):
        """At dealer 17 with 2-card player, REVEAL_PLAYER is NOT legal."""
        info = DealerActionInfoSet(dealer_total=17, dealer_nc=2, is_soft=False, player_nc=2)
        assert DealerAction.REVEAL_PLAYER not in get_legal_dealer_actions(info)

    def test_below_16_forced_hit(self):
        """Dealer below 16: only HIT (forced to reach minimum 16)."""
        info = DealerActionInfoSet(dealer_total=15, dealer_nc=2, is_soft=False, player_nc=3)
        assert get_legal_dealer_actions(info) == [DealerAction.HIT]

    def test_18_stand_only(self):
        """Dealer at 18: STAND only."""
        info = DealerActionInfoSet(dealer_total=18, dealer_nc=2, is_soft=False, player_nc=3)
        assert get_legal_dealer_actions(info) == [DealerAction.STAND]

    def test_20_stand_only(self):
        """Dealer at 20: STAND only."""
        info = DealerActionInfoSet(dealer_total=20, dealer_nc=2, is_soft=False, player_nc=3)
        assert get_legal_dealer_actions(info) == [DealerAction.STAND]

    def test_21_stand_only(self):
        """Dealer at 21: STAND only."""
        info = DealerActionInfoSet(dealer_total=21, dealer_nc=2, is_soft=False, player_nc=2)
        assert get_legal_dealer_actions(info) == [DealerAction.STAND]

    def test_hashable(self):
        """DealerActionInfoSet is hashable."""
        info = DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=3)
        d = {info: [0.33, 0.33, 0.34]}
        assert info in d

    def test_16_has_hit(self):
        """At dealer 16, HIT is always a legal action."""
        info = DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=2)
        assert DealerAction.HIT in get_legal_dealer_actions(info)

    def test_16_has_stand(self):
        """At dealer 16, STAND is always a legal action."""
        info = DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=2)
        assert DealerAction.STAND in get_legal_dealer_actions(info)

    def test_16_player_nc3_exactly_three_actions(self):
        """At dealer 16 with 3+-card player: exactly 3 legal actions."""
        info = DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=3)
        assert len(get_legal_dealer_actions(info)) == 3


# ─── TestMakeHelpers ──────────────────────────────────────────────────────────

class TestMakeHelpers:

    def test_make_player_info_set_ban_luck(self):
        """A+K = Ban Luck, total=21, nc=2, soft."""
        cards = hand('AS', 'KC')
        info = make_player_info_set(cards)
        assert info.total == 21
        assert info.num_cards == 2
        assert isinstance(info, PlayerHitStandInfoSet)

    def test_make_player_info_set_3card(self):
        """3-card hand: nc=3."""
        cards = hand('5C', '6D', '7H')  # 5+6+7 = 18
        info = make_player_info_set(cards)
        assert info.num_cards == 3
        assert info.total == 18
        assert info.is_soft is False

    def test_make_dealer_surrender_info_set_hard_15(self):
        """7+8 = hard 15: is_hard_fifteen=True."""
        cards = hand('7C', '8D')
        info = make_dealer_surrender_info_set(cards)
        assert info.total == 15
        assert info.is_hard_fifteen is True

    def test_make_dealer_surrender_info_set_soft_15(self):
        """A+4 = soft 15: is_hard_fifteen=False."""
        cards = hand('AS', '4C')
        info = make_dealer_surrender_info_set(cards)
        assert info.total == 15
        assert info.is_hard_fifteen is False

    def test_make_dealer_action_info_set_16(self):
        """9+7 dealer at 16, player nc=3."""
        cards = hand('9C', '7D')
        info = make_dealer_action_info_set(cards, player_nc=3)
        assert info.dealer_total == 16
        assert info.dealer_nc == 2
        assert info.player_nc == 3
        assert isinstance(info, DealerActionInfoSet)

    def test_make_dealer_action_info_set_soft_17(self):
        """A+6 = soft 17, player nc=2."""
        cards = hand('AS', '6C')
        info = make_dealer_action_info_set(cards, player_nc=2)
        assert info.dealer_total == 17
        assert info.is_soft is True
        assert info.player_nc == 2

    def test_round_trip_player_legal_actions(self):
        """make_player_info_set → get_legal_player_actions round-trip."""
        cards = hand('9C', '8D')  # 17
        info = make_player_info_set(cards)
        actions = get_legal_player_actions(info)
        assert PlayerAction.HIT in actions
        assert PlayerAction.STAND in actions

    def test_return_types(self):
        """All make_* functions return the correct NamedTuple type."""
        p_info = make_player_info_set(hand('9C', '8D'))
        s_info = make_dealer_surrender_info_set(hand('7C', '8D'))
        a_info = make_dealer_action_info_set(hand('9C', '7D'), player_nc=3)

        assert type(p_info) is PlayerHitStandInfoSet
        assert type(s_info) is DealerSurrenderInfoSet
        assert type(a_info) is DealerActionInfoSet
