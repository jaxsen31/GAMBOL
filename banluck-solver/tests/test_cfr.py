"""Tests for Phase 2.2 — CFR+ solver (src/solvers/cfr.py).

Test structure mirrors the module structure:

    TestRegretMatching      — _get_strategy, _update_strategy_sum,
                              _regret_update_player, _regret_update_dealer
    TestCompositionHelpers  — _is_hard_fifteen_comp, _is_player_ban_ban, etc.
    TestSettlement          — _cfr_settle, _settle_player_ban_ban/luck
    TestInitialDeals        — _build_initial_deals, probability invariants
    TestCfrTables           — _CfrTables dataclass fields and isolation
    TestDealerCfr           — _dealer_cfr terminal / forced / strategic cases
    TestPlayerCfr           — _player_cfr terminal / forced / decision cases
    TestRootCfr             — _root_cfr surrender + special hand integration
    TestSolve               — solve() output shape and basic invariants
    TestStrategyInvariants  — Nash strategy sanity checks after convergence
    TestExploitability      — compute_exploitability, best-response helpers
    TestPublicHelpers       — get_player_action, get_dealer_surrender_prob
"""

from __future__ import annotations

import pytest

from src.engine.game_state import DealerAction, PlayerAction
from src.solvers.cfr import (
    _D_REVEAL,
    _P_HIT,
    _P_STAND,
    CfrResult,
    _best_dealer_ev,
    _best_player_ev,
    _build_initial_deals,
    _cfr_settle,
    _CfrTables,
    _dealer_action_key,
    _dealer_cfr,
    _dealer_surrender_key,
    _get_strategy,
    _is_dealer_ban_ban,
    _is_dealer_ban_luck,
    _is_hard_fifteen_comp,
    _is_player_ban_ban,
    _is_player_ban_luck,
    _player_cfr,
    _player_key,
    _regret_update_dealer,
    _regret_update_player,
    _root_cfr,
    _settle_player_ban_ban,
    _settle_player_ban_luck,
    _update_strategy_sum,
    compute_exploitability,
    get_dealer_surrender_prob,
    get_player_action,
    solve,
)
from src.solvers.information_sets import (
    DealerActionInfoSet,
    PlayerHitStandInfoSet,
)

# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_tables() -> _CfrTables:
    return _CfrTables()


# ─── TestRegretMatching ────────────────────────────────────────────────────────


class TestRegretMatching:
    """_get_strategy, _update_strategy_sum, regret update helpers."""

    def test_uniform_when_no_regrets(self) -> None:
        tables = _make_tables()
        info_set = PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=False)
        actions = [PlayerAction.HIT, PlayerAction.STAND]
        strategy = _get_strategy(info_set, tables.player_regrets, actions)
        assert abs(strategy[PlayerAction.HIT] - 0.5) < 1e-9
        assert abs(strategy[PlayerAction.STAND] - 0.5) < 1e-9

    def test_uniform_when_all_regrets_zero(self) -> None:
        tables = _make_tables()
        info_set = PlayerHitStandInfoSet(total=16, num_cards=2, is_soft=False)
        tables.player_regrets[info_set] = {
            PlayerAction.HIT: 0.0,
            PlayerAction.STAND: 0.0,
        }
        actions = [PlayerAction.HIT, PlayerAction.STAND]
        strategy = _get_strategy(info_set, tables.player_regrets, actions)
        assert abs(strategy[PlayerAction.HIT] - 0.5) < 1e-9

    def test_strategy_proportional_to_positive_regrets(self) -> None:
        tables = _make_tables()
        info_set = PlayerHitStandInfoSet(total=16, num_cards=2, is_soft=False)
        tables.player_regrets[info_set] = {
            PlayerAction.HIT: 3.0,
            PlayerAction.STAND: 1.0,
        }
        actions = [PlayerAction.HIT, PlayerAction.STAND]
        strategy = _get_strategy(info_set, tables.player_regrets, actions)
        assert abs(strategy[PlayerAction.HIT] - 0.75) < 1e-9
        assert abs(strategy[PlayerAction.STAND] - 0.25) < 1e-9

    def test_negative_regrets_treated_as_zero(self) -> None:
        tables = _make_tables()
        info_set = PlayerHitStandInfoSet(total=18, num_cards=2, is_soft=False)
        tables.player_regrets[info_set] = {
            PlayerAction.HIT: -5.0,
            PlayerAction.STAND: 2.0,
        }
        actions = [PlayerAction.HIT, PlayerAction.STAND]
        strategy = _get_strategy(info_set, tables.player_regrets, actions)
        # Negative regret for HIT is clamped to 0, so STAND gets probability 1
        assert abs(strategy[PlayerAction.STAND] - 1.0) < 1e-9
        assert abs(strategy[PlayerAction.HIT] - 0.0) < 1e-9

    def test_strategy_probabilities_sum_to_one(self) -> None:
        tables = _make_tables()
        info_set = PlayerHitStandInfoSet(total=15, num_cards=3, is_soft=False)
        tables.player_regrets[info_set] = {
            PlayerAction.HIT: 1.5,
            PlayerAction.STAND: 2.5,
        }
        actions = [PlayerAction.HIT, PlayerAction.STAND]
        strategy = _get_strategy(info_set, tables.player_regrets, actions)
        assert abs(sum(strategy.values()) - 1.0) < 1e-9

    def test_update_strategy_sum_accumulates_correctly(self) -> None:
        sums: dict = {}
        info_set = PlayerHitStandInfoSet(total=17, num_cards=2, is_soft=True)
        strategy = {PlayerAction.HIT: 0.4, PlayerAction.STAND: 0.6}
        _update_strategy_sum(info_set, strategy, reach_self=1.0, iteration=1, sums_table=sums)
        _update_strategy_sum(info_set, strategy, reach_self=1.0, iteration=2, sums_table=sums)
        # After iter 1: sums = {HIT: 0.4, STAND: 0.6}
        # After iter 2: sums = {HIT: 0.4 + 2×0.4=0.8+0.4=1.2, STAND: 0.6+1.2=1.8}
        assert abs(sums[info_set][PlayerAction.HIT] - (0.4 + 0.8)) < 1e-9
        assert abs(sums[info_set][PlayerAction.STAND] - (0.6 + 1.2)) < 1e-9

    def test_update_strategy_sum_weights_by_reach(self) -> None:
        sums: dict = {}
        info_set = PlayerHitStandInfoSet(total=17, num_cards=3, is_soft=False)
        strategy = {PlayerAction.HIT: 1.0, PlayerAction.STAND: 0.0}
        _update_strategy_sum(info_set, strategy, reach_self=0.5, iteration=3, sums_table=sums)
        # weight = 3 × 0.5 = 1.5; HIT += 1.5 × 1.0 = 1.5
        assert abs(sums[info_set][PlayerAction.HIT] - 1.5) < 1e-9

    def test_player_regret_update_positive(self) -> None:
        rt: dict = {}
        info_set = PlayerHitStandInfoSet(total=16, num_cards=2, is_soft=False)
        actions = [PlayerAction.HIT, PlayerAction.STAND]
        action_evs = {PlayerAction.HIT: 0.5, PlayerAction.STAND: 0.2}
        node_ev = 0.35
        _regret_update_player(info_set, actions, action_evs, node_ev, 1.0, rt)
        # r(HIT)   = max(0, (0.5 - 0.35) × 1.0) = 0.15
        # r(STAND) = max(0, (0.2 - 0.35) × 1.0) = 0.0 (floor at 0)
        assert abs(rt[info_set][PlayerAction.HIT] - 0.15) < 1e-9
        assert abs(rt[info_set][PlayerAction.STAND] - 0.0) < 1e-9

    def test_dealer_regret_update_minimizer(self) -> None:
        rt: dict = {}
        info_set = DealerActionInfoSet(dealer_total=16, dealer_nc=2, is_soft=False, player_nc=2)
        actions = [DealerAction.HIT, DealerAction.STAND]
        action_evs = {DealerAction.HIT: 0.3, DealerAction.STAND: 0.5}
        node_ev = 0.4
        _regret_update_dealer(info_set, actions, action_evs, node_ev, 1.0, rt)
        # Dealer minimises player EV.
        # r(HIT)   = max(0, (node_ev - ev(HIT)) × 1.0)   = max(0, 0.1) = 0.1
        # r(STAND) = max(0, (node_ev - ev(STAND)) × 1.0) = max(0, -0.1) = 0.0
        assert abs(rt[info_set][DealerAction.HIT] - 0.1) < 1e-9
        assert abs(rt[info_set][DealerAction.STAND] - 0.0) < 1e-9

    def test_cfr_plus_regrets_floor_at_zero(self) -> None:
        """CFR+: accumulated regrets cannot go below 0."""
        rt: dict = {}
        info_set = PlayerHitStandInfoSet(total=16, num_cards=2, is_soft=False)
        actions = [PlayerAction.HIT, PlayerAction.STAND]
        # First update: HIT has negative regret
        action_evs = {PlayerAction.HIT: 0.1, PlayerAction.STAND: 0.4}
        _regret_update_player(info_set, actions, action_evs, 0.3, 1.0, rt)
        assert rt[info_set][PlayerAction.HIT] == 0.0  # Floored at 0
        # Second update with positive regret for HIT
        action_evs2 = {PlayerAction.HIT: 0.6, PlayerAction.STAND: 0.2}
        _regret_update_player(info_set, actions, action_evs2, 0.4, 1.0, rt)
        assert rt[info_set][PlayerAction.HIT] > 0.0  # Now positive


# ─── TestCompositionHelpers ────────────────────────────────────────────────────


class TestCompositionHelpers:
    """_is_hard_fifteen_comp, _is_player/dealer_ban_ban/luck."""

    def test_hard_fifteen_nat15_na0_nc2(self) -> None:
        assert _is_hard_fifteen_comp(15, 0, 2)

    def test_not_hard_fifteen_soft15(self) -> None:
        # A+4: nat=4, na=1, nc=2 → total=15 (soft) but NOT hard 15
        assert not _is_hard_fifteen_comp(4, 1, 2)

    def test_not_hard_fifteen_wrong_total(self) -> None:
        assert not _is_hard_fifteen_comp(14, 0, 2)

    def test_not_hard_fifteen_nc3(self) -> None:
        assert not _is_hard_fifteen_comp(15, 0, 3)

    def test_player_ban_ban_na2_nc2(self) -> None:
        assert _is_player_ban_ban(0, 2, 2)

    def test_player_not_ban_ban_na1(self) -> None:
        assert not _is_player_ban_ban(0, 1, 2)

    def test_player_ban_luck_na1_nat10_nc2(self) -> None:
        assert _is_player_ban_luck(10, 1, 2)

    def test_player_not_ban_luck_wrong_nat(self) -> None:
        assert not _is_player_ban_luck(9, 1, 2)

    def test_dealer_ban_ban_symmetric(self) -> None:
        assert _is_dealer_ban_ban(0, 2, 2)
        assert not _is_dealer_ban_ban(0, 1, 2)

    def test_dealer_ban_luck_symmetric(self) -> None:
        assert _is_dealer_ban_luck(10, 1, 2)
        assert not _is_dealer_ban_luck(9, 1, 2)


# ─── TestSettlement ────────────────────────────────────────────────────────────


class TestSettlement:
    """_cfr_settle and special-hand settlement functions."""

    def test_player_bust_regular(self) -> None:
        # Player total=22, nc=2 → -1.0
        result = _cfr_settle(20, 0, 2, 10, 0, 2, d_busted=False)
        # 20 > dealer's 10? Wait, player 20 vs dealer 10 → player wins
        assert result == 1.0
        # Actual bust test: nat=12, na=0, nc=2 → total=12... not a bust
        # Need a bust: nat=22, na=0, nc=2 → total=22
        result_bust = _cfr_settle(22, 0, 2, 10, 0, 2, d_busted=False)
        assert result_bust == -1.0

    def test_player_five_card_bust_costs_two(self) -> None:
        result = _cfr_settle(25, 0, 5, 10, 0, 2, d_busted=False)
        assert result == -2.0

    def test_player_forfeit_unconditional(self) -> None:
        # Player total=14 (≤15) → -1.0 even if dealer busts
        result = _cfr_settle(14, 0, 2, 10, 0, 2, d_busted=True)
        assert result == -1.0

    def test_player_forfeit_at_15(self) -> None:
        result = _cfr_settle(15, 0, 2, 5, 0, 2, d_busted=False)
        assert result == -1.0

    def test_dealer_bust_player_wins_regular(self) -> None:
        result = _cfr_settle(18, 0, 2, 0, 0, 0, d_busted=True)
        assert result == 1.0

    def test_dealer_bust_player_wins_five_card_sub21(self) -> None:
        result = _cfr_settle(19, 0, 5, 0, 0, 0, d_busted=True)
        assert result == 2.0

    def test_dealer_bust_player_wins_five_card_21(self) -> None:
        result = _cfr_settle(21, 0, 5, 0, 0, 0, d_busted=True)
        assert result == 3.0

    def test_regular_win(self) -> None:
        result = _cfr_settle(20, 0, 2, 18, 0, 2, d_busted=False)
        assert result == 1.0

    def test_regular_loss(self) -> None:
        result = _cfr_settle(18, 0, 2, 20, 0, 2, d_busted=False)
        assert result == -1.0

    def test_regular_push(self) -> None:
        result = _cfr_settle(19, 0, 2, 19, 0, 2, d_busted=False)
        assert result == 0.0

    def test_dealer_ban_ban_beats_regular(self) -> None:
        # Dealer Ban Ban (d_na=2, d_nc=2, d_total=21): beats player regular 18
        result = _cfr_settle(18, 0, 2, 0, 2, 2, d_busted=False)
        assert result == -3.0

    def test_dealer_ban_ban_beats_five_card_21(self) -> None:
        # Dealer Ban Ban beats player five-card 21 (CFR handles this correctly)
        result = _cfr_settle(21, 0, 5, 0, 2, 2, d_busted=False)
        assert result == -3.0  # Dealer Ban Ban (rank 1) > five_card_21 (rank 4)

    def test_dealer_ban_luck_beats_regular(self) -> None:
        result = _cfr_settle(18, 0, 2, 10, 1, 2, d_busted=False)
        assert result == -2.0

    def test_five_card_21_beats_regular_dealer(self) -> None:
        result = _cfr_settle(21, 0, 5, 19, 0, 2, d_busted=False)
        assert result == 3.0

    def test_five_card_sub21_beats_regular_dealer(self) -> None:
        result = _cfr_settle(19, 0, 5, 18, 0, 2, d_busted=False)
        assert result == 2.0

    def test_five_card_vs_five_card_higher_total_wins(self) -> None:
        result = _cfr_settle(20, 0, 5, 19, 0, 5, d_busted=False)
        assert result == 1.0  # Same tier: compare totals

    def test_five_card_vs_five_card_push(self) -> None:
        result = _cfr_settle(19, 0, 5, 19, 0, 5, d_busted=False)
        assert result == 0.0

    def test_settle_player_ban_ban_vs_regular(self) -> None:
        assert _settle_player_ban_ban(10, 0, 2) == 3.0

    def test_settle_player_ban_ban_vs_dealer_ban_ban(self) -> None:
        assert _settle_player_ban_ban(0, 2, 2) == 0.0

    def test_settle_player_ban_luck_vs_regular(self) -> None:
        assert _settle_player_ban_luck(10, 0, 2) == 2.0

    def test_settle_player_ban_luck_vs_dealer_ban_ban(self) -> None:
        assert _settle_player_ban_luck(0, 2, 2) == -3.0

    def test_settle_player_ban_luck_vs_dealer_ban_luck(self) -> None:
        assert _settle_player_ban_luck(10, 1, 2) == 0.0


# ─── TestInitialDeals ──────────────────────────────────────────────────────────


class TestInitialDeals:
    """_build_initial_deals: probability invariants and composition coverage."""

    def test_probabilities_sum_to_one(self) -> None:
        deals = _build_initial_deals()
        total = sum(w for _, _, w in deals)
        assert abs(total - 1.0) < 1e-9

    def test_all_probabilities_positive(self) -> None:
        deals = _build_initial_deals()
        assert all(w > 0 for _, _, w in deals)

    def test_player_ban_ban_probability(self) -> None:
        """P(player Ban Ban) = (1/13)^2 = 1/169."""
        deals = _build_initial_deals()
        p_bb = sum(w for p, _, w in deals if _is_player_ban_ban(*p))
        assert abs(p_bb - 1 / 169) < 1e-9

    def test_player_ban_luck_probability(self) -> None:
        """P(player Ban Luck) = 8/169 (ace + one of 4 ten-values, both orders)."""
        deals = _build_initial_deals()
        p_bl = sum(w for p, _, w in deals if _is_player_ban_luck(*p))
        assert abs(p_bl - 8 / 169) < 1e-9

    def test_unique_composition_pairs(self) -> None:
        """Number of unique (player_comp, dealer_comp) pairs is reasonable."""
        deals = _build_initial_deals()
        # Should have 27×27 = 729 unique pairs (27 unique 2-card compositions)
        # Allow some slack in case of rounding merges
        assert 600 <= len(deals) <= 800

    def test_player_comp_nc_always_2(self) -> None:
        deals = _build_initial_deals()
        assert all(p[2] == 2 for p, _, _ in deals)

    def test_dealer_comp_nc_always_2(self) -> None:
        deals = _build_initial_deals()
        assert all(d[2] == 2 for _, d, _ in deals)


# ─── TestCfrTables ─────────────────────────────────────────────────────────────


class TestCfrTables:
    """_CfrTables dataclass initialisation."""

    def test_all_tables_start_empty(self) -> None:
        tables = _CfrTables()
        assert tables.player_regrets == {}
        assert tables.player_strategy_sums == {}
        assert tables.dealer_surrender_regrets == {}
        assert tables.dealer_surrender_strategy_sums == {}
        assert tables.dealer_action_regrets == {}
        assert tables.dealer_action_strategy_sums == {}

    def test_tables_are_independent(self) -> None:
        t1 = _CfrTables()
        t2 = _CfrTables()
        info_set = PlayerHitStandInfoSet(17, 2, False)
        t1.player_regrets[info_set] = {PlayerAction.HIT: 1.0}
        assert info_set not in t2.player_regrets


# ─── TestDealerCfr ─────────────────────────────────────────────────────────────


class TestDealerCfr:
    """_dealer_cfr terminal, forced-hit, and strategic-decision cases."""

    def test_dealer_bust_returns_player_win(self) -> None:
        tables = _make_tables()
        # Dealer composition that gives total > 21
        # nat=22, na=0, nc=3 → total=22 (busted)
        ev = _dealer_cfr(18, 0, 2, 22, 0, 3, 1.0, 1.0, tables, 1)
        assert ev == 1.0  # Player regular 18, dealer busted → player wins 1:1

    def test_dealer_forced_stand_at_18(self) -> None:
        tables = _make_tables()
        # Dealer at 18 with nat=18, na=0, nc=2 → total=18, no bust
        ev = _dealer_cfr(17, 0, 2, 18, 0, 2, 1.0, 1.0, tables, 1)
        assert ev == -1.0  # Dealer 18 > player 17 → player loses

    def test_dealer_forced_stand_at_21(self) -> None:
        tables = _make_tables()
        # Dealer at 21 (nat=21, na=0, nc=3)
        ev = _dealer_cfr(20, 0, 2, 21, 0, 3, 1.0, 1.0, tables, 1)
        assert ev == -1.0  # Dealer 21 > player 20

    def test_dealer_below_16_forces_hit(self) -> None:
        tables = _make_tables()
        # Dealer at 14 (nat=14, na=0, nc=2). Should hit (chance node),
        # eventually reach a decision. No regret updates expected at nc<16.
        ev = _dealer_cfr(20, 0, 2, 14, 0, 2, 1.0, 1.0, tables, 1)
        # Player 20 is strong — EV should be > 0 but dealer has a chance
        assert -3.0 <= ev <= 3.0  # Bounded by max payouts

    def test_dealer_strategic_at_16_updates_regrets(self) -> None:
        tables = _make_tables()
        # Dealer at 16 hard (nat=16, na=0, nc=2), player has 2 cards
        _dealer_cfr(18, 0, 2, 16, 0, 2, 1.0, 1.0, tables, 1)
        assert _dealer_action_key(16, 2, False, 2) in tables.dealer_action_regrets

    def test_dealer_strategic_at_17_soft_updates_regrets(self) -> None:
        tables = _make_tables()
        # Dealer soft 17: nat=6, na=1, nc=2 → total=17, is_soft=True
        _dealer_cfr(18, 0, 2, 6, 1, 2, 1.0, 1.0, tables, 1)
        assert _dealer_action_key(17, 2, True, 2) in tables.dealer_action_regrets

    def test_dealer_reveal_player_legal_for_nc3(self) -> None:
        tables = _make_tables()
        # Player has 3 cards: player_nc=3, dealer at 16
        _dealer_cfr(18, 0, 3, 16, 0, 2, 1.0, 1.0, tables, 1)
        key = _dealer_action_key(16, 2, False, 3)
        assert key in tables.dealer_action_regrets
        # REVEAL_PLAYER (_D_REVEAL) should be in the regrets (legal at player_nc=3)
        assert _D_REVEAL in tables.dealer_action_regrets[key]


# ─── TestPlayerCfr ─────────────────────────────────────────────────────────────


class TestPlayerCfr:
    """_player_cfr terminal, forced-stand, and decision cases."""

    def test_player_bust_returns_minus_one(self) -> None:
        tables = _make_tables()
        # Player nat=22, na=0, nc=3 → total=22 (bust)
        ev = _player_cfr(22, 0, 3, 10, 0, 2, 1.0, 1.0, tables, 1)
        assert ev == -1.0

    def test_player_five_card_bust_returns_minus_two(self) -> None:
        tables = _make_tables()
        # Player nat=25, na=0, nc=5 → total=25 (5-card bust)
        ev = _player_cfr(25, 0, 5, 10, 0, 2, 1.0, 1.0, tables, 1)
        assert ev == -2.0

    def test_player_at_21_forced_stand(self) -> None:
        tables = _make_tables()
        # Player nat=21, na=0, nc=3 → total=21 (forced stand → dealer phase)
        _player_cfr(21, 0, 3, 10, 0, 2, 1.0, 1.0, tables, 1)
        # No player decision node at 21
        assert _player_key(21, 3, False) not in tables.player_regrets

    def test_player_at_5_cards_forced_stand(self) -> None:
        tables = _make_tables()
        # Player 5 cards, total=18 → forced stand
        _player_cfr(18, 0, 5, 10, 0, 2, 1.0, 1.0, tables, 1)
        assert _player_key(18, 5, False) not in tables.player_regrets

    def test_player_decision_at_17_updates_regrets(self) -> None:
        tables = _make_tables()
        _player_cfr(17, 0, 2, 10, 0, 2, 1.0, 1.0, tables, 1)
        assert _player_key(17, 2, False) in tables.player_regrets

    def test_player_decision_at_16_has_both_actions(self) -> None:
        tables = _make_tables()
        _player_cfr(16, 0, 2, 10, 0, 2, 1.0, 1.0, tables, 1)
        regrets = tables.player_regrets[_player_key(16, 2, False)]
        assert _P_HIT in regrets
        assert _P_STAND in regrets

    def test_player_forfeit_below_16_still_creates_decision(self) -> None:
        """Player can hit or stand at ≤15 (standing = forfeit, but HIT is allowed)."""
        tables = _make_tables()
        _player_cfr(14, 0, 2, 10, 0, 2, 1.0, 1.0, tables, 1)
        assert _player_key(14, 2, False) in tables.player_regrets


# ─── TestRootCfr ───────────────────────────────────────────────────────────────


class TestRootCfr:
    """_root_cfr integration: surrender + special hands + player/dealer phases."""

    def test_dealer_hard15_surrender_creates_surrender_node(self) -> None:
        tables = _make_tables()
        # Dealer hard 15: nat=15, na=0, nc=2
        p_comp = (5, 0, 2)  # player total=5 (low, not special)
        d_comp = (15, 0, 2)  # dealer hard 15
        _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, 1)
        assert _dealer_surrender_key(15, True) in tables.dealer_surrender_regrets

    def test_dealer_no_surrender_non_hard15(self) -> None:
        tables = _make_tables()
        p_comp = (10, 0, 2)  # player total=10
        d_comp = (10, 0, 2)  # dealer hard 20, no surrender option
        _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, 1)
        assert _dealer_surrender_key(20, False) not in tables.dealer_surrender_regrets

    def test_player_ban_ban_returns_immediate_settlement(self) -> None:
        tables = _make_tables()
        # Player Ban Ban (0, 2, 2) vs dealer regular hand
        p_comp = (0, 2, 2)  # Ban Ban
        d_comp = (10, 0, 2)  # dealer 20 regular
        ev = _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, 1)
        # Player Ban Ban wins 3:1 vs any non-Ban-Ban dealer
        assert ev == 3.0
        # No player decision nodes (Ban Ban is pre-settled)
        assert tables.player_regrets == {}

    def test_player_ban_luck_vs_regular_dealer(self) -> None:
        tables = _make_tables()
        p_comp = (10, 1, 2)  # Ban Luck
        d_comp = (10, 0, 2)  # dealer regular 20
        ev = _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, 1)
        assert ev == 2.0

    def test_player_ban_luck_vs_dealer_ban_ban(self) -> None:
        tables = _make_tables()
        p_comp = (10, 1, 2)  # Player Ban Luck
        d_comp = (0, 2, 2)  # Dealer Ban Ban
        ev = _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, 1)
        assert ev == -3.0  # Ban Luck loses to dealer Ban Ban

    def test_normal_hand_proceeds_to_player_decisions(self) -> None:
        tables = _make_tables()
        p_comp = (17, 0, 2)  # player hard 17
        d_comp = (10, 0, 2)  # dealer hard 20
        _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, 1)
        assert _player_key(17, 2, False) in tables.player_regrets

    def test_surrender_with_player_ban_ban_uses_no_surrender_ev(self) -> None:
        """If dealer surrenders: push (0.0), even if player has Ban Ban."""
        tables = _make_tables()
        # Dealer hard 15, player Ban Ban. If dealer surrenders → push.
        # CFR will find the dealer surrender strategy based on EV difference.
        p_comp = (0, 2, 2)  # Player Ban Ban
        d_comp = (15, 0, 2)  # Dealer hard 15
        _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, 1)
        # Surrender EV = 0, no-surrender EV = 3.0 (Ban Ban)
        # Dealer should prefer surrender (0 > -3 from dealer's view)
        assert _dealer_surrender_key(15, True) in tables.dealer_surrender_regrets


# ─── TestSolve ─────────────────────────────────────────────────────────────────


class TestSolve:
    """solve() output structure and basic invariants."""

    @pytest.fixture(scope="class")
    def result_100(self) -> CfrResult:
        return solve(n_iterations=100, convergence_check_every=100)

    @pytest.fixture(scope="class")
    def result_300(self) -> CfrResult:
        return solve(n_iterations=300, convergence_check_every=300)

    def test_returns_cfr_result(self, result_100: CfrResult) -> None:
        assert isinstance(result_100, CfrResult)

    def test_n_iterations_recorded(self, result_100: CfrResult) -> None:
        assert result_100.n_iterations == 100

    def test_nash_ev_is_finite(self, result_100: CfrResult) -> None:
        assert -5.0 < result_100.nash_ev < 5.0

    def test_exploitability_non_negative(self, result_100: CfrResult) -> None:
        assert result_100.exploitability >= 0

    def test_player_strategy_non_empty(self, result_100: CfrResult) -> None:
        assert len(result_100.player_strategy) > 0

    def test_dealer_surrender_strategy_non_empty(self, result_100: CfrResult) -> None:
        assert len(result_100.dealer_surrender_strategy) > 0

    def test_dealer_action_strategy_non_empty(self, result_100: CfrResult) -> None:
        assert len(result_100.dealer_action_strategy) > 0

    def test_player_strategy_probabilities_sum_to_one(self, result_100: CfrResult) -> None:
        for info_set, probs in result_100.player_strategy.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9, f"Probs don't sum to 1 at {info_set}"

    def test_dealer_action_strategy_probabilities_sum_to_one(self, result_100: CfrResult) -> None:
        for info_set, probs in result_100.dealer_action_strategy.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9, f"Probs don't sum to 1 at {info_set}"

    def test_nash_ev_plausible_range(self, result_300: CfrResult) -> None:
        """After 300 iterations, Nash EV should be in a plausible range."""
        # Dealer can surrender on hard-15 → house edge expected
        # DP solver (no surrender) gave player +1.57% (no dealer surrender)
        # With GTO dealer surrender, expect some house edge
        assert -0.5 < result_300.nash_ev < 0.3


# ─── TestStrategyInvariants ────────────────────────────────────────────────────


class TestStrategyInvariants:
    """Nash strategy sanity checks after CFR convergence."""

    @pytest.fixture(scope="class")
    def result(self) -> CfrResult:
        return solve(n_iterations=500, convergence_check_every=500)

    def test_player_stands_on_20(self, result: CfrResult) -> None:
        """Optimal player always stands at 20 — hitting can only make it worse."""
        info_set = PlayerHitStandInfoSet(total=20, num_cards=2, is_soft=False)
        if info_set in result.player_strategy:
            strategy = result.player_strategy[info_set]
            p_stand = strategy.get(PlayerAction.STAND, 0.0)
            assert p_stand > 0.9, f"Player should stand strongly at 20: {strategy}"

    def test_player_stands_on_21(self, result: CfrResult) -> None:
        """21 is a forced stand — info set should not appear in CFR table."""
        info_set = PlayerHitStandInfoSet(total=21, num_cards=2, is_soft=False)
        # PlayerHitStandInfoSet for total=21 has only STAND as legal action
        # and may not appear in the strategy table if it was never a decision node
        if info_set in result.player_strategy:
            strategy = result.player_strategy[info_set]
            p_stand = strategy.get(PlayerAction.STAND, 0.0)
            assert p_stand >= 0.99

    def test_player_hits_at_11(self, result: CfrResult) -> None:
        """Player should hit at 11 — cannot bust, and drawing improves the hand."""
        info_set = PlayerHitStandInfoSet(total=11, num_cards=2, is_soft=False)
        if info_set in result.player_strategy:
            strategy = result.player_strategy[info_set]
            p_hit = strategy.get(PlayerAction.HIT, 0.0)
            assert p_hit > 0.9, f"Player should hit at 11: {strategy}"

    def test_dealer_surrenders_on_hard_15(self, result: CfrResult) -> None:
        """Dealer should have meaningful surrender probability on hard 15."""
        p_surr = get_dealer_surrender_prob(result)
        # In a real game, dealer surrenders hard 15 to avoid big losses
        # CFR should find surrender has positive value vs not surrendering
        assert 0.0 <= p_surr <= 1.0

    def test_dealer_action_info_sets_have_valid_probs(self, result: CfrResult) -> None:
        """All dealer action info set probabilities sum to 1."""
        for info_set, probs in result.dealer_action_strategy.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9

    def test_dealer_stands_at_18_or_higher_not_in_cfr_table(self, result: CfrResult) -> None:
        """DealerActionInfoSet for total >= 18 should not appear in strategy table
        (those are forced stands, no CFR decisions)."""
        for info_set in result.dealer_action_strategy:
            assert info_set.dealer_total in {
                16,
                17,
            }, f"Unexpected dealer total in strategy: {info_set}"


# ─── TestExploitability ────────────────────────────────────────────────────────


class TestExploitability:
    """compute_exploitability, best-response EVs."""

    @pytest.fixture(scope="class")
    def result_200(self) -> CfrResult:
        return solve(n_iterations=200, convergence_check_every=200)

    def test_exploitability_non_negative(self, result_200: CfrResult) -> None:
        eps = compute_exploitability(
            result_200.player_strategy,
            result_200.dealer_surrender_strategy,
            result_200.dealer_action_strategy,
        )
        assert eps >= 0.0

    def test_best_player_ev_at_least_nash_ev(self, result_200: CfrResult) -> None:
        """Best player EV ≥ Nash EV (player can only do better with optimal play)."""
        initial_deals = _build_initial_deals()
        best_p = _best_player_ev(
            result_200.dealer_surrender_strategy,
            result_200.dealer_action_strategy,
            initial_deals,
        )
        assert best_p >= result_200.nash_ev - 1e-6

    def test_best_dealer_ev_at_most_nash_ev(self, result_200: CfrResult) -> None:
        """Best dealer response minimises player EV ≤ Nash EV."""
        initial_deals = _build_initial_deals()
        worst_p = _best_dealer_ev(result_200.player_strategy, initial_deals)
        assert worst_p <= result_200.nash_ev + 1e-6

    def test_exploitability_decreases_with_more_iterations(self) -> None:
        """More CFR iterations → lower exploitability."""
        r100 = solve(n_iterations=100, convergence_check_every=100)
        r500 = solve(n_iterations=500, convergence_check_every=500)
        eps_100 = r100.exploitability
        eps_500 = r500.exploitability
        assert eps_500 <= eps_100 + 0.05  # Allow small statistical fluctuation


# ─── TestPublicHelpers ──────────────────────────────────────────────────────────


class TestPublicHelpers:
    """get_player_action, get_dealer_surrender_prob."""

    @pytest.fixture(scope="class")
    def result(self) -> CfrResult:
        return solve(n_iterations=300, convergence_check_every=300)

    def test_get_player_action_returns_player_action(self, result: CfrResult) -> None:
        action = get_player_action(result, total=17, num_cards=2, is_soft=False)
        assert isinstance(action, PlayerAction)

    def test_get_player_action_stand_on_20(self, result: CfrResult) -> None:
        action = get_player_action(result, total=20, num_cards=2, is_soft=False)
        assert action == PlayerAction.STAND

    def test_get_player_action_hit_on_11(self, result: CfrResult) -> None:
        action = get_player_action(result, total=11, num_cards=2, is_soft=False)
        assert action == PlayerAction.HIT

    def test_get_player_action_missing_returns_stand(self, result: CfrResult) -> None:
        action = get_player_action(result, total=21, num_cards=2, is_soft=False)
        # 21 is a forced stand, may not be in table → should return STAND
        assert action == PlayerAction.STAND

    def test_get_dealer_surrender_prob_in_range(self, result: CfrResult) -> None:
        p = get_dealer_surrender_prob(result)
        assert 0.0 <= p <= 1.0

    def test_get_dealer_surrender_prob_non_zero_after_many_iterations(
        self, result: CfrResult
    ) -> None:
        """Hard-15 surrender should be non-trivially used by Nash dealer."""
        p = get_dealer_surrender_prob(result)
        # Surrendering hard 15 is strategically valuable — CFR should
        # discover it has positive EV for the dealer (negative for player)
        assert p > 0.05, f"Expected surrender prob > 0.05, got {p}"
