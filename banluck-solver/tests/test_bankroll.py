"""Tests for src/analysis/bankroll.py — variance and bankroll analysis.

Module-scoped fixture runs simulate_hands(n_hands=10_000, return_payouts=True)
once per test session to keep the suite fast.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest

from src.analysis.bankroll import (
    BankrollRequirement,
    DrawdownStats,
    HorizonProjection,
    RotationAnalysis,
    VarianceStats,
    compute_drawdown_stats,
    compute_fair_rotation,
    compute_horizon_projections,
    compute_variance_stats,
    print_rotation_analysis,
    print_variance_report,
    required_bankroll,
    risk_of_ruin,
)
from src.analysis.simulator import (
    SimulationResult,
    make_dp_player_strategy,
    make_fixed_dealer_strategy,
    simulate_hands,
)

# ─── Module-scoped fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sim_result_with_payouts() -> SimulationResult:
    """Run 10k-hand MC simulation with return_payouts=True (run once per session)."""
    surrender_strat, hit_strat = make_fixed_dealer_strategy()
    return simulate_hands(
        player_strategy=make_dp_player_strategy(reveal_mode=False),
        dealer_surrender_strategy=surrender_strat,
        dealer_hit_strategy=hit_strat,
        n_hands=10_000,
        seed=42,
        reveal_mode=False,
        return_payouts=True,
    )


@pytest.fixture(scope="module")
def sim_result_no_payouts() -> SimulationResult:
    """Run 10k-hand MC simulation with return_payouts=False (default)."""
    surrender_strat, hit_strat = make_fixed_dealer_strategy()
    return simulate_hands(
        player_strategy=make_dp_player_strategy(reveal_mode=False),
        dealer_surrender_strategy=surrender_strat,
        dealer_hit_strategy=hit_strat,
        n_hands=10_000,
        seed=42,
        reveal_mode=False,
        return_payouts=False,
    )


@pytest.fixture(scope="module")
def payouts(sim_result_with_payouts: SimulationResult) -> np.ndarray:
    assert sim_result_with_payouts.payouts is not None
    return sim_result_with_payouts.payouts


@pytest.fixture(scope="module")
def vstats(payouts: np.ndarray) -> VarianceStats:
    return compute_variance_stats(payouts)


# ─── TestSimulateHandsPayouts ─────────────────────────────────────────────────


class TestSimulateHandsPayouts:
    def test_payouts_returned_when_requested(self, sim_result_with_payouts):
        """return_payouts=True attaches a numpy array to result.payouts."""
        assert sim_result_with_payouts.payouts is not None
        assert isinstance(sim_result_with_payouts.payouts, np.ndarray)

    def test_payouts_none_by_default(self, sim_result_no_payouts):
        """return_payouts=False (default) leaves result.payouts as None."""
        assert sim_result_no_payouts.payouts is None

    def test_payouts_mean_matches_mean_ev(self, sim_result_with_payouts):
        """Mean of raw payouts matches the reported mean_ev."""
        arr = sim_result_with_payouts.payouts
        assert arr is not None
        assert abs(float(np.mean(arr)) - sim_result_with_payouts.mean_ev) < 1e-10


# ─── TestComputeVarianceStats ─────────────────────────────────────────────────


class TestComputeVarianceStats:
    def test_returns_variance_stats(self, vstats):
        assert isinstance(vstats, VarianceStats)

    def test_mean_matches_payouts(self, payouts, vstats):
        assert abs(vstats.mean - float(np.mean(payouts))) < 1e-10

    def test_std_positive(self, vstats):
        assert vstats.std > 0

    def test_percentiles_ordered(self, vstats):
        pct = vstats.percentiles
        keys = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
        vals = [pct[k] for k in keys]
        assert vals == sorted(vals)

    def test_skewness_finite(self, vstats):
        assert math.isfinite(vstats.skewness)

    def test_n_hands_matches(self, payouts, vstats):
        assert vstats.n_hands == len(payouts)


# ─── TestRiskOfRuin ───────────────────────────────────────────────────────────


class TestRiskOfRuin:
    def test_positive_edge_returns_between_0_and_1(self):
        ror = risk_of_ruin(bankroll=100.0, edge=0.05, std=1.0)
        assert 0.0 < ror < 1.0

    def test_zero_edge_returns_1(self):
        assert risk_of_ruin(bankroll=100.0, edge=0.0, std=1.0) == 1.0

    def test_negative_edge_returns_1(self):
        assert risk_of_ruin(bankroll=100.0, edge=-0.02, std=1.0) == 1.0

    def test_monotone_in_bankroll(self):
        """Larger bankroll → lower risk of ruin (with positive edge)."""
        r1 = risk_of_ruin(bankroll=50.0, edge=0.05, std=1.0)
        r2 = risk_of_ruin(bankroll=200.0, edge=0.05, std=1.0)
        assert r2 < r1

    def test_monotone_in_edge(self):
        """Larger edge → lower risk of ruin."""
        r1 = risk_of_ruin(bankroll=100.0, edge=0.02, std=1.0)
        r2 = risk_of_ruin(bankroll=100.0, edge=0.10, std=1.0)
        assert r2 < r1


# ─── TestRequiredBankroll ─────────────────────────────────────────────────────


class TestRequiredBankroll:
    def test_returns_bankroll_requirement(self):
        req = required_bankroll(edge=0.05, std=1.0, survival_prob=0.95)
        assert isinstance(req, BankrollRequirement)

    def test_99pct_greater_than_95pct(self):
        r95 = required_bankroll(edge=0.05, std=1.0, survival_prob=0.95)
        r99 = required_bankroll(edge=0.05, std=1.0, survival_prob=0.99)
        assert r99.required_bankroll > r95.required_bankroll

    def test_raises_on_non_positive_edge(self):
        with pytest.raises(ValueError):
            required_bankroll(edge=0.0, std=1.0, survival_prob=0.95)
        with pytest.raises(ValueError):
            required_bankroll(edge=-0.01, std=1.0, survival_prob=0.95)

    def test_positive_result(self):
        req = required_bankroll(edge=0.05, std=1.0, survival_prob=0.95)
        assert req.required_bankroll > 0


# ─── TestHorizonProjections ───────────────────────────────────────────────────


class TestHorizonProjections:
    def test_length_and_type(self, vstats):
        horizons = [100, 1000, 10_000]
        result = compute_horizon_projections(vstats.mean, vstats.std, horizons)
        assert len(result) == 3
        assert all(isinstance(p, HorizonProjection) for p in result)

    def test_ci_widens_with_horizon(self, vstats):
        """CI width (ci_high - ci_low) must increase with N."""
        horizons = [100, 1000, 10_000]
        result = compute_horizon_projections(vstats.mean, vstats.std, horizons)
        widths = [p.ci_high - p.ci_low for p in result]
        assert widths[0] < widths[1] < widths[2]

    def test_expected_profit_scales_linearly(self, vstats):
        """expected_profit must equal n_hands * edge exactly."""
        horizons = [100, 200]
        result = compute_horizon_projections(vstats.mean, vstats.std, horizons)
        assert abs(result[1].expected_profit - 2 * result[0].expected_profit) < 1e-10

    def test_prob_positive_in_unit_interval(self, vstats):
        horizons = [100, 1000]
        result = compute_horizon_projections(vstats.mean, vstats.std, horizons)
        for p in result:
            assert 0.0 <= p.prob_positive <= 1.0


# ─── TestDrawdownStats ────────────────────────────────────────────────────────


class TestDrawdownStats:
    def test_returns_drawdown_stats(self, payouts):
        dd = compute_drawdown_stats(payouts, n_trajectories=100, trajectory_length=200)
        assert isinstance(dd, DrawdownStats)

    def test_drawdowns_nonnegative(self, payouts):
        dd = compute_drawdown_stats(payouts, n_trajectories=100, trajectory_length=200)
        assert dd.mean_max_drawdown >= 0.0
        assert dd.median_max_drawdown >= 0.0
        assert dd.p95_max_drawdown >= 0.0

    def test_p95_ge_median(self, payouts):
        dd = compute_drawdown_stats(payouts, n_trajectories=200, trajectory_length=200)
        assert dd.p95_max_drawdown >= dd.median_max_drawdown

    def test_n_trajectories_matches(self, payouts):
        n = 150
        dd = compute_drawdown_stats(payouts, n_trajectories=n, trajectory_length=200)
        assert dd.n_trajectories == n


# ─── TestFairRotation ────────────────────────────────────────────────────────


class TestFairRotation:
    def test_returns_rotation_analysis(self, vstats):
        rotation = compute_fair_rotation(abs(vstats.mean), vstats.std)
        assert isinstance(rotation, RotationAnalysis)

    def test_recommended_n_positive(self, vstats):
        rotation = compute_fair_rotation(abs(vstats.mean), vstats.std)
        assert rotation.recommended_n > 0

    def test_smaller_edge_longer_rotation(self):
        """Smaller edge → harder to detect → longer rotation interval."""
        r_small = compute_fair_rotation(dealer_edge=0.01, std=1.0)
        r_large = compute_fair_rotation(dealer_edge=0.05, std=1.0)
        assert r_small.recommended_n > r_large.recommended_n

    def test_explanation_nonempty(self, vstats):
        rotation = compute_fair_rotation(abs(vstats.mean), vstats.std)
        assert len(rotation.explanation) > 0


# ─── TestPrintFunctions ───────────────────────────────────────────────────────


class TestPrintFunctions:
    def test_variance_report_nonempty_and_has_headers(self, vstats, payouts):
        reqs = []
        if vstats.mean > 0:
            for sp in [0.95, 0.99]:
                reqs.append(required_bankroll(vstats.mean, vstats.std, sp))
        projections = compute_horizon_projections(vstats.mean, vstats.std)
        dd = compute_drawdown_stats(payouts, n_trajectories=100, trajectory_length=200)
        rotation = compute_fair_rotation(abs(vstats.mean), vstats.std)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_variance_report(vstats, reqs, projections, dd, rotation, label="test")

        output = captured.getvalue()
        assert len(output) > 0
        assert "Distribution Statistics" in output
        assert "Horizon Projections" in output
        assert "Q5" in output

    def test_rotation_analysis_output_nonempty(self, vstats):
        rotation = compute_fair_rotation(abs(vstats.mean), vstats.std)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_rotation_analysis(rotation)

        output = captured.getvalue()
        assert len(output) > 0
        assert "Q5" in output
        assert "Recommended rotation" in output


# ─── needed for test_skewness_finite ─────────────────────────────────────────
import math  # noqa: E402 — placed here to avoid shadowing stdlib in fixtures
