"""Tests for Phase 2.3 — strategy report (src/analysis/strategy_report.py).

Tests verify that each print function produces non-empty output and that the
underlying CfrResult data meets basic invariants (probabilities in range,
Nash EV plausible). Reduced iteration counts keep the test suite fast.
"""

from __future__ import annotations

import pytest

from src.analysis.strategy_report import (
    _DP_EV_REVEAL_OFF,
    _DP_EV_REVEAL_ON,
    _HARD15_FREQ,
    _MC_EV_REVEAL_OFF,
    _MC_EV_REVEAL_ON,
    print_dealer_strategy,
    print_nash_ev,
    print_reveal_advantage,
    print_surrender_strategy,
    print_surrender_value,
)
from src.engine.game_state import DealerAction
from src.solvers.cfr import CfrResult, solve


@pytest.fixture(scope="module")
def result() -> CfrResult:
    return solve(n_iterations=200, convergence_check_every=200)


# ─── print_nash_ev ────────────────────────────────────────────────────────────


class TestPrintNashEv:
    def test_output_non_empty(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_nash_ev(result)
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_output_contains_nash_ev_header(
        self, result: CfrResult, capsys: pytest.CaptureFixture
    ) -> None:
        print_nash_ev(result)
        captured = capsys.readouterr()
        assert "Nash" in captured.out

    def test_nash_ev_plausible_range(self, result: CfrResult) -> None:
        # Reduced iterations → wide range; tightens with convergence.
        assert -1.0 < result.nash_ev < 0.5

    def test_exploitability_non_negative(self, result: CfrResult) -> None:
        assert result.exploitability >= 0.0

    def test_output_contains_baseline_values(
        self, result: CfrResult, capsys: pytest.CaptureFixture
    ) -> None:
        print_nash_ev(result)
        captured = capsys.readouterr()
        # DP baseline strings should appear in the comparison section.
        assert "DP" in captured.out
        assert "MC" in captured.out


# ─── print_surrender_strategy ─────────────────────────────────────────────────


class TestPrintSurrenderStrategy:
    def test_output_non_empty(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_surrender_strategy(result)
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_surrender_prob_in_unit_interval(self, result: CfrResult) -> None:
        for info_set, probs in result.dealer_surrender_strategy.items():
            if info_set.is_hard_fifteen:
                p = probs.get(DealerAction.SURRENDER, 0.0)
                assert 0.0 <= p <= 1.0, f"P(surrender) out of range at {info_set}"

    def test_hard15_info_set_present(self, result: CfrResult) -> None:
        hard15 = [k for k in result.dealer_surrender_strategy if k.is_hard_fifteen]
        assert len(hard15) > 0, "No hard-15 info set in surrender strategy"

    def test_probs_sum_to_one(self, result: CfrResult) -> None:
        for info_set, probs in result.dealer_surrender_strategy.items():
            if info_set.is_hard_fifteen:
                total = sum(probs.values())
                assert abs(total - 1.0) < 1e-9, f"Probs don't sum to 1 at {info_set}"


# ─── print_dealer_strategy ────────────────────────────────────────────────────


class TestPrintDealerStrategy:
    def test_output_non_empty(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_dealer_strategy(result)
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_reveal_prob_in_unit_interval(self, result: CfrResult) -> None:
        for info_set, probs in result.dealer_action_strategy.items():
            p = probs.get(DealerAction.REVEAL_PLAYER, 0.0)
            assert 0.0 <= p <= 1.0, f"P(reveal) out of range at {info_set}"

    def test_reveal_zero_for_two_card_player(self, result: CfrResult) -> None:
        # REVEAL_PLAYER is not legal when player holds 2 cards.
        for info_set, probs in result.dealer_action_strategy.items():
            if info_set.player_nc == 2:
                p = probs.get(DealerAction.REVEAL_PLAYER, 0.0)
                assert p == 0.0, f"P(reveal) non-zero for 2-card player at {info_set}"

    def test_reveal_legal_at_16_with_multicard_player(self, result: CfrResult) -> None:
        # Any DealerActionInfoSet at total 16 with player_nc >= 3 should exist.
        nodes_16 = [
            k for k in result.dealer_action_strategy if k.dealer_total == 16 and k.player_nc >= 3
        ]
        assert len(nodes_16) > 0, "No dealer-16 nodes with 3+-card player found"

    def test_probs_sum_to_one(self, result: CfrResult) -> None:
        for info_set, probs in result.dealer_action_strategy.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9, f"Probs don't sum to 1 at {info_set}"


# ─── print_reveal_advantage ───────────────────────────────────────────────────


class TestPrintRevealAdvantage:
    def test_output_not_empty(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_reveal_advantage(result)
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_has_section_headers(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_reveal_advantage(result)
        captured = capsys.readouterr()
        assert "Reveal Advantage" in captured.out
        assert "GTO Reveal Frequency" in captured.out
        assert "Research Question" in captured.out

    def test_ev_deltas_in_output(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_reveal_advantage(result)
        captured = capsys.readouterr()
        dp_delta_pct = (_DP_EV_REVEAL_ON - _DP_EV_REVEAL_OFF) * 100
        mc_delta_pct = (_MC_EV_REVEAL_ON - _MC_EV_REVEAL_OFF) * 100
        # Check formatted delta strings appear (e.g. "+0.84" and "+0.96")
        assert f"{dp_delta_pct:+.2f}" in captured.out
        assert f"{mc_delta_pct:+.2f}" in captured.out

    def test_gto_frequency_in_unit_interval(self, result: CfrResult) -> None:
        # P(reveal) must be in [0, 1] for every 3+-card node.
        for info_set, probs in result.dealer_action_strategy.items():
            if info_set.player_nc >= 3:
                p = probs.get(DealerAction.REVEAL_PLAYER, 0.0)
                assert 0.0 <= p <= 1.0, f"P(reveal) out of [0,1] at {info_set}"

    def test_breakdown_rows_16_and_17(
        self, result: CfrResult, capsys: pytest.CaptureFixture
    ) -> None:
        print_reveal_advantage(result)
        captured = capsys.readouterr()
        assert "Dealer-16" in captured.out
        assert "Dealer-17" in captured.out

    def test_research_answers_present(
        self, result: CfrResult, capsys: pytest.CaptureFixture
    ) -> None:
        print_reveal_advantage(result)
        captured = capsys.readouterr()
        assert "Q1" in captured.out
        assert "Q3" in captured.out
        assert (
            "indifference" in captured.out.lower()
            or "indifferent" in captured.out.lower()
            or "REVEAL_PLAYER" in captured.out
        )


# ─── print_surrender_value ────────────────────────────────────────────────────


class TestPrintSurrenderValue:
    def test_output_not_empty(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_surrender_value(result)
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_has_section_headers(self, result: CfrResult, capsys: pytest.CaptureFixture) -> None:
        print_surrender_value(result)
        captured = capsys.readouterr()
        assert "Surrender Value" in captured.out
        assert "Research Question" in captured.out

    def test_hard15_frequency_in_output(
        self, result: CfrResult, capsys: pytest.CaptureFixture
    ) -> None:
        print_surrender_value(result)
        captured = capsys.readouterr()
        # Formatted as "7.10%" (12/169 ≈ 7.1006%)
        expected = f"{_HARD15_FREQ * 100:.2f}%"
        assert expected in captured.out

    def test_surrender_prob_in_unit_interval(self, result: CfrResult) -> None:
        from src.solvers.cfr import get_dealer_surrender_prob

        p = get_dealer_surrender_prob(result)
        assert 0.0 <= p <= 1.0, f"P(surrender) out of [0, 1]: {p}"

    def test_effective_rate_in_output(
        self, result: CfrResult, capsys: pytest.CaptureFixture
    ) -> None:
        print_surrender_value(result)
        captured = capsys.readouterr()
        assert (
            "Effective surrender rate" in captured.out
            or "effective surrender rate" in captured.out.lower()
        )

    def test_research_answer_q4_present(
        self, result: CfrResult, capsys: pytest.CaptureFixture
    ) -> None:
        print_surrender_value(result)
        captured = capsys.readouterr()
        assert "Q4" in captured.out
        assert "surrender" in captured.out.lower()
