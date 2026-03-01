"""Tests for Phase 3 — strategy heat maps (src/analysis/heat_maps.py).

Tests verify data-matrix shapes and value invariants (no display required)
plus that each plot function returns a well-formed matplotlib Figure.
The Agg backend is activated before any pyplot import so CI/CD environments
without a display server can run the suite safely.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import

import matplotlib.figure
import numpy as np
import pytest

from src.analysis.heat_maps import (
    build_cfr_heatmap_data,
    build_dp_heatmap_data,
    plot_cfr_strategy_heatmaps,
    plot_dp_strategy_heatmaps,
    plot_strategy_comparison,
    plot_strategy_heatmaps,
)
from src.solvers.cfr import CfrResult, solve


@pytest.fixture(scope="module")
def result() -> CfrResult:
    return solve(n_iterations=50, convergence_check_every=50)


# ─── build_dp_heatmap_data ────────────────────────────────────────────────────


class TestBuildDpHeatmapData:
    def test_shape(self) -> None:
        hard, soft = build_dp_heatmap_data()
        assert hard.shape == (6, 4)
        assert soft.shape == (6, 4)

    def test_values_binary_or_nan(self) -> None:
        hard, soft = build_dp_heatmap_data()
        for mat in (hard, soft):
            for val in mat.flat:
                if not np.isnan(val):
                    assert val in (0.0, 1.0), f"Non-binary value in DP matrix: {val}"

    def test_hard_21_all_stand(self) -> None:
        # Row 5 = total 21; forced STAND everywhere.
        hard, _ = build_dp_heatmap_data()
        for c in range(4):
            val = hard[5, c]
            if not np.isnan(val):
                assert val == 0.0, f"Total=21 hard should be STAND (0.0), got {val}"

    def test_non_empty(self) -> None:
        hard, soft = build_dp_heatmap_data()
        assert not np.all(np.isnan(hard)), "Hard matrix is entirely NaN"
        assert not np.all(np.isnan(soft)), "Soft matrix is entirely NaN"

    def test_reveal_on_same_shape(self) -> None:
        hard, soft = build_dp_heatmap_data(reveal_mode=True)
        assert hard.shape == (6, 4)
        assert soft.shape == (6, 4)


# ─── build_cfr_heatmap_data ───────────────────────────────────────────────────


class TestBuildCfrHeatmapData:
    def test_shape(self, result: CfrResult) -> None:
        hard, soft = build_cfr_heatmap_data(result)
        assert hard.shape == (6, 4)
        assert soft.shape == (6, 4)

    def test_values_in_unit_interval(self, result: CfrResult) -> None:
        hard, soft = build_cfr_heatmap_data(result)
        for mat in (hard, soft):
            for val in mat.flat:
                if not np.isnan(val):
                    assert 0.0 <= val <= 1.0, f"P(HIT) out of [0,1]: {val}"

    def test_non_empty(self, result: CfrResult) -> None:
        hard, _ = build_cfr_heatmap_data(result)
        assert not np.all(np.isnan(hard)), "Hard matrix is entirely NaN"

    def test_hard_21_is_zero(self, result: CfrResult) -> None:
        # Total 21 = forced STAND → P(HIT) must be 0.
        hard, _ = build_cfr_heatmap_data(result)
        for c in range(4):
            val = hard[5, c]
            if not np.isnan(val):
                assert val == 0.0, f"P(HIT) at total=21 should be 0.0, got {val}"


# ─── plot_strategy_heatmaps ───────────────────────────────────────────────────


class TestPlotStrategyHeatmaps:
    def test_returns_figure_binary(self) -> None:
        hard, soft = build_dp_heatmap_data()
        fig = plot_strategy_heatmaps(hard, soft, "Test binary", binary=True, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_module = __import__("matplotlib.pyplot", fromlist=["close"])
        plt_module.close(fig)

    def test_binary_has_two_subplot_axes(self) -> None:
        import matplotlib.pyplot as plt_

        hard, soft = build_dp_heatmap_data()
        fig = plot_strategy_heatmaps(hard, soft, "Test axes", binary=True, show=False)
        subplot_axes = [ax for ax in fig.axes if hasattr(ax, "get_subplotspec")]
        assert len(subplot_axes) == 2
        plt_.close(fig)

    def test_returns_figure_continuous(self) -> None:
        import matplotlib.pyplot as plt_

        hard, soft = build_dp_heatmap_data()
        fig = plot_strategy_heatmaps(hard, soft, "Test continuous", binary=False, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_.close(fig)

    def test_save_path(self, tmp_path: pytest.TempPathFactory) -> None:
        import matplotlib.pyplot as plt_

        hard, soft = build_dp_heatmap_data()
        path = str(tmp_path / "heatmap.png")
        fig = plot_strategy_heatmaps(hard, soft, "Save test", show=False, save_path=path)
        assert os.path.exists(path), "File not created"
        assert os.path.getsize(path) > 0, "File is empty"
        plt_.close(fig)


# ─── plot_dp_strategy_heatmaps ───────────────────────────────────────────────


class TestPlotDpStrategyHeatmaps:
    def test_reveal_off_returns_figure(self) -> None:
        import matplotlib.pyplot as plt_

        fig = plot_dp_strategy_heatmaps(reveal_mode=False, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_.close(fig)

    def test_reveal_on_returns_figure(self) -> None:
        import matplotlib.pyplot as plt_

        fig = plot_dp_strategy_heatmaps(reveal_mode=True, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_.close(fig)


# ─── plot_cfr_strategy_heatmaps ──────────────────────────────────────────────


class TestPlotCfrStrategyHeatmaps:
    def test_returns_figure(self, result: CfrResult) -> None:
        import matplotlib.pyplot as plt_

        fig = plot_cfr_strategy_heatmaps(result, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_.close(fig)


# ─── plot_strategy_comparison ────────────────────────────────────────────────


class TestPlotStrategyComparison:
    def test_returns_figure(self, result: CfrResult) -> None:
        import matplotlib.pyplot as plt_

        fig = plot_strategy_comparison(result, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt_.close(fig)

    def test_has_six_subplot_axes(self, result: CfrResult) -> None:
        import matplotlib.pyplot as plt_

        fig = plot_strategy_comparison(result, show=False)
        subplot_axes = [ax for ax in fig.axes if hasattr(ax, "get_subplotspec")]
        assert len(subplot_axes) >= 6, f"Expected ≥6 subplot axes, got {len(subplot_axes)}"
        plt_.close(fig)

    def test_save_path(self, result: CfrResult, tmp_path: pytest.TempPathFactory) -> None:
        import matplotlib.pyplot as plt_

        path = str(tmp_path / "comparison.png")
        fig = plot_strategy_comparison(result, show=False, save_path=path)
        assert os.path.exists(path), "Comparison file not created"
        assert os.path.getsize(path) > 0, "Comparison file is empty"
        plt_.close(fig)
