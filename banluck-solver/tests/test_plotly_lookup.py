"""Tests for Phase 3 — interactive Plotly lookup tool (src/analysis/plotly_lookup.py).

Tests verify that each public function returns a well-formed go.Figure with the
expected trace count, data invariants, and hover text.  The save helper is
tested against a temporary file path.

No display server is required: Plotly figures are in-memory objects and the
save helper writes HTML without rendering.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import pytest

from src.analysis.plotly_lookup import (
    build_cfr_lookup_figure,
    build_comparison_figure,
    build_dp_lookup_figure,
    save_lookup_html,
)
from src.solvers.cfr import CfrResult, solve


@pytest.fixture(scope="module")
def result() -> CfrResult:
    return solve(n_iterations=50, convergence_check_every=50)


# ─── build_dp_lookup_figure ───────────────────────────────────────────────────


class TestBuildDpLookupFigure:
    def test_returns_figure_reveal_off(self) -> None:
        fig = build_dp_lookup_figure(reveal_mode=False)
        assert isinstance(fig, go.Figure)

    def test_returns_figure_reveal_on(self) -> None:
        fig = build_dp_lookup_figure(reveal_mode=True)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self) -> None:
        fig = build_dp_lookup_figure()
        assert len(fig.data) == 2

    def test_title_contains_reveal_label(self) -> None:
        fig_off = build_dp_lookup_figure(reveal_mode=False)
        fig_on = build_dp_lookup_figure(reveal_mode=True)
        assert "reveal=OFF" in fig_off.layout.title.text
        assert "reveal=ON" in fig_on.layout.title.text

    def test_hard_trace_hover_contains_action(self) -> None:
        fig = build_dp_lookup_figure(reveal_mode=False)
        hard_trace = fig.data[0]
        # Flatten the 6×4 text grid and find non-empty cells.
        flat = [cell for row in hard_trace.text for cell in row if cell]
        assert flat, "No hover text found in hard trace"
        assert any("Action:" in cell for cell in flat)

    def test_ev_margin_in_hover_when_enabled(self) -> None:
        fig = build_dp_lookup_figure(show_ev_margin=True)
        hard_trace = fig.data[0]
        flat = [cell for row in hard_trace.text for cell in row if cell]
        assert any("EV margin:" in cell for cell in flat)

    def test_no_ev_margin_in_hover_when_disabled(self) -> None:
        fig = build_dp_lookup_figure(show_ev_margin=False)
        hard_trace = fig.data[0]
        flat = [cell for row in hard_trace.text for cell in row if cell]
        assert not any("EV margin:" in cell for cell in flat)

    def test_traces_have_zmin_zmax(self) -> None:
        fig = build_dp_lookup_figure()
        for trace in fig.data:
            assert trace.zmin == 0.0
            assert trace.zmax == 1.0


# ─── build_cfr_lookup_figure ─────────────────────────────────────────────────


class TestBuildCfrLookupFigure:
    def test_returns_figure(self, result: CfrResult) -> None:
        fig = build_cfr_lookup_figure(result)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self, result: CfrResult) -> None:
        fig = build_cfr_lookup_figure(result)
        assert len(fig.data) == 2

    def test_title_contains_cfr(self, result: CfrResult) -> None:
        fig = build_cfr_lookup_figure(result)
        assert "CFR" in fig.layout.title.text

    def test_hover_contains_p_hit(self, result: CfrResult) -> None:
        fig = build_cfr_lookup_figure(result)
        hard_trace = fig.data[0]
        flat = [cell for row in hard_trace.text for cell in row if cell]
        assert flat, "No hover text found in hard trace"
        assert any("P(HIT):" in cell for cell in flat)

    def test_z_values_in_unit_interval(self, result: CfrResult) -> None:
        fig = build_cfr_lookup_figure(result)
        for trace in fig.data:
            for row in trace.z:
                for val in row:
                    if val is not None:
                        assert 0.0 <= val <= 1.0, f"P(HIT) out of [0,1]: {val}"


# ─── build_comparison_figure ─────────────────────────────────────────────────


class TestBuildComparisonFigure:
    def test_returns_figure(self, result: CfrResult) -> None:
        fig = build_comparison_figure(result)
        assert isinstance(fig, go.Figure)

    def test_has_six_traces(self, result: CfrResult) -> None:
        fig = build_comparison_figure(result)
        assert len(fig.data) == 6

    def test_title_present(self, result: CfrResult) -> None:
        fig = build_comparison_figure(result)
        assert fig.layout.title.text, "Figure title should not be empty"

    def test_subplot_titles_cover_all_panels(self, result: CfrResult) -> None:
        fig = build_comparison_figure(result)
        # Subplot titles are stored as layout annotations in Plotly.
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert any("reveal=OFF" in t for t in annotation_texts)
        assert any("reveal=ON" in t for t in annotation_texts)
        assert any("CFR" in t for t in annotation_texts)


# ─── save_lookup_html ────────────────────────────────────────────────────────


class TestSaveLookupHtml:
    def test_creates_file(self, tmp_path: pytest.TempPathFactory) -> None:
        fig = build_dp_lookup_figure()
        path = str(tmp_path / "lookup.html")
        save_lookup_html(fig, path)
        assert os.path.exists(path)

    def test_file_nonempty(self, tmp_path: pytest.TempPathFactory) -> None:
        fig = build_dp_lookup_figure()
        path = str(tmp_path / "lookup.html")
        save_lookup_html(fig, path)
        assert os.path.getsize(path) > 0

    def test_file_is_html(self, tmp_path: pytest.TempPathFactory) -> None:
        fig = build_dp_lookup_figure()
        path = str(tmp_path / "lookup.html")
        save_lookup_html(fig, path)
        with open(path) as f:
            content = f.read(200)
        assert "<html" in content.lower() or "<!doctype" in content.lower()
