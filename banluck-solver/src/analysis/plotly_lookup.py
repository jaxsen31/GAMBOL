"""Interactive Plotly strategy lookup tool for the Banluck solver.

Four public functions:

    build_dp_lookup_figure(reveal_mode, show_ev_margin)
        — Interactive hard/soft heatmaps from the DP solver.
    build_cfr_lookup_figure(result)
        — Interactive hard/soft heatmaps showing CFR Nash P(HIT).
    build_comparison_figure(result)
        — 2×3 grid: DP reveal=OFF / DP reveal=ON / CFR Nash for hard and soft.
    save_lookup_html(fig, path)
        — Export any figure to a self-contained HTML file.

All figures are interactive Plotly figures: hover over any cell to see the
hand-state details (total, hand size, hard/soft, action, EV margin).  Figures
open in a browser via ``fig.show()`` or embed in Jupyter notebooks.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.heat_maps import build_cfr_heatmap_data, build_dp_heatmap_data
from src.solvers.cfr import CfrResult

# ─── Constants ────────────────────────────────────────────────────────────────

_TOTALS: list[int] = [16, 17, 18, 19, 20, 21]
_HAND_SIZES: list[int] = [2, 3, 4, 5]
_ROW_LABELS: list[str] = [str(t) for t in _TOTALS]
_COL_LABELS: list[str] = [f"nc={nc}" for nc in _HAND_SIZES]

# Discrete red→green colorscale: 0.0 = STAND (red), 1.0 = HIT (green).
# The step at 0.5 creates a hard binary cutoff.
_BINARY_COLORSCALE: list[list] = [
    [0.0, "#d62728"],
    [0.499, "#d62728"],
    [0.501, "#2ca02c"],
    [1.0, "#2ca02c"],
]

_CFR_COLORSCALE: str = "RdYlGn"


# ─── Hover text builders ──────────────────────────────────────────────────────


def _build_dp_hover(
    data: np.ndarray,
    ev_margin: dict[tuple[int, int, int], float | None] | None,
    is_soft: bool,
) -> list[list[str]]:
    """Return a 6×4 list of hover strings for a DP strategy panel.

    Each non-NaN cell shows Total, Cards, Type (Hard/Soft), Action (HIT/STAND),
    and optionally the EV margin |EV_HIT − EV_STAND|.

    Args:
        data:      (6, 4) matrix: 1.0=HIT, 0.0=STAND, NaN=absent.
        ev_margin: Dict keyed on ``(total, nc, is_soft_int)`` → float or None.
                   Pass None to omit EV margin from hover text.
        is_soft:   True for the soft-totals panel.

    Returns:
        6×4 list of HTML hover strings (empty string for absent cells).
    """
    is_soft_flag = 1 if is_soft else 0
    type_label = "Soft" if is_soft else "Hard"
    rows: list[list[str]] = []
    for r, total in enumerate(_TOTALS):
        row: list[str] = []
        for c, nc in enumerate(_HAND_SIZES):
            val = data[r, c]
            if np.isnan(val):
                row.append("")
                continue

            action = "HIT" if val >= 0.5 else "STAND"
            lines = [
                f"Total: <b>{total}</b>",
                f"Cards: {nc}",
                f"Type: {type_label}",
                f"Action: <b>{action}</b>",
            ]
            if ev_margin is not None:
                margin = ev_margin.get((total, nc, is_soft_flag))
                if margin is not None:
                    lines.append(f"EV margin: {margin:.4f}")
            row.append("<br>".join(lines))
        rows.append(row)
    return rows


def _build_cfr_hover(data: np.ndarray, is_soft: bool) -> list[list[str]]:
    """Return a 6×4 list of hover strings for a CFR strategy panel.

    Each non-NaN cell shows Total, Cards, Type (Hard/Soft), and P(HIT).

    Args:
        data:    (6, 4) matrix: P(HIT) in [0, 1], NaN=absent.
        is_soft: True for the soft-totals panel.

    Returns:
        6×4 list of HTML hover strings (empty string for absent cells).
    """
    type_label = "Soft" if is_soft else "Hard"
    rows: list[list[str]] = []
    for r, total in enumerate(_TOTALS):
        row: list[str] = []
        for c, nc in enumerate(_HAND_SIZES):
            val = data[r, c]
            if np.isnan(val):
                row.append("")
                continue
            lines = [
                f"Total: <b>{total}</b>",
                f"Cards: {nc}",
                f"Type: {type_label}",
                f"P(HIT): <b>{val:.3f}</b>",
                f"Leans: {'HIT' if val >= 0.5 else 'STAND'}",
            ]
            row.append("<br>".join(lines))
        rows.append(row)
    return rows


# ─── Trace builder ─────────────────────────────────────────────────────────────


def _make_heatmap_trace(
    data: np.ndarray,
    hover_text: list[list[str]],
    *,
    colorscale: list[list] | str,
    name: str,
    showscale: bool = True,
    colorbar_title: str = "",
    colorbar_x: float = 1.02,
) -> go.Heatmap:
    """Build one go.Heatmap trace for a strategy panel.

    NaN values in *data* are converted to None so Plotly renders them as
    blank (transparent) cells.

    Args:
        data:           (6, 4) float array; NaN → blank cell.
        hover_text:     6×4 list of HTML hover strings.
        colorscale:     Plotly colorscale (list or named string).
        name:           Trace name (visible in hover/legend).
        showscale:      Whether to show the colorbar for this trace.
        colorbar_title: Label for the colorbar.
        colorbar_x:     Horizontal anchor for the colorbar in [0, 2].

    Returns:
        go.Heatmap instance.
    """
    z = [[None if np.isnan(v) else v for v in row] for row in data.tolist()]
    return go.Heatmap(
        z=z,
        x=_COL_LABELS,
        y=_ROW_LABELS,
        colorscale=colorscale,
        zmin=0.0,
        zmax=1.0,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showscale=showscale,
        colorbar={"title": colorbar_title, "x": colorbar_x},
        name=name,
    )


# ─── Public figure builders ───────────────────────────────────────────────────


def build_dp_lookup_figure(
    reveal_mode: bool = False,
    *,
    show_ev_margin: bool = True,
) -> go.Figure:
    """Build an interactive Plotly figure for DP strategy (hard + soft panels).

    Each cell is coloured red (STAND) or green (HIT).  Hovering over a cell
    shows the optimal action and, when ``show_ev_margin=True``, the EV
    difference |EV_HIT − EV_STAND| for that hand state.

    Args:
        reveal_mode:    If True, use the reveal=ON solver results.
        show_ev_margin: If True, include EV margin in hover text.

    Returns:
        go.Figure with two heatmap traces (hard, soft) in a 1×2 subplot layout.
    """
    from src.solvers.baseline_dp import build_ev_margin_table

    hard, soft = build_dp_heatmap_data(reveal_mode=reveal_mode)
    ev_margin = build_ev_margin_table(reveal_mode) if show_ev_margin else None

    hard_hover = _build_dp_hover(hard, ev_margin, is_soft=False)
    soft_hover = _build_dp_hover(soft, ev_margin, is_soft=True)

    mode_label = "reveal=ON" if reveal_mode else "reveal=OFF"
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Hard totals", "Soft totals"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        _make_heatmap_trace(
            hard,
            hard_hover,
            colorscale=_BINARY_COLORSCALE,
            name="Hard",
            showscale=True,
            colorbar_title="STAND / HIT",
            colorbar_x=1.02,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _make_heatmap_trace(
            soft,
            soft_hover,
            colorscale=_BINARY_COLORSCALE,
            name="Soft",
            showscale=False,
            colorbar_title="",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text=f"DP Strategy Lookup — {mode_label}",
        title_font_size=15,
        height=420,
        width=780,
    )
    fig.update_yaxes(title_text="Player total", col=1)
    fig.update_xaxes(title_text="Hand size")
    return fig


def build_cfr_lookup_figure(result: CfrResult) -> go.Figure:
    """Build an interactive Plotly figure for CFR Nash strategy (hard + soft panels).

    Each cell shows P(HIT) with a continuous RdYlGn colorscale (red=STAND,
    green=HIT, yellow=mixed).  Hover text shows the P(HIT) value and the
    majority-rule lean (HIT if P(HIT) ≥ 0.5, else STAND).

    Args:
        result: CfrResult returned by cfr.solve().

    Returns:
        go.Figure with two heatmap traces (hard, soft) in a 1×2 subplot layout.
    """
    hard, soft = build_cfr_heatmap_data(result)
    hard_hover = _build_cfr_hover(hard, is_soft=False)
    soft_hover = _build_cfr_hover(soft, is_soft=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Hard totals", "Soft totals"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        _make_heatmap_trace(
            hard,
            hard_hover,
            colorscale=_CFR_COLORSCALE,
            name="Hard",
            showscale=True,
            colorbar_title="P(HIT)",
            colorbar_x=1.02,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _make_heatmap_trace(
            soft,
            soft_hover,
            colorscale=_CFR_COLORSCALE,
            name="Soft",
            showscale=False,
            colorbar_title="",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="CFR Nash Strategy Lookup — P(HIT)",
        title_font_size=15,
        height=420,
        width=780,
    )
    fig.update_yaxes(title_text="Player total", col=1)
    fig.update_xaxes(title_text="Hand size")
    return fig


def build_comparison_figure(result: CfrResult) -> go.Figure:
    """Build a 2×3 interactive comparison figure: DP×2 vs CFR Nash.

    Layout::

        Col 1 = DP reveal=OFF   Col 2 = DP reveal=ON   Col 3 = CFR Nash P(HIT)
        Row 1 = Hard totals      Row 2 = Soft totals

    Hover text on each cell shows the action (or P(HIT)) plus the EV margin
    where applicable.

    Args:
        result: CfrResult returned by cfr.solve().

    Returns:
        go.Figure with 6 heatmap traces in a 2×3 subplot layout.
    """
    from src.solvers.baseline_dp import build_ev_margin_table

    hard_off, soft_off = build_dp_heatmap_data(reveal_mode=False)
    hard_on, soft_on = build_dp_heatmap_data(reveal_mode=True)
    hard_cfr, soft_cfr = build_cfr_heatmap_data(result)

    ev_off = build_ev_margin_table(reveal_mode=False)
    ev_on = build_ev_margin_table(reveal_mode=True)

    col_titles = ["DP reveal=OFF", "DP reveal=ON", "CFR Nash P(HIT)"]
    subplot_titles = [f"{ct} — Hard" for ct in col_titles] + [f"{ct} — Soft" for ct in col_titles]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # (row, col, data, hover_text, colorscale, showscale)
    panels: list[tuple] = [
        (1, 1, hard_off, _build_dp_hover(hard_off, ev_off, False), _BINARY_COLORSCALE, False),
        (1, 2, hard_on, _build_dp_hover(hard_on, ev_on, False), _BINARY_COLORSCALE, False),
        (1, 3, hard_cfr, _build_cfr_hover(hard_cfr, False), _CFR_COLORSCALE, True),
        (2, 1, soft_off, _build_dp_hover(soft_off, ev_off, True), _BINARY_COLORSCALE, False),
        (2, 2, soft_on, _build_dp_hover(soft_on, ev_on, True), _BINARY_COLORSCALE, False),
        (2, 3, soft_cfr, _build_cfr_hover(soft_cfr, True), _CFR_COLORSCALE, False),
    ]

    for row, col, data, hover, cscale, showscale in panels:
        cb_title = "P(HIT)" if (cscale == _CFR_COLORSCALE and showscale) else ""
        fig.add_trace(
            _make_heatmap_trace(
                data,
                hover,
                colorscale=cscale,
                name=f"r{row}c{col}",
                showscale=showscale,
                colorbar_title=cb_title,
                colorbar_x=1.02,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text="Banluck Player Strategy Comparison",
        title_font_size=15,
        height=700,
        width=1050,
    )
    for r in range(1, 3):
        fig.update_yaxes(title_text="Player total", row=r, col=1)
    for c in range(1, 4):
        fig.update_xaxes(title_text="Hand size", row=2, col=c)

    return fig


# ─── HTML export ───────────────────────────────────────────────────────────────


def save_lookup_html(fig: go.Figure, path: str) -> None:
    """Save a Plotly figure to a self-contained HTML file.

    The resulting file can be opened in any browser.  Plotly JS is loaded
    from the CDN so the file itself remains compact.

    Args:
        fig:  Any go.Figure produced by this module.
        path: Destination file path (e.g. ``"strategy_lookup.html"``).
    """
    fig.write_html(path, include_plotlyjs="cdn")


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from src.solvers.cfr import solve

    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"Running CFR+ for {n_iter} iterations …")
    result = solve(n_iterations=n_iter)

    print("Building interactive lookup figures …")
    dp_off_fig = build_dp_lookup_figure(reveal_mode=False)
    dp_on_fig = build_dp_lookup_figure(reveal_mode=True)
    cfr_fig = build_cfr_lookup_figure(result)
    cmp_fig = build_comparison_figure(result)

    save_lookup_html(dp_off_fig, "dp_reveal_off_lookup.html")
    save_lookup_html(dp_on_fig, "dp_reveal_on_lookup.html")
    save_lookup_html(cfr_fig, "cfr_lookup.html")
    save_lookup_html(cmp_fig, "strategy_comparison_lookup.html")
    print(
        "Saved: dp_reveal_off_lookup.html, dp_reveal_on_lookup.html, "
        "cfr_lookup.html, strategy_comparison_lookup.html"
    )
