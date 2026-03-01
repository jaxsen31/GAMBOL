"""Strategy heat maps for Banluck solver.

Four public data-builder functions return NumPy matrices that can be used
programmatically or passed to the plot helpers:

    build_dp_heatmap_data(reveal_mode)  — (hard, soft) binary matrices from DP
    build_cfr_heatmap_data(result)      — (hard, soft) P(HIT) matrices from CFR

Three public plot functions render matplotlib figures:

    plot_strategy_heatmaps(hard, soft, title, ...)  — 1×2 figure (hard + soft)
    plot_dp_strategy_heatmaps(reveal_mode, ...)     — convenience DP wrapper
    plot_cfr_strategy_heatmaps(result, ...)         — convenience CFR wrapper
    plot_strategy_comparison(result, ...)           — 2×3 comparison figure

Matrix convention (both builders):
    Shape  : (6, 4) — rows = totals [16, 17, 18, 19, 20, 21],
                       cols = num_cards [2, 3, 4, 5]
    Values : 1.0 = HIT, 0.0 = STAND (DP binary); P(HIT) in [0,1] (CFR)
             np.nan = state absent from strategy table
"""

from __future__ import annotations

import matplotlib
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from src.engine.game_state import PlayerAction
from src.solvers.baseline_dp import Action
from src.solvers.baseline_dp import solve as dp_solve
from src.solvers.cfr import CfrResult

# ─── Constants ────────────────────────────────────────────────────────────────

_TOTALS: list[int] = [16, 17, 18, 19, 20, 21]
_HAND_SIZES: list[int] = [2, 3, 4, 5]
_ROW_LABELS: list[str] = [str(t) for t in _TOTALS]
_COL_LABELS: list[str] = [f"nc={nc}" for nc in _HAND_SIZES]
_NAN_COLOR: str = "#cccccc"


# ─── Colormaps ────────────────────────────────────────────────────────────────


def _make_binary_cmap() -> matplotlib.colors.ListedColormap:
    """Red=STAND (0), Green=HIT (1), grey=absent (NaN)."""
    cmap = matplotlib.colors.ListedColormap(["#d62728", "#2ca02c"])
    cmap.set_bad(color=_NAN_COLOR)
    return cmap


def _make_continuous_cmap() -> matplotlib.colors.Colormap:
    """RdYlGn gradient: red=P(HIT)=0, green=P(HIT)=1, grey=absent (NaN)."""
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(color=_NAN_COLOR)
    return cmap


_BINARY_CMAP: matplotlib.colors.Colormap = _make_binary_cmap()
_CONTINUOUS_CMAP: matplotlib.colors.Colormap = _make_continuous_cmap()


# ─── Data builders ────────────────────────────────────────────────────────────


def build_dp_heatmap_data(reveal_mode: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (hard_matrix, soft_matrix) from the DP solver.

    Each matrix has shape (6, 4): rows = totals [16–21], cols = num_cards [2–5].
    Values: 1.0 = HIT, 0.0 = STAND, np.nan = state absent from strategy table.

    Args:
        reveal_mode: If True, use reveal=ON solver results.

    Returns:
        (hard_matrix, soft_matrix) each of dtype float64, shape (6, 4).
    """
    table = dp_solve(reveal_mode)
    strategy = {k: v[0] for k, v in table.items()}

    hard = np.full((6, 4), np.nan)
    soft = np.full((6, 4), np.nan)

    for r, total in enumerate(_TOTALS):
        for c, nc in enumerate(_HAND_SIZES):
            action_h = strategy.get((total, nc, 0))
            if action_h is not None:
                hard[r, c] = 1.0 if action_h == Action.HIT else 0.0

            action_s = strategy.get((total, nc, 1))
            if action_s is not None:
                soft[r, c] = 1.0 if action_s == Action.HIT else 0.0

    return hard, soft


def build_cfr_heatmap_data(result: CfrResult) -> tuple[np.ndarray, np.ndarray]:
    """Return (hard_matrix, soft_matrix) of P(HIT) from CFR Nash strategy.

    Each matrix has shape (6, 4): rows = totals [16–21], cols = num_cards [2–5].
    Values: P(HIT) in [0, 1], or np.nan for absent info sets.

    Args:
        result: CfrResult returned by cfr.solve().

    Returns:
        (hard_matrix, soft_matrix) each of dtype float64, shape (6, 4).
    """
    hard = np.full((6, 4), np.nan)
    soft = np.full((6, 4), np.nan)

    for info_set, probs in result.player_strategy.items():
        total = info_set.total
        nc = info_set.num_cards
        is_soft = info_set.is_soft

        if total not in _TOTALS or nc not in _HAND_SIZES:
            continue

        r = _TOTALS.index(total)
        c = _HAND_SIZES.index(nc)
        p_hit = probs.get(PlayerAction.HIT, 0.0)

        if is_soft:
            soft[r, c] = p_hit
        else:
            hard[r, c] = p_hit

    return hard, soft


# ─── Rendering helper ─────────────────────────────────────────────────────────


def _render_panel(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    binary: bool,
) -> matplotlib.image.AxesImage:
    """Render one heat-map panel onto *ax* and return the AxesImage.

    Sets axis ticks, tick labels, and cell annotations.  The caller is
    responsible for setting title, xlabel, and ylabel.
    """
    cmap = _BINARY_CMAP if binary else _CONTINUOUS_CMAP
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(_HAND_SIZES)))
    ax.set_xticklabels(_COL_LABELS, fontsize=9)
    ax.set_yticks(range(len(_TOTALS)))
    ax.set_yticklabels(_ROW_LABELS, fontsize=9)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if np.isnan(val):
                continue
            if binary:
                text = "H" if val >= 0.5 else "S"
                text_color = "white"
            else:
                text = f"{val:.2f}"
                text_color = "black" if 0.25 < val < 0.75 else "white"
            ax.text(
                c,
                r,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    return im


# ─── Public plot functions ────────────────────────────────────────────────────


def plot_strategy_heatmaps(
    hard_data: np.ndarray,
    soft_data: np.ndarray,
    title: str,
    *,
    binary: bool = True,
    show: bool = True,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot hard and soft strategy heat maps as a 1×2 figure.

    Args:
        hard_data: (6, 4) array for hard totals.  Values: 1.0=HIT, 0.0=STAND
                   (binary mode) or P(HIT) in [0,1] (continuous).  NaN=absent.
        soft_data: (6, 4) array for soft totals (same value convention).
        title:     Figure suptitle.
        binary:    True  → discrete red/green colormap + "H"/"S" annotations.
                   False → continuous RdYlGn colormap + P(HIT) annotations.
        show:      If True, call plt.show() after rendering.
        save_path: If not None, save the figure to this path before showing.

    Returns:
        matplotlib.figure.Figure.
    """
    fig, (ax_hard, ax_soft) = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    im_hard = _render_panel(ax_hard, hard_data, binary)
    im_soft = _render_panel(ax_soft, soft_data, binary)

    ax_hard.set_title("Hard totals", fontsize=10)
    ax_hard.set_xlabel("Hand size", fontsize=9)
    ax_hard.set_ylabel("Player total", fontsize=9)

    ax_soft.set_title("Soft totals", fontsize=10)
    ax_soft.set_xlabel("Hand size", fontsize=9)
    ax_soft.set_ylabel("Player total", fontsize=9)

    if not binary:
        plt.colorbar(im_hard, ax=ax_hard, label="P(HIT)", fraction=0.046, pad=0.04)
        plt.colorbar(im_soft, ax=ax_soft, label="P(HIT)", fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()

    return fig


def plot_dp_strategy_heatmaps(
    reveal_mode: bool = False,
    *,
    show: bool = True,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Convenience: build DP data and render binary H/S heat maps.

    Args:
        reveal_mode: If True, use reveal=ON solver results.
        show:        If True, call plt.show().
        save_path:   If not None, save to path.

    Returns:
        matplotlib.figure.Figure.
    """
    hard, soft = build_dp_heatmap_data(reveal_mode=reveal_mode)
    mode_label = "reveal=ON" if reveal_mode else "reveal=OFF"
    return plot_strategy_heatmaps(
        hard,
        soft,
        f"DP Solver Strategy  ({mode_label})",
        binary=True,
        show=show,
        save_path=save_path,
    )


def plot_cfr_strategy_heatmaps(
    result: CfrResult,
    *,
    show: bool = True,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Convenience: build CFR data and render P(HIT) heat maps.

    Args:
        result:    CfrResult from cfr.solve().
        show:      If True, call plt.show().
        save_path: If not None, save to path.

    Returns:
        matplotlib.figure.Figure.
    """
    hard, soft = build_cfr_heatmap_data(result)
    return plot_strategy_heatmaps(
        hard,
        soft,
        "CFR Nash Strategy  (P(HIT) by hand state)",
        binary=False,
        show=show,
        save_path=save_path,
    )


def plot_strategy_comparison(
    result: CfrResult,
    *,
    show: bool = True,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Side-by-side comparison: DP reveal=OFF, DP reveal=ON, CFR P(HIT).

    Produces a 2×3 figure:
        Row 0 = hard totals.
        Row 1 = soft totals.
        Col 0 = DP reveal=OFF  (binary H/S).
        Col 1 = DP reveal=ON   (binary H/S).
        Col 2 = CFR Nash P(HIT) (continuous RdYlGn + colorbar).

    Args:
        result:    CfrResult from cfr.solve().
        show:      If True, call plt.show().
        save_path: If not None, save to path.

    Returns:
        matplotlib.figure.Figure with 6 subplot axes.
    """
    hard_off, soft_off = build_dp_heatmap_data(reveal_mode=False)
    hard_on, soft_on = build_dp_heatmap_data(reveal_mode=True)
    hard_cfr, soft_cfr = build_cfr_heatmap_data(result)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Banluck Player Strategy Comparison", fontsize=14, fontweight="bold")

    col_titles = ["DP reveal=OFF", "DP reveal=ON", "CFR Nash P(HIT)"]
    row_names = ["Hard totals", "Soft totals"]
    binaries = [True, True, False]
    data_grid = [
        [hard_off, hard_on, hard_cfr],
        [soft_off, soft_on, soft_cfr],
    ]

    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            data = data_grid[row][col]
            binary = binaries[col]

            im = _render_panel(ax, data, binary)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{row_names[row]}\nTotal", fontsize=9)
            if row == 1:
                ax.set_xlabel("Hand size", fontsize=9)

            if not binary:
                plt.colorbar(im, ax=ax, label="P(HIT)", fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()

    return fig


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from src.solvers.cfr import solve

    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"Running CFR+ for {n_iter} iterations …")
    result = solve(n_iterations=n_iter)

    print("Generating strategy heat maps …")
    plot_dp_strategy_heatmaps(reveal_mode=False, show=False, save_path="dp_reveal_off.png")
    plot_dp_strategy_heatmaps(reveal_mode=True, show=False, save_path="dp_reveal_on.png")
    plot_cfr_strategy_heatmaps(result, show=False, save_path="cfr_strategy.png")
    plot_strategy_comparison(result, show=False, save_path="strategy_comparison.png")
    print("Saved: dp_reveal_off.png, dp_reveal_on.png, cfr_strategy.png, strategy_comparison.png")
