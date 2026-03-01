"""Banluck GTO Solver — Streamlit Dashboard.

Four-tab interactive dashboard for exploring Banluck strategy results:
  Tab 1 — Strategy Heat Maps   (matplotlib, DP + CFR comparison)
  Tab 2 — Interactive Lookup   (Plotly, hover for action + EV margin)
  Tab 3 — Bankroll Analysis    (risk-of-ruin, horizon projections)
  Tab 4 — Strategy Report      (Nash EV, surrender, reveal, Q4 answers)

Run:
    cd banluck-solver
    PYTHONPATH=. streamlit run app.py
"""

from __future__ import annotations

import contextlib
import io

import matplotlib

matplotlib.use("Agg")  # must be set before any other matplotlib imports

import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Banluck GTO Solver",
    page_icon="🃏",
    layout="wide",
)

# ─── Lazy imports (inside functions to keep startup fast) ─────────────────────


@st.cache_resource
def _load_analysis_modules():
    """Import heavy analysis modules once (cached for the process lifetime)."""
    import numpy as np

    from src.analysis.bankroll import (
        compute_drawdown_stats,
        compute_fair_rotation,
        compute_horizon_projections,
        compute_variance_stats,
        print_rotation_analysis,
        print_variance_report,
        required_bankroll,
        risk_of_ruin,
    )
    from src.analysis.heat_maps import (
        build_cfr_heatmap_data,
        build_dp_heatmap_data,
        plot_cfr_strategy_heatmaps,
        plot_dp_strategy_heatmaps,
        plot_strategy_comparison,
    )
    from src.analysis.plotly_lookup import (
        build_cfr_lookup_figure,
        build_comparison_figure,
        build_dp_lookup_figure,
    )
    from src.analysis.simulator import (
        make_dp_player_strategy,
        simulate_hands,
    )
    from src.analysis.strategy_report import (
        print_dealer_strategy,
        print_nash_ev,
        print_reveal_advantage,
        print_surrender_strategy,
        print_surrender_value,
    )
    from src.engine.game_state import (
        _dealer_basic_hit_strategy,
        _dealer_never_surrenders,
    )

    return {
        "np": np,
        "build_dp_heatmap_data": build_dp_heatmap_data,
        "build_cfr_heatmap_data": build_cfr_heatmap_data,
        "plot_dp_strategy_heatmaps": plot_dp_strategy_heatmaps,
        "plot_cfr_strategy_heatmaps": plot_cfr_strategy_heatmaps,
        "plot_strategy_comparison": plot_strategy_comparison,
        "build_dp_lookup_figure": build_dp_lookup_figure,
        "build_cfr_lookup_figure": build_cfr_lookup_figure,
        "build_comparison_figure": build_comparison_figure,
        "compute_variance_stats": compute_variance_stats,
        "risk_of_ruin": risk_of_ruin,
        "required_bankroll": required_bankroll,
        "compute_horizon_projections": compute_horizon_projections,
        "compute_drawdown_stats": compute_drawdown_stats,
        "compute_fair_rotation": compute_fair_rotation,
        "print_variance_report": print_variance_report,
        "print_rotation_analysis": print_rotation_analysis,
        "simulate_hands": simulate_hands,
        "make_dp_player_strategy": make_dp_player_strategy,
        "print_nash_ev": print_nash_ev,
        "print_surrender_strategy": print_surrender_strategy,
        "print_dealer_strategy": print_dealer_strategy,
        "print_reveal_advantage": print_reveal_advantage,
        "print_surrender_value": print_surrender_value,
        "dealer_hit": _dealer_basic_hit_strategy,
        "dealer_no_surr": _dealer_never_surrenders,
    }


@st.cache_resource
def _run_cfr(n_iterations: int):
    """Run CFR+ solver and cache the result (keyed on iteration count)."""
    from src.solvers.cfr import solve

    return solve(n_iterations=n_iterations, convergence_check_every=n_iterations)


# ─── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🃏 Banluck GTO Solver")
    st.markdown("---")

    reveal_mode = st.selectbox(
        "Reveal mode (DP / MC)",
        options=[False, True],
        format_func=lambda v: "ON" if v else "OFF",
        index=0,
    )

    n_cfr_iterations = st.slider(
        "CFR iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )

    run_cfr = st.button("Run CFR Solver", type="primary")

    st.markdown("---")
    n_mc_hands = st.slider(
        "MC hands (bankroll tab)",
        min_value=10_000,
        max_value=200_000,
        value=50_000,
        step=10_000,
    )

    st.markdown("---")
    st.caption("Phase 3 — Banluck GTO Solver")
    st.caption("Engine → DP → CFR+ → Analysis")

# ─── CFR solver result ────────────────────────────────────────────────────────

# Trigger CFR solve if the button was pressed or a cached result exists.
cfr_result = None
if run_cfr or "cfr_result_cached" in st.session_state:
    with st.spinner(f"Running CFR+ ({n_cfr_iterations} iterations) … ~7 s + JIT cold start"):
        cfr_result = _run_cfr(n_cfr_iterations)
    st.session_state["cfr_result_cached"] = True
    st.sidebar.success(
        f"CFR done — Nash EV: {cfr_result.nash_ev:+.4f} | "
        f"Exploitability: {cfr_result.exploitability:.4f}"
    )

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Strategy Heat Maps",
        "Interactive Plotly Lookup",
        "Bankroll Analysis",
        "Strategy Report",
    ]
)

m = _load_analysis_modules()

# ── Tab 1: Strategy Heat Maps ─────────────────────────────────────────────────

with tab1:
    st.header("Strategy Heat Maps")
    st.caption(
        "Rows = player total (16–21) | Cols = num cards (2–5) | "
        "Green = HIT, Red = STAND, Grey = absent"
    )

    st.subheader(f"DP Strategy — reveal_mode={reveal_mode}")
    fig_dp = m["plot_dp_strategy_heatmaps"](reveal_mode=reveal_mode, show=False)
    st.pyplot(fig_dp)

    st.markdown("---")

    if cfr_result is not None:
        st.subheader("CFR Nash Strategy")
        fig_cfr = m["plot_cfr_strategy_heatmaps"](cfr_result, show=False)
        st.pyplot(fig_cfr)

        st.markdown("---")

        st.subheader("DP vs CFR Comparison (reveal=OFF, reveal=ON, CFR Nash)")
        fig_cmp = m["plot_strategy_comparison"](cfr_result, show=False)
        st.pyplot(fig_cmp)
    else:
        st.info("Press **Run CFR Solver** in the sidebar to see CFR heat maps and comparison.")

# ── Tab 2: Interactive Plotly Lookup ─────────────────────────────────────────

with tab2:
    st.header("Interactive Plotly Strategy Lookup")
    st.caption("Hover over any cell to see total, cards, action, and EV margin.")

    st.subheader(f"DP Lookup — reveal_mode={reveal_mode}")
    fig_dp_plotly = m["build_dp_lookup_figure"](reveal_mode=reveal_mode, show_ev_margin=True)
    st.plotly_chart(fig_dp_plotly, use_container_width=True)

    if cfr_result is not None:
        st.markdown("---")
        st.subheader("CFR Nash Lookup — P(HIT) heatmap")
        fig_cfr_plotly = m["build_cfr_lookup_figure"](cfr_result)
        st.plotly_chart(fig_cfr_plotly, use_container_width=True)

        st.markdown("---")
        st.subheader("DP vs CFR Comparison")
        fig_cmp_plotly = m["build_comparison_figure"](cfr_result)
        st.plotly_chart(fig_cmp_plotly, use_container_width=True)
    else:
        st.info("Press **Run CFR Solver** in the sidebar to see CFR and comparison figures.")

# ── Tab 3: Bankroll Analysis ──────────────────────────────────────────────────

with tab3:
    st.header("Bankroll Analysis")
    st.caption(
        "Monte Carlo simulation used to estimate per-hand variance. "
        "Risk-of-ruin and horizon projections computed via CLT."
    )

    with st.spinner(f"Simulating {n_mc_hands:,} hands …"):
        player_strat = m["make_dp_player_strategy"](reveal_mode=reveal_mode)
        sim = m["simulate_hands"](
            player_strat,
            m["dealer_no_surr"],
            m["dealer_hit"],
            n_hands=n_mc_hands,
            seed=42,
            reveal_mode=reveal_mode,
            return_payouts=True,
        )

    # Descriptive stats
    vs = m["compute_variance_stats"](sim.payouts)

    st.subheader("Distribution Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean EV / hand", f"{vs.mean:+.4f}")
    col2.metric("Std dev", f"{vs.std:.4f}")
    col3.metric("Skewness", f"{vs.skewness:.3f}")
    col4.metric("Kurtosis", f"{vs.kurtosis:.3f}")

    import pandas as pd

    pct_df = pd.DataFrame(
        {"Percentile": list(vs.percentiles.keys()), "Value": list(vs.percentiles.values())}
    )
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Risk of Ruin")

    bankrolls = [20, 50, 100, 200]
    ror_rows = []
    for br in bankrolls:
        ror = m["risk_of_ruin"](br, vs.mean, vs.std)
        ror_rows.append(
            {"Bankroll (units)": br, "P(ruin)": f"{ror:.4f}", "P(ruin) %": f"{ror * 100:.2f}%"}
        )
    st.dataframe(pd.DataFrame(ror_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Horizon Projections (CLT)")

    horizons = [100, 500, 1000, 5000, 10000]
    proj = m["compute_horizon_projections"](vs.mean, vs.std, horizons=horizons)
    proj_rows = [
        {
            "Hands": p.n_hands,
            "Expected Profit": f"{p.expected_profit:+.2f}",
            "CI Low": f"{p.ci_low:+.2f}",
            "CI High": f"{p.ci_high:+.2f}",
            "P(profit > 0)": f"{p.prob_positive:.3f}",
        }
        for p in proj
    ]
    st.dataframe(pd.DataFrame(proj_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Fair Dealer Rotation (Q5)")

    rotation = m["compute_fair_rotation"](abs(vs.mean), vs.std)
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended rotation", f"Every {rotation.recommended_n} hands")
    col2.metric("Within-noise N*", f"~{rotation.recommended_n * 4} hands")
    col3.metric("Edge (abs)", f"{rotation.edge * 100:.3f}%/hand")
    st.caption(rotation.explanation)

    st.markdown("---")
    st.subheader("Full Variance Report (stdout capture)")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # required_bankroll() only works when the player has a positive edge.
        # With negative edge (house advantage), bankroll requirement is infinite.
        if vs.mean > 0:
            br_reqs = [m["required_bankroll"](vs.mean, vs.std, sp) for sp in [0.90, 0.95, 0.99]]
        else:
            br_reqs = []
        drawdown = m["compute_drawdown_stats"](
            sim.payouts, n_trajectories=200, trajectory_length=500
        )
        m["print_variance_report"](vs, br_reqs, proj, drawdown, rotation)
        m["print_rotation_analysis"](rotation)
    st.code(buf.getvalue(), language=None)

# ── Tab 4: Strategy Report ────────────────────────────────────────────────────

with tab4:
    st.header("Strategy Report")

    if cfr_result is not None:
        st.caption("CFR+ Nash equilibrium strategies and research question answers.")

        for section_fn, label in [
            (m["print_nash_ev"], "Nash EV Summary"),
            (m["print_surrender_strategy"], "Dealer Surrender Strategy"),
            (m["print_dealer_strategy"], "Dealer Action Strategy at 16/17"),
            (m["print_reveal_advantage"], "Reveal Advantage (Q1 + Q3)"),
            (m["print_surrender_value"], "Hard-15 Surrender Value (Q4)"),
        ]:
            st.subheader(label)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                section_fn(cfr_result)
            st.code(buf.getvalue(), language=None)
    else:
        st.info("Press **Run CFR Solver** in the sidebar to see the strategy report.")
