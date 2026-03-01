"""Variance and bankroll analysis for Banluck.

Provides:
- Distribution statistics (mean, std, skewness, kurtosis, percentiles)
- Risk-of-ruin and required bankroll (classic gambler's ruin formula)
- Horizon projections via CLT (expected profit + confidence intervals)
- Bootstrap drawdown analysis (max drawdown distribution)
- Fair dealer rotation analysis (Q5: every N hands to balance advantage)

Usage (standalone report):
    cd banluck-solver && PYTHONPATH=. python -m src.analysis.bankroll
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# ─── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class VarianceStats:
    """Descriptive statistics for a per-hand payout distribution.

    Attributes:
        mean:        Mean payout per hand (units).
        std:         Sample standard deviation.
        variance:    Sample variance (std**2).
        skewness:    Fisher skewness of the payout distribution.
        kurtosis:    Excess kurtosis (Fisher, normal = 0).
        percentiles: Dict mapping percentile label to value.
                     Keys: 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99'.
        n_hands:     Number of hands in the sample.
    """

    mean: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    percentiles: dict[str, float]
    n_hands: int


@dataclass
class BankrollRequirement:
    """Bankroll required to survive a session at a given confidence level.

    Attributes:
        survival_prob: Target probability of not going broke (e.g. 0.95).
        required_bankroll: Units needed at that survival probability.
        edge:          Mean payout per hand (player edge, positive = advantage).
        std:           Per-hand standard deviation.
        method:        Formula used ('gambler_ruin_approx').
    """

    survival_prob: float
    required_bankroll: float
    edge: float
    std: float
    method: str = "gambler_ruin_approx"


@dataclass
class HorizonProjection:
    """Expected profit and uncertainty at a given number of hands.

    Attributes:
        n_hands:         Number of hands in this horizon.
        expected_profit: n_hands * edge (units).
        ci_low:          Lower bound of confidence interval (units).
        ci_high:         Upper bound of confidence interval (units).
        prob_positive:   Probability that cumulative profit > 0 (CLT).
    """

    n_hands: int
    expected_profit: float
    ci_low: float
    ci_high: float
    prob_positive: float


@dataclass
class DrawdownStats:
    """Max drawdown distribution from bootstrapped trajectories.

    Attributes:
        mean_max_drawdown:   Mean of max-drawdown across trajectories (units).
        median_max_drawdown: Median max-drawdown (units).
        p95_max_drawdown:    95th-percentile max-drawdown (units).
        n_trajectories:      Number of bootstrap trajectories used.
    """

    mean_max_drawdown: float
    median_max_drawdown: float
    p95_max_drawdown: float
    n_trajectories: int


@dataclass
class RotationAnalysis:
    """Fair dealer rotation cadence to balance the position advantage.

    The dealer position in Banluck has a different EV than the player
    position. The signal-to-noise ratio for detecting this advantage
    grows as sqrt(N). We define the "within-noise" threshold as the N
    where the cumulative advantage equals one standard deviation of the
    cumulative payout — i.e. the advantage is statistically indistinct
    from variance over that many hands.

    Recommended rotation: before the advantage becomes clearly detectable,
    typically N/2 or N/4 of the within-noise threshold.

    Attributes:
        edge:              Mean payout per hand (absolute, positive).
        std:               Per-hand standard deviation.
        hands_for_1pct:    Hands until expected profit = 1% of bankroll
                           (informational; uses 100-unit bankroll).
        hands_for_half_pct: Same but for 0.5%.
        recommended_n:     Suggested rotation interval (hands).
        explanation:       Human-readable rationale.
    """

    edge: float
    std: float
    hands_for_1pct: int
    hands_for_half_pct: int
    recommended_n: int
    explanation: str


# ─── Computation functions ────────────────────────────────────────────────────


def compute_variance_stats(payouts: np.ndarray) -> VarianceStats:
    """Compute descriptive statistics for a per-hand payout distribution.

    Args:
        payouts: 1-D float64 array of per-hand payouts.

    Returns:
        VarianceStats with mean, std, variance, skewness, kurtosis,
        percentiles (p1/p5/p25/p50/p75/p95/p99), and n_hands.
    """
    n = len(payouts)
    mean = float(np.mean(payouts))
    std = float(np.std(payouts, ddof=1))
    variance = std**2
    skewness = float(stats.skew(payouts))
    kurt = float(stats.kurtosis(payouts))  # excess (Fisher), normal=0
    pct_values = np.percentile(payouts, [1, 5, 25, 50, 75, 95, 99])
    percentiles = {
        "p1": float(pct_values[0]),
        "p5": float(pct_values[1]),
        "p25": float(pct_values[2]),
        "p50": float(pct_values[3]),
        "p75": float(pct_values[4]),
        "p95": float(pct_values[5]),
        "p99": float(pct_values[6]),
    }
    return VarianceStats(
        mean=mean,
        std=std,
        variance=variance,
        skewness=skewness,
        kurtosis=kurt,
        percentiles=percentiles,
        n_hands=n,
    )


def risk_of_ruin(bankroll: float, edge: float, std: float) -> float:
    """Probability of ruin given a fixed bankroll, edge, and per-hand std.

    Uses the classic gambler's ruin approximation for a random walk:
        RoR = exp(-2 * edge * bankroll / variance)

    This assumes the edge and variance are constant and the walk is
    continuous. Returns 1.0 when edge <= 0 (certain ruin eventually).

    Args:
        bankroll: Starting capital in units (must be > 0).
        edge:     Mean payout per hand (player's perspective, units).
        std:      Per-hand standard deviation (units).

    Returns:
        Probability of ruin in [0, 1].
    """
    if edge <= 0:
        return 1.0
    variance = std**2
    return float(math.exp(-2.0 * edge * bankroll / variance))


def required_bankroll(
    edge: float,
    std: float,
    survival_prob: float,
) -> BankrollRequirement:
    """Invert the ruin formula to find the bankroll for a target survival probability.

    Solves RoR = exp(-2*edge*B/variance) for B:
        B = -variance * ln(1 - survival_prob) / (2 * edge)

    Args:
        edge:          Mean payout per hand (must be > 0).
        std:           Per-hand standard deviation.
        survival_prob: Target survival probability in (0, 1), e.g. 0.95.

    Returns:
        BankrollRequirement with the required bankroll.

    Raises:
        ValueError: If edge <= 0 (ruin formula undefined; bankroll → ∞).
    """
    if edge <= 0:
        raise ValueError(
            f"required_bankroll() requires a positive edge; got edge={edge:.6f}. "
            "With zero or negative edge the gambler's ruin formula gives infinite bankroll."
        )
    variance = std**2
    b = -variance * math.log(1.0 - survival_prob) / (2.0 * edge)
    return BankrollRequirement(
        survival_prob=survival_prob,
        required_bankroll=b,
        edge=edge,
        std=std,
    )


def compute_horizon_projections(
    edge: float,
    std: float,
    horizons: list[int] | None = None,
    confidence: float = 0.95,
) -> list[HorizonProjection]:
    """CLT-based profit projections at multiple hand-count horizons.

    By CLT, cumulative profit after N hands ~ N(N*edge, N*variance).
    Confidence interval: [N*edge ± z * std * sqrt(N)].
    Probability positive: Phi(sqrt(N) * edge / std) when std > 0.

    Args:
        edge:       Mean payout per hand.
        std:        Per-hand standard deviation.
        horizons:   List of hand counts to project. Defaults to
                    [100, 500, 1000, 5000, 10000].
        confidence: Confidence level for the interval (default 0.95).

    Returns:
        List of HorizonProjection, one per horizon, in input order.
    """
    if horizons is None:
        horizons = [100, 500, 1000, 5000, 10_000]

    z = stats.norm.ppf((1.0 + confidence) / 2.0)
    projections = []
    for n in horizons:
        expected = n * edge
        margin = z * std * math.sqrt(n)
        if std > 0:
            prob_pos = float(stats.norm.cdf(math.sqrt(n) * edge / std))
        else:
            prob_pos = 1.0 if edge > 0 else 0.0
        projections.append(
            HorizonProjection(
                n_hands=n,
                expected_profit=expected,
                ci_low=expected - margin,
                ci_high=expected + margin,
                prob_positive=prob_pos,
            )
        )
    return projections


def compute_drawdown_stats(
    payouts: np.ndarray,
    n_trajectories: int = 1000,
    trajectory_length: int = 500,
    seed: int = 0,
) -> DrawdownStats:
    """Bootstrap-resample trajectories and compute max-drawdown distribution.

    Each trajectory is built by sampling (with replacement) from the
    observed payout array, then computing the running cumulative sum and
    the maximum drawdown (peak-to-trough).

    Args:
        payouts:           Observed per-hand payouts (1-D float64).
        n_trajectories:    Number of bootstrap trajectories (default 1000).
        trajectory_length: Hands per trajectory (default 500).
        seed:              NumPy random seed for reproducibility.

    Returns:
        DrawdownStats with mean, median, and p95 max-drawdown, plus count.
    """
    rng = np.random.default_rng(seed)
    max_drawdowns = np.empty(n_trajectories, dtype=np.float64)

    for i in range(n_trajectories):
        sample = rng.choice(payouts, size=trajectory_length, replace=True)
        cumsum = np.cumsum(sample)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_drawdowns[i] = float(np.max(drawdowns))

    return DrawdownStats(
        mean_max_drawdown=float(np.mean(max_drawdowns)),
        median_max_drawdown=float(np.median(max_drawdowns)),
        p95_max_drawdown=float(np.percentile(max_drawdowns, 95)),
        n_trajectories=n_trajectories,
    )


def compute_fair_rotation(dealer_edge: float, std: float) -> RotationAnalysis:
    """Compute Q5: fair dealer rotation interval to balance positional advantage.

    The within-noise threshold N* is where the cumulative advantage equals
    one std of the cumulative random walk:
        N* * |edge| = std * sqrt(N*)
        => N* = (std / |edge|)^2

    Below N* hands, the positional advantage is indistinguishable from
    normal variance. We recommend rotating at N*/4 so that no player
    accrues a statistically detectable advantage from the dealer position.

    Uses a 100-unit reference bankroll for the percentage calculations.

    Args:
        dealer_edge: Absolute per-hand edge of the dealer position (units).
                     Must be positive (take abs value before calling if needed).
        std:         Per-hand standard deviation.

    Returns:
        RotationAnalysis with recommended rotation N and explanation.
    """
    abs_edge = abs(dealer_edge)

    # Within-noise threshold
    if abs_edge > 0:
        within_noise_n = int((std / abs_edge) ** 2)
    else:
        within_noise_n = 10_000  # edge ≈ 0 → rotation frequency arbitrary

    # Hands until expected profit = 1% or 0.5% of a 100-unit bankroll
    reference_bankroll = 100.0
    if abs_edge > 0:
        hands_1pct = int(math.ceil(0.01 * reference_bankroll / abs_edge))
        hands_half_pct = int(math.ceil(0.005 * reference_bankroll / abs_edge))
    else:
        hands_1pct = within_noise_n
        hands_half_pct = within_noise_n

    # Recommend rotating at 1/4 of the within-noise threshold
    recommended_n = max(1, within_noise_n // 4)

    explanation = (
        f"The dealer position advantage of {abs_edge:.4f} units/hand becomes "
        f"statistically detectable (signal ≥ 1σ noise) after {within_noise_n:,} hands. "
        f"Rotating every {recommended_n:,} hands (N*/4) keeps the positional "
        f"advantage well within one standard deviation of the cumulative payout, "
        f"ensuring no player accumulates an unfair share of the dealer advantage "
        f"over a typical session."
    )

    return RotationAnalysis(
        edge=abs_edge,
        std=std,
        hands_for_1pct=hands_1pct,
        hands_for_half_pct=hands_half_pct,
        recommended_n=recommended_n,
        explanation=explanation,
    )


# ─── Output functions ─────────────────────────────────────────────────────────


def print_variance_report(
    stats: VarianceStats,
    bankroll_reqs: list[BankrollRequirement],
    projections: list[HorizonProjection],
    drawdown: DrawdownStats,
    rotation: RotationAnalysis,
    *,
    label: str = "",
) -> str:
    """Format and print a full variance and bankroll report.

    Args:
        stats:         VarianceStats from compute_variance_stats().
        bankroll_reqs: List of BankrollRequirement at different survival probs.
        projections:   List of HorizonProjection from compute_horizon_projections().
        drawdown:      DrawdownStats from compute_drawdown_stats().
        rotation:      RotationAnalysis from compute_fair_rotation().
        label:         Optional label for the header (e.g. 'reveal_mode=OFF').

    Returns:
        The formatted report string (also printed to stdout).
    """
    header = f"Variance & Bankroll Report{' — ' + label if label else ''}"
    lines = [
        "=" * 70,
        header,
        "=" * 70,
        "",
        "── Distribution Statistics ─────────────────────────────────────────",
        f"  Hands simulated : {stats.n_hands:>10,}",
        f"  Mean EV / hand  : {stats.mean:>+10.4f} units  ({stats.mean * 100:+.2f}%)",
        f"  Std deviation   : {stats.std:>10.4f} units",
        f"  Variance        : {stats.variance:>10.4f}",
        f"  Skewness        : {stats.skewness:>10.4f}",
        f"  Excess kurtosis : {stats.kurtosis:>10.4f}",
        "",
        "  Percentiles (units):",
        f"    p1={stats.percentiles['p1']:.2f}  p5={stats.percentiles['p5']:.2f}  "
        f"p25={stats.percentiles['p25']:.2f}  p50={stats.percentiles['p50']:.2f}  "
        f"p75={stats.percentiles['p75']:.2f}  p95={stats.percentiles['p95']:.2f}  "
        f"p99={stats.percentiles['p99']:.2f}",
        "",
        "── Bankroll Requirements ────────────────────────────────────────────",
    ]
    for req in bankroll_reqs:
        lines.append(
            f"  Survival {req.survival_prob * 100:.0f}%   : "
            f"{req.required_bankroll:>8.1f} units  ({req.method})"
        )
    lines += [
        "",
        "── Horizon Projections (CLT, 95% CI) ───────────────────────────────",
        f"  {'Hands':>8}  {'E[profit]':>10}  {'CI low':>10}  {'CI high':>10}  {'P(+)':>6}",
        f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 6}",
    ]
    for p in projections:
        lines.append(
            f"  {p.n_hands:>8,}  {p.expected_profit:>+10.2f}  "
            f"{p.ci_low:>+10.2f}  {p.ci_high:>+10.2f}  {p.prob_positive:>5.1%}"
        )
    lines += [
        "",
        "── Drawdown Analysis (bootstrap) ────────────────────────────────────",
        f"  Trajectories    : {drawdown.n_trajectories:,}",
        f"  Mean max DD     : {drawdown.mean_max_drawdown:.2f} units",
        f"  Median max DD   : {drawdown.median_max_drawdown:.2f} units",
        f"  p95 max DD      : {drawdown.p95_max_drawdown:.2f} units",
        "",
    ]
    lines += print_rotation_analysis(rotation, _return_only=True)
    lines.append("")
    report = "\n".join(lines)
    print(report)
    return report


def print_rotation_analysis(
    rotation: RotationAnalysis,
    *,
    _return_only: bool = False,
) -> list[str] | None:
    """Print (or return) the Q5 dealer rotation analysis.

    Args:
        rotation:     RotationAnalysis from compute_fair_rotation().
        _return_only: Internal flag; if True, return lines without printing.

    Returns:
        List of lines if _return_only=True, else None.
    """
    lines = [
        "── Q5: Fair Dealer Rotation ─────────────────────────────────────────",
        f"  Dealer edge     : {rotation.edge:+.4f} units/hand",
        f"  Per-hand std    : {rotation.std:.4f}",
        f"  Hands for 1% gain  (100-unit bankroll): {rotation.hands_for_1pct:,}",
        f"  Hands for 0.5% gain (100-unit bankroll): {rotation.hands_for_half_pct:,}",
        f"  Recommended rotation: every {rotation.recommended_n:,} hands",
        "",
        f"  {rotation.explanation}",
    ]
    if _return_only:
        return lines
    print("\n".join(lines))
    return None


# ─── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from src.analysis.simulator import (
        make_dp_player_strategy,
        make_fixed_dealer_strategy,
        simulate_hands,
    )

    n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    print(f"Banluck Variance & Bankroll Analysis — {n_hands:,} hands per mode\n")

    surrender_strat, hit_strat = make_fixed_dealer_strategy()
    survival_probs = [0.90, 0.95, 0.99]

    for reveal in (False, True):
        label = f"reveal_mode={'ON' if reveal else 'OFF'}"
        print(f"\nRunning MC simulation: {label} ...")
        result = simulate_hands(
            player_strategy=make_dp_player_strategy(reveal_mode=reveal),
            dealer_surrender_strategy=surrender_strat,
            dealer_hit_strategy=hit_strat,
            n_hands=n_hands,
            seed=42,
            reveal_mode=reveal,
            return_payouts=True,
        )
        assert result.payouts is not None

        vs = compute_variance_stats(result.payouts)
        reqs = []
        if vs.mean > 0:
            for sp in survival_probs:
                reqs.append(required_bankroll(vs.mean, vs.std, sp))
        else:
            print("  (Edge ≤ 0 — bankroll requirements not applicable)")

        projections = compute_horizon_projections(vs.mean, vs.std)
        dd = compute_drawdown_stats(result.payouts, n_trajectories=500)

        # Q5: use absolute edge
        rotation = compute_fair_rotation(abs(vs.mean), vs.std)

        print_variance_report(vs, reqs, projections, dd, rotation, label=label)
