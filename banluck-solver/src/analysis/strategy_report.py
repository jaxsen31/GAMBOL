"""Strategy report for the Banluck CFR+ Nash equilibrium solver.

Five public functions format a CfrResult into human-readable tables for
inspection and comparison with DP/MC baselines.

    print_nash_ev(result)          — Nash EV, exploitability, convergence
    print_surrender_strategy(result) — P(surrender) at hard-15
    print_dealer_strategy(result)  — dealer action probabilities at 16/17
    print_reveal_advantage(result) — reveal EV delta + GTO reveal frequency
    print_surrender_value(result)  — hard-15 surrender frequency & EV impact (Q4)
"""

from __future__ import annotations

from src.engine.game_state import DealerAction
from src.solvers.cfr import CfrResult, get_dealer_surrender_prob

# Hard-15 analytic frequency (infinite-deck, 2-card dealer hand).
# Ordered pairs summing to 15 with no ace:
#   (5, 10-val): 2 × (1/13)(4/13) = 8/169
#   (6, 9):      2 × (1/13)(1/13) = 2/169
#   (7, 8):      2 × (1/13)(1/13) = 2/169
# Total = 12/169
_HARD15_FREQ: float = 12.0 / 169.0

# ─── Known baselines ──────────────────────────────────────────────────────────

# Phase 1.1 DP solver (infinite-deck approximation, no surrender).
_DP_EV_REVEAL_OFF: float = 0.0157
_DP_EV_REVEAL_ON: float = 0.0241

# Phase 2.4 Monte Carlo (real deck, 200k hands, seed=42, no surrender).
_MC_EV_REVEAL_OFF: float = -0.0471
_MC_EV_REVEAL_ON: float = -0.0375


# ─── Public report functions ──────────────────────────────────────────────────

def print_nash_ev(result: CfrResult) -> None:
    """Print Nash EV summary and comparison to DP/MC baselines.

    Args:
        result: CfrResult returned by cfr.solve().
    """
    ev_pct = result.nash_ev * 100
    expl_pct = result.exploitability * 100

    print("=" * 56)
    print("Nash Equilibrium EV Summary")
    print("=" * 56)
    print(f"  Nash EV:         {result.nash_ev:+.4f} units  ({ev_pct:+.2f}%)")
    print(f"  Exploitability:  {result.exploitability:.4f} units  ({expl_pct:.2f}%)")
    print(f"  Iterations:      {result.n_iterations}")
    print(f"  Converged:       {'yes' if result.converged else 'no'}")
    print()
    print("  Comparison to baselines (player perspective):")
    print(f"    DP reveal=OFF  {_DP_EV_REVEAL_OFF:+.4f} ({_DP_EV_REVEAL_OFF*100:+.2f}%)  [infinite-deck]")
    print(f"    DP reveal=ON   {_DP_EV_REVEAL_ON:+.4f} ({_DP_EV_REVEAL_ON*100:+.2f}%)  [infinite-deck]")
    print(f"    MC reveal=OFF  {_MC_EV_REVEAL_OFF:+.4f} ({_MC_EV_REVEAL_OFF*100:+.2f}%)  [real-deck, 200k hands]")
    print(f"    MC reveal=ON   {_MC_EV_REVEAL_ON:+.4f} ({_MC_EV_REVEAL_ON*100:+.2f}%)  [real-deck, 200k hands]")
    print()


def print_surrender_strategy(result: CfrResult) -> None:
    """Print dealer surrender strategy — P(surrender) at each info set.

    Only hard-15 hands have a genuine choice; all other totals force
    P(continue) = 1.0 and are omitted from the output.

    Args:
        result: CfrResult returned by cfr.solve().
    """
    print("=" * 56)
    print("Dealer Surrender Strategy  (hard-15 only)")
    print("=" * 56)
    print(f"  {'Total':>5}  {'P(surrender)':>12}  {'P(continue)':>11}")
    print(f"  {'-----':>5}  {'------------':>12}  {'-----------':>11}")

    hard15_rows = [
        (info_set, probs)
        for info_set, probs in result.dealer_surrender_strategy.items()
        if info_set.is_hard_fifteen
    ]
    hard15_rows.sort(key=lambda kv: kv[0].total)

    if not hard15_rows:
        print("  (no hard-15 info sets found)")
    else:
        for info_set, probs in hard15_rows:
            p_surr = probs.get(DealerAction.SURRENDER, 0.0)
            p_cont = probs.get(DealerAction.HIT, 0.0)  # HIT = "continue" sentinel
            print(f"  {info_set.total:>5}  {p_surr:>12.4f}  {p_cont:>11.4f}")
    print()


def print_dealer_strategy(result: CfrResult) -> None:
    """Print dealer action strategy at total 16/17 (hit/stand/reveal).

    Rows are ordered by (dealer_total, is_soft, dealer_nc, player_nc).
    P(reveal) is 0.0 for player_nc == 2 rows since REVEAL_PLAYER is not
    a legal action when the player holds only 2 cards.

    Args:
        result: CfrResult returned by cfr.solve().
    """
    print("=" * 56)
    print("Dealer Action Strategy at 16/17")
    print("=" * 56)
    print(
        f"  {'Total':>5}  {'Soft':>4}  {'DNC':>3}  {'PNC':>3}  "
        f"{'P(hit)':>6}  {'P(stand)':>8}  {'P(reveal)':>9}"
    )
    print(
        f"  {'-----':>5}  {'----':>4}  {'---':>3}  {'---':>3}  "
        f"{'------':>6}  {'--------':>8}  {'---------':>9}"
    )

    rows = sorted(
        result.dealer_action_strategy.items(),
        key=lambda kv: (
            kv[0].dealer_total,
            kv[0].is_soft,
            kv[0].dealer_nc,
            kv[0].player_nc,
        ),
    )

    for info_set, probs in rows:
        p_hit = probs.get(DealerAction.HIT, 0.0)
        p_stand = probs.get(DealerAction.STAND, 0.0)
        p_rev = probs.get(DealerAction.REVEAL_PLAYER, 0.0)
        soft = "yes" if info_set.is_soft else "no"
        print(
            f"  {info_set.dealer_total:>5}  {soft:>4}  {info_set.dealer_nc:>3}  "
            f"{info_set.player_nc:>3}  {p_hit:>6.4f}  {p_stand:>8.4f}  {p_rev:>9.4f}"
        )
    print()


def print_reveal_advantage(result: CfrResult) -> None:
    """Print reveal advantage analysis and GTO reveal frequency at 16/17.

    Answers PRD Research Questions 1 and 3:
      Q1: How valuable is dealer's selective reveal? (EV delta across methods)
      Q3: What's the GTO dealer reveal strategy at 16/17?

    Section 1 shows EV delta between reveal=OFF and reveal=ON from the player's
    perspective using DP and MC baselines.  Section 2 aggregates P(reveal) from
    CFR Nash across all 3+-card-player DealerActionInfoSet nodes.  Section 3
    states concise answers to Q1 and Q3.

    Args:
        result: CfrResult returned by cfr.solve().
    """
    dp_delta = _DP_EV_REVEAL_ON - _DP_EV_REVEAL_OFF
    mc_delta = _MC_EV_REVEAL_ON - _MC_EV_REVEAL_OFF

    # ── Section 1: EV delta table ────────────────────────────────────────────
    print("=" * 56)
    print("Reveal Advantage — EV delta (player perspective)")
    print("=" * 56)
    print(f"  {'Source':<14}  {'reveal=OFF':>10}  {'reveal=ON':>9}  {'delta':>7}")
    print(f"  {'-'*14}  {'----------':>10}  {'---------':>9}  {'-------':>7}")
    print(
        f"  {'DP (inf-deck)':<14}  {_DP_EV_REVEAL_OFF*100:>+9.2f}%"
        f"  {_DP_EV_REVEAL_ON*100:>+8.2f}%  {dp_delta*100:>+6.2f}%"
    )
    print(
        f"  {'MC (real-deck)':<14}  {_MC_EV_REVEAL_OFF*100:>+9.2f}%"
        f"  {_MC_EV_REVEAL_ON*100:>+8.2f}%  {mc_delta*100:>+6.2f}%"
    )
    print(f"  {'CFR (1v1)':<14}  {'N/A':>10}  {'N/A':>9}  {'N/A':>7}")
    print(f"  (heads-up: reveal ≡ stand — no separate EV decomposition)")
    print()

    # ── Section 2: GTO reveal frequency from CFR ─────────────────────────────
    print("=" * 56)
    print("GTO Reveal Frequency at 16/17  (3+-card player only)")
    print("=" * 56)
    print(
        f"  {'Total':>5}  {'Soft':>4}  {'PNC':>3}"
        f"  {'P(hit)':>6}  {'P(stand)':>8}  {'P(reveal)':>9}"
    )
    print(
        f"  {'-----':>5}  {'----':>4}  {'---':>3}"
        f"  {'------':>6}  {'--------':>8}  {'---------':>9}"
    )

    reveal_nodes = sorted(
        (
            (info_set, probs)
            for info_set, probs in result.dealer_action_strategy.items()
            if info_set.player_nc >= 3
        ),
        key=lambda kv: (
            kv[0].dealer_total,
            kv[0].is_soft,
            kv[0].dealer_nc,
            kv[0].player_nc,
        ),
    )

    all_p_reveal: list[float] = []
    by_dealer_total: dict[int, list[float]] = {}

    for info_set, probs in reveal_nodes:
        p_hit = probs.get(DealerAction.HIT, 0.0)
        p_stand = probs.get(DealerAction.STAND, 0.0)
        p_rev = probs.get(DealerAction.REVEAL_PLAYER, 0.0)
        soft = "yes" if info_set.is_soft else "no"
        print(
            f"  {info_set.dealer_total:>5}  {soft:>4}  {info_set.player_nc:>3}"
            f"  {p_hit:>6.4f}  {p_stand:>8.4f}  {p_rev:>9.4f}"
        )
        all_p_reveal.append(p_rev)
        by_dealer_total.setdefault(info_set.dealer_total, []).append(p_rev)

    avg_reveal = sum(all_p_reveal) / len(all_p_reveal) if all_p_reveal else 0.0
    print()
    print(f"  Overall avg P(reveal) across 3+-card nodes: {avg_reveal:.4f}")
    for total in sorted(by_dealer_total):
        vals = by_dealer_total[total]
        avg_t = sum(vals) / len(vals)
        print(f"  Dealer-{total} avg P(reveal): {avg_t:.4f}")
    print()

    # ── Section 3: Research question answers ─────────────────────────────────
    print("=" * 56)
    print("Research Question Answers")
    print("=" * 56)
    print(
        f"  Q1 (Reveal value): Selective reveal is worth +{dp_delta*100:.2f}%"
        f" (DP, inf-deck) to +{mc_delta*100:.2f}% (MC, real-deck) in a"
        f" multi-player context (player perspective). In 1v1 heads-up, the"
        f" advantage is zero — reveal and stand are payoff-equivalent."
    )
    print(
        f"  Q3 (GTO at 16/17): In heads-up play CFR finds indifference between"
        f" REVEAL_PLAYER and STAND — both settle against the current dealer"
        f" total. Avg P(reveal) = {avg_reveal:.4f} across 3+-card-player nodes."
        f" Multi-player GTO requires an explicit multi-player solver."
    )
    print()


def print_surrender_value(result: CfrResult) -> None:
    """Print hard-15 surrender frequency and EV impact analysis.

    Answers PRD Research Question Q4:
      Q4: How powerful is hard-15 surrender? What % of hands does it save
          the dealer from losing?

    Section 1 shows the analytic hard-15 deal frequency (infinite-deck),
    the GTO P(surrender) from CFR Nash, and the effective surrender rate
    (proportion of all hands where the dealer actually surrenders).
    Section 2 compares Nash EV (surrender available) against the no-surrender
    DP/MC baselines to quantify how much surrender shifts the EV.
    Section 3 states a concise answer to Q4.

    Args:
        result: CfrResult returned by cfr.solve().
    """
    p_surrender = get_dealer_surrender_prob(result)
    effective_rate = _HARD15_FREQ * p_surrender

    # ── Section 1: Hard-15 frequency & surrender probability ─────────────────
    print("=" * 56)
    print("Hard-15 Surrender Value  (dealer perspective)")
    print("=" * 56)
    print(f"  Hard-15 deal frequency (analytic, inf-deck):")
    print(f"    12/169 = {_HARD15_FREQ*100:.2f}% of all 2-card dealer hands")
    print()
    print(f"  GTO P(surrender | hard-15):  {p_surrender:.4f}  ({p_surrender*100:.2f}%)")
    print(f"  Effective surrender rate:    {effective_rate:.4f}  ({effective_rate*100:.2f}%)")
    print(f"    (= hard-15 freq × P(surrender) — % of all hands where dealer surrenders)")
    print()

    # ── Section 2: EV impact (Nash vs no-surrender baselines) ────────────────
    print("=" * 56)
    print("Surrender EV Impact  (player perspective)")
    print("=" * 56)
    print(f"  {'Baseline':<22}  {'EV':>8}  {'vs Nash':>8}")
    print(f"  {'-'*22}  {'--------':>8}  {'--------':>8}")
    nash_ev = result.nash_ev
    baselines = [
        ("DP reveal=OFF (no surr)", _DP_EV_REVEAL_OFF),
        ("DP reveal=ON  (no surr)", _DP_EV_REVEAL_ON),
        ("MC reveal=OFF (no surr)", _MC_EV_REVEAL_OFF),
        ("MC reveal=ON  (no surr)", _MC_EV_REVEAL_ON),
        ("CFR Nash (surr enabled)", nash_ev),
    ]
    for label, ev in baselines:
        delta = nash_ev - ev if label != "CFR Nash (surr enabled)" else 0.0
        delta_str = f"{delta*100:+.2f}%" if label != "CFR Nash (surr enabled)" else "—"
        print(f"  {label:<22}  {ev*100:>+7.2f}%  {delta_str:>8}")
    print()
    print(f"  Note: Nash EV includes surrender as a strategic option for the dealer.")
    print(f"  A negative delta means surrender shifts EV toward the dealer (hurts player).")
    print()

    # ── Section 3: Research question Q4 answer ───────────────────────────────
    print("=" * 56)
    print("Research Question Answer")
    print("=" * 56)
    print(
        f"  Q4 (Hard-15 surrender): Dealer hard-15 arises in {_HARD15_FREQ*100:.2f}%"
        f" of deals. At Nash equilibrium the dealer surrenders with probability"
        f" {p_surrender:.4f}, making the effective surrender rate"
        f" {effective_rate*100:.2f}% of all hands. This converts certain losses"
        f" (e.g. vs Ban Ban/Luck) into pushes — its full EV benefit relative to"
        f" the no-surrender DP baselines is embedded in the Nash EV reported above."
    )
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from src.solvers.cfr import solve

    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"Running CFR+ for {n_iter} iterations …")
    result = solve(n_iterations=n_iter)

    print_nash_ev(result)
    print_surrender_strategy(result)
    print_dealer_strategy(result)
    print_reveal_advantage(result)
    print_surrender_value(result)
