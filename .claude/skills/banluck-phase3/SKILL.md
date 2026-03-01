---
name: banluck-phase3
description: >
  Context and reference for Phase 3 of the Banluck GTO solver project.
  Covers payout structure, baseline EVs, completed analysis modules,
  and pointers to architecture documents.
---

# Banluck Solver — Phase 3 Reference

## Payout Table

| Hand            | Condition              | Player Payout |
|-----------------|------------------------|---------------|
| Ban Ban         | 2 aces                 | 3:1           |
| Ban Luck        | Ace + 10-value         | 2:1           |
| 777             | Three 7s (3 cards)     | 7:1           |
| Five-card 21    | 5 cards totaling 21    | 3:1           |
| Five-card <21   | 5 cards totaling <21   | 2:1           |
| Regular         | All other non-bust      | 1:1           |
| Five-card bust  | 5 cards, total >21     | −2 units      |

Dealer hard-15 surrender → push (overrides ALL outcomes including Ban Ban).
Player ≤15 forfeit → unconditional −1 unit (even on dealer bust).

## Baseline EVs

### DP Solver (infinite-deck approximation, no dealer surrender)

| Mode        | Player EV   |
|-------------|-------------|
| reveal=OFF  | +1.57%      |
| reveal=ON   | +2.41%      |
| delta       | +0.84%      |

### Monte Carlo (real deck, 200k hands, seed=42, no dealer surrender)

| Mode        | Player EV   |
|-------------|-------------|
| reveal=OFF  | −4.71%      |
| reveal=ON   | −3.75%      |
| delta       | +0.96%      |

> The ~6% gap between DP and MC is caused by deck-correlation effects
> (forfeit-forced-hit interaction) — not a bug in either solver.

### CFR+ Nash Equilibrium (1v1, dealer surrender enabled)

Nash EV is reported by `solve()` and printed via `print_nash_ev()`.
In heads-up play, REVEAL_PLAYER ≡ STAND (payoff-indifferent), so the
CFR GTO reveal frequency is not meaningfully comparable to the multi-player
DP/MC reveal deltas above.

## Research Question Answers (Phase 3)

| Q | Question | Answer |
|---|----------|--------|
| 1 | Reveal advantage? | +0.84% DP / +0.96% MC in multi-player; zero in 1v1 |
| 2 | Differs from BJ basic strategy? | TBD (Phase 3 open) |
| 3 | GTO reveal threshold at 16/17? | 1v1: indifferent. Multi-player requires explicit solver. |
| 4 | Hard-15 surrender saves dealer how often? | 7.10% of deals. Effective rate = 7.10% × P(surrender|hard-15) |
| 5 | Fair dealer rotation rate? | Every ~112 hands (N*/4 where N*≈450 = (σ/h)²) |

## Phase 3 Completed Modules

| File | Purpose | Tests |
|------|---------|-------|
| `src/analysis/strategy_report.py` | Nash EV, surrender, reveal, Q4 reports | 14 |
| `src/analysis/heat_maps.py` | Matplotlib strategy heat maps | 19 |
| `src/analysis/plotly_lookup.py` | Interactive Plotly strategy lookup | 20 |
| `src/analysis/bankroll.py` | Variance, risk-of-ruin, bankroll, rotation | 32 |
| `src/analysis/simulator.py` | Monte Carlo validator (also Phase 2.4) | 39 |

## Key Architecture Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Project instructions, engine architecture, test conventions |
| `banluck_PRD_v1.3.md` | Active PRD — all implementation decisions must reference this |
| `NEXT_ACTIONS.md` | Living task log; last entry = current state |
| `src/solvers/baseline_dp.py` | Phase 1.1 DP solver |
| `src/solvers/cfr.py` | Phase 2.2 CFR+ Nash solver (Numba JIT, ~6.7 ms/pass) |
| `src/solvers/information_sets.py` | PlayerHitStandInfoSet, DealerSurrenderInfoSet, DealerActionInfoSet |

## Running the Full Report

```bash
cd banluck-solver
PYTHONPATH=. python -m src.analysis.strategy_report 2000
```

## Running All Tests

```bash
cd banluck-solver
PYTHONPATH=. python -m pytest tests/ -v
```

Expected: 684+ tests passing.

## Architecture Cross-References

- Engine rules → `CLAUDE.md` § "Key Game Rules (Banluck-Specific)"
- Payout rules → `banluck_PRD_v1.3.md` § Special Hand Payouts
- CFR design decisions → `NEXT_ACTIONS.md` § Phase 2.2 CFR+ Solver Reference
- Phase 3 goals → `NEXT_ACTIONS.md` § Phase 3 — Analysis & Insights
