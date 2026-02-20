# NEXT_ACTIONS.md
> Active planning file — updated frequently. Reflects current project state and immediate next steps.
> Last updated: 2026-02-20

---

## Current State

**Phase 1 — Game Engine: COMPLETE ✅**

- All 6 engine modules implemented and tested
- 245/245 unit + integration tests passing
- All 14 PRD edge cases verified correct
- Test command: `cd banluck-solver && PYTHONPATH=. python -m pytest tests/ -v`

---

## Immediate Next Steps — Phase 1.1: Baseline DP Solver

### Goal
Compute the **optimal player strategy** against a fixed (non-strategic) dealer, using backward induction. Output a hit/stand chart keyed on `(player_total, hand_size, dealer_upcard)`.

### Fixed Dealer Strategy (assumption for Phase 1.1)
- Dealer is **forced** to hit until total ≥ 16
- At 16: hit
- At soft 17: hit
- At hard 17+: stand
- At 16/17: reveals 3+-card players first (selective reveal), then acts
- Dealer never surrenders (surrender strategy handled separately in CFR phase)

### Tasks

- [ ] **1.1.1 — `src/solvers/baseline_dp.py`**
  - Define state: `(player_cards: tuple, dealer_upcard: int, deck: tuple)`
  - Implement recursive EV calculation with memoisation
  - Use rank-based abstraction (suits irrelevant for totals) to reduce state space
  - Compute `EV_hit(state)` and `EV_stand(state)` for every reachable player state
  - Output: `optimal_action[player_total][hand_size][dealer_upcard]` → HIT or STAND

- [ ] **1.1.2 — `src/solvers/ev_tables.py`**
  - Pre-compute and store EV lookup tables
  - Format: pandas DataFrame, exportable to CSV
  - EV in units/hand and house edge %

- [ ] **1.1.3 — `src/analysis/simulator.py`**
  - Monte Carlo simulator: 1M+ hands using the DP-computed strategy
  - Cross-validate: simulated EV must match DP output within ±0.1%
  - Sanity checks: stand on 20 > hit on 20; house edge is positive

- [ ] **1.1.4 — Tests**
  - `tests/test_baseline_dp.py` — verify EV calculations on known hand scenarios
  - `tests/test_simulator.py` — statistical validation (χ², convergence)

- [ ] **1.1.5 — Output**
  - Player hit/stand chart (terminal printout + CSV export)
  - House edge summary (units/hand, %)

---

## Phase 2 — CFR+ Full Nash Equilibrium (After Phase 1.1)

### Goal
Compute Nash equilibrium strategies for **both player and dealer** using Counterfactual Regret Minimization (CFR+). This handles the imperfect information from dealer's selective reveal.

### Tasks (high-level, to be broken down when Phase 1.1 is complete)

- [ ] **2.1 — Information set design**
  - Player information set: `(own_cards, dealer_upcard)`
  - Dealer information set: `(own_cards, player_hand_size, revealed_player_outcomes)`
  - Encode as hashable tuples for regret table indexing

- [ ] **2.2 — `src/solvers/cfr.py`**
  - CFR+ regret update loop (Numba JIT for inner loop)
  - Strategy profile averaging
  - Convergence tracking: exploitability < 0.01 units/hand

- [ ] **2.3 — Dealer strategy outputs**
  - Optimal reveal threshold at 16/17: always reveal? Probabilistic?
  - Optimal surrender frequency for hard 15
  - Optimal hit/stand at 16 vs 2-card player

- [ ] **2.4 — Validation**
  - Monte Carlo: 1M hands at equilibrium strategies
  - EV match to CFR solution within ±0.1%

---

## Phase 3 — Analysis & Insights (After Phase 2)

- [ ] Quantify dealer selective reveal advantage (%)
- [ ] Quantify hard 15 surrender value (% of hands saved)
- [ ] Strategy heat maps: player action by (total, dealer upcard)
- [ ] Interactive Plotly lookup tool
- [ ] Variance and bankroll analysis
- [ ] Answer the 5 key research questions from PRD

---

## Research Questions (to answer by Phase 3)

| # | Question | Answer |
|---|----------|--------|
| 1 | How valuable is dealer selective reveal? | TBD |
| 2 | Does optimal play differ from BJ basic strategy? | TBD |
| 3 | GTO dealer reveal threshold at 16/17? | TBD |
| 4 | How often does hard 15 surrender save dealer? | TBD |
| 5 | Fair dealer rotation rate (every N hands)? | TBD |

---

## Deferred (Phase 4+)

- Multi-player solver (4 players + dealer)
- Card counting viability analysis
- GUI application
