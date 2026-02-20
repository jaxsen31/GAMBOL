# NEXT_ACTIONS.md
> Active planning file — updated frequently. Reflects current project state and immediate next steps.
> Last updated: 2026-02-20 (Phase 1.1 broken into 12 atomic tasks)

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

### Design Summary
- **Infinite-deck approximation**: P(rank) = 1/13 for all 13 ranks. Eliminates deck tracking; error vs single-deck < 0.5%.
- **Composition state**: `(non_ace_total, num_aces, num_cards, dealer_upcard_rank)`. Required because Banluck's ace revaluation (11→10 when going from 2 to 3 cards) means `(total, is_soft)` alone can't drive transitions.
- **Selective reveal is a dealer strategic option, not a hard rule.** Two solver runs: `reveal_mode=ON` (dealer always reveals 3+-card players at 16/17, settling them against the pre-hit total) and `reveal_mode=OFF` (all players settle against the same final dealer total). EV delta between runs quantifies the impact before CFR.

### Fixed Dealer Strategy
- Hit until total ≥ 16 (forced), hit at 16, hit soft 17, stand hard 17+.
- Dealer never surrenders (handled in CFR phase).

### Atomic Tasks (each touches 1–2 files, adds < 100 lines)

- [ ] **1.1.A1 — Constants & scaffolding** (`src/solvers/baseline_dp.py`, new file)
  - `Action` enum (HIT, STAND)
  - `RANK_PROB = 1/13`, `RANK_VALUES_SOLVER` (13-element list mirroring `cards.py`)
  - `DealerOutcome` dataclass: `{final_dist: dict[int,float], bust_prob: float, init_16_prob: float, init_17_prob: float, init_16_final_dist: dict, init_17_final_dist: dict}`
  - Module docstring and empty function stubs for the layers below

- [ ] **1.1.A2 — Composition helpers** (`src/solvers/baseline_dp.py`)
  - `_total_from_composition(non_ace_total, num_aces, num_cards) -> int` — must exactly mirror `hand.py:calculate_total` logic
  - `_is_soft_from_composition(non_ace_total, num_aces, num_cards) -> bool`
  - `_transition(nat, na, nc, drawn_rank) -> tuple[int, int, int]` — returns new `(nat, na, nc)` after drawing a rank

- [ ] **1.1.A3 — Tests: composition helpers** (`tests/test_baseline_dp.py`, new file)
  - ~20 cases comparing `_total_from_composition` against `calculate_total()` (hard hands, soft hands, multi-ace, bust)
  - ~10 cases verifying `_is_soft_from_composition`
  - ~10 cases verifying `_transition` for various drawn ranks

- [ ] **1.1.A4 — Dealer outcome distribution** (`src/solvers/baseline_dp.py`)
  - `_dealer_play_recursive(nat, na, nc, memo) -> dict[str|int, float]` — recursive fixed-strategy simulation returning `{total: prob, 'bust': prob}`
  - `compute_dealer_outcomes(upcard_rank) -> DealerOutcome` — enumerates 13 hole cards (P=1/13 each), partitions into init_16/17 vs other paths, returns full distribution

- [ ] **1.1.A5 — Tests: dealer distributions** (`tests/test_baseline_dp.py`)
  - For each of 13 upcards: probabilities sum to 1.0 ± 1e-9
  - Dealer bust probability in range 10–35% for all upcards
  - `init_16_prob + init_17_prob` is in [0, 1] and equals expected value for each upcard

- [ ] **1.1.A6 — Settlement helper** (`src/solvers/baseline_dp.py`)
  - `_settle_ev(player_total, player_nc, dealer_total, dealer_nc, dealer_busted) -> float` — mirrors `rules.py` on abstracted state
  - Handles: player bust → -1.0, forfeit → -1.0, five-card vs five-card at 1:1, special hand type bonuses, regular total comparison

- [ ] **1.1.A7 — EV stand** (`src/solvers/baseline_dp.py`)
  - `_ev_stand(nat, na, nc, dealer_outcome: DealerOutcome, reveal_mode: bool) -> float`
  - total ≤ 15 → -1.0 (unconditional forfeit)
  - nc == 2: always settle vs full final distribution (same in both modes)
  - nc ≥ 3, reveal_mode=ON: weighted sum — P(init_16) × settle(vs 16) + P(init_17) × settle(vs 17) + P(other) × Σ settle(vs final)
  - nc ≥ 3, reveal_mode=OFF: settle vs full final distribution (same as nc=2 path)

- [ ] **1.1.A8 — EV hit & optimal EV** (`src/solvers/baseline_dp.py`)
  - `_ev_hit(nat, na, nc, dealer_outcome, memo) -> float` — enumerate 13 draw ranks, compute `_transition`, recurse; terminal cases: bust → -1.0, five-card ≤21 → `_ev_stand` with bonus type, total=21 → `_ev_stand`
  - `_optimal_ev(nat, na, nc, dealer_upcard_rank, memo) -> tuple[float, Action]` — returns `max(EV_hit, EV_stand)` and the corresponding action

- [ ] **1.1.A9 — Tests: strategy invariants** (`tests/test_baseline_dp.py`)
  - Stand on 20 EV > hit on 20 EV for all 13 upcards
  - Hit always optimal at total ≤ 11 (cannot bust, cannot forfeit)
  - Total ≤ 15 → EV_stand == -1.0 regardless of upcard
  - No action possible at total = 21 (stand is forced)
  - With reveal_mode=ON: EV_stand differs for nc=2 vs nc=3 at same total when dealer upcard makes 16/17 reachable
  - With reveal_mode=OFF: EV_stand is the same for nc=2 and nc=3 at the same total and upcard

- [ ] **1.1.A10 — Pre-settled hand EVs** (`src/solvers/baseline_dp.py`)
  - `_ev_ban_ban(upcard_rank) -> float` — P(dealer also ban_ban) × 0 + P(dealer ban_luck) × -3.0 + rest × +3.0
  - `_ev_ban_luck(upcard_rank) -> float` — similar: P(dealer ban_ban) × -2.0, P(dealer ban_luck) × 0, rest × +2.0
  - `_ev_777_bonus(upcard_rank) -> float` — EV of the 777 payout over final dealer distribution (dealer can only have 2-card hand at reveal, no selective reveal applies)

- [ ] **1.1.A11 — Public API: solve() and compute_house_edge()** (`src/solvers/baseline_dp.py`)
  - `solve(reveal_mode: bool) -> dict` — iterates all 13 upcards, calls `compute_dealer_outcomes`, runs `_optimal_ev` over all reachable starting compositions, returns strategy table keyed on `(total, nc, is_soft, upcard_rank)`
  - `compute_house_edge(strategy_table) -> float` — averages EV over all initial 2-card player deals (13² rank pairs × 13 upcards), includes Ban Ban/Ban Luck/777 contributions
  - Called twice: once with `reveal_mode=True`, once with `reveal_mode=False`

- [ ] **1.1.A12 — print_strategy_chart() + __main__ + end-to-end test** (`src/solvers/baseline_dp.py`, `tests/test_baseline_dp.py`)
  - `print_strategy_chart(strategy_table, label: str)` — terminal grid: rows=totals 16–21, cols=upcards 2–A, cells=H/S
  - `__main__` block: calls `solve(reveal_mode=True)` and `solve(reveal_mode=False)`, prints both charts + house edge for each + EV delta between the two modes
  - Test: `compute_house_edge(solve(reveal_mode=True))` and `solve(reveal_mode=False)` both in range -0.01 to -0.10 (house has edge, but not absurd)

---

## Phase 2 — CFR+ Full Nash Equilibrium (After Phase 1.1)

### Goal
Compute Nash equilibrium strategies for **both player and dealer** using Counterfactual Regret Minimization (CFR+). This handles the imperfect information from dealer's selective reveal.

**Key design decision:** REVEAL_PLAYER is an explicit **dealer action node** in the game tree at 16/17, not a pre-determined rule. The CFR solver determines the GTO reveal frequency (always, never, or a mixed strategy).

### Tasks (high-level, to be broken down when Phase 1.1 is complete)

- [ ] **2.1 — Information set design**
  - Player information set: `(own_cards, dealer_upcard)`
  - Dealer information set: `(own_cards, player_hand_size, revealed_player_outcomes)`
  - At 16/17: dealer action set = `{REVEAL_PLAYER, HIT, STAND}` — REVEAL_PLAYER is a strategic choice
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
