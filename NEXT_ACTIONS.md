# NEXT_ACTIONS.md
> Active planning file — updated frequently. Reflects current project state and immediate next steps.
> Last updated: 2026-02-24 (A1–A13 complete + 5-card bust rule clarified; Phase 1.1 fully done — Phase 2 CFR is next)

---

## Current State

**Phase 1 — Game Engine: COMPLETE ✅**

- All 6 engine modules implemented and tested
- 245/245 unit + integration tests passing
- All 14 PRD edge cases verified correct
- Test command: `cd banluck-solver && PYTHONPATH=. python -m pytest tests/ -v`

**Phase 1.1 — DP Solver Foundation: IN PROGRESS**

- A1 ✅ Constants & scaffolding (`src/solvers/baseline_dp.py` created — `Action`, `RANK_PROB`, `RANK_VALUES_SOLVER`, `DealerOutcome`, stubs A4–A11)
- A2 ✅ Composition helpers (`_total_from_composition`, `_is_soft_from_composition`, `_transition`)
- A3 ✅ Tests: composition helpers (`tests/test_baseline_dp.py`, 50 tests — all passing)
- A4 ✅ Dealer outcome distribution (`_dealer_play_recursive`, `compute_dealer_distribution` — all 13 upcards verified, bust+final_dist=1.0)
- A5 ✅ Tests: dealer distributions (17 tests — probability invariants, bust range, analytical init_16/17 values, conditional dist properties, T/J/Q/K symmetry). **Note: bust range is 10–45%, not 10–35% as originally spec'd — upcard 6 busts at 43.3%.**
- A6 ✅ Settlement helper (`_settle_ev`, `_abstract_hand_type`, `_ABSTRACT_HIERARCHY`, `_ABSTRACT_PAYOUT` — mirrors rules.py on abstracted totals/card counts)
- A7 ✅ EV stand (`_ev_vs_dist`, `_ev_stand` — forfeit, reveal_mode=OFF/nc=2 vs full dist, reveal_mode=ON correction formula)
- A8 ✅ EV hit & optimal EV (`_ev_hit`, `_optimal_ev`, `optimal_action` wired up — backward induction core complete)
- **Design correction**: no dealer upcard in Banluck — added `compute_marginal_dealer_distribution()`, updated all stubs to key on `(player_total, hand_size, is_soft)` not `dealer_upcard`
- A9 ✅ Tests: strategy invariants (19 tests — stand-20 dominance, hit-at-≤11, forfeit-at-≤15, forced-stand-21, reveal-mode nc=2/nc=3 divergence, reveal_mode=OFF symmetry)
- Total: **344/344 tests passing**
- A10 ✅ Pre-settled hand EVs (18 tests; total 362/362)
- A11 ✅ Public API: solve(), compute_house_edge(), wrappers (24 tests; total 386/386)
- A12 ✅ print_strategy_chart() + __main__ + end-to-end tests (9 tests; total **395/395 passing**)
- **5-card bust rule clarified** ✅ — Player/dealer 5-card bust costs **2 units** (symmetric with 5-card bonus). Engine (rules.py) + DP (_ev_hit, _settle_ev) updated. 5 new engine tests + 5 new DP tests; total **404/404 passing**.
  - reveal_mode=OFF player EV: **+1.57%**; reveal_mode=ON: **+2.41%**; delta: **+0.83%**
  - (Was +4.22%/+4.98% before 5-card bust penalty was applied)
  - House edge only emerges in Phase 2 when dealer surrender is added as a CFR option
- A13 ✅ EV margin column in print_strategy_chart() (6 new tests; total **410/410 passing**)
  - `build_ev_margin_table(reveal_mode)` → `{(total, nc, is_soft): float | None}`
  - `print_strategy_chart(..., ev_margin_table=..., show_ev_margin=True)` prints |HIT−STAND| grid
  - Hard-16 nc=4 forces HIT with margin 0.053; nc=4 soft-16 margin 1.822 (five-card bonus dominant)

**Phase 1.1 — COMPLETE ✅**

---

## Immediate Next Steps — Phase 1.1: Baseline DP Solver

### Goal
Compute the **optimal player strategy** against a fixed (non-strategic) dealer, using backward induction. Output a hit/stand chart keyed on `(player_total, hand_size, dealer_upcard)`.

### Design Summary
- **Infinite-deck approximation**: P(rank) = 1/13 for all 13 ranks. Eliminates deck tracking; error vs single-deck < 0.5%.
- **Composition state**: `(non_ace_total, num_aces, num_cards)`. Required because Banluck's ace revaluation (11→10 when going from 2 to 3 cards) means `(total, is_soft)` alone can't drive transitions.
- **No dealer upcard visible to player.** In Banluck all dealer cards are face-down. Player strategy depends only on the player's own hand. The solver uses `compute_marginal_dealer_distribution()` — the dealer distribution averaged over all 13 possible upcard ranks. Strategy table keyed on `(player_total, hand_size, is_soft)` NOT `dealer_upcard`.
- **Selective reveal is a dealer strategic option, not a hard rule.** Two solver runs: `reveal_mode=ON` (dealer always reveals 3+-card players at 16/17, settling them against the pre-hit total) and `reveal_mode=OFF` (all players settle against the same final dealer total). EV delta between runs quantifies the impact before CFR.

### Fixed Dealer Strategy
- Hit until total ≥ 16 (forced), hit at 16, hit soft 17, stand hard 17+.
- Dealer never surrenders (handled in CFR phase).

### Atomic Tasks (each touches 1–2 files, adds < 100 lines)

- [x] **1.1.A1 — Constants & scaffolding** (`src/solvers/baseline_dp.py`, new file)
  - `Action` enum (HIT, STAND)
  - `RANK_PROB = 1/13`, `RANK_VALUES_SOLVER` (13-element list mirroring `cards.py`)
  - `DealerOutcome` dataclass: `{final_dist: dict[int,float], bust_prob: float, init_16_prob: float, init_17_prob: float, init_16_final_dist: dict, init_17_final_dist: dict}`
  - Module docstring and empty function stubs for the layers below

- [x] **1.1.A2 — Composition helpers** (`src/solvers/baseline_dp.py`)
  - `_total_from_composition(non_ace_total, num_aces, num_cards) -> int` — must exactly mirror `hand.py:calculate_total` logic
  - `_is_soft_from_composition(non_ace_total, num_aces, num_cards) -> bool`
  - `_transition(nat, na, nc, drawn_rank) -> tuple[int, int, int]` — returns new `(nat, na, nc)` after drawing a rank

- [x] **1.1.A3 — Tests: composition helpers** (`tests/test_baseline_dp.py`, new file)
  - 20 cases comparing `_total_from_composition` against `calculate_total()` (hard hands, soft hands, multi-ace, bust, five-card)
  - 11 cases verifying `_is_soft_from_composition` (cross-validated against `is_soft()`)
  - 11 cases verifying `_transition` for all rank categories + invariants
  - 8 constants sanity cases (RANK_PROB, RANK_VALUES_SOLVER, Action, DealerOutcome)

- [x] **1.1.A4 — Dealer outcome distribution** (`src/solvers/baseline_dp.py`)
  - `_dealer_play_recursive(nat, na, nc, memo) -> dict[str|int, float]` — recursive fixed-strategy simulation returning `{total: prob, 'bust': prob}`
  - `compute_dealer_outcomes(upcard_rank) -> DealerOutcome` — enumerates 13 hole cards (P=1/13 each), partitions into init_16/17 vs other paths, returns full distribution

- [x] **1.1.A5 — Tests: dealer distributions** (`tests/test_baseline_dp.py`)
  - For each of 13 upcards: probabilities sum to 1.0 ± 1e-9
  - Dealer bust probability in range 10–35% for all upcards
  - `init_16_prob + init_17_prob` is in [0, 1] and equals expected value for each upcard

- [x] **1.1.A6 — Settlement helper** (`src/solvers/baseline_dp.py`)
  - `_settle_ev(player_total, player_nc, dealer_total, dealer_nc, dealer_busted) -> float` — mirrors `rules.py` on abstracted state
  - Handles: player bust → -1.0, forfeit → -1.0, five-card vs five-card at 1:1, special hand type bonuses, regular total comparison

- [x] **1.1.A7 — EV stand** (`src/solvers/baseline_dp.py`)
  - `_ev_stand(nat, na, nc, dealer_outcome: DealerOutcome, reveal_mode: bool) -> float`
  - total ≤ 15 → -1.0 (unconditional forfeit)
  - nc == 2: always settle vs full final distribution (same in both modes)
  - nc ≥ 3, reveal_mode=ON: weighted sum — P(init_16) × settle(vs 16) + P(init_17) × settle(vs 17) + P(other) × Σ settle(vs final)
  - nc ≥ 3, reveal_mode=OFF: settle vs full final distribution (same as nc=2 path)

- [x] **1.1.A8 — EV hit & optimal EV** (`src/solvers/baseline_dp.py`)
  - `_ev_hit(nat, na, nc, dealer_outcome, memo) -> float` — enumerate 13 draw ranks, compute `_transition`, recurse; terminal cases: bust → -1.0, five-card ≤21 → `_ev_stand` with bonus type, total=21 → `_ev_stand`
  - `_optimal_ev(nat, na, nc, dealer_upcard_rank, memo) -> tuple[float, Action]` — returns `max(EV_hit, EV_stand)` and the corresponding action

- [x] **1.1.A9 — Tests: strategy invariants** (`tests/test_baseline_dp.py`)
  - Stand on 20 EV > hit on 20 EV for all 13 upcards
  - Hit always optimal at total ≤ 11 (cannot bust, cannot forfeit)
  - Total ≤ 15 → EV_stand == -1.0 regardless of upcard
  - No action possible at total = 21 (stand is forced)
  - With reveal_mode=ON: EV_stand differs for nc=2 vs nc=3 at same total when dealer upcard makes 16/17 reachable
  - With reveal_mode=OFF: EV_stand is the same for nc=2 and nc=3 at the same total and upcard

- [x] **1.1.A10 — Pre-settled hand EVs** (`src/solvers/baseline_dp.py`)
  - `_ev_ban_ban()` → (168/169)×3 ≈ +2.982; `_ev_ban_luck()` → 317/169 ≈ +1.876; `_ev_777()` → 1101/169 ≈ +6.515
  - All three settle against dealer's 2-card initial hand (before dealer plays) using infinite-deck 2-card probabilities
  - Constants `_P_DEALER_BAN_BAN = 1/169`, `_P_DEALER_BAN_LUCK = 8/169` added
  - **Design note**: spec had `upcard_rank` param + hierarchy errors — corrected per actual rules.py and no-upcard design
  - 18 new tests in `TestPreSettledHandEvs`; total: **362/362 passing**

- [x] **1.1.A11 — Public API: solve() and compute_house_edge()** (`src/solvers/baseline_dp.py`)
  - `solve(reveal_mode)` → `{(total, nc, is_soft): (Action, ev)}` — enumerates all reachable states, shared memo, representative compositions for nc=2 are exact
  - `compute_house_edge(strategy_table)` → float — iterates 169 rank pairs; Ban Ban/Ban Luck use special EVs, all others use table lookup
  - Wrappers implemented: `build_strategy_chart`, `build_ev_table`, `run_dp_solver`, `compare_reveal_modes`
  - **Design finding**: Phase 1.1 (no dealer surrender) gives player +4.2% edge; house edge only emerges in Phase 2 when dealer surrender is added as a CFR strategic option. A12 spec bound [-0.10, -0.01] was based on dealer surrender being active — corrected in tests.
  - 24 new tests in `TestPublicAPI`; total: **386/386 passing**

- [x] **1.1.A12 — print_strategy_chart() + __main__ + end-to-end test** ✅
- [x] **1.1.A13 — EV margin column in print_strategy_chart()** ✅
  - `build_ev_margin_table(reveal_mode)` computes `|EV_HIT − EV_STAND|` for all states
  - `print_strategy_chart(..., ev_margin_table=..., show_ev_margin=True)` prints second grid below H/S
  - `__main__` calls both and prints EV margin grids alongside strategy charts
  - 6 tests: non-negative margins, forced-stand=None, hard-20>hard-16, nc=4 margin>0.5, print on/off

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
