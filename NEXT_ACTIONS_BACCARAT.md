# Thai Baccarat (Pok Deng) — Next Actions

## Status: Scaffolding complete, planning not yet started

---

## Installed Subagents (`.claude/agents/`)

Four specialized subagents are installed project-wide. Claude Code invokes them
automatically when tasks match their descriptions. You can also request them
explicitly (e.g. "use the performance-engineer agent to profile this").

| Agent | Model | When it activates |
|---|---|---|
| `python-pro` | Sonnet | NumPy, Numba JIT, pytest fixtures, type-safe Python |
| `performance-engineer` | Sonnet | Profiling, bottleneck identification, benchmarking |
| `code-reviewer` | **Opus** | Post-phase code quality, security, best-practice audit |
| `debugger` | Sonnet | Root-cause analysis, stack traces, solver correctness bugs |

---

## Step 0 — GSD Project Planning

**Run `/gsd:new-project` in a fresh Claude Code session** from the repo root (`/workspaces/GAMBOL`).

- GSD v1.22.0 is already installed at `.claude/commands/gsd/`
- GSD will read `thai-baccarat-solver/docs/pok_deng_rules.md` and generate planning artifacts
- When prompted, scope the solver to **heads-up (1 player vs banker)** first

### Expected outputs

**In the repo root (main folder):**
- `baccarat_PRD.md` — product requirements doc (game rules, scope, success criteria)
- Updated `NEXT_ACTIONS_BACCARAT.md` — this file, with Phase 1+ task list filled in

**Inside `thai-baccarat-solver/` (internal GSD reference):**
- `PROJECT.md` — goals, constraints, milestone map
- `ROADMAP.md` — phased milestone plan
- `.planning/` — per-phase specs and task breakdowns
  - Note: `.planning/` is GSD's internal tracking equivalent; `NEXT_ACTIONS_BACCARAT.md` is the human-readable summary that should stay in sync with it

### After GSD setup completes

Update `CLAUDE.md` to:
- Replace all remaining Banluck-specific references in the "Project Context" and "Next Phase" sections with Baccarat equivalents (the Thai Baccarat solver is now the active project)
- Add `baccarat_PRD.md` to the PRD Version Control section
- Update the Session Start rule so Baccarat sessions read `NEXT_ACTIONS_BACCARAT.md` as the primary doc

---

## What's Already Done

- `thai-baccarat-solver/` directory scaffolded (src/engine, src/solvers, src/analysis, tests, notebooks)
- `docs/pok_deng_rules.md` — full Pok Deng rules written
- `pyproject.toml` + `requirements.txt` created (matches banluck-solver config)
- CI: parallel `test-thai-baccarat` job added to `.github/workflows/ci.yml`
- Pre-commit: bandit scope extended to `thai-baccarat-solver/src/`
- `CLAUDE.md` updated with Thai Baccarat section

---

## Phase 1 — Engine

Modules to build (all in `thai-baccarat-solver/src/engine/`):

- `cards.py` — card encoding (same int 0–51 scheme as banluck)
- `deck.py` — NumPy int8 deck
- `hand.py` — hand value (`sum mod 10`), Pok detection
- `combinations.py` — classify 3-card combos: Tong (×5), Sam Lueang (×3), Dok Jik (×2), Nam (×2)
- `rules.py` — settlement: Pok priority, multiplier resolution
- `game_state.py` — full hand flow (deal → player action → banker action → settle)

**Subagent guidance:**
- `python-pro` — invoke for NumPy deck implementation, pytest conftest setup, type annotations
- `debugger` — invoke if settlement edge cases produce unexpected results in tests

**After Phase 1 complete → invoke `code-reviewer`** for a full engine audit before
building the solver on top of it. Fix all findings before Phase 2.

---

## Phase 2 — Solver

### 2.1 Baseline DP Solver
- State: `(player_value, player_has_combo, banker_upcard)` — heads-up, infinite-deck approximation
- Strategy keyed on player value + combo type + banker upcard
- Output: strategy chart + house edge (two runs: standard vs combo-adjusted)

### 2.2 CFR+ GTO Solver
- Player actions: hit / stand
- Banker actions: hit / stand (+ potential reveal at boundary totals)
- Full Nash equilibrium → `CfrResult` (mirrors banluck cfr.py pattern)

**⚠ Performance lesson from Banluck (read before implementing):**
The Banluck CFR solver required 4 separate optimization sessions (C1→C2→C3b→C3c)
to go from 4.3 s/pass → 6.7 ms/pass. Avoid that journey:

1. After the first working CFR implementation, **immediately invoke `performance-engineer`**
   before writing any tests. Let it profile and recommend the right tree representation.
2. Target architecture: tabular CSR tree (built once) + Numba `@njit` kernel.
   See `banluck-solver/src/solvers/cfr.py` for the proven pattern.
3. `python-pro` — invoke for Numba JIT kernel implementation and NumPy array structuring.
4. `debugger` — invoke if regret/strategy convergence looks wrong (check CFR+ flooring,
   linear averaging, and settlement sign conventions).

**After Phase 2 complete → invoke `code-reviewer`** (Opus) before Phase 3.

---

## Phase 3 — Analysis

- Monte Carlo cross-check vs DP (±0.1% EV target)
- EV tables and strategy heat maps (`src/analysis/heat_maps.py` pattern from banluck)
- Multiplier impact analysis — how Tong/Sam Lueang/Dok Jik/Nam shift player EV
- Interactive Plotly lookup tool (`src/analysis/plotly_lookup.py` pattern from banluck)
- Variance and bankroll analysis
- Streamlit dashboard (4-tab layout: heat maps / Plotly / bankroll / strategy report)

**Subagent guidance:**
- `python-pro` — invoke for Plotly figure construction and Matplotlib heat map setup
- `performance-engineer` — invoke if Monte Carlo simulation is slow (>30 s for 200k hands)

**After Phase 3 complete:**
1. Invoke `code-reviewer` for final audit
2. Run `/simplify` on all Phase 3 modules
3. Update `NEXT_ACTIONS_BACCARAT.md` status to COMPLETE
