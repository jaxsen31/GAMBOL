# Thai Baccarat (Pok Deng) — Next Actions

## Status: Scaffolding complete, planning not yet started

---

## Immediate Next Step

**Run `/gsd:new-project` in a fresh Claude Code session.**

- Open a new chat (current session is at ~20% token budget)
- GSD v1.22.0 is already installed at `.claude/commands/gsd/`
- GSD will read `docs/pok_deng_rules.md` and generate:
  - `PROJECT.md` — goals, constraints, success criteria
  - `ROADMAP.md` — phased milestone plan
  - `.planning/` — phase specs and task breakdowns
- When prompted, scope the solver to **heads-up (1 player vs banker)** first

---

## What's Already Done

- `thai-baccarat-solver/` directory scaffolded (src/engine, src/solvers, src/analysis, tests, notebooks)
- `docs/pok_deng_rules.md` — full Pok Deng rules written
- `pyproject.toml` + `requirements.txt` created (matches banluck-solver config)
- CI: parallel `test-thai-baccarat` job added to `.github/workflows/ci.yml`
- Pre-commit: bandit scope extended to `thai-baccarat-solver/src/`
- `CLAUDE.md` updated with Thai Baccarat section

---

## After GSD Planning Is Done

Phase 1 — Engine:
- `src/engine/cards.py` — card encoding (same int 0–51 scheme as banluck)
- `src/engine/deck.py` — NumPy int8 deck
- `src/engine/hand.py` — hand value (`sum mod 10`), Pok detection
- `src/engine/combinations.py` — classify 3-card combos: Tong (×5), Sam Lueang (×3), Dok Jik (×2), Nam (×2)
- `src/engine/rules.py` — settlement: Pok priority, multiplier resolution
- `src/engine/game_state.py` — full hand flow (deal → player action → banker action → settle)

Phase 2 — Solver:
- Baseline DP solver (infinite-deck approximation, heads-up)
- Strategy keyed on `(player_value, player_has_combo, banker_upcard)`
- CFR+ for full GTO (player hit/stand + banker hit/stand)

Phase 3 — Analysis:
- Monte Carlo cross-check vs DP (±0.1% EV target)
- EV tables and strategy heat maps
- Multiplier impact analysis (how combos shift player EV)
