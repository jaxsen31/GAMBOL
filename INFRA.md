# INFRA.md — Infrastructure & Tooling Status
> Tracks one-time setup tasks for CI, code quality, and security tooling.
> Each task is atomic: one command or one file change. Check the box when done.
> Last updated: 2026-03-01

---

## Status summary

| Area | Status |
|---|---|
| Config files created | ✅ Done |
| Tools installed locally | ✅ Done (2026-03-01) |
| Lint violations fixed | ✅ Done (2026-03-01) |
| Security scans verified | ✅ Done (2026-03-01) |
| CI green on GitHub | ⬜ Pending |
| Pre-commit active locally | ✅ Done (2026-03-01) |
| /simplify workflow adopted | ⬜ Pending |

---

## A — Install tools locally
> Do these once in a terminal. All subsequent tasks depend on them.

- [x] **A1** — `pip install ruff` (linter + formatter)
- [x] **A2** — `pip install bandit` (security scanner)
- [x] **A3** — `pip install pip-audit` (dependency CVE scanner)
- [x] **A4** — `pip install pre-commit` (git hook manager)

---

## B — Fix any existing lint violations
> Run from repo root. Ruff will auto-fix what it can; review the rest manually.

- [x] **B1** — `ruff check banluck-solver/src/ --fix` — auto-fix lint issues in source
- [x] **B2** — `ruff check banluck-solver/tests/ --fix` — auto-fix lint issues in tests
- [x] **B3** — `ruff format banluck-solver/src/ banluck-solver/tests/` — apply formatting
- [x] **B4** — `ruff check banluck-solver/src/ banluck-solver/tests/` — confirm 0 violations remaining
- [x] **B5** — commit: `style: apply ruff lint + format fixes; activate pre-commit hooks`

---

## C — Verify security scans pass
> These should pass clean on first run. Fix any findings before enabling CI.

- [x] **C1** — `pip-audit --requirement banluck-solver/requirements.txt` — 0 CVEs ✅
- [x] **C2** — `bandit -r banluck-solver/src/ -ll --skip B101,B311 -f txt` — 0 issues ✅

---

## D — Activate pre-commit hooks
> One-time per dev machine. Hooks run automatically on every `git commit` after this.

- [x] **D1** — `pre-commit install` (run from repo root `/workspaces/GAMBOL/`)
- [x] **D2** — `pre-commit run --all-files` — all 9 hooks green (fixed bandit `pass_filenames: false`)
- [x] **D3** — hooks fired clean on the B5 commit above

---

## E — Verify CI green on GitHub
> After B and C pass locally, push and watch the Actions tab.

- [ ] **E1** — `git push origin main` — triggers the CI workflow on GitHub
- [ ] **E2** — Open `https://github.com/jaxsen31/GAMBOL/actions` and confirm both Python 3.11 and 3.12 jobs are green
- [ ] **E3** — If any job fails, read the log, fix locally, push again

---

## F — Adopt /simplify skill for new modules
> The `/simplify` skill reviews newly written code for reuse, quality, and efficiency.
> Use it in Claude Code after completing each new analysis module.

- [ ] **F1** — After writing the Interactive Plotly lookup tool: run `/simplify` in Claude Code and apply suggestions
- [ ] **F2** — After writing variance/bankroll analysis module: run `/simplify` in Claude Code and apply suggestions
- [ ] **F3** — Make `/simplify` a personal habit: run it before every commit on new source files

---

## Notes

- Config files already created (this session): `.github/workflows/ci.yml`, `banluck-solver/.bandit`, `.pre-commit-config.yaml`, ruff config in `banluck-solver/pyproject.toml`
- CI coverage threshold: 75% (`--cov-fail-under=75` in workflow only — not in local pytest defaults, to keep local runs fast)
- Numba JIT compile adds ~6 s to first CI test run per matrix entry; subsequent runs use cache. CI timeout is set to 20 minutes.
- If pip-audit reports a CVE, check whether the fix is in a minor version bump — usually `pip install --upgrade <package>` + update `requirements.txt` resolves it.
