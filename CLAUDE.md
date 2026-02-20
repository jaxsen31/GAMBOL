# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Always prioritize the constraints defined in `banluck_PRD_v1.3.md` (the active PRD). Any implementation decision that conflicts with the PRD must be flagged and resolved before proceeding.

## Session Start

At the start of every session: read the last entry in `NEXT_ACTIONS.md`, summarize it, and ask the user for their priority for the day before doing anything else.

## PRD Version Control

The active PRD is always at the repo root. Superseded versions live in `archive/`:
- `archive/banluck_PRD_v1.1.md` — archived
- `archive/banluck_PRD_v1.2.md` — archived
- `banluck_PRD_v1.3.md` — **active**

When a new PRD version is provided, move the current active version to `archive/` before placing the new one at the root.

## Commands

All commands run from `banluck-solver/` with `PYTHONPATH=.` set:

```bash
# Run all tests
cd banluck-solver && PYTHONPATH=. python -m pytest tests/ -v

# Run a single test file
cd banluck-solver && PYTHONPATH=. python -m pytest tests/test_rules.py -v

# Run a specific test by name
cd banluck-solver && PYTHONPATH=. python -m pytest tests/ -k "test_name" -v

# Run with coverage
cd banluck-solver && PYTHONPATH=. python -m pytest tests/ --cov=src -v
```

## Architecture

The project is a GTO solver for Banluck (Chinese Blackjack). Code lives in `banluck-solver/src/`.

### Engine layer (`src/engine/`) — pure game logic, no solver concerns

- **`cards.py`** — Card encoding: integer 0–51 where `card // 4` = rank index (0=2 … 12=A), `card % 4` = suit. String I/O via `str_to_card`/`card_to_str`. All internal code uses integers; strings only at boundaries.
- **`deck.py`** — NumPy int8 array of length 52 (1=available, 0=dealt). Functions: `create_deck()`, `deal_card(deck)`, `deal_specific_card(deck, card)`, `build_deck_from_hands(*hands)`.
- **`hand.py`** — `calculate_total(cards)` with ace resolution. Ace logic: 2-card hand → 10 or 11; 3+-card hand → 1 or 10 (greedy, maximise without busting).
- **`special_hands.py`** — `classify_hand(cards)` returns one of: `'ban_ban'`, `'ban_luck'`, `'777'`, `'five_card_21'`, `'five_card_sub21'`, `'regular'`, `'bust'`. Hierarchy rank: 1 (strongest) to 7 (weakest).
- **`rules.py`** — `settle_hand(player_cards, dealer_cards, dealer_surrendered, dealer_busted)` → `(Outcome, payout_units)`. Payout is signed floats from player's perspective. Settlement priority: dealer surrender → player bust → player ≤15 forfeit → hand comparison.
- **`game_state.py`** — `play_hand(deck, ...)` orchestrates the full hand flow. `settle_with_selective_reveal(...)` implements the selective reveal mechanic. Strategies are passed as callables. Whether to use selective reveal is controlled by the caller — the engine supports both reveal and no-reveal modes.

### Solver layer (`src/solvers/`) — currently empty, Phase 1.1 target

Planned: `baseline_dp.py` (backward induction, fixed dealer, two runs: reveal_mode=ON vs OFF) and `ev_tables.py`.

### Analysis layer (`src/analysis/`) — currently empty

Planned: Monte Carlo simulator, EV tables, Plotly visualizations.

## Key Game Rules (Banluck-Specific)

These differ from standard Blackjack and are fully implemented in the engine:

1. **Ace valuation**: 2-card hand → ace = 10 or 11; 3+-card hand → ace = 1 or 10
2. **Dealer hard-15 surrender** → push, overrides ALL outcomes including Ban Ban
3. **Player ≤15 forfeit** → unconditional loss, even on dealer bust
4. **Selective reveal at dealer 16/17**: the dealer **may optionally** reveal 3+-card players before hitting — this is a strategic choice, not a hard rule. When used: 3+-card players settle against the dealer's *current* (pre-hit) total; 2-card players settle against the dealer's *final* total. The solver explores both reveal_mode=ON and reveal_mode=OFF to quantify EV impact; Phase 2 (CFR) models REVEAL_PLAYER as an explicit dealer action node.
5. **Five-card vs five-card**: winner determined by total at 1:1 (not the bonus multiplier)
6. **Ban Ban / Ban Luck**: immediately settled after initial deal, never played further

## Special Hand Payouts

| Hand | Condition | Payout |
|------|-----------|--------|
| Ban Ban | 2 aces | 3:1 |
| Ban Luck | Ace + 10-value | 2:1 |
| 777 | Three 7s (3 cards) | 7:1 |
| Five-card 21 | 5 cards totaling 21 | 3:1 |
| Five-card <21 | 5 cards totaling <21 | 2:1 |
| Regular | All other non-bust | 1:1 |

## Testing Conventions

- Tests use the `hand(*card_strs)` helper from `conftest.py` to build hands from strings like `hand('AS', 'KC')`.
- The `h` fixture exposes `hand()` for use in test functions.
- The `fresh_deck` fixture returns a full 52-card deck.
- Edge cases from the PRD are in `tests/test_edge_cases.py` (14 cases, all must pass).

## Next Phase: 1.1 Baseline DP Solver

See `NEXT_ACTIONS.md` for the full task list. Key design decisions already made:
- State: `(player_cards: tuple, dealer_upcard: int, deck: tuple)`
- Dealer strategy: fixed (hit ≤16, hit soft 17, stand hard 17+)
- Two runs: `reveal_mode=ON` (dealer always reveals 3+-card players at 16/17) and `reveal_mode=OFF` (all players settle against final total)
- Output: two strategy charts + EV comparison, keyed on `(player_total, hand_size, dealer_upcard)`
- Validation: Monte Carlo cross-check within ±0.1% EV
