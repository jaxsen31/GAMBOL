# GAMBOL
GTO Ban luck and maybe Thai Baccarat 

# Banluck Solver

A game-theoretically optimal strategy solver for Banluck (Chinese Blackjack), built using dynamic programming and Counterfactual Regret Minimization (CFR+).

## What is Banluck?

Banluck is a Chinese Blackjack variant played commonly during CNY. It differs from standard Blackjack in several key ways:
- Special hands with multiplied payouts (Ban Ban 3:1, Ban Luck 2:1, 777 7:1, Five-card hands)
- Dealer can surrender on hard 15, voiding all bets including special hands
- Dealer uses a selective reveal strategy — opening weak players before deciding to draw
- Ace valuation changes depending on hand size (2-card vs 3+ card hands)

## Project Goal

Determine Nash equilibrium strategies for both player and dealer in a heads-up game, and quantify the dealer's structural advantage from the selective reveal mechanic.

## Project Structure
```
banluck-solver/
├── src/
│   ├── engine/          # Pure game logic (deck, hand evaluation, rules, state)
│   ├── solvers/         # DP baseline + CFR+ equilibrium solver
│   └── analysis/        # EV calculator, Monte Carlo simulator, visualizations
├── tests/               # Unit tests for all 14 edge cases + integration tests
├── notebooks/           # Exploratory Jupyter notebooks
└── docs/                # PRD and strategy tables
```

## Development Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Game engine + unit tests (245/245 passing) | ✅ Complete |
| 1.1 | Baseline DP solver (fixed dealer) | 🔄 Active |
| 2 | CFR+ full Nash equilibrium | ⏭️ Upcoming |
| 3 | Analysis, variance, strategy charts | ⏭️ Upcoming |

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/banluck-solver.git
cd banluck-solver
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests
```bash
pytest tests/ -v
```

## Tech Stack

- **Python 3.11+**
- **NumPy** — array operations and deck representation
- **Numba** — JIT compilation for CFR hot loops
- **SciPy** — optimization and convergence checks
- **Plotly** — interactive strategy lookup tool
- **pytest** — unit testing

## Key Research Questions

1. How valuable is the dealer's selective reveal in % edge?
2. Does optimal play differ meaningfully from standard Blackjack basic strategy?
3. What is the GTO dealer reveal threshold at hard 16/17?
4. How often does hard 15 surrender save the dealer from a losing hand?

## License

MIT
