# Product Requirements Document: Banluck Optimal Strategy Solver

## Project Overview

**Project Name:** Banluck Solver
**Version:** 1.3
**Author:** Jax
**Date:** February 2026
**Objective:** Develop a computational solver to determine game-theoretically optimal strategies for both player and dealer (banker) in the Chinese Blackjack variant "Banluck" with specific house rules.

**Changelog from v1.2:**
- Selective reveal reclassified from a mandatory rule to a **dealer strategic option**
- Section 3.5 reframed: dealer *may choose* to reveal 3+-card players before hitting at 16/17 — whether to do so is a key research question
- Phase 1.1 updated: two solver runs (reveal_mode=ON vs OFF) to immediately quantify EV impact before CFR
- Phase 2 updated: REVEAL_PLAYER is now an explicit dealer action node in the game tree, not a pre-determined rule
- Section 8 (Research Questions) updated to reflect the new framing of selective reveal
- v1.2 PRD superseded by this document

---

## 1. Game Specification — Complete Ruleset

### 1.1 Equipment & Setup
- **Deck:** Single standard 52-card deck (no jokers)
- **Deck Refresh:** Complete shuffle after every hand
- **Players:** Heads-up only (1 player vs 1 dealer/banker). Multi-player deferred to future scope.
- **Betting:** Variable unit bets allowed. Solver uses $1 unit bet for calculations.

### 1.2 Card Values
- **Number cards (2–9):** Face value
- **Face cards (J, Q, K, 10):** 10 points
- **Ace valuation:**
  - **With 2 cards in hand:** Ace = **10 or 11** (player/dealer choice, whichever is more favourable without busting)
  - **With 3+ cards in hand:** Ace = **1 or 10** (player/dealer choice, whichever is more favourable without busting)

**Implementation note:** Aces are resolved greedily (highest non-busting value first). With multiple aces, each is resolved in sequence — this correctly produces Ban Ban total of 21 (11+10) for a 2-card A-A hand.

### 1.3 Card Encoding (Numba-compatible)
```python
# Cards are integers 0–51
# Rank index = card // 4  (0=2, 1=3, ..., 7=9, 8=10, 9=J, 10=Q, 11=K, 12=A)
# Suit index = card % 4   (0=C, 1=D, 2=H, 3=S)
RANK_VALUES: list[int | None] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, None]  # Ace=None
```

### 1.4 Objective
Achieve a total closest to 21 without exceeding it, beat the dealer's total, and exploit special hand bonuses.

---

## 2. Special Hands & Payouts

| Hand | Definition | Payout | Timing | Notes |
|------|-----------|--------|--------|-------|
| **Ban Ban** | Two Aces (initial deal) | 3:1 | Immediate reveal | Total = 21 (10+11). Beats all non-Ban-Ban hands. Ban Ban vs Ban Ban = push. |
| **Ban Luck** | Ace + {10,J,Q,K} (initial deal) | 2:1 | Immediate reveal | Beats all regular 21s. Ban Luck vs Ban Luck = push. |
| **777** | Dealt (7,7), draw third 7 | 7:1 | After drawing 3rd card | Suit-independent. Beats dealer 21. |
| **Five-card <21** | 5 cards totaling <21 | 2:1 | After 5th card drawn | Solver always announces (see §3.4). |
| **Five-card =21** | 5 cards totaling exactly 21 | 3:1 | After 5th card drawn | Solver always announces (see §3.4). |
| **Regular Win** | Beat dealer, no special hand | 1:1 | Showdown | Standard payout. |

**Special Hand Hierarchy (Strongest → Weakest):**
1. Ban Ban (immediate)
2. Ban Luck (immediate)
3. 777 (beats dealer 21)
4. Five-card 21 (3:1)
5. Five-card <21 (2:1)
6. Regular 21
7. Regular <21

**Five-card vs Five-card:** When both hands are five-card, the winner is determined by total comparison, paid at 1:1 (regular win). The five-card bonus multiplier only applies against a non-five-card opponent.

**Dealer bonuses are fully symmetric:** Dealer collects at the same multipliers (Ban Ban 3:1 collected from player, etc.).

---

## 3. Gameplay Flow

### 3.1 Initial Deal
1. Players place bets
2. Dealer shuffles deck, one player cuts
3. Dealer deals 2 cards face-down to each player and themselves
4. Dealer and player simultaneously check for special hands

### 3.2 CRITICAL: Dealer Hard 15 Surrender (Highest Priority)
- If dealer is dealt **hard 15** (two cards totaling 15, with no Ace), dealer may **surrender**
- **Hard 15 definition:** exactly 2 cards, total = 15, hand contains no Ace whatsoever (e.g. 6-9, 7-8, 5-10). A soft 15 (A-4) is NOT a hard 15 and does not qualify.
- Dealer surrender **overrides ALL settlements including Ban Ban and Ban Luck**
- All bets push (returned), hand is voided and re-dealt
- This is checked **before** any special hand payouts are finalized
- Dealer is not forced to surrender — it is a strategic choice

### 3.3 Special Hand Reveals (If dealer does not surrender)
- Player with **Ban Ban** or **Ban Luck** immediately reveals for instant payout
- Settled players are out of the hand

### 3.4 Player Actions
**Players act sequentially.**

- **Hand ≤15:** Player may hit or stand, BUT hands ≤15 are **unconditionally forfeited at showdown** — player loses their bet even if the dealer busts. There is no scenario in which a player standing on ≤15 wins.
- **Hand 16–20:** May hit or stand
- **Hand 21:** Must stand
- **Hand >21:** Bust — lose bet immediately
- **Soft 17 (e.g. A-6 as 2-card hand):** May stand
- **Maximum 5 cards:** Cannot draw a 6th card
- **Five-card announcement:** In the solver, this is modelled as **always-announce** — a rational agent always announces before drawing the 5th card. The bonus forfeiture rule is irrelevant in the solver context.

### 3.5 Dealer Actions & Selective Reveal

**Minimum 16 Rule:** Dealer must reach at least 16 points.

**Selective Reveal — A Dealer Strategic Option:**
At totals of 16 or 17 (including soft 17), the dealer **may choose** to reveal 3+-card players before deciding to hit or stand. The rationale: those players may have busted, allowing the dealer to collect those bets before drawing a potentially-busting card, then face the remaining (typically stronger) 2-card hands.

**Whether selective reveal is optimal is a key research question answered by the solver.** The Phase 1.1 baseline runs the game with reveal forced ON and reveal forced OFF to quantify the EV impact. Phase 2 (CFR) models REVEAL_PLAYER as an explicit dealer action node so the equilibrium strategy can be computed.

**Mechanics of selective reveal (when the dealer chooses to use it):**
1. Dealer reveals and immediately settles **all players with 3+ cards** (pay wins, collect losses/busts)
2. Those players are now out of the hand
3. Dealer then decides: **hit or stand** against remaining 2-card players
4. If dealer hits and improves (e.g. reaches 18–21), reveal remaining players and settle
5. If dealer hits and busts, all remaining players win

**Dealer has 18, 19, or 20:**
- Reveal and settle **all players simultaneously**
- Dealer stands (no incentive to draw)

**Dealer has 21:**
- Reveal all players
- Collect from non-special hands; pay special hands at their multipliers

**Dealer Bust (after drawing):**
- All remaining (unrevealed, active) players win
- Special hands receive their multiplier payouts
- Hands ≤15 still forfeit unconditionally — dealer bust does not save them

### 3.6 Showdown & Settlement

**Settlement priority (highest to lowest):**
1. Dealer hard 15 surrender → push (overrides ALL)
2. Player bust (>21) → player loses 1 unit
3. Player ≤15 forfeit → player loses 1 unit (unconditional, even on dealer bust)
4. Hand hierarchy comparison → winner's multiplier applied
5. Total comparison → regular hands compared by total

**Push Conditions:**
- Equal point totals push (bet returned)
- **Exception:** Special hands beat regular hands of the same point value
- **Special hand mirror pushes:** Ban Ban vs Ban Ban = push. Ban Luck vs Ban Luck = push.
- **Five-card vs Five-card:** Same total = push. Different totals = higher total wins at 1:1.

---

## 4. Technical Requirements

### 4.1 State Space Definition

```python
from dataclasses import dataclass
from enum import Enum, auto

class Phase(Enum):
    DEAL = auto()
    PLAYER_ACTION = auto()
    DEALER_REVEAL = auto()
    DEALER_ACTION = auto()
    SETTLEMENT = auto()

@dataclass(frozen=True)  # Frozen = hashable for DP memoisation
class GameState:
    player_cards: tuple[int, ...]
    dealer_cards: tuple[int, ...]
    deck_remaining: tuple[int, ...]    # 52-length tuple, 1 = in deck, 0 = dealt
    phase: Phase
    player_announced_five_card: bool   # Always True in solver (always-announce rule)
    player_active: bool                # False if settled early (Ban Ban/Ban Luck/bust)
```

### 4.2 Deck Representation

```python
import numpy as np
deck: np.ndarray = np.ones(52, dtype=np.int8)  # 1=available, 0=dealt
```

### 4.3 Strategy Representation

```python
# Player: maps (own_hand, deck_state, phase) → action probabilities
π_player: dict[tuple, dict[PlayerAction, float]]

# Dealer: maps (own_hand, deck_state, revealed_outcomes) → action probabilities
π_dealer: dict[tuple, dict[DealerAction, float]]

class PlayerAction(Enum):
    HIT = auto()
    STAND = auto()

class DealerAction(Enum):
    REVEAL_PLAYER = auto()   # Strategic choice at 16/17 — not a forced rule
    HIT = auto()
    STAND = auto()
    SURRENDER = auto()

class Outcome(Enum):
    WIN = auto()
    LOSS = auto()
    PUSH = auto()
```

### 4.4 Solver Algorithms

**Phase 1.1: Baseline Player Strategy**
- Method: Backward induction / Dynamic programming
- Assumption: Dealer follows fixed strategy (hit at 16/soft-17, stand at hard-17+)
- **Two baseline runs:**
  - `reveal_mode=ON` — dealer always uses selective reveal at 16/17
  - `reveal_mode=OFF` — dealer never uses selective reveal (2-card and 3+-card players settle against the same final dealer total)
- Output: Two optimal player hit/stand charts + EV comparison, keyed on `(player_total, hand_size, dealer_upcard)`
- The EV delta between runs quantifies the impact of selective reveal before CFR is built

**Phase 2: Full Equilibrium**
- Method: Counterfactual Regret Minimization (CFR+)
- Handles: Imperfect information; **REVEAL_PLAYER is an explicit dealer action node at 16/17**, not a hard rule
- Dealer's information set at 16/17 includes the choice to reveal or not
- Output: Nash equilibrium strategies for both player and dealer, including the GTO reveal frequency

**Phase 3: Validation**
- Monte Carlo simulation (1M+ hands)
- Cross-validation against hand-calculated edge cases
- Convergence testing (exploitability < 0.01 units/hand)

### 4.5 EV Reporting Standard
- All EV figures in **units per hand** (e.g. −0.023 units/hand)
- All house edge figures as **percentage** (e.g. 2.3% house edge)
- Variance reported as standard deviation in units

### 4.6 Technical Stack

```
numpy       — array operations, deck representation (int8 arrays)
numba       — JIT compilation for hot loops (CFR iterations)
scipy       — optimization routines for convergence checks
matplotlib  — strategy heat maps, EV convergence curves
seaborn     — styled visualizations
plotly      — interactive strategy lookup tool
pandas      — EV tables and reporting
pytest      — unit testing
```

---

## 5. Project Knowledge & Instructions for AI Assistants

### 5.1 User Context
- **Name:** Jax, 26-year-old Design Engineer at Dyson
- **Background:** Mechanical engineering, poker player (tournament winner), understands GTO/Nash equilibrium
- **Technical:** Python proficient, familiar with probability, EV, variance
- **Goal:** Optimize heads-up CNY banluck strategy, understand WHY strategies are optimal

### 5.2 Code Style Preferences
- **Always provide full code files** when making changes
- Include debugging checklists with code
- Mathematical rigor over shortcuts
- Clear variable naming (no single letters except loop indices)
- Type hints for all functions
- Docstrings with examples
- **No change summaries unless explicitly requested**

### 5.3 Domain Terminology

| Banluck Term | Game Theory Equivalent | Poker Analog |
|--------------|----------------------|--------------|
| Ban Ban | Premium hand (AA) | Pocket Aces |
| Ban Luck | Premium hand (A+T) | Big Slick |
| Dealer opens player | Information revelation | Opponent shows hand |
| Hard 15 surrender | Dealer escape mechanism | Hand cancellation |
| Selective reveal | Imperfect information / dealer action node | Partial board reveal |

### 5.4 Critical Implementation Gotchas

**Ace Valuation (CORRECT implementation):**
```python
def resolve_ace(hand_size: int, current_total: int) -> int:
    """
    With 2 cards: ace = 10 or 11
    With 3+ cards: ace = 1 or 10
    Always choose highest value that doesn't bust.
    """
    if hand_size == 2:
        return 11 if current_total + 11 <= 21 else 10
    else:
        return 10 if current_total + 10 <= 21 else 1
```

**Greedy multi-ace resolution:** Process aces sequentially, each at the highest non-busting value. A-A-A in a 3-card hand = 10+10+1 = 21 (not 12). This is mathematically correct and passes all tests.

**Hard 15 Detection:**
```python
def is_hard_fifteen(hand_cards: tuple[int, ...]) -> bool:
    if len(hand_cards) != 2:
        return False
    ranks = [card // 4 for card in hand_cards]
    if 12 in ranks:  # rank index 12 = Ace — disqualifies soft hands
        return False
    total = sum(RANK_VALUES[r] for r in ranks)
    return total == 15
```

**Settlement priority (unconditional forfeit before comparison):**
```python
def settle_hand(player_cards, dealer_cards, dealer_surrendered, dealer_busted):
    if dealer_surrendered:   return PUSH, 0
    if player_total > 21:    return LOSS, -1   # bust
    if player_total <= 15:   return LOSS, -1   # unconditional forfeit
    # ... hand comparison
```

**Selective reveal at 16/17 (when dealer chooses to use it):**
- 3+-card player: settle against dealer's INITIAL hand, BEFORE dealer hits
- 2-card player: settle against dealer's FINAL hand, AFTER dealer acts
- When reveal_mode=OFF: all players settle against the same final dealer total

### 5.5 Critical Edge Cases — All 14 Pass ✅

```python
# Edge Case 1: Dealer hard 15 overrides Ban Ban → push
# Edge Case 2: Ban Ban total = 21 (A-A two-card: 11+10=21)
# Edge Case 3: Soft 15 (A-4) is NOT a hard 15
# Edge Case 4: Five-card vs five-card same total → push
# Edge Case 5: Five-card vs five-card different total → higher wins at 1:1
# Edge Case 6: 777 beats dealer 21 at 7:1
# Edge Case 7: 777 is suit-independent
# Edge Case 8: Player ≤15 forfeits even on dealer bust
# Edge Case 9: Player ≤15 forfeits vs any dealer total (no push at 15 vs 15)
# Edge Case 10: Ban Ban vs Ban Ban → push
# Edge Case 11: Ban Luck vs Ban Luck → push
# Edge Case 12: A-A-A (3-card) = 21, no spurious bust
# Edge Case 13: Dealer at 17 reveals 3+-card player BEFORE deciding to hit (reveal_mode=ON)
# Edge Case 14: Solver always-announce five-card bonus always applies
```

### 5.6 Implemented Project Structure

```
banluck-solver/
├── src/
│   ├── engine/
│   │   ├── cards.py          # Card constants, encoding, rank/suit helpers
│   │   ├── deck.py           # Deck creation, deal_card, shuffle
│   │   ├── hand.py           # Hand evaluation, ace resolution, total calculation
│   │   ├── special_hands.py  # Ban Ban, Ban Luck, 777, five-card detection
│   │   ├── rules.py          # Settlement, payout calculation, hand comparison
│   │   └── game_state.py     # GameState dataclass, Phase enum, play_hand()
│   ├── solvers/
│   │   ├── baseline_dp.py    # Phase 1.1: backward induction [TODO]
│   │   ├── cfr.py            # Phase 2: CFR+ solver [TODO]
│   │   └── ev_tables.py      # Pre-computed EV lookups [TODO]
│   └── analysis/
│       ├── simulator.py      # Monte Carlo validation [TODO]
│       ├── ev_calculator.py  # EV reporting [TODO]
│       └── visualize.py      # Charts and interactive tools [TODO]
├── tests/
│   ├── conftest.py           # Shared fixtures (hand() helper)
│   ├── test_cards.py         # 33 tests ✅
│   ├── test_deck.py          # 23 tests ✅
│   ├── test_hand.py          # 37 tests ✅
│   ├── test_special_hands.py # 55 tests ✅
│   ├── test_rules.py         # 62 tests ✅
│   └── test_edge_cases.py    # 35 tests (all 14 edge cases) ✅
├── requirements.txt
└── pyproject.toml
```

### 5.7 Naming Conventions

```python
RANK_NAMES = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
SUIT_NAMES = ['C','D','H','S']

class PlayerAction(Enum): HIT = auto(); STAND = auto()
class DealerAction(Enum): REVEAL_PLAYER = auto(); HIT = auto(); STAND = auto(); SURRENDER = auto()
class Outcome(Enum): WIN = auto(); LOSS = auto(); PUSH = auto()

PAYOUT_MULTIPLIERS = {
    'ban_ban': 3, 'ban_luck': 2, '777': 7,
    'five_card_21': 3, 'five_card_sub21': 2, 'regular': 1,
}
```

### 5.8 Rank Values Reference

```python
# Index: 0  1  2  3  4  5  6  7   8   9   10  11  12
# Rank:  2  3  4  5  6  7  8  9  10   J   Q   K   A
RANK_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, None]
```

---

## 6. Deliverables

### Phase 1 (Weeks 1–3): Game Engine + Tests
- ✅ Formal game specification (PRD v1.0 → v1.3)
- ✅ Game engine: `cards.py`, `deck.py`, `hand.py`, `special_hands.py`, `rules.py`, `game_state.py`
- ✅ 245 unit tests for all 14 edge cases — all passing
- ✅ `NEXT_ACTIONS.md` active planning document

### Phase 1.1: Baseline Solver (Active Next Milestone)
- ⏭️ Backward induction DP solver (`baseline_dp.py`)
- ⏭️ Two strategy charts: reveal_mode=ON vs reveal_mode=OFF
- ⏭️ EV delta between the two modes
- ⏭️ EV tables (`ev_tables.py`)
- ⏭️ Monte Carlo cross-validation (`simulator.py`)

### Phase 2 (Weeks 4–8): Full Nash Equilibrium
- ⏭️ CFR+ solver implementation (`cfr.py`)
- ⏭️ REVEAL_PLAYER modelled as explicit dealer action node at 16/17
- ⏭️ GTO reveal frequency output (always? probabilistic? never?)
- ⏭️ Complete strategy tables for player and dealer
- ⏭️ EV calculations for all game states

### Phase 3 (Weeks 9–12): Analysis & Insights
- ⏭️ Quantify dealer positional advantage from selective reveal
- ⏭️ Variance analysis and bankroll requirements
- ⏭️ Sensitivity analysis (rule variants)
- ⏭️ Interactive strategy lookup tool (Plotly)

### Phase 4 (Future — Deferred)
- ⏭️ Multi-player solver (4 players + dealer)
- ⏭️ Card counting viability analysis
- ⏭️ GUI application

---

## 7. Success Criteria

### Correctness
- [x] All 14 edge cases produce correct payouts (245/245 tests pass)
- [ ] Strategies converge to equilibrium (exploitability → 0)
- [ ] Simulated EV matches theoretical calculations (±0.1%)
- [ ] Results replicate across different random seeds

### Performance
- [ ] Baseline solver completes in <1 hour
- [ ] Full CFR+ solver converges in <24 hours
- [ ] Strategy lookup is instant (<10ms)

### Insights
- [ ] Quantify dealer advantage from selective reveal (ON vs OFF EV delta)
- [ ] Determine GTO dealer reveal frequency at 16/17 (from CFR)
- [ ] Determine if card counting is viable with per-hand shuffles (expected: no)
- [ ] Answer "Should I hit on 16?" definitively for all heads-up scenarios

---

## 8. Key Research Questions

1. **How valuable is dealer's selective reveal?** Quantify the EV delta between reveal_mode=ON and reveal_mode=OFF in percentage terms
2. **Does optimal play differ significantly from blackjack basic strategy?**
3. **What's the GTO dealer reveal strategy at 16/17?** The solver treats REVEAL_PLAYER as an explicit action node — does equilibrium produce always-reveal, never-reveal, or a mixed strategy?
4. **How powerful is hard 15 surrender?** What % of hands does it save the dealer from losing?
5. **What's the fair rotation rate for dealer position?** (Every N hands to balance advantage)

---

## 9. Development Phases

**Week 1–2:** Game engine + comprehensive testing ✅ COMPLETE
**Week 3–4:** Baseline solver (two charts: reveal ON vs OFF) ← ACTIVE
**Week 5–8:** CFR+ implementation (full Nash equilibrium, GTO reveal frequency)
**Week 9+:** Analysis, visualization, practical insights

---

## 10. Reference Materials

**Game Theory:**
- "Regret Minimization in Games with Incomplete Information" (Zinkevich et al., 2007)
- "Solving Imperfect Information Games Using Decomposition" (Burch et al., 2014)

**Implementations:**
- CFR tutorial: http://modelai.gettysburg.edu/2013/cfr/cfr.pdf

**Comparison Baseline:**
- "The Theory of Blackjack" by Peter Griffin

---

## 11. Current Status

✅ **Phase 1 complete — game engine fully implemented**
✅ **245/245 tests passing — all 14 edge cases verified**
✅ **PRD updated to v1.3 — selective reveal reclassified as dealer strategic option**
✅ **NEXT_ACTIONS.md created as active planning document**
⏭️ **Active next step: Phase 1.1 — `baseline_dp.py` (two runs: reveal_mode=ON and OFF)**

---

**This PRD (v1.3) is the canonical specification. The game engine is complete and frozen. Do not modify engine modules without a corresponding test update. Begin Phase 1.1 with `src/solvers/baseline_dp.py`.**
