# Product Requirements Document: Banluck Optimal Strategy Solver

## Project Overview

**Project Name:** Banluck Solver
**Version:** 1.1
**Author:** Jax
**Date:** February 2026
**Objective:** Develop a computational solver to determine game-theoretically optimal strategies for both player and dealer (banker) in the Chinese Blackjack variant "Banluck" with specific house rules.

**Changelog from v1.0:**
- Clarified dealer selective reveal logic for hard 16, hard 17, and soft 17 (Gaps 1 & 8)
- Confirmed player ≤15 forfeit is unconditional, including on dealer bust (Gap 2)
- Confirmed 5-card announcement is always-announce in solver context (Gap 3)
- Scope locked to heads-up (1 player vs dealer) for all phases (Gap 4)
- Confirmed Ban Ban total = 21, immediately settled, never played (Gap 5)
- Confirmed hard 15 definition: any 2-card 15 with no ace (Gap 6)
- Confirmed 777 is suit-independent (Gap 7)
- Confirmed Ban Ban vs Ban Ban = push; Ban Luck vs Ban Luck = push (Gap 9)
- EV reporting standardized: units per hand + house edge % (Gap 10)
- Deck representation standardized to numpy int arrays for Numba compatibility (Gap 11)

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

### 1.3 Objective
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

---

## 3. Gameplay Flow

### 3.1 Initial Deal
1. Players place bets
2. Dealer shuffles deck, one player cuts
3. Dealer deals 2 cards face-down to each player and themselves
4. Dealer and player simultaneously check for special hands

### 3.2 CRITICAL: Dealer Hard 15 Surrender (Highest Priority)
- If dealer is dealt **hard 15** (two cards totaling 15, with no Ace — i.e. no ace present in the hand at all), dealer may **surrender**
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
- **Five-card announcement:** A player with 4 cards intending to draw a 5th must announce before drawing. In the solver, this is modelled as **always-announce** — a rational agent always announces, so no strategic decision is required here. The bonus forfeiture rule (failure to announce) is irrelevant in the solver context.

### 3.5 Dealer Actions & Selective Reveal

**Minimum 16 Rule:** Dealer must reach at least 16 points.

**Selective Reveal Logic — Core Principle:**
The dealer's selective reveal is a strategic tool. At totals of 16 or 17 (including soft 17), the dealer is likely to lose against a 2-card player hand (which is assumed to be ≥16). The dealer therefore reveals 3+ card players first — these players may have busted — to collect those bets, then draws a card to try to improve before facing the stronger 2-card hands.

**Dealer has hard 16, hard 17, or soft 17:**
1. Reveal and immediately settle **all players with 3+ cards** (pay wins, collect losses/busts)
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

**Push Conditions:**
- Equal point totals push (bet returned)
- **Exception:** Special hands beat regular hands of the same point value:
  - Ban Ban (21) beats Regular 21
  - Ban Luck (21) beats Regular 21
  - Five-card 21 beats Regular 21
- **Special hand mirror pushes:** Ban Ban vs Ban Ban = push. Ban Luck vs Ban Luck = push.
- **Five-card vs Five-card:** Same total = push. Different totals = higher total wins.

**Unconditional Forfeit:**
- Player with a final total ≤15 loses their bet regardless of any dealer outcome, including dealer bust. No exceptions.

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
    player_cards: tuple[int, ...]      # Card indices 0–51 (rank = card // 4, suit = card % 4)
    dealer_cards: tuple[int, ...]      # Same encoding
    deck_remaining: tuple[int, ...]    # 52-length tuple, 1 = in deck, 0 = dealt
    phase: Phase
    player_announced_five_card: bool   # Always True in solver (always-announce rule)
    player_active: bool                # False if settled early (Ban Ban/Ban Luck/bust)
```

**Card Encoding:**
```python
# Cards are integers 0–51
# Rank index = card // 4  (0=2, 1=3, ..., 7=9, 8=10, 9=J, 10=Q, 11=K, 12=A)
# Suit index = card % 4   (0=C, 1=D, 2=H, 3=S)
# Value lookup: RANK_VALUES = [2,3,4,5,6,7,8,9,10,10,10,10,None]  # Ace=None, resolved contextually

RANK_VALUES: list[int | None] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, None]  # index 12 = Ace
```

**Why integer encoding:** Python strings are incompatible with Numba JIT and slow for large-scale DP/CFR iterations. All card logic operates on integers. String representations (`'AS'`, `'KH'`) are used only at human-readable I/O boundaries.

### 4.2 Deck Representation

```python
import numpy as np

# Internal: numpy array of length 52
# 1 = card available in deck, 0 = card dealt
deck: np.ndarray = np.ones(52, dtype=np.int8)

# For Numba-compatible hot loops, convert to tuple for GameState hashing,
# operate on numpy arrays during simulation
```

### 4.3 Strategy Representation

**Player Strategy:**
```python
# Maps (own_hand: tuple, deck_state: tuple, phase: Phase) → action probabilities
π_player: dict[tuple, dict[PlayerAction, float]]

class PlayerAction(Enum):
    HIT = auto()
    STAND = auto()
```

**Dealer Strategy:**
```python
# Maps (own_hand: tuple, deck_state: tuple, revealed_outcomes: tuple) → action probabilities
π_dealer: dict[tuple, dict[DealerAction, float]]

class DealerAction(Enum):
    REVEAL_PLAYER = auto()   # Open 3+ card player and settle
    HIT = auto()
    STAND = auto()
    SURRENDER = auto()       # Hard 15 only
```

### 4.4 Solver Algorithms

**Phase 1: Baseline Player Strategy (Weeks 1–3)**
- Method: Backward induction / Dynamic programming
- Assumption: Dealer follows fixed strategy (reveal 3-card players at 16/17, hit, stand on 18+)
- Output: Optimal player hit/stand chart

**Phase 2: Full Equilibrium (Weeks 4–8)**
- Method: Counterfactual Regret Minimization (CFR → CFR+)
- Handles: Imperfect information from dealer selective reveals
- Output: Nash equilibrium strategies for both player and dealer

**Phase 3: Validation**
- Monte Carlo simulation (1M+ hands)
- Cross-validation against hand-calculated edge cases
- Convergence testing (exploitability < 0.01 units/hand)

### 4.5 EV Reporting Standard

- All EV figures reported in **units per hand** (e.g. −0.023 units/hand)
- All house edge figures reported as **percentage** (e.g. 2.3% house edge)
- Variance reported as standard deviation in units

### 4.6 Technical Stack

**Primary Language:** Python 3.11+

**Core Libraries:**
```
numpy       — array operations, deck representation (int8 arrays)
numba       — JIT compilation for hot loops (requires numpy arrays, not dicts/strings)
scipy       — optimization routines for convergence checks
matplotlib  — strategy heat maps, EV convergence curves
seaborn     — styled visualizations
plotly      — interactive strategy lookup tool
pandas      — EV tables and reporting
pytest      — unit testing (built in parallel with engine)
```

**Architecture Note on Numba:** Numba JIT cannot operate on Python dicts, strings, or objects. All performance-critical paths (deal loops, hand evaluation, CFR iterations) must operate on `numpy` arrays or Python numeric primitives. The `deck_remaining` state is a `numpy.ndarray` during computation and converted to a `tuple` only for hashing in the DP memoisation table.

**Do not use OpenSpiel:** The overhead of implementing a custom game in OpenSpiel's abstraction layer outweighs its benefits for this project. A hand-rolled CFR in NumPy is faster to debug, easier to validate against the rules, and produces clearer insight into why the equilibrium is what it is.

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
| Selective reveal | Imperfect information | Partial board reveal |

User understands: GTO, Nash equilibrium, EV, variance, bankroll requirements, exploitative vs unexploitable play.

### 5.4 Critical Implementation Gotchas

**Deck Representation (Numba-compatible):**
```python
# WRONG: String-based deck, incompatible with Numba
deck = ['AS', 'KH', '7D', ...]

# CORRECT: Integer-indexed numpy array
deck = np.ones(52, dtype=np.int8)  # 1=available, 0=dealt

def deal_card(deck: np.ndarray) -> int:
    """Draw a random available card from the deck."""
    available = np.where(deck == 1)[0]
    card = np.random.choice(available)
    deck[card] = 0
    return int(card)
```

**Ace Valuation:**
```python
# WRONG: Hard-coding ace value
if card == 'A':
    value = 11  # Ignores hand size rule

# CORRECT: Context-aware ace resolution
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

**Hard 15 Detection:**
```python
# CORRECT: No ace present at all — not just "no usable ace"
def is_hard_fifteen(hand_cards: tuple[int, ...]) -> bool:
    """
    Hard 15: exactly 2 cards, total = 15, contains no Ace.
    Soft 15 (A-4) does NOT qualify.
    """
    if len(hand_cards) != 2:
        return False
    ranks = [card // 4 for card in hand_cards]
    has_ace = 12 in ranks  # rank index 12 = Ace
    if has_ace:
        return False
    total = sum(RANK_VALUES[r] for r in ranks)
    return total == 15
```

**Dealer Selective Reveal Order:**
```python
# CORRECT: Reveal 3+ card players BEFORE hitting at 16/17
def dealer_action_sequence(dealer_hand, player_hand, deck):
    dealer_total = calculate_total(dealer_hand)

    if dealer_total in (16, 17):  # Includes soft 17
        # Step 1: Reveal and settle 3+ card players first
        if len(player_hand) >= 3:
            outcome = compare_hands(dealer_hand, player_hand)
            settle(player, outcome)
            player.is_active = False

        # Step 2: Dealer now decides to hit against remaining 2-card players
        # (Optimal strategy determined by solver)
        dealer_hits = should_dealer_hit(dealer_hand, deck)
        if dealer_hits:
            new_card = deal_card(deck)
            dealer_hand.append(new_card)

        # Step 3: Reveal and settle remaining active players
        if player.is_active:
            outcome = compare_hands(dealer_hand, player_hand)
            settle(player, outcome)

# WRONG: Revealing all players before hitting
def dealer_action_wrong(dealer_hand, player_hand, deck):
    outcome = compare_hands(dealer_hand, player_hand)  # Premature!
    settle(player, outcome)
    dealer_hits = should_dealer_hit(dealer_hand, deck)
    # Too late — player already settled
```

**Unconditional Forfeit for ≤15:**
```python
# CORRECT: Check forfeit before any other settlement logic
def settle_player(player_total, player_hand_size, dealer_total, dealer_busted, ...):
    if player_total <= 15:
        return Outcome.LOSS  # Unconditional. Dealer bust does not help.
    # ... rest of settlement logic
```

### 5.5 Critical Edge Cases — Must Have Unit Tests

```python
# Edge Case 1: Dealer hard 15 overrides Ban Ban
def test_dealer_hard_15_cancels_ban_ban():
    """Player reveals Ban Ban. Dealer has hard 15 (e.g. 6-9) and surrenders.
    Expected: Push — all bets returned, Ban Ban win voided.
    Hard 15 surrender is checked BEFORE special hand payouts."""

# Edge Case 2: Ban Ban total
def test_ban_ban_total_is_21():
    """Player has A-A. Expected: Ban Ban, total = 21 (one ace=11, one=10).
    Hand is immediately settled, never played further."""

# Edge Case 3: Soft 15 is NOT a hard 15
def test_soft_fifteen_not_hard_fifteen():
    """Dealer has A-4 (soft 15, total = 15). Expected: NOT a hard 15.
    Dealer may not surrender on soft 15."""

# Edge Case 4: Five-card vs five-card — same total
def test_five_card_vs_five_card_push():
    """Both player and dealer have 5 cards totaling 19. Expected: Push."""

# Edge Case 5: Five-card vs five-card — different totals
def test_five_card_vs_five_card_higher_wins():
    """Player 5-card 20, dealer 5-card 19. Expected: Player wins at 1:1
    (regular win, not 2:1, since dealer also has 5 cards)."""

# Edge Case 6: 777 beats dealer 21
def test_777_beats_dealer_21():
    """Player has 7-7-7 (any suits). Dealer has J-J-A (21).
    Expected: Player wins 7:1."""

# Edge Case 7: 777 suit independence
def test_777_any_suits():
    """Player has 7C-7D-7H. Player has 7S-7C-7D. Both are valid 777s."""

# Edge Case 8: Player ≤15 forfeits even on dealer bust
def test_player_fifteen_forfeits_on_dealer_bust():
    """Player stands on 15. Dealer draws and busts.
    Expected: Player still loses. Dealer bust provides no relief."""

# Edge Case 9: Player ≤15 forfeits even on dealer ≤15
def test_player_fifteen_forfeits_regardless():
    """Player stands on 14. Dealer stands on 14 (equal total).
    Expected: Player loses. Not a push — forfeit is unconditional."""

# Edge Case 10: Ban Ban vs Ban Ban push
def test_ban_ban_vs_ban_ban_push():
    """Player A-A, dealer A-A. Expected: Push."""

# Edge Case 11: Ban Luck vs Ban Luck push
def test_ban_luck_vs_ban_luck_push():
    """Player A-K, dealer A-Q. Expected: Push."""

# Edge Case 12: Multiple aces in 3+ card hand
def test_three_aces():
    """Player has A-A-A (3 cards). Ace values: 1 or 10 each.
    Best achievable total without busting: A(10)+A(1)+A(1) = 12, or A(1)+A(1)+A(1) = 3.
    Verify total calculation is 13 (10+1+1+... depends on context), not bust."""

# Edge Case 13: Dealer selective reveal at 17, then improves to 21
def test_dealer_reveals_then_improves():
    """Dealer has 17. Reveals 3-card player who busted (collects).
    Dealer hits to 21. Reveals 2-card player with 20.
    Expected: Dealer collects from both. 3-card bust settled first, 2-card loss settled after improvement."""

# Edge Case 14: Solver always announces five-card
def test_solver_always_announces_five_card():
    """Player with 4 cards always treated as having announced before drawing 5th.
    5-card bonus always applies in solver context."""
```

### 5.6 Project Structure

```
banluck-solver/
├── docs/
│   ├── PRD_v1.1.md
│   ├── game_rules.md
│   └── strategy_tables/
├── src/
│   ├── engine/
│   │   ├── deck.py          # numpy int8 array deck, deal_card(), card_to_rank()
│   │   ├── hand.py          # Hand evaluation, ace logic, special hand detection
│   │   ├── rules.py         # Payout calculation, hand comparison, forfeit logic
│   │   └── game_state.py    # Frozen dataclass state, phase transitions
│   ├── solvers/
│   │   ├── baseline_dp.py   # Phase 1: backward induction, fixed dealer strategy
│   │   ├── cfr.py           # Phase 2: CFR+ solver
│   │   └── ev_tables.py     # Pre-computed EV lookup tables
│   ├── analysis/
│   │   ├── simulator.py     # Monte Carlo validation (1M+ hands)
│   │   ├── ev_calculator.py # EV in units, house edge %
│   │   └── visualize.py     # Strategy heat maps, plotly interactive tool
│   └── main.py
├── tests/
│   ├── test_deck.py
│   ├── test_hand_evaluation.py
│   ├── test_special_hands.py
│   ├── test_dealer_surrender.py
│   ├── test_dealer_reveal.py
│   ├── test_five_card_hands.py
│   ├── test_ace_valuation.py
│   └── test_settlement.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_baseline_strategy.ipynb
│   └── 03_cfr_analysis.ipynb
├── requirements.txt
└── README.md
```

### 5.7 Naming Conventions

```python
# Card encoding (integers)
player_cards: tuple[int, ...] = (48, 11)  # e.g. Ace of Spades (48), 3 of Hearts (11)
dealer_cards: tuple[int, ...] = (24, 28)

# Human-readable conversion (I/O only)
RANK_NAMES = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
SUIT_NAMES = ['C','D','H','S']
def card_to_str(card: int) -> str:
    return RANK_NAMES[card // 4] + SUIT_NAMES[card % 4]

# Totals
player_total: int = 21
dealer_total: int = 17

# Actions
from enum import Enum, auto
class PlayerAction(Enum):
    HIT = auto()
    STAND = auto()

class DealerAction(Enum):
    REVEAL_PLAYER = auto()
    HIT = auto()
    STAND = auto()
    SURRENDER = auto()

# Outcomes
class Outcome(Enum):
    WIN = auto()
    LOSS = auto()
    PUSH = auto()

# Payouts
payout_multipliers: dict[str, int] = {
    'ban_ban': 3,
    'ban_luck': 2,
    '777': 7,
    'five_card_21': 3,
    'five_card_sub21': 2,
    'regular_win': 1,
}
```

### 5.8 Rank Values Reference

```python
# Rank index 0–12 maps to card ranks 2–A
# Index: 0  1  2  3  4  5  6  7   8   9   10  11  12
# Rank:  2  3  4  5  6  7  8  9  10   J   Q   K   A
RANK_VALUES: list[int | None] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, None]
# Ace (index 12) is None — always resolved contextually via resolve_ace()

# Count of each rank in a fresh deck (4 suits each)
# Ranks 0–11 have 4 copies. Ace (rank 12) has 4 copies.
DECK_RANK_COUNTS: np.ndarray = np.array([4]*13, dtype=np.int8)
```

---

## 6. Deliverables

### Phase 1 (Weeks 1–3): Game Engine + Baseline Solver
- ✅ Formal game specification (this PRD)
- ⏭️ Game engine: `deck.py`, `hand.py`, `rules.py`, `game_state.py`
- ⏭️ Comprehensive unit tests for all 14 edge cases
- ⏭️ Baseline player strategy (vs fixed dealer) via backward induction
- ⏭️ Player hit/stand chart

### Phase 2 (Weeks 4–8): Full Nash Equilibrium
- ⏭️ CFR+ solver implementation
- ⏭️ Dealer optimal reveal strategy
- ⏭️ Complete strategy tables for player and dealer
- ⏭️ EV calculations for all game states (in units + %)

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
- [ ] All 14 edge cases produce correct payouts
- [ ] Strategies converge to equilibrium (exploitability → 0)
- [ ] Simulated EV matches theoretical calculations (±0.1%)
- [ ] Results replicate across different random seeds

### Performance
- [ ] Baseline solver completes in <1 hour
- [ ] Full CFR+ solver converges in <24 hours
- [ ] Strategy lookup is instant (<10ms)

### Insights
- [ ] Quantify dealer advantage from selective reveal (%)
- [ ] Identify optimal dealer reveal threshold
- [ ] Determine if card counting is viable with per-hand shuffles (expected: no)
- [ ] Answer "Should I hit on 16?" definitively for all heads-up scenarios

---

## 8. Key Research Questions

1. **How valuable is dealer's selective reveal?** Quantify the edge in percentage terms
2. **Does optimal play differ significantly from blackjack basic strategy?**
3. **What's the GTO dealer reveal strategy at 16/17?** Always reveal 3-card players? Probabilistic mixing?
4. **How powerful is hard 15 surrender?** What % of hands does it save the dealer from losing?
5. **What's the fair rotation rate for dealer position?** (Every N hands to balance advantage)

---

## 9. Development Phases

**Week 1–2:** Game engine + comprehensive testing (100% rule coverage)
**Week 3–4:** Baseline solver (player-only strategy, fixed dealer)
**Week 5–8:** CFR+ implementation (full Nash equilibrium)
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

✅ **All rules clarified and finalized (v1.1)**
✅ **PRD complete — all 11 gaps from review resolved**
✅ **Implementation guidance and gotchas documented**
⏭️ **Ready to begin Step 1.1: `deck.py` + `hand.py` implementation**
⏭️ **Next: Game engine with comprehensive unit tests for all 14 edge cases**

---

**This PRD is the complete canonical specification. Begin implementation with the game engine (`deck.py` → `hand.py` → `rules.py` → `game_state.py`), building unit tests in parallel. Do not begin solver work until all 14 edge case tests pass.**
