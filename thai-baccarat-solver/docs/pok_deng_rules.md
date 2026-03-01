# Pok Deng (ป๊อกเด้ง) — Thai Baccarat Rules

## Overview

Pok Deng is a Thai card game played between one **banker** (เจ้ามือ) and one to
sixteen **players** (ผู้เล่น). Each player competes independently against the
banker; players do not compete against each other. The game uses a single
standard 52-card deck.

---

## Card Values

| Card            | Value |
|-----------------|-------|
| 2–9             | Face value (2–9) |
| 10, J, Q, K     | 0     |
| Ace             | 1     |

A **hand value** is the units digit of the sum of the cards:

```
hand_value = (sum of card values) mod 10
```

Examples: A+9 = 10 mod 10 = **0**; 7+8 = 15 mod 10 = **5**; K+3+6 = 9 mod 10 = **9**.

---

## Deal

1. The banker deals **2 cards** face-down to each player and to themselves.
2. Special hands (**Pok**) are checked immediately after the deal.
3. Players act before the banker.

---

## Special Hands (Pok)

A **Pok** is a 2-card natural and is resolved immediately — no further cards
may be drawn.

| Hand   | Condition            | Notes                        |
|--------|----------------------|------------------------------|
| Pok 9  | 2-card total = 9     | Highest natural              |
| Pok 8  | 2-card total = 8     | Second highest natural       |

If both player and banker have Pok hands: higher Pok wins; equal Pok → **push**
(tie, no exchange).

---

## Player Action Phase

Players who do **not** hold a Pok may:

- **Stand** (หยุด) — keep their 2-card hand.
- **Hit** (เพิ่ม) — draw exactly **one** additional card (total 3 cards).

Players may not draw more than one extra card. The decision is made privately
before the banker reveals their own hand.

### Player Standing Ranges (common convention)

- Always stand on 7, 8, 9 (strong hands).
- Always hit on 0, 1, 2 (weak hands).
- 3–6: player's discretion (GTO solver target).

---

## Banker Action Phase

After all players have acted, the banker reveals their hand and may also:

- **Stand** — keep 2-card hand.
- **Hit** — draw exactly one additional card.

The banker's decision affects all remaining players simultaneously.

### Banker Standing Ranges (common convention)

- Stand on 7, 8, 9.
- Hit on 0–4 (must hit, house rule varies).
- 5–6: banker's discretion.

---

## Special Hand Combinations (Multipliers)

These apply to **3-card hands** and increase the payout multiplier:

| Combination        | Thai Name      | Condition                                 | Multiplier |
|--------------------|----------------|-------------------------------------------|------------|
| Three of a Kind    | ตอง (Tong)     | All 3 cards same rank                     | ×5         |
| Straight Flush     | สามเลือด (Sam Lueang) | 3 cards same suit, consecutive rank | ×3         |
| Straight           | ดอกจิก (Dok Jik) | 3 consecutive ranks, any suit          | ×2         |
| Flush (same suit)  | น้ำ (Nam)      | All 3 cards same suit (not straight)      | ×2         |

**Note**: Multipliers are symmetric — if a player holds a Tong and wins, they
receive 5× the bet; if they lose, they pay 5× the bet.

**When both sides have multipliers**: multiply them together.
Example: player Tong (×5) beats banker Nam (×2) → net multiplier = ×10.

### Rank Ordering for Straights

For straight detection, Ace = 1 only. Consecutive ranks use the natural card
order: A(1)–2–3–4–5–6–7–8–9–10–J–Q–K.

Wrapping (K–A–2) is **not** a straight.

---

## Settlement

Settlement is player-vs-banker, one-on-one:

| Outcome                | Result                                     |
|------------------------|--------------------------------------------|
| Player total > banker  | Player wins; banker pays player's bet × multiplier |
| Player total < banker  | Banker wins; player pays bet × multiplier  |
| Equal totals           | **Push** — no money changes hands          |

**Multiplier resolution order**:
1. Identify each side's combination multiplier (1 if none).
2. Final multiplier = player_mult × banker_mult (or just the higher side's, by some variants — specify per game).
3. Apply to base bet.

---

## Settlement Priority

1. **Pok vs Pok** — higher total wins; tie → push.
2. **Pok vs non-Pok** — Pok wins regardless of non-Pok total.
3. **Both non-Pok** — compare hand values; higher wins. Equal → push.

---

## Bust Rule

There is **no bust** in Pok Deng. Hand value is always taken mod 10, so any
number of cards results in a valid value 0–9.

---

## Betting

- Each player places a bet before the deal.
- The banker must have enough chips to cover all players' maximum bets.
- Banker position rotates (or stays by convention) after each round.

---

## Key Differences from Standard Baccarat

| Feature              | Standard Baccarat       | Pok Deng                        |
|----------------------|-------------------------|---------------------------------|
| Players vs banker    | Fixed shoe, no choice   | Player chooses to hit/stand     |
| Third-card rule      | Rigid tableau           | Voluntary (player); semi-free (banker) |
| Special combos       | None                    | Tong, Sam Lueang, Dok Jik, Nam  |
| Number of players    | Typically 1–8           | 1–16 vs banker                  |
| Ace value            | 1 always                | 1 always (mod 10 scoring)       |

---

## Solver Scope

The GTO solver targets the **heads-up (1 player vs banker)** variant to keep
the game tree tractable. Multi-player variants can be approximated by running
heads-up EV computations per player seat independently (players do not interact
strategically with each other).

### Decision Points

1. **Player hit/stand** — given player's 2-card hand value (0–9) and banker's
   upcard (0–9 by rank class).
2. **Banker hit/stand** — given banker's 2-card hand value (0–9), with
   knowledge of all player hand sizes and values.

### State Representation (planned)

- Player hand: `(card1, card2)` or `(card1, card2, card3)` — integer encoding.
- Banker hand: same encoding.
- Deck: NumPy int8 array (1=available, 0=dealt), or infinite-deck approximation.
- Multiplier flags: derived from hand composition.

---

## References

- Common Thai Pok Deng house rules (Bangkok casinos, 2020–2024).
- Wikipedia: [Pok Deng](https://en.wikipedia.org/wiki/Pok_Deng).
