# Banluck GTO Solver — Key Findings

> Generated 2026-03-01 from the Banluck GTO Solver (Phase 1 → Phase 3 complete).
> Engine: backward-induction DP + CFR+ Nash equilibrium (Numba-accelerated, ~6.7 ms/pass).
> Monte Carlo validation: 200,000 hands, real deck, seed 42.

---

## House Edge

| Scenario | Player EV | Notes |
|---|---|---|
| Infinite-deck, no dealer surrender, reveal=OFF | **+1.57%** (player edge) | DP theoretical approximation |
| Infinite-deck, no dealer surrender, reveal=ON | **+2.41%** (player edge) | DP theoretical approximation |
| Real-deck MC (200k hands), no dealer surrender, reveal=OFF | **−4.71%** (house edge) | Best empirical baseline |
| Real-deck MC (200k hands), no dealer surrender, reveal=ON | **−3.75%** (house edge) | Best empirical baseline |
| CFR Nash, dealer surrender enabled, reveal=OFF equivalent | **≈ −5% to −6%** (house edge) | Surrender adds ~0.3–1.3% to house edge |

**Why infinite-deck and real-deck differ by ~6.3 percentage points (even though both have no surrender):**
The DP solver assumes an infinite, never-depleting deck (each rank always drawn at probability 1/13). In a real 52-card deck, three effects compound: (1) *Forfeit-forced-hit correlation* — players at ≤15 must keep hitting or face automatic loss, consuming more cards than normal; this depletes the deck in a way that systematically hurts the player. (2) *Special hand depletion* — aces and 10-value cards dealt to players are unavailable for Ban Ban/Ban Luck combinations, slightly reducing the player's best payouts. (3) *Strategy mismatch* — the DP strategy is tuned for infinite-deck probabilities and is slightly suboptimal on a real deck. The forfeit-forced-hit correlation is the dominant driver (~6% overestimate in the infinite-deck model).

**Why dealer surrender only adds ~0.3–1.3% to house edge (not a large jump):**
Dealer hard-15 occurs in only 7.10% of deals (12/169 starting pairs), capping the maximum possible impact. Even then, the dealer surrenders with a mixed GTO probability — not always. Surrender converts certain big losses (vs Ban Ban/Luck) into pushes, but the limited frequency of hard-15 means the total effect is modest.

---

## Player Strategy — Where Banluck Differs From Blackjack

### Hard hands (2–3 cards)
- **Hard 17+**: Always STAND — identical to blackjack basic strategy.
- **Hard 16**: STAND on 2 or 3 cards — same as blackjack (in heads-up, no upcard to adjust against).

### Hard hands (4 cards) — the big divergence
- **Hard 16 with 4 cards: HIT** — unique to Banluck. Any card 2–5 gives you a 5-card sub-21 (2:1 bonus), a 5 gives 5-card 21 (3:1 bonus). The bonus EV makes hitting correct even though it busts on 6–K.
- **Hard 17–21 with 4 cards: STAND** — the bust risk outweighs the five-card bonus.

### Soft hands — most dramatic departure from blackjack

| Hand | 2 cards | 3 cards | 4 cards |
|---|---|---|---|
| Soft 16–18 | HIT | HIT | **HIT** |
| Soft 19 | STAND | HIT | **HIT** |
| Soft 20 | STAND | STAND | **HIT** |
| Soft 21 | STAND | STAND | STAND |

- **All 4-card soft hands below 21: always HIT** — in blackjack you'd stand on soft 18, 19, 20. In Banluck, the five-card bonus is so valuable that hitting any 4-card soft hand (that isn't 21) is always correct.
- **3-card soft 19 and soft 20: HIT** — also diverges from blackjack (where these always stand). Getting a 5th card for a bonus outweighs the risk.
- **3-card soft 16–18: HIT** — consistent with blackjack for the weaker soft hands, but for a different reason (ace in 3+-card hands is worth only 1 or 10, not 11, so soft hands are weaker than they look in BJ terms).

**The rule of thumb**: If you have 4 cards and aren't at 21, **always hit** regardless of your total. The five-card bonus is that powerful.

---

## Dealer Strategy (Nash Equilibrium)

### Hard-15 surrender
- A dealer hard-15 hand occurs in **7.10% of all deals** (analytic: 12/169 starting pairs).
- At Nash equilibrium, the dealer surrenders with a GTO probability — this converts certain big losses (e.g., vs Ban Ban/Ban Luck) into pushes.
- Surrendering is the single largest lever the dealer has for reducing player EV.

### At 16/17 (reveal decision)
- In **1v1 heads-up play**, REVEAL and STAND are **payoff-equivalent** — the dealer is indifferent. CFR finds mixed strategies with no reveal advantage.
- The selective reveal only has strategic value in **multi-player games** (3+ players), where revealing settles some players against the current total before a hit.

---

## Dealer Selective Reveal — How Valuable Is It?

| Context | Value to Dealer |
|---|---|
| 1v1 heads-up | **0%** — pure indifference |
| Multi-player (DP, infinite-deck) | **+0.84%** house edge increase |
| Multi-player (MC, real-deck) | **+0.96%** house edge increase |

The reveal converts EV for the dealer by settling 3+-card players against the dealer's *pre-hit* total. If the dealer then busts, those already-settled players still lose — this asymmetry is the source of the advantage.

---

## Bankroll & Variance

- **Std deviation per hand**: ~1.0 units (driven by occasional Ban Ban 3:1, 777 7:1 payouts).
- **Risk of ruin** at a 20-unit bankroll: very high — see the Bankroll tab in the dashboard for exact figures at your unit size.
- **Horizon projections**: At 1,000 hands the expected loss is ~47 units (at 1 unit/hand), with the 95% confidence interval spanning roughly ±62 units around that.

---

## Fair Dealer Rotation (Q5)

Rotate the dealer every **~112 hands**.

This is derived from N*/4 where N* ≈ 450 — the number of hands at which cumulative house edge first exceeds one standard deviation of variance noise. The formula is N* = (σ/h)², where σ ≈ 1.0 (per-hand std dev) and h ≈ 4.71%/hand (edge magnitude).

Rotating more frequently than this is fair to all players; rotating less allows systematic variance to accumulate in a way that becomes statistically detectable.

---

## Special Hand Payouts (Reference)

| Hand | Condition | Payout |
|---|---|---|
| Ban Ban | 2 aces | 3:1 |
| Ban Luck | Ace + 10-value card | 2:1 |
| 777 | Three 7s (exactly 3 cards) | 7:1 |
| Five-card 21 | 5 cards totalling exactly 21 | 3:1 |
| Five-card <21 | 5 cards totalling <21 (no bust) | 2:1 |
| Regular win | All other non-bust wins | 1:1 |

Note: Player forfeit (total ≤15 after all hits) is an **unconditional loss** — even if the dealer subsequently busts.

---

## Practical Takeaways for a Banluck Player

1. **The house always wins long-term** (~4.7% edge with real deck + dealer surrender) — this is a negative-EV game for players regardless of strategy.
2. **Never stand on a 4-card soft hand** below 21. The five-card bonus makes hitting mandatory every time.
3. **Hit hard 16 with exactly 4 cards** — the only 4-card hard hand where hitting is correct.
4. **Stand hard 17+ regardless of card count** (5-card hands are already settled).
5. **The dealer's hard-15 surrender hurts you more than you might expect** — it converts your biggest wins (Ban Ban/Luck vs a surrendering dealer) into pushes.
6. **Selective reveal is a multi-player concern** — in 1v1 it makes no difference; in group play, the dealer revealing before hitting at 16/17 costs the player roughly 1% EV.
7. **Short sessions reduce expected loss** — with σ ≈ 1.0 and edge ≈ −4.7%, you're roughly as likely to be up as down for the first ~100 hands. Beyond ~450 hands, the house edge dominates variance and a losing outcome becomes highly probable.
