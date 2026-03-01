"""
Monte Carlo simulator for Banluck strategy validation.

Simulates full Banluck hands using the game engine and accumulates per-hand
payouts to produce EV statistics with confidence intervals.

Primary use: cross-validate the Phase 1.1 DP solver results. The DP solver
uses an infinite-deck approximation; Monte Carlo uses the real 52-card deck
(reshuffled every hand). Discrepancy should be < 0.3%; the PRD target is
±0.1% at 1M hands.

Known Phase 1.1 targets (infinite-deck, fixed dealer, no surrender):
    reveal_mode=OFF → player EV ≈ +0.0157 units/hand  (+1.57%)
    reveal_mode=ON  → player EV ≈ +0.0241 units/hand  (+2.41%)

Key implementation notes:
    - play_hand() and settle_with_selective_reveal() both hardcode payout=-1.0
      for ANY player bust, including 5-card busts which should be -2.0.
      _fix_five_card_bust_payout() corrects this.
    - reveal_mode=OFF uses play_hand() directly.
    - reveal_mode=ON replicates play_hand() but routes the dealer+settlement
      phase through settle_with_selective_reveal().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.engine.deck import create_deck, deal_card
from src.engine.game_state import (
    DealerHitStrategy,
    DealerSurrenderStrategy,
    HandResult,
    PlayerAction,
    PlayerStrategy,
    _dealer_basic_hit_strategy,
    _dealer_never_surrenders,
    play_hand,
    settle_with_selective_reveal,
)
from src.engine.hand import calculate_total, is_soft
from src.engine.rules import Outcome, settle_hand
from src.engine.special_hands import classify_hand, is_ban_ban, is_ban_luck

# ─── Result type ──────────────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """Aggregate statistics from a Monte Carlo simulation run.

    Attributes:
        n_hands:       Number of hands simulated.
        mean_ev:       Mean payout per hand in units (positive = player wins).
        std_ev:        Sample standard deviation of per-hand payouts.
        ci_95_low:     Lower bound of 95% confidence interval for mean_ev.
        ci_95_high:    Upper bound of 95% confidence interval for mean_ev.
        house_edge_pct: House edge as a percentage: -mean_ev * 100.
                        Positive = house advantage; negative = player advantage.
        reveal_mode:   Whether selective reveal was active for this run.
        n_wins:        Hands where payout > 0.
        n_losses:      Hands where payout < 0.
        n_pushes:      Hands where payout == 0.
        payouts:       Raw per-hand payout array (float64, length n_hands), or None
                       if simulate_hands() was called with return_payouts=False.
    """

    n_hands: int
    mean_ev: float
    std_ev: float
    ci_95_low: float
    ci_95_high: float
    house_edge_pct: float
    reveal_mode: bool
    n_wins: int
    n_losses: int
    n_pushes: int
    payouts: np.ndarray | None = None

    def __str__(self) -> str:
        sign = "+" if self.mean_ev >= 0 else ""
        edge_sign = "-" if self.house_edge_pct < 0 else "+"
        return (
            f"Hands: {self.n_hands:,} | "
            f"EV: {sign}{self.mean_ev:.4f} ({sign}{self.mean_ev * 100:.2f}%) | "
            f"95% CI: [{self.ci_95_low:.4f}, {self.ci_95_high:.4f}] | "
            f"House edge: {edge_sign}{abs(self.house_edge_pct):.2f}% | "
            f"Reveal: {'ON' if self.reveal_mode else 'OFF'}"
        )


# ─── Payout correction ────────────────────────────────────────────────────────


def _fix_five_card_bust_payout(result: HandResult) -> float:
    """Correct the 5-card bust payout that the engine hardcodes incorrectly.

    Both play_hand() and settle_with_selective_reveal() return payout=-1.0 for
    ANY player bust as an early exit, even when the player has exactly 5 cards
    (which should cost 2 units per the five-card bust penalty rule).

    We identify a 5-card bust by: payout == -1.0 AND len(player_cards) == 5
    AND total > 21. We do NOT change forfeit payouts (total ≤ 15, payout=-1.0)
    because forfeit requires total ≤ 15, which cannot overlap with bust.

    Args:
        result: HandResult from play_hand or settle_with_selective_reveal.

    Returns:
        Corrected payout float.
    """
    if result.payout == -1.0 and len(result.player_cards) == 5:
        if calculate_total(result.player_cards) > 21:
            return -2.0
    return result.payout


# ─── Reveal-mode-ON hand simulation ──────────────────────────────────────────


def _simulate_hand_reveal_on(
    deck: np.ndarray,
    player_strategy: PlayerStrategy,
    dealer_surrender_strategy: DealerSurrenderStrategy,
    dealer_hit_strategy: DealerHitStrategy,
) -> HandResult:
    """Simulate one hand with selective reveal active.

    Replicates the play_hand() flow but routes the dealer phase through
    settle_with_selective_reveal(), so 3+-card players are settled against the
    dealer's initial 16/17 total (before the dealer hits).

    Args:
        deck:                     Fresh 52-card deck (will be mutated).
        player_strategy:          Callable(player_cards, dealer_upcard, deck) → PlayerAction.
        dealer_surrender_strategy: Callable(dealer_cards) → bool.
        dealer_hit_strategy:      Callable(dealer_cards, deck) → bool.

    Returns:
        HandResult for this hand.
    """
    # Deal initial hands
    player_cards = (deal_card(deck), deal_card(deck))
    dealer_cards = (deal_card(deck), deal_card(deck))
    dealer_upcard = dealer_cards[0]

    # Dealer hard-15 surrender (highest priority — overrides all)
    if dealer_surrender_strategy(dealer_cards):
        return HandResult(
            player_cards=player_cards,
            dealer_cards=dealer_cards,
            outcome=Outcome.PUSH,
            payout=0.0,
            player_hand_type=classify_hand(player_cards),
            dealer_hand_type=classify_hand(dealer_cards),
            dealer_surrendered=True,
            dealer_busted=False,
        )

    # Immediate special hand settlement (Ban Ban / Ban Luck never played further)
    if is_ban_ban(player_cards) or is_ban_luck(player_cards):
        outcome, payout = settle_hand(
            player_cards,
            dealer_cards,
            dealer_surrendered=False,
            dealer_busted=False,
        )
        return HandResult(
            player_cards=player_cards,
            dealer_cards=dealer_cards,
            outcome=outcome,
            payout=payout,
            player_hand_type=classify_hand(player_cards),
            dealer_hand_type=classify_hand(dealer_cards),
            dealer_surrendered=False,
            dealer_busted=False,
        )

    # Player action phase
    player_cards_list = list(player_cards)
    while True:
        total = calculate_total(tuple(player_cards_list))
        if total > 21 or len(player_cards_list) == 5 or total == 21:
            break
        action = player_strategy(tuple(player_cards_list), dealer_upcard, deck)
        if action == PlayerAction.STAND:
            break
        player_cards_list.append(deal_card(deck))

    player_cards = tuple(player_cards_list)

    # Dealer phase + settlement via selective reveal
    result, _final_dealer = settle_with_selective_reveal(
        player_cards, dealer_cards, deck, dealer_hit_strategy
    )
    return result


# ─── Core simulation loop ─────────────────────────────────────────────────────


def simulate_hands(
    player_strategy: PlayerStrategy,
    dealer_surrender_strategy: DealerSurrenderStrategy,
    dealer_hit_strategy: DealerHitStrategy,
    n_hands: int = 100_000,
    seed: int | None = 42,
    reveal_mode: bool = False,
    return_payouts: bool = False,
) -> SimulationResult:
    """Simulate n_hands of Banluck and return aggregate statistics.

    Each hand is played with a fresh shuffled deck. Per-hand payouts are
    collected and summarised into mean EV, standard deviation, 95% confidence
    interval, and win/loss/push counts.

    Args:
        player_strategy:          Callable matching PlayerStrategy signature.
        dealer_surrender_strategy: Callable matching DealerSurrenderStrategy.
        dealer_hit_strategy:      Callable matching DealerHitStrategy.
        n_hands:                  Number of hands to simulate.
        seed:                     NumPy random seed for reproducibility. None
                                  for a non-deterministic run.
        reveal_mode:              If False, use play_hand() (no selective reveal).
                                  If True, use selective reveal: 3+-card players
                                  settle against dealer's initial 16/17 total.
        return_payouts:           If True, attach raw per-hand payout array
                                  (float64, length n_hands) to SimulationResult.payouts.
                                  Defaults to False to avoid extra memory allocation.

    Returns:
        SimulationResult with EV statistics for the run.
    """
    if seed is not None:
        np.random.seed(seed)

    payouts: list[float] = []
    n_wins = n_losses = n_pushes = 0

    for _ in range(n_hands):
        deck = create_deck()

        if reveal_mode:
            result = _simulate_hand_reveal_on(
                deck, player_strategy, dealer_surrender_strategy, dealer_hit_strategy
            )
        else:
            result = play_hand(
                deck,
                player_strategy=player_strategy,
                dealer_surrender_strategy=dealer_surrender_strategy,
                dealer_hit_strategy=dealer_hit_strategy,
            )

        payout = _fix_five_card_bust_payout(result)
        payouts.append(payout)

        if payout > 0:
            n_wins += 1
        elif payout < 0:
            n_losses += 1
        else:
            n_pushes += 1

    arr = np.array(payouts, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    ci_margin = 1.96 * std / math.sqrt(n_hands)

    return SimulationResult(
        n_hands=n_hands,
        mean_ev=mean,
        std_ev=std,
        ci_95_low=mean - ci_margin,
        ci_95_high=mean + ci_margin,
        house_edge_pct=-mean * 100.0,
        reveal_mode=reveal_mode,
        n_wins=n_wins,
        n_losses=n_losses,
        n_pushes=n_pushes,
        payouts=arr if return_payouts else None,
    )


# ─── Strategy factories ───────────────────────────────────────────────────────


def make_fixed_dealer_strategy() -> tuple[DealerSurrenderStrategy, DealerHitStrategy]:
    """Return the fixed Phase 1.1 baseline dealer strategy pair.

    Surrender strategy: never surrender (hard-15 surrender is a CFR option;
    the Phase 1.1 baseline models a dealer who cannot surrender).

    Hit strategy: hit when total < 16 (forced minimum), hit at hard/soft 16,
    hit soft 17, stand hard 17+. Matches the _dealer_basic_hit_strategy
    definition in game_state.py.

    Returns:
        (dealer_surrender_strategy, dealer_hit_strategy) tuple of callables.
    """
    return _dealer_never_surrenders, _dealer_basic_hit_strategy


def make_dp_player_strategy(reveal_mode: bool = False) -> PlayerStrategy:
    """Return a player strategy callable wrapping the Phase 1.1 DP solver.

    Calls solve(reveal_mode) once at construction time and caches the result.
    The returned callable maps (player_cards, dealer_upcard, deck) → PlayerAction.

    The DP strategy is keyed on (total, nc, is_soft_flag) where is_soft_flag
    is 0 or 1 (integer). The dealer_upcard argument is IGNORED — Banluck has
    all dealer cards face-down and the DP solver already accounts for this by
    using the marginal dealer distribution.

    Args:
        reveal_mode: Which DP strategy table to load (ON or OFF).

    Returns:
        PlayerStrategy callable.
    """
    # Import here to avoid circular deps and to keep the solve() call lazy.
    from src.solvers.baseline_dp import Action, solve

    table = solve(reveal_mode)

    def _strategy(
        player_cards: tuple[int, ...],
        dealer_upcard: int,  # ignored in Banluck
        deck: np.ndarray,  # ignored by DP strategy
    ) -> PlayerAction:
        total = calculate_total(player_cards)
        nc = len(player_cards)
        is_soft_flag = int(is_soft(player_cards))

        # Terminal cases — always stand
        if total > 21 or total == 21 or nc == 5:
            return PlayerAction.STAND

        key = (total, nc, is_soft_flag)
        entry = table.get(key)
        if entry is None:
            # Fallback: stand on anything not in the table
            return PlayerAction.STAND

        action_dp, _ = entry
        return PlayerAction.HIT if action_dp == Action.HIT else PlayerAction.STAND

    return _strategy


def make_simple_player_strategy(stand_threshold: int = 17) -> PlayerStrategy:
    """Return a simple threshold player strategy.

    Stand on total >= stand_threshold, hit otherwise. Useful as a baseline
    to verify that optimal play outperforms naive play.

    Args:
        stand_threshold: Total at which the player starts standing.

    Returns:
        PlayerStrategy callable.
    """

    def _strategy(
        player_cards: tuple[int, ...],
        dealer_upcard: int,
        deck: np.ndarray,
    ) -> PlayerAction:
        total = calculate_total(player_cards)
        return PlayerAction.STAND if total >= stand_threshold else PlayerAction.HIT

    return _strategy


# ─── Validation convenience ───────────────────────────────────────────────────


def run_validation(
    n_hands: int = 200_000,
    seed: int = 42,
) -> dict[str, SimulationResult]:
    """Run both reveal modes with optimal DP strategies and return results.

    Uses the Phase 1.1 DP solver strategies and the fixed baseline dealer.
    Results should match Phase 1.1 EV targets within ±0.5% (absolute):
        reveal_mode=OFF → mean_ev ≈ +0.0157
        reveal_mode=ON  → mean_ev ≈ +0.0241

    Args:
        n_hands: Hands per run.
        seed:    NumPy random seed. Each mode gets a separate seeded run.

    Returns:
        {'reveal_off': SimulationResult, 'reveal_on': SimulationResult}
    """
    surrender_strat, hit_strat = make_fixed_dealer_strategy()

    result_off = simulate_hands(
        player_strategy=make_dp_player_strategy(reveal_mode=False),
        dealer_surrender_strategy=surrender_strat,
        dealer_hit_strategy=hit_strat,
        n_hands=n_hands,
        seed=seed,
        reveal_mode=False,
    )
    result_on = simulate_hands(
        player_strategy=make_dp_player_strategy(reveal_mode=True),
        dealer_surrender_strategy=surrender_strat,
        dealer_hit_strategy=hit_strat,
        n_hands=n_hands,
        seed=seed,
        reveal_mode=True,
    )
    return {"reveal_off": result_off, "reveal_on": result_on}


# ─── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Banluck Monte Carlo Validation — 200,000 hands per mode\n")
    results = run_validation(n_hands=200_000)
    print(f"reveal_mode=OFF: {results['reveal_off']}")
    print(f"reveal_mode=ON:  {results['reveal_on']}")
    delta = results["reveal_on"].mean_ev - results["reveal_off"].mean_ev
    print(f"\nEV delta (ON − OFF): {delta:+.4f} ({delta * 100:+.2f}%)")
    print("\nPhase 1.1 DP targets:")
    print("  reveal_mode=OFF → +0.0157 (+1.57%)")
    print("  reveal_mode=ON  → +0.0241 (+2.41%)")
    print("  delta           →  +0.0084 (+0.84%)")
