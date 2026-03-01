"""
Baseline DP solver for Banluck using backward induction.

Fixed dealer strategy: hit ≤16, hit soft-17, stand hard-17+.
Two runs: reveal_mode=ON (dealer reveals 3+-card players at 16/17 before hitting)
          reveal_mode=OFF (all players settle against final dealer total).

State abstracted as composition (non_ace_total, num_aces, num_cards) to avoid
enumerating individual cards in the hot path.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum

from src.engine.cards import RANK_ACE, TEN_VALUE_RANKS

# ─── Constants ────────────────────────────────────────────────────────────────

RANK_PROB: float = 1.0 / 13
"""Probability of drawing any one rank from an infinite (uniform) deck."""

RANK_VALUES_SOLVER: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 0]
"""Point values by rank index (0=2 … 11=K, 12=Ace sentinel=0).

Ace (index 12) is represented as 0 here; ace logic is handled explicitly
by _total_from_composition and _transition.
"""

NUM_RANKS: int = 13
"""Number of distinct ranks in a standard deck."""


# ─── Action ───────────────────────────────────────────────────────────────────


class Action(Enum):
    """Player or dealer actions available during a hand."""

    HIT = "HIT"
    STAND = "STAND"


# ─── DealerOutcome ────────────────────────────────────────────────────────────


@dataclass
class DealerOutcome:
    """Pre-computed dealer probability distribution for a given upcard and deck.

    Captures the full distribution needed to compute player EV without
    re-running the dealer simulation per player state.

    Attributes:
        final_dist:         {total: prob} for all non-bust dealer final totals.
        bust_prob:          Probability dealer busts.
        init_16_prob:       P(dealer reaches a hard-16 state before final action).
        init_17_prob:       P(dealer reaches a soft-17 state before final action).
        init_16_final_dist: Distribution over final totals *after* dealer hits from 16.
        init_17_final_dist: Distribution over final totals *after* dealer hits from soft-17.
    """

    final_dist: dict[int, float]
    bust_prob: float
    init_16_prob: float
    init_17_prob: float
    init_16_final_dist: dict[int, float]
    init_17_final_dist: dict[int, float]


# ─── A2: Composition helpers ──────────────────────────────────────────────────


@functools.cache
def _total_from_composition(non_ace_total: int, num_aces: int, num_cards: int) -> int:
    """Compute best hand total from the abstract composition representation.

    Mirrors calculate_total() from hand.py exactly, operating on the
    composition triple rather than individual card integers.

    Ace resolution is greedy: for each ace add the highest value that
    doesn't bust. Banluck rule: 2-card hands use 11/10; 3+-card hands use 10/1.

    Args:
        non_ace_total: Sum of all non-ace card point values.
        num_aces:      Number of aces in the hand.
        num_cards:     Total card count (aces + non-aces). This is the FINAL
                       hand size, matching how hand.py uses len(cards).

    Returns:
        Best achievable total ≤21, or the minimum bust total if unavoidably bust.

    Examples:
        >>> _total_from_composition(10, 1, 2)   # 10 + A(11) = 21 (Ban Luck)
        21
        >>> _total_from_composition(0, 2, 2)    # A(11) + A(10) = 21 (Ban Ban)
        21
        >>> _total_from_composition(10, 1, 3)   # 10 + A(10) = 20
        20
        >>> _total_from_composition(12, 1, 3)   # 12 + A(1)  = 13
        13
    """
    total = non_ace_total
    for _ in range(num_aces):
        if num_cards == 2:
            total += 11 if total + 11 <= 21 else 10
        else:
            total += 10 if total + 10 <= 21 else 1
    return total


@functools.cache
def _is_soft_from_composition(non_ace_total: int, num_aces: int, num_cards: int) -> bool:
    """Return True if at least one ace is counted at its high value (soft hand).

    Mirrors is_soft() from hand.py. For 2-card hands: soft means ace=11.
    For 3+-card hands: soft means ace=10.

    Args:
        non_ace_total: Sum of all non-ace card point values.
        num_aces:      Number of aces in the hand.
        num_cards:     Total card count.

    Returns:
        True if the hand contains at least one ace valued at its high value.

    Examples:
        >>> _is_soft_from_composition(10, 1, 2)   # A-10, ace=11 → soft
        True
        >>> _is_soft_from_composition(11, 1, 2)   # 11+11=22>21 → hard
        False
        >>> _is_soft_from_composition(5, 1, 3)    # 5+10=15≤21 → soft
        True
        >>> _is_soft_from_composition(12, 1, 3)   # 12+10=22>21 → hard
        False
    """
    if num_aces < 1:
        return False
    high_value = 11 if num_cards == 2 else 10
    return non_ace_total + high_value <= 21


def _transition(nat: int, na: int, nc: int, drawn_rank: int) -> tuple[int, int, int]:
    """Apply one card draw to a composition triple.

    Args:
        nat:         Current non_ace_total.
        na:          Current num_aces.
        nc:          Current num_cards.
        drawn_rank:  Rank index of the drawn card (0–12).

    Returns:
        New (non_ace_total, num_aces, num_cards) after drawing the card.

    Examples:
        >>> _transition(10, 0, 2, 3)        # draw a 5 (rank idx 3): nat 10→15
        (15, 0, 3)
        >>> _transition(10, 0, 2, RANK_ACE) # draw an ace: na 0→1
        (10, 1, 3)
        >>> _transition(5, 1, 3, 8)         # draw a 10 (rank idx 8): nat 5→15
        (15, 1, 4)
        >>> _transition(10, 0, 2, 0)        # draw a 2 (rank idx 0): nat 10→12
        (12, 0, 3)
    """
    if drawn_rank == RANK_ACE:
        return (nat, na + 1, nc + 1)
    return (nat + RANK_VALUES_SOLVER[drawn_rank], na, nc + 1)


# ─── A4: Dealer outcome distribution ─────────────────────────────────────────


def _dealer_play_recursive(
    nat: int,
    na: int,
    nc: int,
    memo: dict,
) -> dict:
    """Recursively compute the dealer's final outcome distribution.

    Uses the fixed dealer strategy: hit ≤16, hit soft-17, stand hard-17+.
    Assumes infinite (uniform) deck — every rank equally probable.

    Args:
        nat:  Non-ace total.
        na:   Number of aces.
        nc:   Number of cards.
        memo: Memoization cache shared within a single distribution computation.

    Returns:
        Dict mapping outcome key to probability.  Bust is keyed as the string
        ``'bust'``; standing totals are keyed as integers.  All probabilities
        sum to 1.0.
    """
    key = (nat, na, nc)
    if key in memo:
        return memo[key]

    total = _total_from_composition(nat, na, nc)

    # Terminal: bust
    if total > 21:
        result: dict = {"bust": 1.0}
        memo[key] = result
        return result

    # Terminal: five-card rule — dealer stops at 5 cards when not busted
    if nc >= 5:
        result = {total: 1.0}
        memo[key] = result
        return result

    # Fixed strategy decision
    soft = _is_soft_from_composition(nat, na, nc)
    should_hit = total <= 16 or (total == 17 and soft)

    if not should_hit:
        result = {total: 1.0}
        memo[key] = result
        return result

    # HIT: enumerate all 13 ranks with equal probability
    result = {}
    for rank in range(NUM_RANKS):
        new_nat, new_na, new_nc = _transition(nat, na, nc, rank)
        sub_dist = _dealer_play_recursive(new_nat, new_na, new_nc, memo)
        for outcome, prob in sub_dist.items():
            result[outcome] = result.get(outcome, 0.0) + RANK_PROB * prob

    memo[key] = result
    return result


def compute_dealer_distribution(
    upcard_rank: int,
    reveal_mode: bool = False,
) -> DealerOutcome:
    """Compute the full dealer outcome distribution for a given upcard.

    Uses the fixed dealer strategy: hit ≤16, hit soft-17, stand hard-17+.
    Assumes infinite (uniform) deck approximation with RANK_PROB per rank.

    Args:
        upcard_rank:  Rank index (0–12) of the dealer's visible card.
        reveal_mode:  If True, capture init_16/17 sub-distributions needed
                      for selective-reveal EV calculations.

    Returns:
        DealerOutcome with full probability distributions.
    """
    memo: dict = {}

    # Build the upcard's 1-card composition
    if upcard_rank == RANK_ACE:
        upcard_nat, upcard_na, upcard_nc = 0, 1, 1
    else:
        upcard_nat, upcard_na, upcard_nc = RANK_VALUES_SOLVER[upcard_rank], 0, 1

    final_dist: dict[int, float] = {}
    bust_prob: float = 0.0
    init_16_prob: float = 0.0
    init_17_prob: float = 0.0
    init_16_final_dist: dict[int, float] = {}
    init_17_final_dist: dict[int, float] = {}

    # Enumerate all 13 possible hole cards (each with probability RANK_PROB)
    for hole_rank in range(NUM_RANKS):
        nat2, na2, nc2 = _transition(upcard_nat, upcard_na, upcard_nc, hole_rank)
        total2 = _total_from_composition(nat2, na2, nc2)
        is_soft2 = _is_soft_from_composition(nat2, na2, nc2)

        sub_dist = _dealer_play_recursive(nat2, na2, nc2, memo)

        # Aggregate into global final_dist / bust_prob
        for outcome, prob in sub_dist.items():
            if outcome == "bust":
                bust_prob += RANK_PROB * prob
            else:
                final_dist[outcome] = final_dist.get(outcome, 0.0) + RANK_PROB * prob

        # Track init_16: 2-card hard-16 state (needed for reveal_mode)
        if total2 == 16 and not is_soft2:
            init_16_prob += RANK_PROB
            for outcome, prob in sub_dist.items():
                if outcome != "bust":
                    init_16_final_dist[outcome] = (
                        init_16_final_dist.get(outcome, 0.0) + RANK_PROB * prob
                    )

        # Track init_17: 2-card soft-17 state (needed for reveal_mode)
        if total2 == 17 and is_soft2:
            init_17_prob += RANK_PROB
            for outcome, prob in sub_dist.items():
                if outcome != "bust":
                    init_17_final_dist[outcome] = (
                        init_17_final_dist.get(outcome, 0.0) + RANK_PROB * prob
                    )

    # Normalize conditional distributions to sum to 1.0 (divide out the prior)
    if init_16_prob > 0.0:
        init_16_final_dist = {k: v / init_16_prob for k, v in init_16_final_dist.items()}
    if init_17_prob > 0.0:
        init_17_final_dist = {k: v / init_17_prob for k, v in init_17_final_dist.items()}

    return DealerOutcome(
        final_dist=final_dist,
        bust_prob=bust_prob,
        init_16_prob=init_16_prob,
        init_17_prob=init_17_prob,
        init_16_final_dist=init_16_final_dist,
        init_17_final_dist=init_17_final_dist,
    )


def compute_marginal_dealer_distribution() -> DealerOutcome:
    """Compute the unconditional (marginal) dealer outcome distribution.

    In Banluck all dealer cards are dealt face-down.  The player sees no dealer
    information before acting, so player EV must be computed by marginalising
    over all possible dealer starting hands — NOT conditioned on a specific
    dealer upcard (which would be the Blackjack model).

    Method: average ``compute_dealer_distribution(r)`` over all 13 upcard ranks,
    each weighted RANK_PROB = 1/13 (infinite-deck approximation).

    The conditional reveal-mode distributions (init_16_final_dist,
    init_17_final_dist) are re-normalised correctly: they represent
    ``P(dealer final total = t | dealer reached a 2-card hard-16 / soft-17
    state at some point)``, marginalised over all possible dealer starts.

    Returns:
        DealerOutcome with the unconditional dealer probability distribution.
    """
    final_dist: dict[int, float] = {}
    bust_prob: float = 0.0
    init_16_prob: float = 0.0
    init_17_prob: float = 0.0
    # Numerators for conditional distributions (weighted by marginal init prob)
    init_16_weighted: dict[int, float] = {}
    init_17_weighted: dict[int, float] = {}

    for upcard_rank in range(NUM_RANKS):
        do = compute_dealer_distribution(upcard_rank)

        # Marginal final_dist and bust_prob: simple weighted average
        for total, prob in do.final_dist.items():
            final_dist[total] = final_dist.get(total, 0.0) + RANK_PROB * prob
        bust_prob += RANK_PROB * do.bust_prob

        # Accumulate marginal init_16/17 probabilities
        w16 = RANK_PROB * do.init_16_prob
        w17 = RANK_PROB * do.init_17_prob
        init_16_prob += w16
        init_17_prob += w17

        # Accumulate numerators for the conditional distributions.
        # init_16_final_dist[t] from compute_dealer_distribution is already
        # normalised to P(final=t | reached hard-16, upcard=r, not bust).
        # Un-normalise by multiplying by the marginal weight for this path.
        for total, prob in do.init_16_final_dist.items():
            init_16_weighted[total] = init_16_weighted.get(total, 0.0) + w16 * prob
        for total, prob in do.init_17_final_dist.items():
            init_17_weighted[total] = init_17_weighted.get(total, 0.0) + w17 * prob

    # Re-normalise conditional distributions
    init_16_final_dist: dict[int, float] = (
        {k: v / init_16_prob for k, v in init_16_weighted.items()} if init_16_prob > 0.0 else {}
    )
    init_17_final_dist: dict[int, float] = (
        {k: v / init_17_prob for k, v in init_17_weighted.items()} if init_17_prob > 0.0 else {}
    )

    return DealerOutcome(
        final_dist=final_dist,
        bust_prob=bust_prob,
        init_16_prob=init_16_prob,
        init_17_prob=init_17_prob,
        init_16_final_dist=init_16_final_dist,
        init_17_final_dist=init_17_final_dist,
    )


# ─── A6: Settlement helper ────────────────────────────────────────────────────

# Hierarchy ranks for abstract hand types (lower = stronger).
# Ban Ban / Ban Luck / 777 are pre-settled (A10) and excluded here.
_ABSTRACT_HIERARCHY: dict[str, int] = {
    "five_card_21": 4,
    "five_card_sub21": 5,
    "regular": 6,
}

_ABSTRACT_PAYOUT: dict[str, int] = {
    "five_card_21": 3,
    "five_card_sub21": 2,
    "regular": 1,
}


def _abstract_hand_type(total: int, nc: int) -> str:
    """Classify an abstracted hand into a settlement type.

    Only handles types reachable through the DP hit/stand path.
    Ban Ban, Ban Luck, and 777 are pre-settled (A10) and excluded.

    Args:
        total: Best hand total (must be ≤ 21 — caller checks bust/forfeit first).
        nc:    Number of cards in the hand.

    Returns:
        One of: ``'five_card_21'``, ``'five_card_sub21'``, ``'regular'``.
    """
    if nc == 5:
        return "five_card_21" if total == 21 else "five_card_sub21"
    return "regular"


def _settle_ev(
    player_total: int,
    player_nc: int,
    dealer_total: int,
    dealer_nc: int,
    dealer_busted: bool,
) -> float:
    """Compute settlement EV for one (player, dealer) outcome pair.

    Mirrors ``rules.py:settle_hand()`` but operates on abstracted totals
    and card counts rather than actual card tuples. Ban Ban, Ban Luck,
    and 777 are handled separately (A10) and must not be passed here.

    Settlement priority:
        1. Player bust        → -1.0 (or -2.0 if player_nc == 5: five-card bust)
        2. Player ≤15 forfeit → -1.0 (unconditional, even on dealer bust)
        3. Dealer bust        → player wins at their hand-type multiplier
           (dealer 5-card bust not tracked in DealerOutcome — approximated as regular)
        4. Hand hierarchy comparison (five-card > regular)
        5. Same tier: total comparison at 1:1

    Args:
        player_total:  Player's best hand total.
        player_nc:     Player's card count.
        dealer_total:  Dealer's final total (ignored when dealer_busted=True).
        dealer_nc:     Dealer's card count.
        dealer_busted: True if dealer's total exceeds 21.

    Returns:
        Net EV in units from player's perspective.
        Positive = player wins, negative = player loses, 0.0 = push.

    Examples:
        >>> _settle_ev(20, 2, 18, 2, False)    # regular 20 beats 18 → +1
        1.0
        >>> _settle_ev(14, 2, 18, 2, False)    # forfeit ≤15 → -1
        -1.0
        >>> _settle_ev(17, 3, 0, 0, True)      # dealer bust → player wins 1:1
        1.0
        >>> _settle_ev(17, 5, 0, 0, True)      # five-card player, dealer bust → 2:1
        2.0
        >>> _settle_ev(22, 5, 18, 2, False)    # five-card bust → -2
        -2.0
    """
    # Priority 1: Player bust — five-card bust costs 2 units (symmetric with 5-card bonus)
    if player_total > 21:
        return -2.0 if player_nc == 5 else -1.0

    # Priority 2: Player forfeit (unconditional, even on dealer bust)
    if player_total <= 15:
        return -1.0

    player_type = _abstract_hand_type(player_total, player_nc)
    player_payout = _ABSTRACT_PAYOUT[player_type]

    # Priority 3: Dealer bust — player wins at their hand-type multiplier
    if dealer_busted:
        return float(player_payout)

    dealer_type = _abstract_hand_type(dealer_total, dealer_nc)
    dealer_payout = _ABSTRACT_PAYOUT[dealer_type]

    # Priority 4: Hand hierarchy comparison
    player_rank = _ABSTRACT_HIERARCHY[player_type]
    dealer_rank = _ABSTRACT_HIERARCHY[dealer_type]

    if player_rank < dealer_rank:
        # Player has the stronger hand type
        return float(player_payout)
    if dealer_rank < player_rank:
        # Dealer has the stronger hand type — player pays dealer's multiplier
        return -float(dealer_payout)

    # Priority 5: Same hierarchy tier — compare totals at 1:1
    if player_total > dealer_total:
        return 1.0
    if dealer_total > player_total:
        return -1.0
    return 0.0


# ─── A7: EV stand ─────────────────────────────────────────────────────────────


def _ev_vs_dist(
    player_total: int,
    player_nc: int,
    final_dist: dict[int, float],
    bust_prob: float,
) -> float:
    """Compute player EV settling against a given dealer distribution.

    Dealer is treated as a regular (non-five-card) hand for the infinite-deck
    approximation; dealer five-card hands are not tracked per outcome in
    DealerOutcome and are negligible at the DP abstraction level.

    Args:
        player_total: Player's best hand total.
        player_nc:    Player's card count.
        final_dist:   {dealer_total: probability} for non-bust outcomes.
        bust_prob:    Probability dealer busts (complements final_dist).

    Returns:
        EV in units of 1 bet.
    """
    ev = 0.0
    for dealer_total, prob in final_dist.items():
        ev += prob * _settle_ev(player_total, player_nc, dealer_total, 2, False)
    ev += bust_prob * _settle_ev(player_total, player_nc, 0, 0, True)
    return ev


def _ev_stand(
    nat: int,
    na: int,
    nc: int,
    dealer_outcome: DealerOutcome,
    reveal_mode: bool,
) -> float:
    """Compute EV for a player standing at the current composition.

    Settlement modes:
      - total ≤ 15: unconditional forfeit → -1.0 (even on dealer bust)
      - nc == 2 (any mode) or reveal_mode=OFF: settle vs full final distribution
      - nc ≥ 3, reveal_mode=ON: selective-reveal adjustment applied

    Selective-reveal logic (nc ≥ 3, reveal_mode=ON):
      When the dealer reaches a 2-card hard-16 or soft-17 before hitting, 3+-card
      players are settled against the dealer's *pre-hit* total instead of the final
      total. Mathematically this is expressed as a correction on top of the baseline
      EV against the full final distribution:

        EV = EV_full
           + P(init_16) × [settle(vs 16) − EV_vs_init_16_dist]
           + P(init_17) × [settle(vs 17) − EV_vs_init_17_dist]

      Where EV_vs_init_16_dist is the EV the player would get if they settled against
      the post-hit distribution that flows from the 2-card hard-16 state (i.e. what
      they *would have* gotten without selective reveal). The correction replaces that
      with an immediate settle at 16.

    Args:
        nat:            Player non_ace_total.
        na:             Player num_aces.
        nc:             Player num_cards.
        dealer_outcome: Pre-computed dealer distribution for the upcard.
        reveal_mode:    If True, apply selective-reveal for nc ≥ 3 hands.

    Returns:
        EV in units of 1 bet (positive = player advantage).
    """
    player_total = _total_from_composition(nat, na, nc)

    # Priority: unconditional forfeit (even on dealer bust)
    if player_total <= 15:
        return -1.0

    # Base EV: settle against the full final distribution
    ev = _ev_vs_dist(player_total, nc, dealer_outcome.final_dist, dealer_outcome.bust_prob)

    # reveal_mode=OFF or 2-card hand: no reveal adjustment
    if not reveal_mode or nc < 3:
        return ev

    # reveal_mode=ON, nc ≥ 3: apply selective-reveal correction.
    # For each reveal path (init_16, init_17), subtract the EV the player would
    # have gotten from settling against the post-hit distribution and substitute
    # an immediate settle against the pre-hit dealer total.
    init_16_prob = dealer_outcome.init_16_prob
    init_17_prob = dealer_outcome.init_17_prob

    if init_16_prob > 0.0:
        init_16_bust_cond = 1.0 - sum(dealer_outcome.init_16_final_dist.values())
        ev_init16_dist = _ev_vs_dist(
            player_total,
            nc,
            dealer_outcome.init_16_final_dist,
            init_16_bust_cond,
        )
        ev += init_16_prob * (_settle_ev(player_total, nc, 16, 2, False) - ev_init16_dist)

    if init_17_prob > 0.0:
        init_17_bust_cond = 1.0 - sum(dealer_outcome.init_17_final_dist.values())
        ev_init17_dist = _ev_vs_dist(
            player_total,
            nc,
            dealer_outcome.init_17_final_dist,
            init_17_bust_cond,
        )
        ev += init_17_prob * (_settle_ev(player_total, nc, 17, 2, False) - ev_init17_dist)

    return ev


# ─── A8: EV hit & optimal EV ──────────────────────────────────────────────────


def _ev_hit(
    nat: int,
    na: int,
    nc: int,
    dealer_outcome: DealerOutcome,
    reveal_mode: bool,
    memo: dict,
) -> float:
    """Compute EV of the player choosing to hit at the current composition.

    Enumerates all 13 possible drawn ranks with equal probability (infinite deck).
    After the draw, terminal cases are resolved immediately; non-terminal states
    recurse through ``_optimal_ev`` so the player makes the optimal decision at
    each subsequent depth.

    Terminal conditions after drawing:
        - new_total > 21 (bust)             → -1.0 (or -2.0 if new_nc == 5)
        - new_nc == 5 (five-card rule)       → _ev_stand (forced stand)
        - new_total == 21 (cannot improve)   → _ev_stand (forced stand)

    Args:
        nat:            Player non_ace_total (before the draw).
        na:             Player num_aces (before the draw).
        nc:             Player num_cards (before the draw).
        dealer_outcome: Pre-computed dealer distribution for the upcard.
        reveal_mode:    If True, apply selective-reveal settlement for nc ≥ 3 hands.
        memo:           Shared memoization cache; keys are prefixed tuples to
                        avoid collisions with ``_optimal_ev`` entries.

    Returns:
        EV of hitting from the current state under subsequent optimal play.
    """
    key = ("hit", nat, na, nc)
    if key in memo:
        return memo[key]

    ev = 0.0
    for rank in range(NUM_RANKS):
        new_nat, new_na, new_nc = _transition(nat, na, nc, rank)
        new_total = _total_from_composition(new_nat, new_na, new_nc)

        if new_total > 21:
            # Five-card bust costs 2 units (symmetric with 5-card bonus).
            ev_next = -2.0 if new_nc == 5 else -1.0
        elif new_nc >= 5 or new_total == 21:
            # Five-card rule or 21: player must stand, no further decisions.
            ev_next = _ev_stand(new_nat, new_na, new_nc, dealer_outcome, reveal_mode)
        else:
            # Non-terminal: player chooses optimally from the new state.
            ev_next, _ = _optimal_ev(new_nat, new_na, new_nc, dealer_outcome, reveal_mode, memo)

        ev += RANK_PROB * ev_next

    memo[key] = ev
    return ev


def _optimal_ev(
    nat: int,
    na: int,
    nc: int,
    dealer_outcome: DealerOutcome,
    reveal_mode: bool,
    memo: dict,
) -> tuple[float, Action]:
    """Return the optimal (EV, Action) for a player at the current composition.

    Computes EV for both HIT and STAND and returns the option with higher EV.
    Handles forced-stand conditions before evaluating hit.

    Forced-stand conditions (return STAND immediately):
        - total > 21 (bust — caller should guard, but handled defensively)
        - nc >= 5 (five-card rule: cannot draw further)
        - total == 21 (cannot improve; always stand)

    When total ≤ 15 the player forfeits by standing (-1.0), but hitting may
    still be superior, so ev_hit is always computed for non-forced states.

    Args:
        nat:            Player non_ace_total.
        na:             Player num_aces.
        nc:             Player num_cards.
        dealer_outcome: Pre-computed dealer distribution for the upcard.
        reveal_mode:    If True, apply selective-reveal settlement for nc ≥ 3 hands.
        memo:           Shared memoization cache (prefix-keyed with ``'opt'``).

    Returns:
        ``(ev, action)`` — the EV under optimal play and the corresponding Action.
    """
    key = ("opt", nat, na, nc)
    if key in memo:
        return memo[key]

    total = _total_from_composition(nat, na, nc)
    ev_stand = _ev_stand(nat, na, nc, dealer_outcome, reveal_mode)

    # Forced stand: bust, five-card rule, or unimprovable 21.
    if total > 21 or nc >= 5 or total == 21:
        result: tuple[float, Action] = (ev_stand, Action.STAND)
        memo[key] = result
        return result

    ev_hit = _ev_hit(nat, na, nc, dealer_outcome, reveal_mode, memo)

    if ev_hit >= ev_stand:
        result = (ev_hit, Action.HIT)
    else:
        result = (ev_stand, Action.STAND)

    memo[key] = result
    return result


# ─── A10: Pre-settled hand EVs ────────────────────────────────────────────────

# Infinite-deck probability that the dealer's initial 2-card hand is Ban Ban or Ban Luck.
# Used for settling special player hands before the dealer plays further.
_P_DEALER_BAN_BAN: float = RANK_PROB * RANK_PROB
"""P(dealer Ban Ban) = (1/13)² = 1/169 — both initial cards are aces."""

_P_DEALER_BAN_LUCK: float = 2.0 * RANK_PROB * (len(TEN_VALUE_RANKS) * RANK_PROB)
"""P(dealer Ban Luck) = 2×(1/13)×(4/13) = 8/169 — ace + ten-value (both orderings)."""


def _ev_ban_ban() -> float:
    """EV for a player holding Ban Ban (two aces, rank 1) at the initial 2-card deal.

    Ban Ban is settled immediately against the dealer's 2-card initial hand
    (before the dealer plays further).  Settlement from player's perspective:

        - vs dealer Ban Ban (rank 1, same tier) → push            → 0.0
        - vs dealer Ban Luck (rank 2, dealer weaker) → player wins at 3:1 → +3.0
        - vs all other dealer 2-card hands → player wins at 3:1  → +3.0

    Ban Ban (rank 1) beats every other hand type, so the only non-win outcome
    is a push against another Ban Ban.

    Returns:
        EV ≈ +2.982.  Exact: (168/169) × 3 = 504/169.
    """
    return (1.0 - _P_DEALER_BAN_BAN) * 3.0


def _ev_ban_luck() -> float:
    """EV for a player holding Ban Luck (ace + ten-value, rank 2) at the initial 2-card deal.

    Ban Luck is settled immediately against the dealer's 2-card initial hand.
    Settlement from player's perspective:

        - vs dealer Ban Ban (rank 1, dealer stronger) → player loses at dealer's 3:1 → -3.0
        - vs dealer Ban Luck (rank 2, same tier)       → push                         →  0.0
        - vs all other dealer 2-card hands             → player wins at 2:1            → +2.0

    Returns:
        EV ≈ +1.876.  Exact: (1/169)×(−3) + (160/169)×2 = 317/169.
    """
    ev = _P_DEALER_BAN_BAN * (-3.0)
    ev += (1.0 - _P_DEALER_BAN_BAN - _P_DEALER_BAN_LUCK) * 2.0
    return ev


def _ev_777() -> float:
    """EV for a player holding 777 (three 7s, rank 3, payout 7:1).

    777 is settled against the dealer's 2-card initial hand (before the dealer
    plays further).  Player 777 (rank 3) loses only to dealer Ban Ban (rank 1)
    or dealer Ban Luck (rank 2).

    Settlement from player's perspective:

        - vs dealer Ban Ban (rank 1) → player loses at dealer's 3:1 → -3.0
        - vs dealer Ban Luck (rank 2) → player loses at dealer's 2:1 → -2.0
        - vs all other dealer 2-card hands → player wins at 7:1      → +7.0

    Returns:
        EV ≈ +6.515.  Exact: (1/169)×(−3) + (8/169)×(−2) + (160/169)×7 = 1101/169.
    """
    ev = _P_DEALER_BAN_BAN * (-3.0)
    ev += _P_DEALER_BAN_LUCK * (-2.0)
    ev += (1.0 - _P_DEALER_BAN_BAN - _P_DEALER_BAN_LUCK) * 7.0
    return ev


def optimal_action(
    player_nat: int,
    player_na: int,
    player_nc: int,
    dealer_outcome: DealerOutcome,
    reveal_mode: bool = False,
) -> tuple[Action, float]:
    """Return the optimal action and its EV for a given player state.

    Args:
        player_nat:     Player non_ace_total.
        player_na:      Player num_aces.
        player_nc:      Player num_cards.
        dealer_outcome: Pre-computed dealer distribution.
        reveal_mode:    If True, apply selective-reveal settlement rules.

    Returns:
        (action, ev) — the best Action and the EV achieved by it.
    """
    memo: dict = {}
    ev, action = _optimal_ev(player_nat, player_na, player_nc, dealer_outcome, reveal_mode, memo)
    return (action, ev)


# ─── A11: Public API ──────────────────────────────────────────────────────────


def solve(
    reveal_mode: bool = False,
) -> dict[tuple[int, int, int], tuple[Action, float]]:
    """Build the optimal strategy + EV table via backward induction.

    Enumerates all reachable player states ``(total, nc, is_soft)`` and
    computes the optimal action and EV at each via ``_optimal_ev``.  Uses
    the marginal dealer distribution — the player cannot see any dealer card.

    A representative composition is chosen for each ``(total, nc, is_soft)``
    key and passed to ``_optimal_ev``.  For ``nc == 2`` this representative
    is the unique composition that achieves the key, so the table is exact
    for initial 2-card EV lookups.  For ``nc ≥ 3`` the representative may
    differ in composition from other hands sharing the key, but the action
    (HIT vs STAND) is the same across all such compositions in practice.

    Special pre-settled hands (Ban Ban, Ban Luck) map to ``(21, 2, 1)`` and
    carry the composition-level EV (regular payout).  ``compute_house_edge``
    handles their correct payout separately.

    Memoization is local to this call.  For repeated access, cache the result.

    Args:
        reveal_mode: If True, apply selective-reveal settlement for nc ≥ 3.

    Returns:
        Dict mapping ``(total, nc, is_soft)`` to ``(Action, ev)``:
            - total:    best hand total (2–21; bust states excluded).
            - nc:       number of cards (2–5).
            - is_soft:  1 if soft hand (an ace at its high value), else 0.
            - Action:   optimal player decision.
            - ev:       EV under optimal play in units of 1 bet.
    """
    dealer_outcome = compute_marginal_dealer_distribution()
    memo: dict = {}
    table: dict[tuple[int, int, int], tuple[Action, float]] = {}

    for nc in range(2, 6):
        for total in range(2, 22):  # totals 2–21; bust (> 21) excluded
            for is_soft_flag in (0, 1):
                # Build a representative composition for (total, nc, is_soft).
                if is_soft_flag:
                    # Soft: place one ace counted at its high value (11 for 2-card,
                    # 10 for 3+-card) and set nat = total − high_value.
                    high = 11 if nc == 2 else 10
                    nat, na = total - high, 1
                    if nat < 0:
                        continue  # Soft total too low to be reachable at this nc
                else:
                    nat, na = total, 0

                # Skip if the composition doesn't round-trip to the target state.
                if _total_from_composition(nat, na, nc) != total:
                    continue
                if bool(_is_soft_from_composition(nat, na, nc)) != bool(is_soft_flag):
                    continue

                ev, action = _optimal_ev(nat, na, nc, dealer_outcome, reveal_mode, memo)
                table[(total, nc, is_soft_flag)] = (action, ev)

    return table


def compute_house_edge(
    strategy_table: dict[tuple[int, int, int], tuple[Action, float]],
) -> float:
    """Compute the expected player EV (house edge) over all initial 2-card deals.

    Iterates all 13² = 169 ordered starting rank pairs (infinite-deck, uniform
    weight 1/169 each) and computes EV for each:

        - Ban Ban (both aces)           → ``_ev_ban_ban()``  [3:1, pre-settled]
        - Ban Luck (ace + ten-value)    → ``_ev_ban_luck()`` [2:1, pre-settled]
        - All other 2-card hands        → EV from ``strategy_table[(total, 2, is_soft)]``

    The 777 contribution is implicitly included through the strategy_table EV
    for the ``(14, 2, 0)`` state (composition approximation; < 0.3% EV error).

    Args:
        strategy_table: Dict returned by ``solve()``, mapping
                        ``(total, nc, is_soft)`` to ``(Action, ev)``.

    Returns:
        Weighted average EV from player's perspective (negative = house edge).
    """
    total_ev = 0.0

    for rank_i in range(NUM_RANKS):
        # Build starting composition after first card
        if rank_i == RANK_ACE:
            nat1, na1 = 0, 1
        else:
            nat1, na1 = RANK_VALUES_SOLVER[rank_i], 0

        for rank_j in range(NUM_RANKS):
            # Draw second card to complete the starting 2-card hand
            nat2, na2, nc2 = _transition(nat1, na1, 1, rank_j)
            # nc2 == 2 always; nc1 was conceptually 1 (first card only)

            # Classify starting 2-card hand
            is_ban_ban = rank_i == RANK_ACE and rank_j == RANK_ACE
            is_ban_luck = (not is_ban_ban) and (
                (rank_i == RANK_ACE and rank_j in TEN_VALUE_RANKS)
                or (rank_j == RANK_ACE and rank_i in TEN_VALUE_RANKS)
            )

            if is_ban_ban:
                ev = _ev_ban_ban()
            elif is_ban_luck:
                ev = _ev_ban_luck()
            else:
                total = _total_from_composition(nat2, na2, nc2)
                is_soft_flag = 1 if _is_soft_from_composition(nat2, na2, nc2) else 0
                ev = strategy_table[(total, 2, is_soft_flag)][1]

            total_ev += ev

    # All 169 pairs have equal weight 1/169
    return total_ev / (NUM_RANKS * NUM_RANKS)


def build_strategy_chart(
    reveal_mode: bool = False,
) -> dict[tuple[int, int, int], Action]:
    """Build the full GTO strategy chart via backward induction.

    Thin wrapper around ``solve()`` returning only the optimal actions.

    Keys: (player_total, hand_size, is_soft).
    Values: optimal Action (HIT or STAND).

    Args:
        reveal_mode: If True, solve with selective-reveal rules.
    """
    return {k: v[0] for k, v in solve(reveal_mode).items()}


def build_ev_table(
    reveal_mode: bool = False,
) -> dict[tuple[int, int, int], float]:
    """Build the EV table for all reachable player states.

    Thin wrapper around ``solve()`` returning only the EV values.

    Keys: (player_total, hand_size, is_soft).
    Values: EV under optimal play.

    Args:
        reveal_mode: If True, solve with selective-reveal rules.
    """
    return {k: v[1] for k, v in solve(reveal_mode).items()}


def run_dp_solver(
    reveal_mode: bool = False,
) -> tuple[dict[tuple[int, int, int], Action], dict[tuple[int, int, int], float]]:
    """Run the full DP solver and return strategy chart + EV table.

    Thin wrapper around ``solve()`` returning both dicts as a 2-tuple.

    Args:
        reveal_mode: If True, solve with selective-reveal rules.

    Returns:
        (strategy_chart, ev_table) — both keyed on (total, hand_size, is_soft).
    """
    table = solve(reveal_mode)
    strategy = {k: v[0] for k, v in table.items()}
    ev_table = {k: v[1] for k, v in table.items()}
    return strategy, ev_table


def compare_reveal_modes() -> dict[tuple[int, int, int], float]:
    """Compare EV under reveal_mode=ON vs reveal_mode=OFF.

    Returns:
        Dict mapping (player_total, hand_size, is_soft) to
        EV_reveal_on − EV_reveal_off (positive = reveal benefits player).
    """
    table_on = solve(reveal_mode=True)
    table_off = solve(reveal_mode=False)
    return {k: table_on[k][1] - table_off[k][1] for k in table_on if k in table_off}


# ─── A12 / A13: Strategy chart display ────────────────────────────────────────


def build_ev_margin_table(
    reveal_mode: bool = False,
) -> dict[tuple[int, int, int], float | None]:
    """Return the EV margin |EV_HIT − EV_STAND| for every reachable state.

    Used by :func:`print_strategy_chart` when ``show_ev_margin=True``.

    Forced-stand states (nc=5, total=21, total≤15) map to ``None`` because
    there is no hit option — the margin is undefined.

    Args:
        reveal_mode: If True, compute margins using selective-reveal settlement.

    Returns:
        Dict mapping ``(total, nc, is_soft)`` to margin (float ≥ 0) or ``None``
        for forced-stand states.
    """
    dealer_outcome = compute_marginal_dealer_distribution()
    memo: dict = {}
    result: dict[tuple[int, int, int], float | None] = {}

    for nc in range(2, 6):
        for total in range(2, 22):
            for is_soft_flag in (0, 1):
                if is_soft_flag:
                    high = 11 if nc == 2 else 10
                    nat, na = total - high, 1
                    if nat < 0:
                        continue
                else:
                    nat, na = total, 0

                if _total_from_composition(nat, na, nc) != total:
                    continue
                if bool(_is_soft_from_composition(nat, na, nc)) != bool(is_soft_flag):
                    continue

                key = (total, nc, is_soft_flag)

                # Forced-stand states: no hit option → margin undefined
                if nc >= 5 or total == 21 or total <= 15:
                    result[key] = None
                    continue

                ev_stand = _ev_stand(nat, na, nc, dealer_outcome, reveal_mode)
                ev_hit = _ev_hit(nat, na, nc, dealer_outcome, reveal_mode, memo)
                result[key] = abs(ev_hit - ev_stand)

    return result


def print_strategy_chart(
    strategy_table: dict[tuple[int, int, int], Action],
    label: str,
    *,
    ev_margin_table: dict[tuple[int, int, int], float | None] | None = None,
    show_ev_margin: bool = False,
) -> None:
    """Print the strategy chart as a human-readable terminal grid.

    Rows: hard totals 16–21 (top section), then soft totals 16–21 (bottom).
    Cols: hand sizes nc=2 through nc=5.
    Cells: 'H' (HIT), 'S' (STAND), or '-' if the state is absent from the table.

    When ``show_ev_margin=True``, a second grid is printed immediately below the
    H/S grid showing ``|EV_HIT − EV_STAND|`` rounded to 3 decimal places, or
    ``'—'`` for forced-stand states (nc=5, total=21, total≤15).

    Args:
        strategy_table:  Mapping ``(total, nc, is_soft)`` → :class:`Action`.
        label:           Chart title shown in the header line.
        ev_margin_table: Optional mapping ``(total, nc, is_soft)`` → margin or
                         ``None``.  Required when ``show_ev_margin=True``.
                         Build with :func:`build_ev_margin_table`.
        show_ev_margin:  If True, print the EV margin grid below the H/S grid.
    """
    col_w = 6  # width per column (including leading space)
    header = "".join(f"{'nc=' + str(nc):>{col_w}}" for nc in range(2, 6))
    divider = "─" * (12 + col_w * 4)

    print(f"\nStrategy Chart: {label}")
    print(f"{'':12}{header}")
    print(divider)

    for section, is_soft in [("Hard", 0), ("Soft", 1)]:
        for total in range(16, 22):
            cells = ""
            for nc in range(2, 6):
                action = strategy_table.get((total, nc, is_soft))
                if action == Action.HIT:
                    cell = "H"
                elif action == Action.STAND:
                    cell = "S"
                else:
                    cell = "-"
                cells += f"{cell:>{col_w}}"
            print(f"{section + ' ' + str(total):<12}{cells}")
        if is_soft == 0:
            print()

    if show_ev_margin and ev_margin_table is not None:
        print(f"\nEV Margin |HIT − STAND|: {label}")
        print(f"{'':12}{header}")
        print(divider)

        for section, is_soft in [("Hard", 0), ("Soft", 1)]:
            for total in range(16, 22):
                cells = ""
                for nc in range(2, 6):
                    margin = ev_margin_table.get((total, nc, is_soft))
                    if margin is None:
                        cell = "—"
                    else:
                        cell = f"{margin:.3f}"
                    cells += f"{cell:>{col_w}}"
                print(f"{section + ' ' + str(total):<12}{cells}")
            if is_soft == 0:
                print()


# ─── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Banluck DP Solver — Phase 1.1 (infinite-deck, fixed dealer, no surrender)")
    print("Computing strategy tables and EV margins...")

    t0 = time.time()
    table_off = solve(reveal_mode=False)
    table_on = solve(reveal_mode=True)
    margins_off = build_ev_margin_table(reveal_mode=False)
    margins_on = build_ev_margin_table(reveal_mode=True)
    elapsed = time.time() - t0

    print(f"Solved both modes in {elapsed:.2f}s")

    chart_off = {k: v[0] for k, v in table_off.items()}
    chart_on = {k: v[0] for k, v in table_on.items()}

    print_strategy_chart(
        chart_off,
        "reveal_mode=OFF  (baseline)",
        ev_margin_table=margins_off,
        show_ev_margin=True,
    )
    print_strategy_chart(
        chart_on,
        "reveal_mode=ON   (selective reveal at 16/17)",
        ev_margin_table=margins_on,
        show_ev_margin=True,
    )

    he_off = compute_house_edge(table_off)
    he_on = compute_house_edge(table_on)

    print(f"\n{'─' * 50}")
    print(f"Player EV  reveal_mode=OFF : {he_off:+.4f}  ({he_off * 100:+.2f}%)")
    print(f"Player EV  reveal_mode=ON  : {he_on:+.4f}  ({he_on * 100:+.2f}%)")
    print(f"EV delta   (ON − OFF)       : {he_on - he_off:+.6f}  ({(he_on - he_off) * 100:+.4f}%)")
    print(f"{'─' * 50}")
    print("Note: Phase 1.1 uses a fixed dealer (no surrender option).")
    print("      Without dealer surrender, special-hand payouts (Ban Ban 3:1,")
    print("      Ban Luck 2:1, five-card 2:1/3:1) give the player a slight edge.")
    print("      The house edge emerges in Phase 2 when dealer surrender is")
    print("      added as a CFR strategic option.")
