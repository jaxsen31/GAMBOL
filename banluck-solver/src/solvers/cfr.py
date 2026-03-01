"""CFR+ solver for Banluck Nash equilibrium (Phase 2.2).

Finds Nash equilibrium strategies for both player and dealer using
Counterfactual Regret Minimization Plus (CFR+).

Game-theory summary
-------------------
Banluck is a two-player zero-sum sequential game with imperfect information:
  - Player (maximiser) sees only their own hand; dealer cards are face-down.
  - Dealer (minimiser) sees their own cards and the player's card count.

Decision points and information sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PlayerHitStandInfoSet(total, num_cards, is_soft)
      Player's hit/stand decision. Dealer hand is completely hidden.

  DealerSurrenderInfoSet(total, is_hard_fifteen)
      Dealer's surrender check immediately after the initial deal.
      Only hard-15 hands qualify; soft-15 (A+4) does NOT.

  DealerActionInfoSet(dealer_total, dealer_nc, is_soft, player_nc)
      Dealer's strategic choice at total 16 or 17 (including soft 17).
      Legal actions: HIT, STAND, and REVEAL_PLAYER (when player_nc >= 3).

Algorithm: CFR+
~~~~~~~~~~~~~~~
  1. Regrets are floored at 0 after each update (positive-only regrets).
  2. Strategy sums are weighted by iteration number (linear averaging).
  3. Nash equilibrium ← average strategy profile at convergence.
  4. Convergence: total exploitability (eps_player + eps_dealer) < threshold.

Infinite-deck approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  P(any rank) = 1/13. Removes deck-state tracking.
  State is a composition triple (non_ace_total, num_aces, num_cards).
  Error vs single-deck < 0.5%.

Note on REVEAL_PLAYER in heads-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  In heads-up play (one player vs dealer), REVEAL_PLAYER has the same
  mechanical effect as STAND when player has 3+ cards: both settle the
  player at the current dealer total. CFR converges to indifference
  between them. The action is retained for completeness; the CFR will
  find the equilibrium reveal frequency (which equals the stand frequency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

try:
    import numba

    _NUMBA_AVAILABLE: bool = True
except ImportError:
    _NUMBA_AVAILABLE = False

from src.engine.game_state import DealerAction, PlayerAction
from src.solvers.baseline_dp import (
    NUM_RANKS,
    RANK_PROB,
    _is_soft_from_composition,
    _total_from_composition,
    _transition,
)
from src.solvers.information_sets import (
    DealerActionInfoSet,
    DealerSurrenderInfoSet,
    PlayerHitStandInfoSet,
    get_legal_dealer_actions,
    get_legal_dealer_surrender_actions,
    get_legal_player_actions,
)

# Rank index for Ace (same as baseline_dp.RANK_ACE).
_RANK_ACE: int = 12

# ─── Type aliases ──────────────────────────────────────────────────────────────

# Composition triple: (non_ace_total, num_aces, num_cards).
Comp = tuple[int, int, int]

PlayerStrategyDict = dict[PlayerHitStandInfoSet, dict[PlayerAction, float]]
DealerSurrenderStrategyDict = dict[DealerSurrenderInfoSet, dict[DealerAction, float]]
DealerActionStrategyDict = dict[DealerActionInfoSet, dict[DealerAction, float]]


# ─── Result type ───────────────────────────────────────────────────────────────


@dataclass
class CfrResult:
    """Output of the CFR+ solver.

    Attributes:
        player_strategy:           Average Nash strategy for the player.
                                   Maps PlayerHitStandInfoSet → {action: prob}.
        dealer_surrender_strategy: Average Nash strategy for dealer surrender.
                                   Maps DealerSurrenderInfoSet → {action: prob}.
        dealer_action_strategy:    Average Nash strategy for dealer at 16/17.
                                   Maps DealerActionInfoSet → {action: prob}.
        n_iterations:              Number of CFR+ iterations completed.
        exploitability:            Total exploitability in units per hand.
                                   Computed as best_player_ev - worst_player_ev.
        converged:                 True if exploitability < threshold at end.
        nash_ev:                   Expected player EV under average strategies,
                                   averaged over all initial deal compositions.
    """

    player_strategy: PlayerStrategyDict
    dealer_surrender_strategy: DealerSurrenderStrategyDict
    dealer_action_strategy: DealerActionStrategyDict
    n_iterations: int
    exploitability: float
    converged: bool
    nash_ev: float


# ─── CFR+ internal accumulators ────────────────────────────────────────────────


@dataclass
class _CfrTables:
    """Mutable CFR+ state: regret and strategy-sum accumulators.

    Regret tables:    info_set → {action: cumulative_positive_regret}
    Strategy-sum tables: info_set → {action: weighted_probability_sum}

    Both are updated in-place during each CFR+ iteration.
    """

    player_regrets: dict[PlayerHitStandInfoSet, dict[PlayerAction, float]] = field(
        default_factory=dict
    )
    player_strategy_sums: dict[PlayerHitStandInfoSet, dict[PlayerAction, float]] = field(
        default_factory=dict
    )
    dealer_surrender_regrets: dict[DealerSurrenderInfoSet, dict[DealerAction, float]] = field(
        default_factory=dict
    )
    dealer_surrender_strategy_sums: dict[DealerSurrenderInfoSet, dict[DealerAction, float]] = field(
        default_factory=dict
    )
    dealer_action_regrets: dict[DealerActionInfoSet, dict[DealerAction, float]] = field(
        default_factory=dict
    )
    dealer_action_strategy_sums: dict[DealerActionInfoSet, dict[DealerAction, float]] = field(
        default_factory=dict
    )


@dataclass
class _NumbaTables:
    """CFR+ accumulators as NumPy arrays for the Numba JIT kernel (C3c).

    Arrays are indexed by precomputed integer slot indices assigned once when
    building the flat numpy tree (see _build_numba_arrays).  Updated in-place
    during each iteration by _numba_cfr_kernel.

    Attributes:
        player_regrets:       float64[n_player_slots, 2]  — HIT / STAND regrets.
        player_strategy_sums: float64[n_player_slots, 2]  — weighted strategy sums.
        surr_regrets:         float64[n_surr_slots, 2]    — SURRENDER / CONTINUE.
        surr_strategy_sums:   float64[n_surr_slots, 2].
        act_regrets:          float64[n_act_slots, 3]     — HIT / STAND / REVEAL.
        act_strategy_sums:    float64[n_act_slots, 3].
    """

    player_regrets: np.ndarray
    player_strategy_sums: np.ndarray
    surr_regrets: np.ndarray
    surr_strategy_sums: np.ndarray
    act_regrets: np.ndarray
    act_strategy_sums: np.ndarray


# ─── Internal action constants and info-set key converters ────────────────────
# Plain int keys for _CfrTables dicts — avoids Enum.__hash__ overhead in hot path.

# Player action indices
_P_HIT: int = 0
_P_STAND: int = 1

# Dealer action indices (at 16/17 strategic nodes)
_D_HIT: int = 0
_D_STAND: int = 1
_D_REVEAL: int = 2  # REVEAL_PLAYER

# Dealer surrender action indices
_D_SURRENDER: int = 0
_D_CONTINUE: int = 1  # DealerAction.HIT used as "no-surrender" sentinel

# Reverse mappings (int → Enum) used only in _extract_*_avg_strategy at end of solve()
_INT_TO_PLAYER_ACTION: dict[int, PlayerAction] = {
    _P_HIT: PlayerAction.HIT,
    _P_STAND: PlayerAction.STAND,
}
_INT_TO_DEALER_ACTION: dict[int, DealerAction] = {
    _D_HIT: DealerAction.HIT,
    _D_STAND: DealerAction.STAND,
    _D_REVEAL: DealerAction.REVEAL_PLAYER,
}
_INT_TO_SURRENDER_ACTION: dict[int, DealerAction] = {
    _D_SURRENDER: DealerAction.SURRENDER,
    _D_CONTINUE: DealerAction.HIT,
}


def _player_key(total: int, nc: int, is_soft: bool) -> tuple[int, int, int]:
    """Convert player info-set parameters to a plain int-tuple dict key."""
    return (total, nc, int(is_soft))


def _dealer_surrender_key(total: int, is_hard_fifteen: bool) -> tuple[int, int]:
    """Convert dealer surrender info-set parameters to a plain int-tuple dict key."""
    return (total, int(is_hard_fifteen))


def _dealer_action_key(
    dealer_total: int, dealer_nc: int, is_soft: bool, player_nc: int
) -> tuple[int, int, int, int]:
    """Convert dealer action info-set parameters to a plain int-tuple dict key."""
    return (dealer_total, dealer_nc, int(is_soft), player_nc)


# ─── Regret matching + ─────────────────────────────────────────────────────────


def _get_strategy(
    info_set: NamedTuple,
    regret_table: dict,
    legal_actions: list,
) -> dict:
    """Return the current strategy for an information set (CFR+ regret matching).

    Strategy is proportional to positive regrets. Falls back to uniform if
    all regrets are ≤ 0 (including the initial state before any accumulation).

    Args:
        info_set:      Hashable information set (NamedTuple key).
        regret_table:  Maps info_set → {action: cumulative regret}.
        legal_actions: List of legal actions at this information set.

    Returns:
        Dict mapping each legal action to its strategy probability.
    """
    regrets = regret_table.get(info_set, {})
    pos = [max(0.0, regrets.get(a, 0.0)) for a in legal_actions]
    total = sum(pos)
    if total <= 0.0:
        p = 1.0 / len(legal_actions)
        return dict.fromkeys(legal_actions, p)
    return {a: r / total for a, r in zip(legal_actions, pos, strict=True)}


def _update_strategy_sum(
    info_set: NamedTuple,
    strategy: dict,
    reach_self: float,
    iteration: int,
    sums_table: dict,
) -> None:
    """Accumulate the weighted strategy for linear averaging (CFR+).

    CFR+ strategy averaging: weight = iteration × reach_self.
    This gives the time-weighted average strategy that converges to Nash.

    Args:
        info_set:    Hashable information set key.
        strategy:    Current strategy {action: probability}.
        reach_self:  Reach probability from the acting player's own choices.
        iteration:   Current iteration number (1-indexed).
        sums_table:  Dict to update in-place.
    """
    weight = float(iteration) * reach_self
    if info_set not in sums_table:
        sums_table[info_set] = {}
    d = sums_table[info_set]
    for action, prob in strategy.items():
        d[action] = d.get(action, 0.0) + weight * prob


def _regret_update_player(
    info_set: NamedTuple,
    legal_actions: list,
    action_evs: dict,
    node_ev: float,
    reach_dealer: float,
    regret_table: dict,
) -> None:
    """Apply CFR+ regret update for a PLAYER (maximiser) decision node.

    Regret = how much more player could have won by choosing action a.
    Update: R[I][a] = max(0, R[I][a] + (ev(a) - node_ev) × reach_dealer)

    Args:
        info_set:      Player information set.
        legal_actions: List of legal player actions.
        action_evs:    Dict {action: EV if this action is taken}.
        node_ev:       Expected EV under current strategy.
        reach_dealer:  Reach probability from the dealer (opponent).
        regret_table:  Dict to update in-place.
    """
    if info_set not in regret_table:
        regret_table[info_set] = {}
    r = regret_table[info_set]
    for a in legal_actions:
        instant = (action_evs[a] - node_ev) * reach_dealer
        r[a] = max(0.0, r.get(a, 0.0) + instant)


def _regret_update_dealer(
    info_set: NamedTuple,
    legal_actions: list,
    action_evs: dict,
    node_ev: float,
    reach_player: float,
    regret_table: dict,
) -> None:
    """Apply CFR+ regret update for a DEALER (minimiser) decision node.

    Dealer regrets taking an action that resulted in higher player EV.
    Update: R[I][a] = max(0, R[I][a] + (node_ev - ev(a)) × reach_player)

    Args:
        info_set:      Dealer information set.
        legal_actions: List of legal dealer actions.
        action_evs:    Dict {action: player EV if this dealer action is taken}.
        node_ev:       Player's expected EV under current dealer strategy.
        reach_player:  Reach probability from the player (opponent).
        regret_table:  Dict to update in-place.
    """
    if info_set not in regret_table:
        regret_table[info_set] = {}
    r = regret_table[info_set]
    for a in legal_actions:
        instant = (node_ev - action_evs[a]) * reach_player  # Dealer is minimiser
        r[a] = max(0.0, r.get(a, 0.0) + instant)


# ─── Composition helpers ───────────────────────────────────────────────────────


def _is_hard_fifteen_comp(nat: int, na: int, nc: int) -> bool:
    """True if composition is a qualifying hard-15 hand.

    Hard 15: exactly 2 cards, total = 15, no ace (na == 0).
    Soft 15 (A+4: nat=4, na=1, nc=2 → total=15) does NOT qualify.

    Args:
        nat: Non-ace total.
        na:  Number of aces.
        nc:  Total card count.

    Returns:
        True if the composition is a dealer hard-15.
    """
    return nc == 2 and na == 0 and nat == 15


def _is_player_ban_ban(p_nat: int, p_na: int, p_nc: int) -> bool:
    """True if the player has Ban Ban (two aces, 2-card hand)."""
    return p_na == 2 and p_nc == 2


def _is_player_ban_luck(p_nat: int, p_na: int, p_nc: int) -> bool:
    """True if the player has Ban Luck (ace + ten-value, 2-card hand)."""
    return p_na == 1 and p_nc == 2 and p_nat == 10


def _is_dealer_ban_ban(d_nat: int, d_na: int, d_nc: int) -> bool:
    """True if the dealer has Ban Ban (two aces, 2-card hand)."""
    return d_na == 2 and d_nc == 2


def _is_dealer_ban_luck(d_nat: int, d_na: int, d_nc: int) -> bool:
    """True if the dealer has Ban Luck (ace + ten-value, 2-card hand)."""
    return d_na == 1 and d_nc == 2 and d_nat == 10


# ─── Settlement ────────────────────────────────────────────────────────────────


def _cfr_settle(
    p_nat: int,
    p_na: int,
    p_nc: int,
    d_nat: int,
    d_na: int,
    d_nc: int,
    d_busted: bool,
) -> float:
    """Settlement EV from the player's perspective, using full compositions.

    Unlike baseline_dp._settle_ev, this function correctly classifies dealer
    Ban Ban and Ban Luck using the full dealer composition (d_nat, d_na, d_nc),
    rather than treating all dealer 2-card totals of 21 as 'regular'.

    Player Ban Ban and Ban Luck are never passed here — they are resolved before
    the player action phase begins. Player 777 is treated as a regular 3-card 21
    (same approximation as the DP solver; EV error < 0.3%).

    Settlement priority:
        1. Player bust (>21)    → -2.0 (5-card bust) or -1.0
        2. Player forfeit (≤15) → -1.0 (unconditional, even on dealer bust)
        3. Dealer bust          → player wins at their hand-type multiplier
        4. Hand hierarchy       → higher-ranked hand wins at its multiplier
        5. Same tier            → higher total wins at 1:1; equal totals push

    Special hand hierarchy ranks (lower = stronger):
        1=Ban Ban, 2=Ban Luck, 4=five_card_21, 5=five_card_sub21, 6=regular

    Args:
        p_nat, p_na, p_nc: Player composition.
        d_nat, d_na, d_nc: Dealer composition.
        d_busted:          True if dealer's total exceeds 21.

    Returns:
        Net EV in units (positive = player wins, negative = player loses).

    Examples:
        >>> _cfr_settle(10, 1, 5, 5, 0, 2, False)  # player five-card 21 vs regular
        3.0
        >>> _cfr_settle(14, 0, 2, 10, 0, 2, False)  # forfeit ≤15
        -1.0
        >>> _cfr_settle(18, 0, 2, 0, 2, 2, False)  # regular 18 vs dealer Ban Ban
        -3.0
    """
    p_total = _total_from_composition(p_nat, p_na, p_nc)
    d_total = _total_from_composition(d_nat, d_na, d_nc)

    # Priority 1: Player bust
    if p_total > 21:
        return -2.0 if p_nc == 5 else -1.0

    # Priority 2: Player forfeit (even on dealer bust)
    if p_total <= 15:
        return -1.0

    # Determine player hand class: (hierarchy_rank, payout_multiplier)
    if p_nc == 5:
        p_rank, p_pay = (4, 3) if p_total == 21 else (5, 2)
    else:
        p_rank, p_pay = 6, 1  # regular

    # Priority 3: Dealer bust → player wins at their multiplier
    if d_busted:
        return float(p_pay)

    # Determine dealer hand class: (hierarchy_rank, payout_multiplier)
    if _is_dealer_ban_ban(d_nat, d_na, d_nc):
        d_rank, d_pay = 1, 3
    elif _is_dealer_ban_luck(d_nat, d_na, d_nc):
        d_rank, d_pay = 2, 2
    elif d_nc == 5:
        d_rank, d_pay = (4, 3) if d_total == 21 else (5, 2)
    else:
        d_rank, d_pay = 6, 1  # regular

    # Priority 4: Hand hierarchy comparison (lower rank = stronger)
    if p_rank < d_rank:
        return float(p_pay)
    if d_rank < p_rank:
        return -float(d_pay)

    # Priority 5: Same tier → compare totals at 1:1
    if p_total > d_total:
        return 1.0
    if d_total > p_total:
        return -1.0
    return 0.0  # Push


def _settle_player_ban_ban(d_nat: int, d_na: int, d_nc: int) -> float:
    """EV for player with Ban Ban settled against dealer's 2-card initial hand.

    Ban Ban (hierarchy rank 1) beats all hands except dealer Ban Ban (push).

    Args:
        d_nat, d_na, d_nc: Dealer's initial 2-card composition.

    Returns:
        EV from player's perspective (+3.0 win, 0.0 push).
    """
    if _is_dealer_ban_ban(d_nat, d_na, d_nc):
        return 0.0  # Ban Ban vs Ban Ban → push
    return 3.0  # Ban Ban beats everything else → +3 units


def _settle_player_ban_luck(d_nat: int, d_na: int, d_nc: int) -> float:
    """EV for player with Ban Luck settled against dealer's 2-card initial hand.

    Ban Luck (hierarchy rank 2): loses to dealer Ban Ban (-3), pushes vs
    dealer Ban Luck (0), wins 2:1 vs everything else (+2).

    Args:
        d_nat, d_na, d_nc: Dealer's initial 2-card composition.

    Returns:
        EV from player's perspective.
    """
    if _is_dealer_ban_ban(d_nat, d_na, d_nc):
        return -3.0  # Ban Luck vs Ban Ban → player loses at 3:1
    if _is_dealer_ban_luck(d_nat, d_na, d_nc):
        return 0.0  # Ban Luck vs Ban Luck → push
    return 2.0  # Ban Luck beats all others → +2 units


# ─── Forced-hit distribution cache ────────────────────────────────────────────


def _compute_forced_hit_dist(
    d_nat: int,
    d_na: int,
    d_nc: int,
    memo: dict[Comp, dict[Comp, float]],
) -> dict[Comp, float]:
    """Compute prob distribution over dealer states reachable via forced hits.

    Starting from (d_nat, d_na, d_nc) with total < 16, enumerate all paths
    through forced hits (pure chance nodes) and return the probability-weighted
    distribution over states where total >= 16 or > 21 (bust).

    This collapses the exponential forced-hit subtree into a flat distribution,
    making each CFR dealer call O(|output_states|) instead of O(13^depth).

    Args:
        d_nat, d_na, d_nc: Dealer's current composition.
        memo:              Shared memoization dict (keyed by Comp triple).

    Returns:
        Dict mapping (d_nat, d_na, d_nc) → probability for each reachable
        final state (total >= 16 or busted). Probabilities sum to 1.0.
    """
    key = (d_nat, d_na, d_nc)
    if key in memo:
        return memo[key]

    d_total = _total_from_composition(d_nat, d_na, d_nc)

    # Base case: already at a terminal-for-forced-hits state
    if d_total > 21 or d_total >= 16:
        result: dict[Comp, float] = {key: 1.0}
        memo[key] = result
        return result

    # Forced hit: dealer draws each rank with probability 1/13
    result = {}
    for rank in range(NUM_RANKS):
        nd_nat, nd_na, nd_nc = _transition(d_nat, d_na, d_nc, rank)
        sub = _compute_forced_hit_dist(nd_nat, nd_na, nd_nc, memo)
        for state, prob in sub.items():
            result[state] = result.get(state, 0.0) + RANK_PROB * prob

    memo[key] = result
    return result


# Module-level cache: populated lazily on first access.
# Maps Comp → {Comp: probability} for dealer forced-hit distributions.
_FORCED_HIT_CACHE: dict[Comp, dict[Comp, float]] = {}


def _forced_hit_dist(d_nat: int, d_na: int, d_nc: int) -> dict[Comp, float]:
    """Cached lookup for forced-hit distribution (see _compute_forced_hit_dist)."""
    key = (d_nat, d_na, d_nc)
    if key not in _FORCED_HIT_CACHE:
        _compute_forced_hit_dist(d_nat, d_na, d_nc, _FORCED_HIT_CACHE)
    return _FORCED_HIT_CACHE[key]


# ─── Initial deal enumeration ──────────────────────────────────────────────────


def _build_initial_deals() -> list[tuple[Comp, Comp, float]]:
    """Enumerate all initial 2-card composition pairs with probabilities.

    Uses the infinite-deck approximation: each rank has probability 1/13.
    All 13^4 = 28,561 ordered rank-quad combinations are grouped by their
    (player_comp, dealer_comp) composition pair to avoid redundant traversals.

    Returns:
        List of (player_comp, dealer_comp, probability) triples.
        Probabilities sum to 1.0.

    Example:
        >>> deals = _build_initial_deals()
        >>> abs(sum(w for _, _, w in deals) - 1.0) < 1e-10
        True
    """
    deals: dict[tuple[Comp, Comp], float] = {}
    for pr1 in range(NUM_RANKS):
        p1 = _transition(0, 0, 0, pr1)
        for pr2 in range(NUM_RANKS):
            p_comp = _transition(*p1, pr2)
            for dr1 in range(NUM_RANKS):
                d1 = _transition(0, 0, 0, dr1)
                for dr2 in range(NUM_RANKS):
                    d_comp = _transition(*d1, dr2)
                    key = (p_comp, d_comp)
                    deals[key] = deals.get(key, 0.0) + (RANK_PROB**4)
    return [(p, d, w) for (p, d), w in deals.items()]


# ─── Average strategy extraction ───────────────────────────────────────────────


def _extract_avg_strategy(
    sums_table: dict,
    legal_actions_fn,
) -> dict:
    """Extract the normalised average strategy from accumulated sums.

    Args:
        sums_table:        Info-set → {action: weighted probability sum}.
        legal_actions_fn:  Callable(info_set) → list[action].

    Returns:
        Dict mapping info_set → {action: average probability}.
    """
    result = {}
    for info_set, action_sums in sums_table.items():
        legal = legal_actions_fn(info_set)
        total = sum(action_sums.get(a, 0.0) for a in legal)
        if total <= 0.0:
            p = 1.0 / len(legal)
            result[info_set] = dict.fromkeys(legal, p)
        else:
            result[info_set] = {a: action_sums.get(a, 0.0) / total for a in legal}
    return result


# ─── Tabular CFR: precomputed game tree ────────────────────────────────────────
#
# Builds the complete Banluck game tree ONCE as a flat list of _TreeNode objects
# in BFS (topological) order.  Each CFR+ iteration then does three array scans:
#   1. Forward pass  — propagate reach_p and reach_d from initial deals.
#   2. Backward pass — compute action EVs bottom-up from current strategies.
#   3. Batch update  — one regret + strategy-sum update per decision node.
#
# Speedup vs recursive: each unique node is visited ONCE per iteration instead
# of O(N_initial_deals) times.  With ~25 k nodes vs 207 k recursive visits the
# expected speedup is ~8–30×, bringing per-pass time from 2.65 s to < 0.1 s.

# Node-type constants
_NT_PLAYER: int = 0  # player hit/stand decision
_NT_DEALER_SURR: int = 1  # dealer hard-15 surrender decision
_NT_DEALER_ACT: int = 2  # dealer action at strategic total (16 or 17)
_NT_TERMINAL: int = 3  # terminal node (fixed EV, no decisions)


@dataclass
class _TreeNode:
    """A node in the precomputed Banluck game tree.

    Attributes:
        idx:           Index in the flat node list.
        node_type:     One of _NT_PLAYER / _NT_DEALER_SURR / _NT_DEALER_ACT
                       / _NT_TERMINAL.
        p_nat, p_na, p_nc: Player composition triple.
        d_nat, d_na, d_nc: Dealer composition triple.
        info_set_key:  Int-tuple key into _CfrTables (None = no updates).
        legal_actions: Legal action ints (empty for terminals).
        transitions:   {action_int: [(child_idx, prob), ...]} for non-terminals.
                       prob values are *chance* probabilities (RANK_PROB or
                       forced-hit probs); they weight the EV but are NOT folded
                       into reach_p / reach_d.
        terminal_ev:   Fixed EV for _NT_TERMINAL nodes; 0.0 otherwise.
    """

    idx: int
    node_type: int
    p_nat: int
    p_na: int
    p_nc: int
    d_nat: int
    d_na: int
    d_nc: int
    info_set_key: tuple
    legal_actions: list
    transitions: dict  # int -> list[(int, float)]
    terminal_ev: float


def _build_game_tree() -> tuple[list[_TreeNode], list]:
    """Build the Banluck game tree as a flat list of _TreeNode objects.

    Uses BFS starting from all 729 initial deals so nodes are stored in
    topological order (parents before children).  This order enables a single
    forward scan for reach probabilities and a single reversed scan for EVs.

    Key design choices
    ~~~~~~~~~~~~~~~~~~
    * Forced-stand player states (total = 21 or nc = 5) are INLINED into the
      parent's HIT transitions rather than stored as separate nodes.  This
      ensures every _NT_PLAYER node is a genuine hit/stand decision.
    * ``_forced_hit_dist`` is used uniformly to expand "dealer phase" entries:
      it correctly handles the initial dealer comp, dealer totals < 16 (forced
      draw), and totals >= 16 (already at a strategic / terminal state).
    * Terminal nodes are deduplicated by EV value.  Since all _cfr_settle
      results are integer multiples of 1.0, there are at most 7 unique
      terminal EVs: {-3, -2, -1, 0, 1, 2, 3}.

    Returns:
        (nodes, initial_entries) where:
          nodes:           flat list of _TreeNode in BFS topological order.
          initial_entries: list of (node_idx_or_None, deal_prob, imm_ev) for
                           each of the 729 initial deals.  node_idx_or_None is
                           None when the deal resolves immediately (Ban Ban /
                           Ban Luck) with EV = imm_ev.
    """
    from collections import deque

    nodes: list[_TreeNode] = []
    node_map: dict = {}  # (p_nat,p_na,p_nc, d_nat,d_na,d_nc, nt) -> idx
    terminal_ev_map: dict[float, int] = {}  # ev -> idx
    queue: deque = deque()

    # ── Terminal node factory (deduplicated by EV) ──────────────────────────
    def _get_terminal(ev: float) -> int:
        if ev not in terminal_ev_map:
            idx = len(nodes)
            nodes.append(
                _TreeNode(
                    idx=idx,
                    node_type=_NT_TERMINAL,
                    p_nat=0,
                    p_na=0,
                    p_nc=0,
                    d_nat=0,
                    d_na=0,
                    d_nc=0,
                    info_set_key=(),
                    legal_actions=[],
                    transitions={},
                    terminal_ev=ev,
                )
            )
            terminal_ev_map[ev] = idx
        return terminal_ev_map[ev]

    # ── Generic decision-node factory ───────────────────────────────────────
    def _get_or_create(
        nt: int, p_nat: int, p_na: int, p_nc: int, d_nat: int, d_na: int, d_nc: int
    ) -> int:
        key = (p_nat, p_na, p_nc, d_nat, d_na, d_nc, nt)
        if key in node_map:
            return node_map[key]
        idx = len(nodes)
        if nt == _NT_PLAYER:
            p_total = _total_from_composition(p_nat, p_na, p_nc)
            p_soft = _is_soft_from_composition(p_nat, p_na, p_nc)
            info_key = _player_key(p_total, p_nc, p_soft)
            legal = [_P_HIT, _P_STAND]
        elif nt == _NT_DEALER_SURR:
            d_total = _total_from_composition(d_nat, d_na, d_nc)
            info_key = _dealer_surrender_key(d_total, True)
            legal = [_D_SURRENDER, _D_CONTINUE]
        else:  # _NT_DEALER_ACT
            d_total = _total_from_composition(d_nat, d_na, d_nc)
            d_soft = _is_soft_from_composition(d_nat, d_na, d_nc)
            info_key = _dealer_action_key(d_total, d_nc, d_soft, p_nc)
            legal = [_D_HIT, _D_STAND] if p_nc < 3 else [_D_HIT, _D_STAND, _D_REVEAL]
        node = _TreeNode(
            idx=idx,
            node_type=nt,
            p_nat=p_nat,
            p_na=p_na,
            p_nc=p_nc,
            d_nat=d_nat,
            d_na=d_na,
            d_nc=d_nc,
            info_set_key=info_key,
            legal_actions=legal,
            transitions={},
            terminal_ev=0.0,
        )
        nodes.append(node)
        node_map[key] = idx
        queue.append(node)
        return idx

    # ── Dealer-phase transition helper ──────────────────────────────────────
    # Expands "player is done, dealer has comp (d_nat,d_na,d_nc)" into a list
    # of (child_idx, prob) via _forced_hit_dist.
    def _dealer_phase_trans(
        p_nat: int, p_na: int, p_nc: int, d_nat: int, d_na: int, d_nc: int
    ) -> list:
        result = []
        for (nd_nat, nd_na, nd_nc), prob in _forced_hit_dist(d_nat, d_na, d_nc).items():
            nd_total = _total_from_composition(nd_nat, nd_na, nd_nc)
            if nd_total > 21:
                ev = _cfr_settle(p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc, d_busted=True)
                result.append((_get_terminal(ev), prob))
            elif nd_total >= 18:
                ev = _cfr_settle(p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc, d_busted=False)
                result.append((_get_terminal(ev), prob))
            else:
                child = _get_or_create(_NT_DEALER_ACT, p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc)
                result.append((child, prob))
        return result

    # ── After-surrender entry resolver ──────────────────────────────────────
    # Returns (node_idx_or_None, immediate_ev).
    # node_idx_or_None = None → immediate terminal with EV = immediate_ev.
    def _after_surr_entry(p_nat: int, p_na: int, p_nc: int, d_nat: int, d_na: int, d_nc: int):
        if _is_player_ban_ban(p_nat, p_na, p_nc):
            return None, _settle_player_ban_ban(d_nat, d_na, d_nc)
        if _is_player_ban_luck(p_nat, p_na, p_nc):
            return None, _settle_player_ban_luck(d_nat, d_na, d_nc)
        # Normal hand → player decision node (p_total in [4,20] for nc=2)
        return _get_or_create(_NT_PLAYER, p_nat, p_na, p_nc, d_nat, d_na, d_nc), 0.0

    # ── Seed the BFS from all 729 initial deals ──────────────────────────────
    initial_deals = _build_initial_deals()
    initial_entries = []
    for p_comp, d_comp, prob in initial_deals:
        p_nat, p_na, p_nc = p_comp
        d_nat, d_na, d_nc = d_comp
        if _is_hard_fifteen_comp(d_nat, d_na, d_nc):
            idx = _get_or_create(_NT_DEALER_SURR, p_nat, p_na, p_nc, d_nat, d_na, d_nc)
            initial_entries.append((idx, prob, 0.0))
        else:
            node_idx, imm_ev = _after_surr_entry(p_nat, p_na, p_nc, d_nat, d_na, d_nc)
            initial_entries.append((node_idx, prob, imm_ev))

    # ── BFS: build transitions for every queued decision node ───────────────
    while queue:
        node = queue.popleft()
        nt = node.node_type
        p_nat, p_na, p_nc = node.p_nat, node.p_na, node.p_nc
        d_nat, d_na, d_nc = node.d_nat, node.d_na, node.d_nc

        if nt == _NT_DEALER_SURR:
            # SURRENDER → push (EV = 0)
            node.transitions[_D_SURRENDER] = [(_get_terminal(0.0), 1.0)]
            # CONTINUE → after-surrender routing
            cont_idx, cont_ev = _after_surr_entry(p_nat, p_na, p_nc, d_nat, d_na, d_nc)
            if cont_idx is None:
                node.transitions[_D_CONTINUE] = [(_get_terminal(cont_ev), 1.0)]
            else:
                node.transitions[_D_CONTINUE] = [(cont_idx, 1.0)]

        elif nt == _NT_PLAYER:
            # STAND → dealer phase (forced hits from initial dealer comp)
            node.transitions[_P_STAND] = _dealer_phase_trans(p_nat, p_na, p_nc, d_nat, d_na, d_nc)
            # HIT → draw a rank (chance node, RANK_PROB = 1/13)
            hit_trans: list = []
            for rank in range(NUM_RANKS):
                np_nat, np_na, np_nc = _transition(p_nat, p_na, p_nc, rank)
                np_total = _total_from_composition(np_nat, np_na, np_nc)
                if np_total > 21:
                    ev = -2.0 if np_nc == 5 else -1.0
                    hit_trans.append((_get_terminal(ev), RANK_PROB))
                elif np_total == 21 or np_nc == 5:
                    # Forced stand: inline dealer phase directly
                    for child_idx, d_prob in _dealer_phase_trans(
                        np_nat, np_na, np_nc, d_nat, d_na, d_nc
                    ):
                        hit_trans.append((child_idx, RANK_PROB * d_prob))
                else:
                    child = _get_or_create(_NT_PLAYER, np_nat, np_na, np_nc, d_nat, d_na, d_nc)
                    hit_trans.append((child, RANK_PROB))
            node.transitions[_P_HIT] = hit_trans

        else:  # _NT_DEALER_ACT
            # STAND and REVEAL both settle at the current dealer total
            settle_ev = _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False)
            term_idx = _get_terminal(settle_ev)
            node.transitions[_D_STAND] = [(term_idx, 1.0)]
            if _D_REVEAL in node.legal_actions:
                node.transitions[_D_REVEAL] = [(term_idx, 1.0)]
            # HIT → draw a rank → use _forced_hit_dist to expand
            hit_trans = []
            for rank in range(NUM_RANKS):
                nd_nat, nd_na, nd_nc = _transition(d_nat, d_na, d_nc, rank)
                for child_idx, d_prob in _dealer_phase_trans(
                    p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc
                ):
                    hit_trans.append((child_idx, RANK_PROB * d_prob))
            node.transitions[_D_HIT] = hit_trans

    return nodes, initial_entries


# Module-level game tree cache (built once on first call to solve / _cfr_pass).
_GAME_TREE_NODES: list[_TreeNode] | None = None
_GAME_TREE_ENTRIES: list | None = None  # [(node_idx_or_None, prob, imm_ev)]


def _get_or_build_tree() -> tuple[list[_TreeNode], list]:
    """Return the cached game tree, building it on first call."""
    global _GAME_TREE_NODES, _GAME_TREE_ENTRIES
    if _GAME_TREE_NODES is None:
        _GAME_TREE_NODES, _GAME_TREE_ENTRIES = _build_game_tree()
    return _GAME_TREE_NODES, _GAME_TREE_ENTRIES


# ─── Numba flat-array tree (C3c) ────────────────────────────────────────────────

# Cached flat numpy arrays derived from the game tree (built once).
_NUMBA_ARRAYS: tuple | None = None


def _build_numba_arrays(
    nodes: list[_TreeNode],
    initial_entries: list,
) -> tuple:
    """Convert the Python game tree into flat NumPy arrays for the Numba kernel.

    Called once (result is cached in _NUMBA_ARRAYS).  All variable-length
    transition lists are stored in CSR (compressed-sparse-row) format using
    two 2-D pointer arrays: ``trans_start[i, j]`` and ``trans_end[i, j]``
    give the half-open range ``[start, end)`` in the flat ``trans_child`` /
    ``trans_prob`` arrays for node ``i``, action ``j``.

    Legal actions are always ``0 … n_actions[i]-1`` (verified by construction
    in _build_game_tree), so action index ``j`` maps directly to action int ``j``.

    Returns a 17-tuple:
        (node_cat, terminal_ev, info_slot, n_actions,
         trans_start, trans_end, trans_child, trans_prob,
         init_node_idx, init_prob, init_imm_ev,
         n_player_slots, n_surr_slots, n_act_slots,
         player_rev, surr_rev, act_rev)

    where ``*_rev`` dicts map slot_index → int-tuple info-set key (for
    strategy extraction at the end of solve()).
    """
    n = len(nodes)
    MAX_ACT = 3

    # ── Assign info-set slots (one table per node type) ──────────────────────
    player_slot_map: dict = {}  # int-tuple key → slot index
    surr_slot_map: dict = {}
    act_slot_map: dict = {}

    for node in nodes:
        if node.node_type == _NT_TERMINAL:
            continue
        key = node.info_set_key
        if node.node_type == _NT_PLAYER:
            if key not in player_slot_map:
                player_slot_map[key] = len(player_slot_map)
        elif node.node_type == _NT_DEALER_SURR:
            if key not in surr_slot_map:
                surr_slot_map[key] = len(surr_slot_map)
        else:
            if key not in act_slot_map:
                act_slot_map[key] = len(act_slot_map)

    # ── Per-node arrays ──────────────────────────────────────────────────────
    node_cat = np.empty(n, dtype=np.int32)
    terminal_ev_arr = np.zeros(n, dtype=np.float64)
    info_slot_arr = np.full(n, -1, dtype=np.int32)
    n_actions_arr = np.zeros(n, dtype=np.int32)

    for node in nodes:
        i = node.idx
        node_cat[i] = node.node_type
        if node.node_type == _NT_TERMINAL:
            terminal_ev_arr[i] = node.terminal_ev
        else:
            n_actions_arr[i] = len(node.legal_actions)
            if node.node_type == _NT_PLAYER:
                info_slot_arr[i] = player_slot_map[node.info_set_key]
            elif node.node_type == _NT_DEALER_SURR:
                info_slot_arr[i] = surr_slot_map[node.info_set_key]
            else:
                info_slot_arr[i] = act_slot_map[node.info_set_key]

    # ── Count total transitions ──────────────────────────────────────────────
    total_trans = sum(
        len(node.transitions.get(a, [])) for node in nodes for a in node.legal_actions
    )

    # ── CSR transition arrays ────────────────────────────────────────────────
    trans_start_arr = np.zeros((n, MAX_ACT), dtype=np.int32)
    trans_end_arr = np.zeros((n, MAX_ACT), dtype=np.int32)
    trans_child_arr = np.zeros(total_trans, dtype=np.int32)
    trans_prob_arr = np.zeros(total_trans, dtype=np.float64)

    ptr = 0
    for node in nodes:
        i = node.idx
        for j, a in enumerate(node.legal_actions):
            trans_start_arr[i, j] = ptr
            for child_idx, prob in node.transitions[a]:
                trans_child_arr[ptr] = child_idx
                trans_prob_arr[ptr] = prob
                ptr += 1
            trans_end_arr[i, j] = ptr
        # Unused action slots: start == end == ptr (empty range)
        for j in range(len(node.legal_actions), MAX_ACT):
            trans_start_arr[i, j] = ptr
            trans_end_arr[i, j] = ptr

    # ── Initial entries ──────────────────────────────────────────────────────
    n_init = len(initial_entries)
    init_node_idx_arr = np.full(n_init, -1, dtype=np.int32)
    init_prob_arr = np.zeros(n_init, dtype=np.float64)
    init_imm_ev_arr = np.zeros(n_init, dtype=np.float64)

    for e, (node_idx, prob, imm_ev) in enumerate(initial_entries):
        init_prob_arr[e] = prob
        init_imm_ev_arr[e] = imm_ev
        if node_idx is not None:
            init_node_idx_arr[e] = node_idx

    # ── Reverse slot maps (slot → key) for extraction ─────────────────────────
    player_rev = {v: k for k, v in player_slot_map.items()}
    surr_rev = {v: k for k, v in surr_slot_map.items()}
    act_rev = {v: k for k, v in act_slot_map.items()}

    return (
        node_cat,
        terminal_ev_arr,
        info_slot_arr,
        n_actions_arr,
        trans_start_arr,
        trans_end_arr,
        trans_child_arr,
        trans_prob_arr,
        init_node_idx_arr,
        init_prob_arr,
        init_imm_ev_arr,
        len(player_slot_map),
        len(surr_slot_map),
        len(act_slot_map),
        player_rev,
        surr_rev,
        act_rev,
    )


def _get_or_build_numba_arrays() -> tuple:
    """Return cached flat numpy arrays, building them on first call."""
    global _NUMBA_ARRAYS
    if _NUMBA_ARRAYS is None:
        nodes, entries = _get_or_build_tree()
        _NUMBA_ARRAYS = _build_numba_arrays(nodes, entries)
    return _NUMBA_ARRAYS


def _cfr_pass_tabular(tables: _CfrTables, iteration: int) -> float:
    """Run one CFR+ iteration using the precomputed game tree.

    Replaces the recursive ``_cfr_pass`` with three linear array scans:

    1. **Strategy computation** — read current regrets once per decision node.
    2. **Forward pass** — propagate reach_p and reach_d from initial deals.
       *Chance probabilities (RANK_PROB, forced-hit probs) are NOT included
       in reach_p / reach_d* — they appear only in action-EV weighting.
    3. **Backward pass** — compute action EVs bottom-up (reversed BFS order).
    4. **Batch update** — one regret + strategy-sum update per decision node
       using the accumulated reach_p / reach_d sums.

    Mathematical equivalence with ``_cfr_pass_recursive``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For a given state s (identified by its composition), the recursive code
    may visit it N times per iteration (once per initial deal that routes
    through it), each time with the same action EVs (guaranteed by the
    within-pass memo) and different reach probs r_1 … r_N.

    Because the sign of (action_ev − node_ev) is fixed for all visits (same
    cached EVs), the CFR+ max(0, …) floor applied N times gives the same
    result as applying it once with the summed reach Σ r_i.  Hence batch
    accumulation is *mathematically identical* to per-visit updates.

    Args:
        tables:    Mutable CFR+ accumulators (updated in-place).
        iteration: Current iteration number (1-indexed).

    Returns:
        Probability-weighted average player EV for this iteration.
    """
    nodes, initial_entries = _get_or_build_tree()
    n = len(nodes)

    # ── Step 1: Compute current strategies ──────────────────────────────────
    strategies: list = [None] * n
    for node in nodes:
        nt = node.node_type
        if nt == _NT_TERMINAL:
            continue
        if nt == _NT_PLAYER:
            strategies[node.idx] = _get_strategy(
                node.info_set_key, tables.player_regrets, node.legal_actions
            )
        elif nt == _NT_DEALER_SURR:
            strategies[node.idx] = _get_strategy(
                node.info_set_key, tables.dealer_surrender_regrets, node.legal_actions
            )
        else:
            strategies[node.idx] = _get_strategy(
                node.info_set_key, tables.dealer_action_regrets, node.legal_actions
            )

    # ── Step 2: Forward pass — accumulate reach_p / reach_d ─────────────────
    reach_p = [0.0] * n
    reach_d = [0.0] * n
    imm_ev_total = 0.0

    for node_idx, prob, imm_ev in initial_entries:
        if node_idx is None:
            imm_ev_total += prob * imm_ev
        else:
            # Initial reach before any decisions = 1.0 per initial deal.
            # (The deal probability 'prob' is a *chance* probability; it is
            # used only in the final EV weighting, not in reach accumulation.)
            reach_p[node_idx] += 1.0
            reach_d[node_idx] += 1.0

    for node in nodes:  # BFS order = topological (parents before children)
        nt = node.node_type
        if nt == _NT_TERMINAL:
            continue
        rp = reach_p[node.idx]
        rd = reach_d[node.idx]
        if rp == 0.0 and rd == 0.0:
            continue  # unreachable node
        s = strategies[node.idx]
        for a in node.legal_actions:
            if nt == _NT_PLAYER:
                child_rp = rp * s[a]
                child_rd = rd
            else:  # dealer node (SURR or ACT): dealer's own choices affect reach_d
                child_rp = rp
                child_rd = rd * s[a]
            for child_idx, _prob in node.transitions[a]:
                # _prob is a chance prob; NOT folded into reach_p / reach_d
                if nodes[child_idx].node_type != _NT_TERMINAL:
                    reach_p[child_idx] += child_rp
                    reach_d[child_idx] += child_rd

    # ── Step 3: Backward pass — compute EVs bottom-up ───────────────────────
    ev: list = [0.0] * n
    action_evs_all: list = [None] * n

    # Initialise terminal EVs (they are constant, independent of strategies)
    for node in nodes:
        if node.node_type == _NT_TERMINAL:
            ev[node.idx] = node.terminal_ev

    # Reversed BFS = bottom-up (children processed before parents)
    for node in reversed(nodes):
        nt = node.node_type
        if nt == _NT_TERMINAL:
            continue
        s = strategies[node.idx]
        aevs: dict = {}
        for a in node.legal_actions:
            aev = 0.0
            for child_idx, prob in node.transitions[a]:
                aev += prob * ev[child_idx]
            aevs[a] = aev
        node_ev = 0.0
        for a in node.legal_actions:
            node_ev += s[a] * aevs[a]
        ev[node.idx] = node_ev
        action_evs_all[node.idx] = aevs

    # ── Step 4: Batch regret + strategy-sum updates ──────────────────────────
    for node in nodes:
        nt = node.node_type
        if nt == _NT_TERMINAL:
            continue
        s = strategies[node.idx]
        aevs = action_evs_all[node.idx]
        node_ev_val = ev[node.idx]
        rp = reach_p[node.idx]
        rd = reach_d[node.idx]

        if nt == _NT_PLAYER:
            _update_strategy_sum(
                node.info_set_key,
                s,
                rp,
                iteration,
                tables.player_strategy_sums,
            )
            _regret_update_player(
                node.info_set_key,
                node.legal_actions,
                aevs,
                node_ev_val,
                rd,
                tables.player_regrets,
            )
        elif nt == _NT_DEALER_SURR:
            _update_strategy_sum(
                node.info_set_key,
                s,
                rd,
                iteration,
                tables.dealer_surrender_strategy_sums,
            )
            _regret_update_dealer(
                node.info_set_key,
                node.legal_actions,
                aevs,
                node_ev_val,
                rp,
                tables.dealer_surrender_regrets,
            )
        else:  # _NT_DEALER_ACT
            _update_strategy_sum(
                node.info_set_key,
                s,
                rd,
                iteration,
                tables.dealer_action_strategy_sums,
            )
            _regret_update_dealer(
                node.info_set_key,
                node.legal_actions,
                aevs,
                node_ev_val,
                rp,
                tables.dealer_action_regrets,
            )

    # ── Step 5: Compute probability-weighted EV for this iteration ───────────
    total_ev = imm_ev_total
    for node_idx, prob, _imm_ev in initial_entries:
        if node_idx is not None:
            total_ev += prob * ev[node_idx]
    return total_ev


# ─── Numba JIT kernel (C3c) ────────────────────────────────────────────────────

# _NT_* constants duplicated as plain ints so the @njit function needs no
# module-level closure (Numba requires all referenced values to be scalar
# literals or array arguments).
_NB_PLAYER: int = 0  # == _NT_PLAYER
_NB_SURR: int = 1  # == _NT_DEALER_SURR
_NB_ACT: int = 2  # == _NT_DEALER_ACT
_NB_TERMINAL: int = 3  # == _NT_TERMINAL


def _make_numba_kernel():
    """Return the @njit CFR+ kernel, or None if Numba is unavailable."""
    if not _NUMBA_AVAILABLE:
        return None

    NB_PLAYER = _NB_PLAYER
    NB_SURR = _NB_SURR

    @numba.njit(cache=True)
    def _kernel(
        n_nodes,
        node_cat,  # int32[n]   — 0=player, 1=surr, 2=act, 3=terminal
        terminal_ev,  # float64[n]
        info_slot,  # int32[n]
        n_actions,  # int32[n]
        trans_start,  # int32[n, 3]
        trans_end,  # int32[n, 3]
        trans_child,  # int32[total_trans]
        trans_prob,  # float64[total_trans]
        init_node_idx,  # int32[n_init]
        init_prob,  # float64[n_init]
        init_imm_ev,  # float64[n_init]
        player_regrets,  # float64[n_player, 2]  — modified in-place
        player_strategy_sums,  # float64[n_player, 2]  — modified in-place
        surr_regrets,  # float64[n_surr, 2]    — modified in-place
        surr_strategy_sums,  # float64[n_surr, 2]    — modified in-place
        act_regrets,  # float64[n_act, 3]     — modified in-place
        act_strategy_sums,  # float64[n_act, 3]     — modified in-place
        iteration,  # int64
    ):
        """CFR+ inner loop: strategy → forward → backward → update → EV."""
        n_init = init_node_idx.shape[0]

        # ── Step 1: Compute current strategies ───────────────────────────────
        strategies = np.zeros((n_nodes, 3))

        for i in range(n_nodes):
            nc = n_actions[i]
            if nc == 0:
                continue
            cat = node_cat[i]
            slot = info_slot[i]

            reg_sum = 0.0
            if cat == NB_PLAYER:
                for j in range(nc):
                    r = player_regrets[slot, j]
                    if r > 0.0:
                        reg_sum += r
            elif cat == NB_SURR:
                for j in range(nc):
                    r = surr_regrets[slot, j]
                    if r > 0.0:
                        reg_sum += r
            else:
                for j in range(nc):
                    r = act_regrets[slot, j]
                    if r > 0.0:
                        reg_sum += r

            if reg_sum > 0.0:
                inv = 1.0 / reg_sum
                if cat == NB_PLAYER:
                    for j in range(nc):
                        r = player_regrets[slot, j]
                        strategies[i, j] = r * inv if r > 0.0 else 0.0
                elif cat == NB_SURR:
                    for j in range(nc):
                        r = surr_regrets[slot, j]
                        strategies[i, j] = r * inv if r > 0.0 else 0.0
                else:
                    for j in range(nc):
                        r = act_regrets[slot, j]
                        strategies[i, j] = r * inv if r > 0.0 else 0.0
            else:
                inv_nc = 1.0 / nc
                for j in range(nc):
                    strategies[i, j] = inv_nc

        # ── Step 2: Forward pass — accumulate reach_p / reach_d ──────────────
        reach_p = np.zeros(n_nodes)
        reach_d = np.zeros(n_nodes)
        imm_ev_total = 0.0

        for e in range(n_init):
            ni = init_node_idx[e]
            if ni < 0:
                imm_ev_total += init_prob[e] * init_imm_ev[e]
            else:
                reach_p[ni] += 1.0
                reach_d[ni] += 1.0

        for i in range(n_nodes):
            nc = n_actions[i]
            if nc == 0:
                continue
            rp = reach_p[i]
            rd = reach_d[i]
            if rp == 0.0 and rd == 0.0:
                continue
            cat = node_cat[i]
            for j in range(nc):
                if cat == NB_PLAYER:
                    child_rp = rp * strategies[i, j]
                    child_rd = rd
                else:
                    child_rp = rp
                    child_rd = rd * strategies[i, j]
                ts = trans_start[i, j]
                te = trans_end[i, j]
                for k in range(ts, te):
                    ci = trans_child[k]
                    if n_actions[ci] > 0:
                        reach_p[ci] += child_rp
                        reach_d[ci] += child_rd

        # ── Step 3: Backward pass — compute EVs bottom-up ────────────────────
        ev = np.zeros(n_nodes)
        action_evs = np.zeros((n_nodes, 3))

        for i in range(n_nodes):
            if n_actions[i] == 0:
                ev[i] = terminal_ev[i]

        for i in range(n_nodes - 1, -1, -1):
            nc = n_actions[i]
            if nc == 0:
                continue
            node_ev_val = 0.0
            for j in range(nc):
                ts = trans_start[i, j]
                te = trans_end[i, j]
                aev = 0.0
                for k in range(ts, te):
                    aev += trans_prob[k] * ev[trans_child[k]]
                action_evs[i, j] = aev
                node_ev_val += strategies[i, j] * aev
            ev[i] = node_ev_val

        # ── Step 4: Batch regret + strategy-sum updates ───────────────────────
        for i in range(n_nodes):
            nc = n_actions[i]
            if nc == 0:
                continue
            cat = node_cat[i]
            slot = info_slot[i]
            rp = reach_p[i]
            rd = reach_d[i]
            node_ev_val = ev[i]

            if cat == NB_PLAYER:
                # Player = maximiser: regret weighted by reach_d (opponent)
                w = float(iteration) * rp
                for j in range(nc):
                    player_strategy_sums[slot, j] += w * strategies[i, j]
                    instant = rd * (action_evs[i, j] - node_ev_val)
                    new_r = player_regrets[slot, j] + instant
                    player_regrets[slot, j] = new_r if new_r > 0.0 else 0.0
            elif cat == NB_SURR:
                # Dealer = minimiser: regret = node_ev - action_ev, weighted by reach_p
                w = float(iteration) * rd
                for j in range(nc):
                    surr_strategy_sums[slot, j] += w * strategies[i, j]
                    instant = rp * (node_ev_val - action_evs[i, j])
                    new_r = surr_regrets[slot, j] + instant
                    surr_regrets[slot, j] = new_r if new_r > 0.0 else 0.0
            else:
                # Dealer action = minimiser
                w = float(iteration) * rd
                for j in range(nc):
                    act_strategy_sums[slot, j] += w * strategies[i, j]
                    instant = rp * (node_ev_val - action_evs[i, j])
                    new_r = act_regrets[slot, j] + instant
                    act_regrets[slot, j] = new_r if new_r > 0.0 else 0.0

        # ── Step 5: Probability-weighted EV ──────────────────────────────────
        total_ev = imm_ev_total
        for e in range(n_init):
            ni = init_node_idx[e]
            if ni >= 0:
                total_ev += init_prob[e] * ev[ni]
        return total_ev

    return _kernel


# Module-level compiled kernel (built on first call to _cfr_pass_numba).
_NUMBA_KERNEL = None


def _cfr_pass_numba(numba_arrays: tuple, tables: _NumbaTables, iteration: int) -> float:
    """Run one CFR+ iteration via the Numba JIT kernel.

    Args:
        numba_arrays: Tuple returned by _get_or_build_numba_arrays().
        tables:       _NumbaTables instance (updated in-place).
        iteration:    Current 1-indexed iteration number.

    Returns:
        Probability-weighted average player EV for this iteration.
    """
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        _NUMBA_KERNEL = _make_numba_kernel()

    (
        node_cat,
        terminal_ev,
        info_slot,
        n_actions,
        trans_start,
        trans_end,
        trans_child,
        trans_prob,
        init_node_idx,
        init_prob,
        init_imm_ev,
        _n_p,
        _n_s,
        _n_a,
        _pr,
        _sr,
        _ar,
    ) = numba_arrays

    return _NUMBA_KERNEL(
        len(node_cat),
        node_cat,
        terminal_ev,
        info_slot,
        n_actions,
        trans_start,
        trans_end,
        trans_child,
        trans_prob,
        init_node_idx,
        init_prob,
        init_imm_ev,
        tables.player_regrets,
        tables.player_strategy_sums,
        tables.surr_regrets,
        tables.surr_strategy_sums,
        tables.act_regrets,
        tables.act_strategy_sums,
        iteration,
    )


# ─── Dealer action phase CFR ───────────────────────────────────────────────────


def _dealer_cfr(
    p_nat: int,
    p_na: int,
    p_nc: int,
    d_nat: int,
    d_na: int,
    d_nc: int,
    reach_p: float,
    reach_d: float,
    tables: _CfrTables,
    iteration: int,
    memo: dict | None = None,
) -> float:
    """CFR traversal through the dealer action phase.

    Dealer must reach total ≥ 16 (forced hits below 16), then has strategic
    decisions at 16/17 (including soft 17). At ≥ 18 or bust, the game is
    terminal.

    Args:
        p_nat, p_na, p_nc: Player's final composition (fixed after player acts).
        d_nat, d_na, d_nc: Dealer's current composition.
        reach_p:           Player's reach probability contribution.
        reach_d:           Dealer's reach probability contribution.
        tables:            Mutable CFR+ accumulators.
        iteration:         Current CFR+ iteration number (1-indexed).
        memo:              Within-pass EV cache (keyed by game state).
                           Avoids recomputing action EVs for repeated states.

    Returns:
        Expected player EV from this dealer state onwards.
    """
    d_total = _total_from_composition(d_nat, d_na, d_nc)

    # Terminal: dealer bust
    if d_total > 21:
        return _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=True)

    # Terminal: dealer at ≥18 — forced stand, settle immediately
    if d_total >= 18:
        return _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False)

    # Dealer below 16: forced hit (pure chance node, no strategic decisions).
    # Use precomputed distribution to collapse the exponential subtree.
    if d_total < 16:
        ev = 0.0
        for (nd_nat, nd_na, nd_nc), prob in _forced_hit_dist(d_nat, d_na, d_nc).items():
            ev += prob * _dealer_cfr(
                p_nat,
                p_na,
                p_nc,
                nd_nat,
                nd_na,
                nd_nc,
                reach_p,
                reach_d,
                tables,
                iteration,
                memo,
            )
        return ev

    # Dealer at 16 or 17 (including soft 17): strategic decision
    d_soft = _is_soft_from_composition(d_nat, d_na, d_nc)
    d_key = _dealer_action_key(d_total, d_nc, d_soft, p_nc)
    # HIT and STAND are always legal; REVEAL_PLAYER is legal when player has 3+ cards.
    legal_int = [_D_HIT, _D_STAND] if p_nc < 3 else [_D_HIT, _D_STAND, _D_REVEAL]

    # Single legal action (defensive — shouldn't occur at 16/17 with player)
    if len(legal_int) == 1:
        a = legal_int[0]
        if a == _D_HIT:
            ev = 0.0
            for rank in range(NUM_RANKS):
                nd_nat, nd_na, nd_nc = _transition(d_nat, d_na, d_nc, rank)
                ev += RANK_PROB * _dealer_cfr(
                    p_nat,
                    p_na,
                    p_nc,
                    nd_nat,
                    nd_na,
                    nd_nc,
                    reach_p,
                    reach_d,
                    tables,
                    iteration,
                    memo,
                )
            return ev
        # _D_STAND or _D_REVEAL: terminal for player in heads-up
        return _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False)

    # Within-pass memoization: check if action EVs already computed for this state.
    # Key: dealer strategic state — p_comp and d_comp uniquely determine the subtree.
    memo_key = (p_nat, p_na, p_nc, d_nat, d_na, d_nc, "d")
    if memo is not None and memo_key in memo:
        cached_action_evs, cached_node_ev = memo[memo_key]
        # Recompute strategy from current regrets (fast), do regret/sum updates.
        strategy = _get_strategy(d_key, tables.dealer_action_regrets, legal_int)
        _update_strategy_sum(
            d_key, strategy, reach_d, iteration, tables.dealer_action_strategy_sums
        )
        node_ev = sum(strategy[a] * cached_action_evs[a] for a in legal_int)
        _regret_update_dealer(
            d_key,
            legal_int,
            cached_action_evs,
            node_ev,
            reach_p,
            tables.dealer_action_regrets,
        )
        return node_ev

    # Get dealer's current strategy from regrets
    strategy = _get_strategy(d_key, tables.dealer_action_regrets, legal_int)

    # Update strategy sums (weighted by dealer's own reach)
    _update_strategy_sum(d_key, strategy, reach_d, iteration, tables.dealer_action_strategy_sums)

    # Compute EV for each legal action
    action_evs: dict[int, float] = {}
    for a in legal_int:
        if a in (_D_STAND, _D_REVEAL):
            # Both settle player at current dealer total (same in heads-up)
            action_evs[a] = _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False)
        else:
            # _D_HIT: draw a card (chance node)
            hit_ev = 0.0
            for rank in range(NUM_RANKS):
                nd_nat, nd_na, nd_nc = _transition(d_nat, d_na, d_nc, rank)
                hit_ev += RANK_PROB * _dealer_cfr(
                    p_nat,
                    p_na,
                    p_nc,
                    nd_nat,
                    nd_na,
                    nd_nc,
                    reach_p,
                    reach_d * strategy[a],
                    tables,
                    iteration,
                    memo,
                )
            action_evs[a] = hit_ev

    # Node EV under current dealer strategy
    node_ev = sum(strategy[a] * action_evs[a] for a in legal_int)

    # Update dealer regrets (dealer is minimiser)
    _regret_update_dealer(
        d_key,
        legal_int,
        action_evs,
        node_ev,
        reach_p,
        tables.dealer_action_regrets,
    )

    # Cache action EVs for subsequent visits with same (p, d) state
    if memo is not None:
        memo[memo_key] = (action_evs, node_ev)

    return node_ev


# ─── Player action phase CFR ───────────────────────────────────────────────────


def _player_cfr(
    p_nat: int,
    p_na: int,
    p_nc: int,
    d_nat: int,
    d_na: int,
    d_nc: int,
    reach_p: float,
    reach_d: float,
    tables: _CfrTables,
    iteration: int,
    memo: dict | None = None,
) -> float:
    """CFR traversal through the player action phase.

    At each player decision point, CFR+ regrets are updated. Chance nodes
    (card draws) enumerate all 13 ranks with probability 1/13 each.

    Args:
        p_nat, p_na, p_nc: Player's current composition.
        d_nat, d_na, d_nc: Dealer's initial 2-card composition (fixed).
        reach_p:           Player's reach probability contribution.
        reach_d:           Dealer's reach probability contribution.
        tables:            Mutable CFR+ accumulators.
        iteration:         Current CFR+ iteration number (1-indexed).
        memo:              Within-pass EV cache (keyed by game state).
                           Avoids recomputing action EVs for repeated states.

    Returns:
        Expected player EV from this player state onwards.
    """
    p_total = _total_from_composition(p_nat, p_na, p_nc)

    # Terminal: player bust
    if p_total > 21:
        return -2.0 if p_nc == 5 else -1.0

    # Forced stand: player at 21 or has 5 cards → proceed to dealer phase
    if p_total == 21 or p_nc == 5:
        return _dealer_cfr(
            p_nat,
            p_na,
            p_nc,
            d_nat,
            d_na,
            d_nc,
            reach_p,
            reach_d,
            tables,
            iteration,
            memo,
        )

    # Player decision node
    p_soft = _is_soft_from_composition(p_nat, p_na, p_nc)
    p_key = _player_key(p_total, p_nc, p_soft)
    legal_int = [_P_HIT, _P_STAND]

    # Within-pass memoization: check if action EVs already computed for this state.
    # The (p_comp, d_comp) pair uniquely identifies the subtree under fixed strategies.
    memo_key = (p_nat, p_na, p_nc, d_nat, d_na, d_nc, "p")
    if memo is not None and memo_key in memo:
        cached_action_evs, cached_node_ev = memo[memo_key]
        # Recompute strategy from current regrets (fast), do regret/sum updates.
        strategy = _get_strategy(p_key, tables.player_regrets, legal_int)
        _update_strategy_sum(p_key, strategy, reach_p, iteration, tables.player_strategy_sums)
        node_ev = sum(strategy[a] * cached_action_evs[a] for a in legal_int)
        _regret_update_player(
            p_key,
            legal_int,
            cached_action_evs,
            node_ev,
            reach_d,
            tables.player_regrets,
        )
        return node_ev

    # Get current strategy
    strategy = _get_strategy(p_key, tables.player_regrets, legal_int)

    # Update strategy sums (weighted by player's own reach)
    _update_strategy_sum(p_key, strategy, reach_p, iteration, tables.player_strategy_sums)

    # Compute EV for each legal action
    action_evs: dict[int, float] = {}
    for a in legal_int:
        if a == _P_STAND:
            action_evs[a] = _dealer_cfr(
                p_nat,
                p_na,
                p_nc,
                d_nat,
                d_na,
                d_nc,
                reach_p * strategy[a],
                reach_d,
                tables,
                iteration,
                memo,
            )
        else:
            # _P_HIT: draw a card (chance node)
            hit_ev = 0.0
            for rank in range(NUM_RANKS):
                np_nat, np_na, np_nc = _transition(p_nat, p_na, p_nc, rank)
                hit_ev += RANK_PROB * _player_cfr(
                    np_nat,
                    np_na,
                    np_nc,
                    d_nat,
                    d_na,
                    d_nc,
                    reach_p * strategy[a],
                    reach_d,
                    tables,
                    iteration,
                    memo,
                )
            action_evs[a] = hit_ev

    # Node EV under current player strategy
    node_ev = sum(strategy[a] * action_evs[a] for a in legal_int)

    # Update player regrets (player is maximiser)
    _regret_update_player(
        p_key,
        legal_int,
        action_evs,
        node_ev,
        reach_d,
        tables.player_regrets,
    )

    # Cache action EVs for subsequent visits with same (p, d) state
    if memo is not None:
        memo[memo_key] = (action_evs, node_ev)

    return node_ev


# ─── Root traversal ────────────────────────────────────────────────────────────


def _after_surrender(
    p_nat: int,
    p_na: int,
    p_nc: int,
    d_nat: int,
    d_na: int,
    d_nc: int,
    reach_p: float,
    reach_d: float,
    tables: _CfrTables,
    iteration: int,
    memo: dict | None = None,
) -> float:
    """Continue the game after the dealer's surrender check.

    Checks for player special hands (Ban Ban / Ban Luck), then proceeds
    to the player action phase.

    Returns:
        Expected player EV.
    """
    # Player Ban Ban: immediate settlement against dealer's 2-card hand
    if _is_player_ban_ban(p_nat, p_na, p_nc):
        return _settle_player_ban_ban(d_nat, d_na, d_nc)

    # Player Ban Luck: immediate settlement against dealer's 2-card hand
    if _is_player_ban_luck(p_nat, p_na, p_nc):
        return _settle_player_ban_luck(d_nat, d_na, d_nc)

    # Normal hand: proceed to player action phase
    return _player_cfr(
        p_nat,
        p_na,
        p_nc,
        d_nat,
        d_na,
        d_nc,
        reach_p,
        reach_d,
        tables,
        iteration,
        memo,
    )


def _root_cfr(
    p_comp: Comp,
    d_comp: Comp,
    reach_p: float,
    reach_d: float,
    tables: _CfrTables,
    iteration: int,
    memo: dict | None = None,
) -> float:
    """CFR traversal from a specific (player_comp, dealer_comp) initial state.

    Handles the dealer surrender decision first, then routes to the
    player action phase. The initial deal is treated as a chance node
    externally (caller weights by initial deal probability).

    Args:
        p_comp:    Player's 2-card initial composition.
        d_comp:    Dealer's 2-card initial composition.
        reach_p:   Player's reach probability contribution (starts at 1.0).
        reach_d:   Dealer's reach probability contribution (starts at 1.0).
        tables:    Mutable CFR+ accumulators.
        iteration: Current CFR+ iteration number (1-indexed).
        memo:      Within-pass EV cache passed through the entire call chain.

    Returns:
        Expected player EV from this initial state.
    """
    p_nat, p_na, p_nc = p_comp
    d_nat, d_na, d_nc = d_comp
    d_total = _total_from_composition(d_nat, d_na, d_nc)

    # === Dealer surrender check (highest priority) ===
    is_hard_15 = _is_hard_fifteen_comp(d_nat, d_na, d_nc)

    if is_hard_15:
        # Dealer has a strategic choice: surrender or continue
        s_key = _dealer_surrender_key(d_total, True)
        surr_legal_int = [_D_SURRENDER, _D_CONTINUE]
        strategy = _get_strategy(s_key, tables.dealer_surrender_regrets, surr_legal_int)
        _update_strategy_sum(
            s_key,
            strategy,
            reach_d,
            iteration,
            tables.dealer_surrender_strategy_sums,
        )

        action_evs: dict[int, float] = {}
        for a in surr_legal_int:
            if a == _D_SURRENDER:
                action_evs[a] = 0.0  # Push: all bets returned
            else:
                # _D_CONTINUE (DealerAction.HIT used as "no-surrender" sentinel)
                action_evs[a] = _after_surrender(
                    p_nat,
                    p_na,
                    p_nc,
                    d_nat,
                    d_na,
                    d_nc,
                    reach_p,
                    reach_d * strategy[a],
                    tables,
                    iteration,
                    memo,
                )

        node_ev = sum(strategy[a] * action_evs[a] for a in surr_legal_int)
        _regret_update_dealer(
            s_key,
            surr_legal_int,
            action_evs,
            node_ev,
            reach_p,
            tables.dealer_surrender_regrets,
        )
        return node_ev
    else:
        # No surrender option — continue directly
        return _after_surrender(
            p_nat,
            p_na,
            p_nc,
            d_nat,
            d_na,
            d_nc,
            reach_p,
            reach_d,
            tables,
            iteration,
            memo,
        )


# ─── CFR+ main loop ────────────────────────────────────────────────────────────


def _cfr_pass_recursive(
    initial_deals: list[tuple[Comp, Comp, float]],
    tables: _CfrTables,
    iteration: int,
) -> float:
    """Run one full CFR+ pass (recursive implementation, kept for reference).

    This is the original recursive implementation.  The default pass used by
    ``solve()`` is ``_cfr_pass_tabular``, which is ~30× faster.

    Args:
        initial_deals: List of (p_comp, d_comp, probability) from
                       _build_initial_deals().
        tables:        Mutable CFR+ accumulators (updated in-place).
        iteration:     Current iteration number (1-indexed).

    Returns:
        Probability-weighted average player EV for this iteration.
    """
    total_ev = 0.0
    memo: dict = {}  # Within-pass EV cache: reuse action EVs for repeated (p,d) states
    for p_comp, d_comp, prob in initial_deals:
        ev = _root_cfr(p_comp, d_comp, 1.0, 1.0, tables, iteration, memo)
        total_ev += prob * ev
    return total_ev


# ─── Per-type average strategy extraction ──────────────────────────────────────
# These replace the generic _extract_avg_strategy in solve(). They convert
# int-keyed internal tables back to NamedTuple+Enum keys for the public API.


def _extract_player_avg_strategy(sums_table: dict) -> PlayerStrategyDict:
    """Extract normalised average player strategy from int-keyed sums table."""
    result: PlayerStrategyDict = {}
    int_legal = [_P_HIT, _P_STAND]
    for (total, nc, is_soft_int), action_sums in sums_table.items():
        info_set = PlayerHitStandInfoSet(total=total, num_cards=nc, is_soft=bool(is_soft_int))
        total_w = sum(action_sums.get(a, 0.0) for a in int_legal)
        if total_w <= 0.0:
            prob = 1.0 / len(int_legal)
            result[info_set] = {_INT_TO_PLAYER_ACTION[a]: prob for a in int_legal}
        else:
            result[info_set] = {
                _INT_TO_PLAYER_ACTION[a]: action_sums.get(a, 0.0) / total_w for a in int_legal
            }
    return result


def _extract_surrender_avg_strategy(sums_table: dict) -> DealerSurrenderStrategyDict:
    """Extract normalised average dealer surrender strategy from int-keyed sums table."""
    result: DealerSurrenderStrategyDict = {}
    int_legal = [_D_SURRENDER, _D_CONTINUE]  # Only hard-15 creates surrender entries
    for (total, is_hard_int), action_sums in sums_table.items():
        info_set = DealerSurrenderInfoSet(total=total, is_hard_fifteen=bool(is_hard_int))
        total_w = sum(action_sums.get(a, 0.0) for a in int_legal)
        if total_w <= 0.0:
            prob = 1.0 / len(int_legal)
            result[info_set] = {_INT_TO_SURRENDER_ACTION[a]: prob for a in int_legal}
        else:
            result[info_set] = {
                _INT_TO_SURRENDER_ACTION[a]: action_sums.get(a, 0.0) / total_w for a in int_legal
            }
    return result


def _extract_dealer_action_avg_strategy(sums_table: dict) -> DealerActionStrategyDict:
    """Extract normalised average dealer action strategy from int-keyed sums table."""
    result: DealerActionStrategyDict = {}
    for (d_total, d_nc, is_soft_int, p_nc), action_sums in sums_table.items():
        info_set = DealerActionInfoSet(
            dealer_total=d_total, dealer_nc=d_nc, is_soft=bool(is_soft_int), player_nc=p_nc
        )
        int_legal = [_D_HIT, _D_STAND] if p_nc < 3 else [_D_HIT, _D_STAND, _D_REVEAL]
        total_w = sum(action_sums.get(a, 0.0) for a in int_legal)
        if total_w <= 0.0:
            prob = 1.0 / len(int_legal)
            result[info_set] = {_INT_TO_DEALER_ACTION[a]: prob for a in int_legal}
        else:
            result[info_set] = {
                _INT_TO_DEALER_ACTION[a]: action_sums.get(a, 0.0) / total_w for a in int_legal
            }
    return result


# ─── Numpy → strategy extraction helpers (C3c) ─────────────────────────────────


def _extract_player_avg_from_numpy(
    sums: np.ndarray,
    rev: dict,
) -> PlayerStrategyDict:
    """Convert numpy player strategy-sum array to PlayerStrategyDict."""
    sums_dict: dict = {}
    for slot in range(sums.shape[0]):
        key = rev[slot]
        sums_dict[key] = {_P_HIT: float(sums[slot, 0]), _P_STAND: float(sums[slot, 1])}
    return _extract_player_avg_strategy(sums_dict)


def _extract_surrender_avg_from_numpy(
    sums: np.ndarray,
    rev: dict,
) -> DealerSurrenderStrategyDict:
    """Convert numpy dealer-surrender strategy-sum array to DealerSurrenderStrategyDict."""
    sums_dict: dict = {}
    for slot in range(sums.shape[0]):
        key = rev[slot]
        sums_dict[key] = {_D_SURRENDER: float(sums[slot, 0]), _D_CONTINUE: float(sums[slot, 1])}
    return _extract_surrender_avg_strategy(sums_dict)


def _extract_dealer_action_avg_from_numpy(
    sums: np.ndarray,
    rev: dict,
) -> DealerActionStrategyDict:
    """Convert numpy dealer-action strategy-sum array to DealerActionStrategyDict."""
    sums_dict: dict = {}
    for slot in range(sums.shape[0]):
        key = rev[slot]  # (d_total, d_nc, is_soft_int, p_nc)
        p_nc = key[3]
        int_legal = [_D_HIT, _D_STAND] if p_nc < 3 else [_D_HIT, _D_STAND, _D_REVEAL]
        sums_dict[key] = {a: float(sums[slot, j]) for j, a in enumerate(int_legal)}
    return _extract_dealer_action_avg_strategy(sums_dict)


def solve(
    n_iterations: int = 1000,
    convergence_check_every: int = 100,
    exploitability_threshold: float = 0.01,
) -> CfrResult:
    """Run CFR+ and return Nash equilibrium strategies.

    Iterates CFR+ updates over all initial deal compositions. Periodically
    computes exploitability to check convergence.

    Args:
        n_iterations:            Number of CFR+ iterations to run.
        convergence_check_every: Check exploitability every N iterations.
        exploitability_threshold: Target exploitability (units/hand).
                                  Marks ``converged=True`` if reached.

    Returns:
        CfrResult with Nash strategies, exploitability, and nash_ev.

    Examples:
        >>> result = solve(n_iterations=500)
        >>> result.nash_ev < 0.1        # player doesn't have huge edge
        True
        >>> result.n_iterations
        500
    """
    initial_deals = _build_initial_deals()
    nash_ev = 0.0
    exploitability = float("inf")
    converged = False

    if _NUMBA_AVAILABLE:
        # ── Numba path (C3c) ─────────────────────────────────────────────────
        na = _get_or_build_numba_arrays()
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            n_player_slots,
            n_surr_slots,
            n_act_slots,
            player_rev,
            surr_rev,
            act_rev,
        ) = na

        nt = _NumbaTables(
            player_regrets=np.zeros((n_player_slots, 2)),
            player_strategy_sums=np.zeros((n_player_slots, 2)),
            surr_regrets=np.zeros((n_surr_slots, 2)),
            surr_strategy_sums=np.zeros((n_surr_slots, 2)),
            act_regrets=np.zeros((n_act_slots, 3)),
            act_strategy_sums=np.zeros((n_act_slots, 3)),
        )

        for iteration in range(1, n_iterations + 1):
            nash_ev = _cfr_pass_numba(na, nt, iteration)

            if iteration % convergence_check_every == 0:
                avg_player = _extract_player_avg_from_numpy(nt.player_strategy_sums, player_rev)
                avg_surrender = _extract_surrender_avg_from_numpy(nt.surr_strategy_sums, surr_rev)
                avg_action = _extract_dealer_action_avg_from_numpy(nt.act_strategy_sums, act_rev)
                exploitability = compute_exploitability(
                    avg_player, avg_surrender, avg_action, initial_deals
                )
                if exploitability < exploitability_threshold:
                    converged = True
                    break

        player_strategy = _extract_player_avg_from_numpy(nt.player_strategy_sums, player_rev)
        dealer_surrender_strategy = _extract_surrender_avg_from_numpy(
            nt.surr_strategy_sums, surr_rev
        )
        dealer_action_strategy = _extract_dealer_action_avg_from_numpy(
            nt.act_strategy_sums, act_rev
        )

    else:
        # ── Pure-Python tabular fallback ─────────────────────────────────────
        tables = _CfrTables()

        for iteration in range(1, n_iterations + 1):
            nash_ev = _cfr_pass_tabular(tables, iteration)

            if iteration % convergence_check_every == 0:
                avg_player = _extract_player_avg_strategy(tables.player_strategy_sums)
                avg_surrender = _extract_surrender_avg_strategy(
                    tables.dealer_surrender_strategy_sums
                )
                avg_action = _extract_dealer_action_avg_strategy(tables.dealer_action_strategy_sums)
                exploitability = compute_exploitability(
                    avg_player, avg_surrender, avg_action, initial_deals
                )
                if exploitability < exploitability_threshold:
                    converged = True
                    break

        player_strategy = _extract_player_avg_strategy(tables.player_strategy_sums)
        dealer_surrender_strategy = _extract_surrender_avg_strategy(
            tables.dealer_surrender_strategy_sums
        )
        dealer_action_strategy = _extract_dealer_action_avg_strategy(
            tables.dealer_action_strategy_sums
        )

    return CfrResult(
        player_strategy=player_strategy,
        dealer_surrender_strategy=dealer_surrender_strategy,
        dealer_action_strategy=dealer_action_strategy,
        n_iterations=n_iterations,
        exploitability=exploitability,
        converged=converged,
        nash_ev=nash_ev,
    )


# ─── Best-response and exploitability ──────────────────────────────────────────


def _br_dealer_phase(
    p_nat: int,
    p_na: int,
    p_nc: int,
    d_nat: int,
    d_na: int,
    d_nc: int,
    dealer_action_strategy: DealerActionStrategyDict,
) -> float:
    """Dealer phase EV against a fixed dealer action strategy (for best-response).

    Args:
        p_nat, p_na, p_nc: Player's final composition.
        d_nat, d_na, d_nc: Dealer's current composition.
        dealer_action_strategy: Fixed dealer strategy at 16/17.

    Returns:
        Expected player EV.
    """
    d_total = _total_from_composition(d_nat, d_na, d_nc)

    if d_total > 21:
        return _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=True)

    if d_total >= 18:
        return _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False)

    if d_total < 16:
        ev = 0.0
        for (nd_nat, nd_na, nd_nc), prob in _forced_hit_dist(d_nat, d_na, d_nc).items():
            ev += prob * _br_dealer_phase(
                p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc, dealer_action_strategy
            )
        return ev

    # At 16 or 17: use fixed dealer action strategy
    d_soft = _is_soft_from_composition(d_nat, d_na, d_nc)
    info_set = DealerActionInfoSet(
        dealer_total=d_total, dealer_nc=d_nc, is_soft=d_soft, player_nc=p_nc
    )
    legal_actions = get_legal_dealer_actions(info_set)

    if info_set in dealer_action_strategy:
        strategy = dealer_action_strategy[info_set]
    else:
        p_uniform = 1.0 / len(legal_actions)
        strategy = dict.fromkeys(legal_actions, p_uniform)

    ev = 0.0
    for a in legal_actions:
        prob = strategy.get(a, 0.0)
        if prob <= 0.0:
            continue
        if a in (DealerAction.STAND, DealerAction.REVEAL_PLAYER):
            ev += prob * _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False)
        else:
            hit_ev = 0.0
            for rank in range(NUM_RANKS):
                nd_nat, nd_na, nd_nc = _transition(d_nat, d_na, d_nc, rank)
                hit_ev += RANK_PROB * _br_dealer_phase(
                    p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc, dealer_action_strategy
                )
            ev += prob * hit_ev
    return ev


def _best_player_ev(
    dealer_surrender_strategy: DealerSurrenderStrategyDict,
    dealer_action_strategy: DealerActionStrategyDict,
    initial_deals: list[tuple[Comp, Comp, float]],
) -> float:
    """Compute best player EV against fixed dealer average strategies.

    Player makes optimal hit/stand decisions via backward induction.
    Dealer follows the provided fixed strategies (mixed).

    Args:
        dealer_surrender_strategy: Fixed dealer surrender strategy.
        dealer_action_strategy:    Fixed dealer action strategy at 16/17.
        initial_deals:             Precomputed initial deal list.

    Returns:
        Best player EV (player's perspective), averaged over initial deals.
    """
    memo: dict = {}

    def player_ev(p_nat: int, p_na: int, p_nc: int, d_nat: int, d_na: int, d_nc: int) -> float:
        key = (p_nat, p_na, p_nc, d_nat, d_na, d_nc)
        if key in memo:
            return memo[key]
        p_total = _total_from_composition(p_nat, p_na, p_nc)
        if p_total > 21:
            result = -2.0 if p_nc == 5 else -1.0
            memo[key] = result
            return result
        if p_total == 21 or p_nc == 5:
            result = _br_dealer_phase(p_nat, p_na, p_nc, d_nat, d_na, d_nc, dealer_action_strategy)
            memo[key] = result
            return result
        # Player picks optimally
        ev_stand = _br_dealer_phase(p_nat, p_na, p_nc, d_nat, d_na, d_nc, dealer_action_strategy)
        ev_hit = 0.0
        for rank in range(NUM_RANKS):
            np_nat, np_na, np_nc = _transition(p_nat, p_na, p_nc, rank)
            ev_hit += RANK_PROB * player_ev(np_nat, np_na, np_nc, d_nat, d_na, d_nc)
        result = max(ev_stand, ev_hit)
        memo[key] = result
        return result

    total_ev = 0.0
    for p_comp, d_comp, prob in initial_deals:
        p_nat, p_na, p_nc = p_comp
        d_nat, d_na, d_nc = d_comp
        d_total = _total_from_composition(d_nat, d_na, d_nc)

        # Dealer surrender check
        is_hard_15 = _is_hard_fifteen_comp(d_nat, d_na, d_nc)
        surrender_info = DealerSurrenderInfoSet(total=d_total, is_hard_fifteen=is_hard_15)
        surrender_legal = get_legal_dealer_surrender_actions(surrender_info)

        if DealerAction.SURRENDER in surrender_legal:
            if surrender_info in dealer_surrender_strategy:
                s = dealer_surrender_strategy[surrender_info]
            else:
                s = {a: 1.0 / len(surrender_legal) for a in surrender_legal}
            p_surr = s.get(DealerAction.SURRENDER, 0.0)
            p_cont = 1.0 - p_surr
            ev = p_surr * 0.0  # push
            if p_cont > 0.0:
                if _is_player_ban_ban(p_nat, p_na, p_nc):
                    after_ev = _settle_player_ban_ban(d_nat, d_na, d_nc)
                elif _is_player_ban_luck(p_nat, p_na, p_nc):
                    after_ev = _settle_player_ban_luck(d_nat, d_na, d_nc)
                else:
                    after_ev = player_ev(p_nat, p_na, p_nc, d_nat, d_na, d_nc)
                ev += p_cont * after_ev
        else:
            if _is_player_ban_ban(p_nat, p_na, p_nc):
                ev = _settle_player_ban_ban(d_nat, d_na, d_nc)
            elif _is_player_ban_luck(p_nat, p_na, p_nc):
                ev = _settle_player_ban_luck(d_nat, d_na, d_nc)
            else:
                ev = player_ev(p_nat, p_na, p_nc, d_nat, d_na, d_nc)

        total_ev += prob * ev
    return total_ev


def _best_dealer_ev(
    player_strategy: PlayerStrategyDict,
    initial_deals: list[tuple[Comp, Comp, float]],
) -> float:
    """Compute minimum player EV when dealer plays best response to fixed player.

    Dealer makes optimal choices (minimising player EV) via backward induction.
    Player follows the provided fixed strategy.

    Args:
        player_strategy: Fixed player strategy.
        initial_deals:   Precomputed initial deal list.

    Returns:
        Minimum player EV (= best dealer response utility for dealer).
    """
    # We compute this by running the game tree with:
    # - Player following player_strategy
    # - Dealer making optimal choices (min player EV)
    memo_player: dict = {}
    memo_dealer: dict = {}

    def player_done_ev(p_nat: int, p_na: int, p_nc: int, d_nat: int, d_na: int, d_nc: int) -> float:
        """EV with player done at p_comp, dealer making optimal decisions."""
        key = (p_nat, p_na, p_nc, d_nat, d_na, d_nc)
        if key in memo_dealer:
            return memo_dealer[key]

        d_total = _total_from_composition(d_nat, d_na, d_nc)

        if d_total > 21:
            result = _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=True)
            memo_dealer[key] = result
            return result

        if d_total >= 18:
            result = _cfr_settle(p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False)
            memo_dealer[key] = result
            return result

        if d_total < 16:
            ev = 0.0
            for (nd_nat, nd_na, nd_nc), prob in _forced_hit_dist(d_nat, d_na, d_nc).items():
                ev += prob * player_done_ev(p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc)
            memo_dealer[key] = ev
            return ev

        # At 16/17: dealer picks action that MINIMISES player EV
        d_soft = _is_soft_from_composition(d_nat, d_na, d_nc)
        info_set = DealerActionInfoSet(
            dealer_total=d_total, dealer_nc=d_nc, is_soft=d_soft, player_nc=p_nc
        )
        legal_actions = get_legal_dealer_actions(info_set)

        action_evs_local: dict = {}
        for a in legal_actions:
            if a in (DealerAction.STAND, DealerAction.REVEAL_PLAYER):
                action_evs_local[a] = _cfr_settle(
                    p_nat, p_na, p_nc, d_nat, d_na, d_nc, d_busted=False
                )
            else:
                hit_ev = 0.0
                for rank in range(NUM_RANKS):
                    nd_nat, nd_na, nd_nc = _transition(d_nat, d_na, d_nc, rank)
                    hit_ev += RANK_PROB * player_done_ev(p_nat, p_na, p_nc, nd_nat, nd_na, nd_nc)
                action_evs_local[a] = hit_ev

        # Dealer is minimiser: pick the action with lowest player EV
        result = min(action_evs_local.values())
        memo_dealer[key] = result
        return result

    def player_phase_ev(
        p_nat: int, p_na: int, p_nc: int, d_nat: int, d_na: int, d_nc: int
    ) -> float:
        """Player EV when player follows fixed strategy."""
        key = (p_nat, p_na, p_nc, d_nat, d_na, d_nc)
        if key in memo_player:
            return memo_player[key]

        p_total = _total_from_composition(p_nat, p_na, p_nc)

        if p_total > 21:
            result = -2.0 if p_nc == 5 else -1.0
            memo_player[key] = result
            return result

        if p_total == 21 or p_nc == 5:
            result = player_done_ev(p_nat, p_na, p_nc, d_nat, d_na, d_nc)
            memo_player[key] = result
            return result

        p_soft = _is_soft_from_composition(p_nat, p_na, p_nc)
        info_set = PlayerHitStandInfoSet(total=p_total, num_cards=p_nc, is_soft=p_soft)
        legal_actions = get_legal_player_actions(info_set)

        if info_set in player_strategy:
            strategy = player_strategy[info_set]
        else:
            p_uniform = 1.0 / len(legal_actions)
            strategy = dict.fromkeys(legal_actions, p_uniform)

        ev = 0.0
        for a in legal_actions:
            prob = strategy.get(a, 0.0)
            if prob <= 0.0:
                continue
            if a == PlayerAction.STAND:
                ev += prob * player_done_ev(p_nat, p_na, p_nc, d_nat, d_na, d_nc)
            else:
                hit_ev = 0.0
                for rank in range(NUM_RANKS):
                    np_nat, np_na, np_nc = _transition(p_nat, p_na, p_nc, rank)
                    hit_ev += RANK_PROB * player_phase_ev(np_nat, np_na, np_nc, d_nat, d_na, d_nc)
                ev += prob * hit_ev

        memo_player[key] = ev
        return ev

    total_ev = 0.0
    for p_comp, d_comp, prob in initial_deals:
        p_nat, p_na, p_nc = p_comp
        d_nat, d_na, d_nc = d_comp
        d_total = _total_from_composition(d_nat, d_na, d_nc)

        # Dealer surrender: dealer picks optimally (minimise player EV)
        is_hard_15 = _is_hard_fifteen_comp(d_nat, d_na, d_nc)
        surrender_info = DealerSurrenderInfoSet(total=d_total, is_hard_fifteen=is_hard_15)
        surrender_legal = get_legal_dealer_surrender_actions(surrender_info)

        if DealerAction.SURRENDER in surrender_legal:
            # Dealer surrender option: check both actions, pick min player EV
            if _is_player_ban_ban(p_nat, p_na, p_nc):
                ev_no_surr = _settle_player_ban_ban(d_nat, d_na, d_nc)
            elif _is_player_ban_luck(p_nat, p_na, p_nc):
                ev_no_surr = _settle_player_ban_luck(d_nat, d_na, d_nc)
            else:
                ev_no_surr = player_phase_ev(p_nat, p_na, p_nc, d_nat, d_na, d_nc)
            # Surrender EV = 0.0 (push)
            ev = min(0.0, ev_no_surr)
        else:
            if _is_player_ban_ban(p_nat, p_na, p_nc):
                ev = _settle_player_ban_ban(d_nat, d_na, d_nc)
            elif _is_player_ban_luck(p_nat, p_na, p_nc):
                ev = _settle_player_ban_luck(d_nat, d_na, d_nc)
            else:
                ev = player_phase_ev(p_nat, p_na, p_nc, d_nat, d_na, d_nc)

        total_ev += prob * ev
    return total_ev


def compute_exploitability(
    player_strategy: PlayerStrategyDict,
    dealer_surrender_strategy: DealerSurrenderStrategyDict,
    dealer_action_strategy: DealerActionStrategyDict,
    initial_deals: list[tuple[Comp, Comp, float]] | None = None,
) -> float:
    """Compute the total exploitability of the given strategy profile.

    Exploitability = (best_player_ev - nash_ev) + (nash_ev - worst_player_ev)
                   = best_player_ev - worst_player_ev

    Where:
      - best_player_ev   = max player EV against fixed dealer strategy
      - worst_player_ev  = min player EV when dealer plays best response

    At Nash equilibrium, exploitability = 0.

    Args:
        player_strategy:           Player's strategy profile.
        dealer_surrender_strategy: Dealer's surrender strategy.
        dealer_action_strategy:    Dealer's action strategy at 16/17.
        initial_deals:             Precomputed deal list (recomputed if None).

    Returns:
        Total exploitability in units per hand (non-negative).

    Examples:
        >>> result = solve(n_iterations=500)
        >>> eps = compute_exploitability(
        ...     result.player_strategy,
        ...     result.dealer_surrender_strategy,
        ...     result.dealer_action_strategy,
        ... )
        >>> eps >= 0
        True
    """
    if initial_deals is None:
        initial_deals = _build_initial_deals()

    best_player = _best_player_ev(dealer_surrender_strategy, dealer_action_strategy, initial_deals)
    worst_player = _best_dealer_ev(player_strategy, initial_deals)

    return max(0.0, best_player - worst_player)


# ─── Public strategy helpers ───────────────────────────────────────────────────


def get_player_action(
    result: CfrResult,
    total: int,
    num_cards: int,
    is_soft: bool,
) -> PlayerAction:
    """Look up the Nash equilibrium player action for a given hand state.

    Args:
        result:    CfrResult from solve().
        total:     Player's best hand total.
        num_cards: Number of cards in hand.
        is_soft:   True if hand contains an ace counted at its high value.

    Returns:
        Recommended PlayerAction (HIT or STAND).
        Returns STAND if the state is not in the strategy table.
    """
    info_set = PlayerHitStandInfoSet(total=total, num_cards=num_cards, is_soft=is_soft)
    if info_set not in result.player_strategy:
        return PlayerAction.STAND
    strategy = result.player_strategy[info_set]
    return max(strategy, key=strategy.__getitem__)


def get_dealer_surrender_prob(result: CfrResult) -> float:
    """Return the Nash equilibrium surrender probability for dealer hard 15.

    Args:
        result: CfrResult from solve().

    Returns:
        Probability that dealer surrenders on hard 15. Returns 0.0 if
        the hard-15 info set has no accumulated strategy.
    """
    info_set = DealerSurrenderInfoSet(total=15, is_hard_fifteen=True)
    if info_set not in result.dealer_surrender_strategy:
        return 0.0
    strategy = result.dealer_surrender_strategy[info_set]
    return strategy.get(DealerAction.SURRENDER, 0.0)
