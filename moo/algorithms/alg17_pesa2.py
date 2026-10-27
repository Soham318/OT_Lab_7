"""
alg17_pesa2.py: PESA-II (Pareto Envelope-based Selection Algorithm II)

Lecture Family: Modern Elitist Multiobjective Evolutionary Algorithms (Part 3 & Bonus Module 5E, Chapter 5)
Original Citation: Corne, D. W., Jerram, N. R., Knowles, J. D., & Oates, M. J. (2001).
"PESA-II: Region-based Selection in Evolutionary Multiobjective Optimization."
Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2001), pp. 283-290. Morgan Kaufmann.

Algorithmic Logic:
1. Architecture:
       - Internal population P of size N_I (used for offspring generation).
       - External archive (Pareto envelope) of maximum capacity N_E.
2. Adaptive Hypergrid & Squeeze Factor:
       - The objective space is partitioned into K^M hyperboxes (cells), where K is grid divisions per dimension.
       - The cell coordinate for solution i is:
             c_m(i) = floor( K * (f_m(i) - min_m) / (max_m - min_m + eps) ).
       - Squeeze factor of a hyperbox: The count of solutions currently occupying that hyperbox.
3. Region-Based Selection (Distinguishing Hallmark of PESA-II):
       - Instead of selecting individuals, PESA-II selects regions (hyperboxes)!
       - Two non-empty hyperboxes are picked at random for a binary tournament.
       - The hyperbox with the LOWER squeeze factor (less crowded) wins.
       - One individual is selected uniformly at random from the winning hyperbox.
4. Genetic Variation & Archive Update:
       - Selected parents generate offspring via crossover (SBX/OX) and mutation (PM/Swap).
       - Each offspring is evaluated and tested against the external archive:
             If dominated by any archive member, discard.
             Otherwise, insert into archive and remove all archive members dominated by the offspring.
             If archive exceeds capacity N_E, remove an individual from the hyperbox with the HIGHEST squeeze factor.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import constrained_dominates, dominates
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


class AdaptiveHypergrid:
    """Manages hypergrid cell assignment and squeeze factors for PESA-II archive."""

    def __init__(self, n_divs: int = 10) -> None:
        self.n_divs = n_divs

    def compute_cells(self, F: np.ndarray) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
        """Maps each solution to a 1D grid cell index.

        Returns:
            cell_indices: shape (len(F),) array of integer cell IDs.
            cells_to_members: dict mapping cell_id -> list of solution indices.
            squeeze_factors: shape (len(F),) squeeze factor for each solution.
        """
        N, M = F.shape
        if N == 0:
            return np.empty(0, dtype=int), {}, np.empty(0, dtype=int)

        lb = np.min(F, axis=0)
        ub = np.max(F, axis=0)
        span = np.maximum(1e-12, ub - lb)

        # Coordinate in [0, n_divs - 1] along each axis
        coords = np.floor(self.n_divs * (F - lb) / span).astype(int)
        coords = np.clip(coords, 0, self.n_divs - 1)

        # Flatten multi-dimensional coordinates to a unique integer cell ID
        multipliers = (self.n_divs ** np.arange(M))
        cell_ids = np.dot(coords, multipliers)

        cells_to_members: dict[int, list[int]] = {}
        for idx, cid in enumerate(cell_ids):
            if cid not in cells_to_members:
                cells_to_members[cid] = []
            cells_to_members[cid].append(idx)

        squeeze_factors = np.zeros(N, dtype=int)
        for idx, cid in enumerate(cell_ids):
            squeeze_factors[idx] = len(cells_to_members[cid])

        return cell_ids, cells_to_members, squeeze_factors


def region_based_selection(
    cells_to_members: dict[int, list[int]],
    rng: np.random.Generator,
) -> int:
    """PESA-II Region-based binary tournament selection:

    Picks two non-empty hyperboxes; the one with smaller squeeze factor wins.
    Returns the index of a randomly chosen member from the winning hyperbox.
    """
    non_empty_cids = list(cells_to_members.keys())
    if len(non_empty_cids) == 0:
        return 0
    if len(non_empty_cids) == 1:
        return int(rng.choice(cells_to_members[non_empty_cids[0]]))

    # Binary tournament between two hyperboxes
    cid1, cid2 = rng.choice(non_empty_cids, size=2, replace=False)
    sq1 = len(cells_to_members[cid1])
    sq2 = len(cells_to_members[cid2])

    if sq1 < sq2:
        winner_cid = cid1
    elif sq2 < sq1:
        winner_cid = cid2
    else:
        winner_cid = cid1 if rng.random() < 0.5 else cid2

    return int(rng.choice(cells_to_members[winner_cid]))


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 50,
    archive_size: int = 100,
    n_divs: int = 10,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes PESA-II internal population and external archive."""
    rng = np.random.default_rng(seed)
    effective_pop_size = max(1, min(pop_size, budget))
    pop = random_population(problem, effective_pop_size, rng)

    F_list, CV_list = [], []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    # Initialize archive with non-dominated individuals from initial population
    arch_X, arch_F, arch_CV = extract_non_dominated(pop, F, CV, feas_tol=problem.feas_tol)
    if len(arch_X) > archive_size:
        keep = rng.choice(len(arch_X), size=archive_size, replace=False)
        arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop_size,
        "archive_size": archive_size,
        "n_divs": n_divs,
        "grid": AdaptiveHypergrid(n_divs=n_divs),
        "X": pop,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_pop_size,
        "budget": budget,
        "iter": 0,
    }
    return state


def step(
    state: dict,
    problem: Problem,
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
) -> dict:
    """Executes one generation of PESA-II."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N_I = state["pop_size"]
    N_E = state["archive_size"]
    feas_tol = problem.feas_tol

    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    if len(arch_X) < 2:
        return state

    # 1. Compute grid cells and squeeze factors on current archive
    grid = state["grid"]
    cell_ids, cells_to_members, squeeze_factors = grid.compute_cells(arch_F)

    # 2. Region-based selection for mating pool
    mating_pool_indices = []
    for _ in range(N_I):
        p_idx = region_based_selection(cells_to_members, rng)
        mating_pool_indices.append(p_idx)

    mating_pool = arch_X[mating_pool_indices]

    # 3. Variation (SBX + PM)
    offspring_list = []
    is_perm = (problem.var_type == "permutation")

    for i in range(0, N_I, 2):
        p1 = mating_pool[i]
        p2 = mating_pool[(i + 1) % N_I]

        if is_perm:
            c1, c2 = order_crossover(p1, p2, rng)
            c1 = swap_mutation(c1, rng)
            c2 = swap_mutation(c2, rng)
        else:
            c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, prob_c=prob_c, rng=rng)
            c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m, rng=rng)
            c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=eta_m, rng=rng)

            if problem.var_type == "integer":
                c1 = np.round(c1)
                c2 = np.round(c2)
            elif problem.var_type == "binary":
                c1 = (c1 >= 0.5).astype(float)
                c2 = (c2 >= 0.5).astype(float)

        offspring_list.append(c1)
        if len(offspring_list) < N_I:
            offspring_list.append(c2)

    n_to_eval = min(len(offspring_list), min(N_I, rem_budget))
    X_off = np.array(offspring_list[:n_to_eval])

    # 4. Evaluate offspring
    F_off_list, CV_off_list = [], []
    for ind in X_off:
        f, cv, _ = problem.evaluate(ind)
        F_off_list.append(f)
        CV_off_list.append(cv)

    F_off = np.array(F_off_list, dtype=float)
    CV_off = np.array(CV_off_list, dtype=float)

    state["n_evals"] += n_to_eval
    state["iter"] += 1

    # 5. Archive update: incorporate each offspring into Pareto envelope
    cur_X = list(arch_X)
    cur_F = list(arch_F)
    cur_CV = list(arch_CV)

    for cand_idx in range(len(X_off)):
        x_c = X_off[cand_idx]
        f_c = F_off[cand_idx]
        cv_c = CV_off[cand_idx]

        # Check if candidate is dominated by any archive member
        dominated_by_arch = False
        dominated_in_arch: list[int] = []

        for a_idx in range(len(cur_X)):
            if constrained_dominates(cur_F[a_idx], cur_CV[a_idx], f_c, cv_c, feas_tol=feas_tol):
                dominated_by_arch = True
                break
            if constrained_dominates(f_c, cv_c, cur_F[a_idx], cur_CV[a_idx], feas_tol=feas_tol):
                dominated_in_arch.append(a_idx)

        if not dominated_by_arch:
            # Remove solutions dominated by new candidate
            for rem_idx in sorted(dominated_in_arch, reverse=True):
                cur_X.pop(rem_idx)
                cur_F.pop(rem_idx)
                cur_CV.pop(rem_idx)

            cur_X.append(x_c)
            cur_F.append(f_c)
            cur_CV.append(cv_c)

            # If capacity exceeded, remove solution from the most crowded hyperbox
            if len(cur_X) > N_E:
                tmp_F = np.array(cur_F)
                _, cells_map, _ = grid.compute_cells(tmp_F)
                max_sq = max(len(members) for members in cells_map.values())
                crowded_cids = [cid for cid, members in cells_map.items() if len(members) == max_sq]
                target_cid = int(rng.choice(crowded_cids))
                drop_idx = int(rng.choice(cells_map[target_cid]))

                cur_X.pop(drop_idx)
                cur_F.pop(drop_idx)
                cur_CV.pop(drop_idx)

    state["arch_X"] = np.array(cur_X)
    state["arch_F"] = np.array(cur_F)
    state["arch_CV"] = np.array(cur_CV)
    state["X"] = X_off
    state["F"] = F_off
    state["CV"] = CV_off

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    pop_size: int = 50,
    archive_size: int = 100,
    n_divs: int = 10,
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    **kwargs,
) -> dict:
    """Executes the full PESA-II optimization run up to the evaluation budget."""
    state = initialize(
        problem,
        seed=seed,
        pop_size=pop_size,
        archive_size=archive_size,
        n_divs=n_divs,
        budget=budget,
        **kwargs,
    )
    state["budget"] = budget

    while state["n_evals"] < budget and len(state["arch_X"]) >= 2:
        prev_evals = state["n_evals"]
        step(state, problem, prob_c=prob_c, eta_c=eta_c, eta_m=eta_m)
        if state["n_evals"] == prev_evals:
            break

    X_opt, F_opt, CV_opt = extract_non_dominated(
        state["arch_X"], state["arch_F"], state["arch_CV"], feas_tol=problem.feas_tol
    )
    F_opt_nat = problem.to_natural(F_opt)

    return {
        "X_opt": X_opt,
        "F_opt": F_opt,
        "F_opt_nat": F_opt_nat,
        "CV_opt": CV_opt,
        "n_evals": state["n_evals"],
        "state": state,
    }
