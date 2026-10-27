"""
alg35_r_nsga2.py: R-NSGA-II (Reference Point-Based NSGA-II)

Lecture Family: Many-Objective, Indicator, Preference & Gradient Methods (Bonus Module 5E, Chapter 5)
Original Citation: Deb, K., & Sundar, J. (2006).
"Reference Point Based Multi-Objective Optimization Using Evolutionary Algorithms."
Proceedings of the 8th Annual Conference on Genetic and Evolutionary Computation (GECCO '06), 635-642.

Algorithmic Logic:
1. Decision Maker Reference Points:
       Set of aspiration reference points bar{Z} = {z_bar_1, ..., z_bar_P} in objective space.
2. Normalized Euclidean Preference Distance:
       d(x, z_bar_p) = sqrt( sum_{m=1}^M ((f_m(x) - z_bar_p,m) / (z_m^nad - z_m^* + 1e-12))^2 )
       Overall preference distance: D(x) = min_p d(x, z_bar_p).
3. Non-Dominated Sorting:
       Partition pool R_t = P_t U Q_t into Pareto fronts F_1, F_2, ...
4. Preference-Guided Comparison & epsilon-Clearing:
       For candidates on the same front:
       - If |D(x_1) - D(x_2)| > epsilon_clear:
             The solution closer to a reference point (smaller D(x)) is preferred.
       - Else (|D(x_1) - D(x_2)| <= epsilon_clear):
             Standard crowding distance tie-breaker is used to preserve diversity around the ROI.
5. Focused Environmental Selection:
       Fills population prioritizing solutions in the Region of Interest (ROI) around reference points.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, crowding_distance
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def compute_preference_distances(
    F: np.ndarray,
    ref_points: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> np.ndarray:
    """Computes minimum normalized Euclidean distance D(x) to any supplied reference point."""
    N, M = F.shape
    P = len(ref_points)

    scale = nadir - ideal
    scale = np.where(scale < 1e-12, 1.0, scale)

    F_norm = (F - ideal.reshape(1, -1)) / scale.reshape(1, -1)
    ref_norm = (ref_points - ideal.reshape(1, -1)) / scale.reshape(1, -1)

    # Distances from each solution to each reference point: (N, P)
    diff = F_norm[:, None, :] - ref_norm[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Minimum distance across all reference points
    D = np.min(dist_matrix, axis=1)
    return D


def preference_tournament_select(
    i: int,
    j: int,
    ranks: np.ndarray,
    pref_dist: np.ndarray,
    crowd_dist: np.ndarray,
    eps_clear: float = 0.05,
    rng: np.random.Generator | None = None,
) -> int:
    """Preference-guided binary tournament selection."""
    if rng is None:
        rng = np.random.default_rng()

    # Lower rank wins
    if ranks[i] < ranks[j]:
        return i
    if ranks[j] < ranks[i]:
        return j

    # Same rank: check preference distance difference
    d_diff = abs(pref_dist[i] - pref_dist[j])
    if d_diff > eps_clear:
        # Significantly closer to reference point wins
        return i if pref_dist[i] < pref_dist[j] else j

    # Within clearing threshold: larger crowding distance wins for diversity
    if crowd_dist[i] > crowd_dist[j]:
        return i
    if crowd_dist[j] > crowd_dist[i]:
        return j

    return i if rng.random() < 0.5 else j


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes R-NSGA-II population, reference points, and evaluations within budget."""
    rng = np.random.default_rng(seed)
    pop_size = kwargs.get("pop_size", 50)
    budget = kwargs.get("budget", 10000)
    eps_clear = kwargs.get("eps_clear", 0.05)

    # Reference points
    ref_points = kwargs.get("ref_points", None)
    if ref_points is None:
        # Default to problem reference point or center
        if hasattr(problem, "ref_point"):
            ref_points = np.array([problem.ref_point])
        else:
            ref_points = np.array([0.5 * (problem.ideal_point + problem.nadir_point)])
    else:
        ref_points = np.asarray(ref_points, dtype=float)
        if ref_points.ndim == 1:
            ref_points = ref_points.reshape(1, -1)

    effective_pop = max(2, min(pop_size, budget))
    pop = random_population(problem, effective_pop, rng)

    F_list = []
    CV_list = []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    ideal = np.min(F, axis=0)
    nadir = np.max(F, axis=0)
    pref_dist = compute_preference_distances(F, ref_points, ideal, nadir)

    fronts = fast_non_dominated_sort(F, CV=CV, use_deb=True, feas_tol=problem.feas_tol)
    ranks = np.zeros(effective_pop, dtype=int)
    crowd_dist = np.zeros(effective_pop, dtype=float)
    for r_idx, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = r_idx
        cd = crowding_distance(F, front)
        for i, idx in enumerate(front):
            crowd_dist[idx] = cd[i]

    arch_X, arch_F, arch_CV = extract_non_dominated(pop, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop,
        "budget": budget,
        "ref_points": ref_points,
        "eps_clear": eps_clear,
        "ideal": ideal,
        "nadir": nadir,
        "X": pop,
        "F": F,
        "CV": CV,
        "ranks": ranks,
        "pref_dist": pref_dist,
        "crowd_dist": crowd_dist,
        "fronts": fronts,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "archive_X": arch_X,
        "archive_F": arch_F,
        "archive_CV": arch_CV,
        "n_evals": effective_pop,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one generation of R-NSGA-II."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    feas_tol = problem.feas_tol
    ref_points = state["ref_points"]
    eps_clear = state["eps_clear"]

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    ranks = state["ranks"]
    pref_dist = state["pref_dist"]
    crowd_dist = state["crowd_dist"]

    # 1. Preference tournament selection
    mating_indices = []
    for _ in range(N):
        i, j = rng.choice(N, size=2, replace=False)
        w = preference_tournament_select(i, j, ranks, pref_dist, crowd_dist, eps_clear=eps_clear, rng=rng)
        mating_indices.append(w)

    mating_pool = X[mating_indices]

    # 2. Variation
    offspring = []
    is_perm = (problem.var_type == "permutation")

    for i in range(0, N, 2):
        p1 = mating_pool[i]
        p2 = mating_pool[(i + 1) % N]

        if is_perm:
            c1, c2 = order_crossover(p1, p2, rng)
            c1 = swap_mutation(c1, rng)
            c2 = swap_mutation(c2, rng)
        else:
            c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=15.0, prob_c=0.9, rng=rng)
            c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=20.0, rng=rng)
            c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=20.0, rng=rng)
            if problem.var_type == "integer":
                c1 = np.round(c1)
                c2 = np.round(c2)
            elif problem.var_type == "binary":
                c1 = (c1 >= 0.5).astype(float)
                c2 = (c2 >= 0.5).astype(float)

        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    n_to_eval = min(len(offspring), min(N, rem_budget))
    X_off = np.array(offspring[:n_to_eval])

    F_off_list = []
    CV_off_list = []
    for ind in X_off:
        f, cv, _ = problem.evaluate(ind)
        F_off_list.append(f)
        CV_off_list.append(cv)

    F_off = np.array(F_off_list, dtype=float)
    CV_off = np.array(CV_off_list, dtype=float)

    state["n_evals"] += n_to_eval
    state["iter"] += 1

    # 3. Combine pool
    X_comb = np.vstack((X, X_off))
    F_comb = np.vstack((F, F_off))
    CV_comb = np.concatenate((CV, CV_off))

    ideal = np.min(F_comb, axis=0)
    nadir = np.max(F_comb, axis=0)
    state["ideal"] = ideal
    state["nadir"] = nadir

    pref_dist_comb = compute_preference_distances(F_comb, ref_points, ideal, nadir)

    # 4. Environmental Selection
    fronts = fast_non_dominated_sort(F_comb, CV=CV_comb, use_deb=True, feas_tol=feas_tol)

    selected_indices = []
    for front in fronts:
        if len(selected_indices) + len(front) <= N:
            selected_indices.extend(front)
        else:
            needed = N - len(selected_indices)
            cd_front = crowding_distance(F_comb, front)
            p_front = pref_dist_comb[front]

            # Sort borderline front by preference distance, using clearing distance
            order = sorted(
                range(len(front)),
                key=lambda idx: (p_front[idx] // eps_clear, -cd_front[idx])
            )
            chosen = [front[idx] for idx in order[:needed]]
            selected_indices.extend(chosen)
            break

    sel = np.array(selected_indices, dtype=int)
    X_new = X_comb[sel]
    F_new = F_comb[sel]
    CV_new = CV_comb[sel]

    # Re-compute ranks and crowding on selected population
    new_fronts = fast_non_dominated_sort(F_new, CV=CV_new, use_deb=True, feas_tol=feas_tol)
    new_ranks = np.zeros(N, dtype=int)
    new_cd = np.zeros(N, dtype=float)
    for r_idx, front in enumerate(new_fronts):
        for idx in front:
            new_ranks[idx] = r_idx
        cd = crowding_distance(F_new, front)
        for i, idx in enumerate(front):
            new_cd[idx] = cd[i]

    new_p_dist = compute_preference_distances(F_new, ref_points, ideal, nadir)

    # 5. Update Archive
    comb_arch_X = np.vstack((state["archive_X"], X_off))
    comb_arch_F = np.vstack((state["archive_F"], F_off))
    comb_arch_CV = np.concatenate((state["archive_CV"], CV_off))

    arch_X, arch_F, arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )
    if len(arch_X) > 200:
        cd = crowding_distance(arch_F, list(range(len(arch_F))))
        order = np.argsort(-cd)
        keep = order[:200]
        arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

    state["X"] = X_new
    state["F"] = F_new
    state["CV"] = CV_new
    state["ranks"] = new_ranks
    state["crowd_dist"] = new_cd
    state["pref_dist"] = new_p_dist
    state["fronts"] = new_fronts
    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs R-NSGA-II until budget is exhausted."""
    state = initialize(problem, seed=seed, budget=budget, **kwargs)

    while state["n_evals"] < budget:
        state = step(state, problem)

    arch_X = state["archive_X"]
    arch_F = state["archive_F"]
    arch_CV = state["archive_CV"]

    is_feas = (arch_CV <= problem.feas_tol)
    if np.any(is_feas):
        opt_X = arch_X[is_feas]
        opt_F = arch_F[is_feas]
        opt_CV = arch_CV[is_feas]
    else:
        opt_X = arch_X
        opt_F = arch_F
        opt_CV = arch_CV

    return {
        "X_opt": opt_X,
        "F_opt": opt_F,
        "CV_opt": opt_CV,
        "n_evals": state["n_evals"],
        "history": [],
        "state": state,
    }
