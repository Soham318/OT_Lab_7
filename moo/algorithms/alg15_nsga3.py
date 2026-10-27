"""
alg15_nsga3.py: NSGA-III (Reference-Point Based Non-Dominated Sorting GA for Many-Objective Optimization)

Lecture Family: Modern Elitist Multiobjective Evolutionary Algorithms (Part 3 & Bonus Module 5E, Chapter 5)
Original Citation: Deb, K., & Jain, H. (2014).
"An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints."
IEEE Transactions on Evolutionary Computation, 18(4), 577-601.

Algorithmic Logic:
1. Reference Points:
       Generate structured reference points Z^r on the unit hyperplane using the Das-Dennis method
       (simplex lattice (p + M - 1) choose (M - 1)).
2. Population Initialization & Variation:
       - Random population P_0 of size N evaluated within budget.
       - Offspring Q_t generated via SBX crossover and polynomial mutation.
       - Combine parents and offspring: R_t = P_t U Q_t (size 2N).
3. Non-Dominated Sorting:
       Partition R_t into Pareto fronts F_1, F_2, ... using Deb's constrained sorting.
       Collect full fronts until adding front F_L would exceed target size N.
4. Normalization and Association:
       - Update ideal point z* = min_{x in R_t} f_m(x).
       - Translate: f_m'(x) = f_m(x) - z_m*.
       - Extreme points found via Achievement Scalarizing Function (ASF).
       - Intercepts a_m calculated via linear system (or max objective if singular).
       - Normalized objectives: f_m^n(x) = f_m'(x) / a_m.
       - Associate each solution with its closest reference line (minimum perpendicular distance d_perp).
5. Niche-Preservation Selection:
       - Compute niche count rho_j (number of solutions already selected associated with reference point j).
       - Identify reference points with minimum rho_j.
       - If rho_j == 0, select candidate in F_L with smallest d_perp.
       - If rho_j >= 1, select candidate in F_L randomly.
       - Repeat until exactly N solutions are selected for P_{t+1}.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, das_dennis
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def compute_asf(F_trans: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Computes Achievement Scalarizing Function for identifying extreme points."""
    w = np.where(weights < 1e-12, 1e-12, weights)
    return np.max(F_trans / w, axis=1)


def find_hyperplane_intercepts(F_trans: np.ndarray) -> np.ndarray:
    """Finds extreme points and computes intercepts a_m on translated objectives."""
    N, M = F_trans.shape
    extreme_points = np.zeros((M, M))

    for m in range(M):
        w = np.full(M, 1e-6)
        w[m] = 1.0
        asf_vals = compute_asf(F_trans, w)
        extreme_points[m] = F_trans[np.argmin(asf_vals)]

    # Attempt to solve for intercepts: extreme_points * a_inv = 1
    try:
        if np.linalg.matrix_rank(extreme_points) == M:
            b = np.ones(M)
            a_inv = np.linalg.solve(extreme_points, b)
            if np.all(a_inv > 1e-6):
                return 1.0 / a_inv
    except np.linalg.LinAlgError:
        pass

    # Fallback: maximum values along each translated objective
    return np.maximum(1e-4, np.max(F_trans, axis=0))


def associate_to_reference_points(
    F_norm: np.ndarray,
    ref_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Associates each solution with its closest reference point by perpendicular distance.

    Returns:
        assoc_ref: index of closest reference point for each solution (shape N,)
        d_perp: perpendicular distance to associated reference line (shape N,)
    """
    N = len(F_norm)
    K = len(ref_points)

    # Normalize reference vectors to unit length
    ref_norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
    ref_norms = np.where(ref_norms < 1e-12, 1.0, ref_norms)
    ref_unit = ref_points / ref_norms

    # Projected length onto each reference line: (N, K)
    proj = np.dot(F_norm, ref_unit.T)
    proj = np.maximum(0.0, proj)

    # Perpendicular distance matrix: d_perp^2 = ||f||^2 - proj^2
    f_norm_sq = np.sum(F_norm ** 2, axis=1, keepdims=True)
    dist_sq = np.maximum(0.0, f_norm_sq - proj ** 2)
    dist_perp = np.sqrt(dist_sq)

    assoc_ref = np.argmin(dist_perp, axis=1)
    min_dist = np.min(dist_perp, axis=1)
    return assoc_ref, min_dist


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int | None = None,
    p_ref: int = 12,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes NSGA-III state, structured reference points, and initial population."""
    rng = np.random.default_rng(seed)
    M = problem.n_obj

    # Generate Das-Dennis reference points on unit simplex
    # Choose p such that number of reference points matches target pop_size
    ref_points = das_dennis(p=p_ref, M=M)
    K_ref = len(ref_points)

    if pop_size is None:
        # Canonical NSGA-III sets population size to smallest multiple of 4 >= K_ref
        pop_size = int(np.ceil(K_ref / 4.0) * 4)

    effective_pop_size = max(1, min(pop_size, budget))
    pop = random_population(problem, effective_pop_size, rng)

    F_list = []
    CV_list = []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop_size,
        "ref_points": ref_points,
        "ideal_point": np.min(F, axis=0) if len(F) > 0 else problem.ideal_point.copy(),
        "X": pop,
        "F": F,
        "CV": CV,
        "archive_X": pop.copy(),
        "archive_F": F.copy(),
        "archive_CV": CV.copy(),
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
    """Executes one generation of NSGA-III."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    if N < 2:
        return state

    feas_tol = problem.feas_tol
    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    ref_points = state["ref_points"]
    K_ref = len(ref_points)

    # 1. Binary tournament selection (random pairing)
    mating_pool_indices = rng.choice(N, size=N, replace=True)
    mating_pool = X[mating_pool_indices]

    # 2. Genetic variation (SBX + Polynomial Mutation)
    offspring_list = []
    is_perm = (problem.var_type == "permutation")

    for i in range(0, N, 2):
        p1 = mating_pool[i]
        p2 = mating_pool[(i + 1) % N]

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
        if len(offspring_list) < N:
            offspring_list.append(c2)

    n_to_eval = min(len(offspring_list), min(N, rem_budget))
    X_off = np.array(offspring_list[:n_to_eval])

    # 3. Evaluate offspring
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

    # 4. Combine parent and offspring: R_t = P_t U Q_t
    X_comb = np.vstack((X, X_off))
    F_comb = np.vstack((F, F_off))
    CV_comb = np.concatenate((CV, CV_off))

    # 5. Non-dominated sorting
    fronts = fast_non_dominated_sort(F_comb, CV=CV_comb, use_deb=True, feas_tol=feas_tol)

    # 6. Environmental selection using reference points
    chosen_indices: list[int] = []
    last_front: list[int] = []

    for front in fronts:
        if len(chosen_indices) + len(front) <= N:
            chosen_indices.extend(front)
        else:
            last_front = front
            break

    if len(chosen_indices) < N and len(last_front) > 0:
        # Normalization across all solutions in S_t = chosen U last_front
        S_t = chosen_indices + last_front
        F_st = F_comb[S_t]

        # Update ideal point z*
        current_ideal = np.min(F_comb, axis=0)
        state["ideal_point"] = np.minimum(state.get("ideal_point", current_ideal), current_ideal)
        z_star = state["ideal_point"]

        # Translated objectives
        F_trans = np.maximum(0.0, F_st - z_star)

        # Compute hyperplane intercepts
        intercepts = find_hyperplane_intercepts(F_trans)
        F_norm = F_trans / np.where(intercepts < 1e-12, 1.0, intercepts)

        # Association with reference points
        assoc_ref, dist_perp = associate_to_reference_points(F_norm, ref_points)

        # Map back from S_t local indices to combined indices
        st_to_comb = {local_idx: comb_idx for local_idx, comb_idx in enumerate(S_t)}
        comb_to_st = {comb_idx: local_idx for local_idx, comb_idx in enumerate(S_t)}

        # Compute initial niche counts rho_j for already chosen solutions
        rho = np.zeros(K_ref, dtype=int)
        for idx in chosen_indices:
            loc = comb_to_st[idx]
            rho[assoc_ref[loc]] += 1

        # Candidate pool from last front
        cand_local_indices = [comb_to_st[idx] for idx in last_front]
        cand_ref_map: dict[int, list[int]] = {j: [] for j in range(K_ref)}
        for loc in cand_local_indices:
            cand_ref_map[assoc_ref[loc]].append(loc)

        # Iteratively select N - len(chosen_indices) solutions
        needed = N - len(chosen_indices)
        active_ref_indices = list(range(K_ref))

        while needed > 0 and len(active_ref_indices) > 0:
            # Find reference points with minimum niche count
            min_rho = np.min([rho[j] for j in active_ref_indices])
            min_ref_list = [j for j in active_ref_indices if rho[j] == min_rho]
            j_bar = int(rng.choice(min_ref_list))

            cands_for_j = cand_ref_map[j_bar]
            if len(cands_for_j) == 0:
                active_ref_indices.remove(j_bar)
                continue

            if rho[j_bar] == 0:
                # Select candidate with minimum perpendicular distance
                best_cand = cands_for_j[np.argmin([dist_perp[c] for c in cands_for_j])]
            else:
                # Select random candidate associated with j_bar
                best_cand = int(rng.choice(cands_for_j))

            chosen_indices.append(st_to_comb[best_cand])
            cand_ref_map[j_bar].remove(best_cand)
            rho[j_bar] += 1
            needed -= 1

    # Truncate or pad to exactly N
    chosen_indices = chosen_indices[:N]
    state["X"] = X_comb[chosen_indices]
    state["F"] = F_comb[chosen_indices]
    state["CV"] = CV_comb[chosen_indices]

    # Update non-dominated external archive
    combined_arch_X = np.vstack((state["archive_X"], X_off))
    combined_arch_F = np.vstack((state["archive_F"], F_off))
    combined_arch_CV = np.concatenate((state["archive_CV"], CV_off))

    arch_X, arch_F, arch_CV = extract_non_dominated(
        combined_arch_X, combined_arch_F, combined_arch_CV, feas_tol=feas_tol
    )
    if len(arch_X) > 200:
        keep = rng.choice(len(arch_X), size=200, replace=False)
        arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    pop_size: int | None = None,
    p_ref: int = 12,
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    **kwargs,
) -> dict:
    """Executes the full NSGA-III optimization run up to the evaluation budget."""
    state = initialize(
        problem,
        seed=seed,
        pop_size=pop_size,
        p_ref=p_ref,
        budget=budget,
        **kwargs,
    )
    state["budget"] = budget

    while state["n_evals"] < budget and state["pop_size"] >= 2:
        prev_evals = state["n_evals"]
        step(state, problem, prob_c=prob_c, eta_c=eta_c, eta_m=eta_m)
        if state["n_evals"] == prev_evals:
            break

    X_opt, F_opt, CV_opt = extract_non_dominated(
        state["archive_X"], state["archive_F"], state["archive_CV"], feas_tol=problem.feas_tol
    )
    F_opt_nat = problem.to_natural(F_opt)

    return {
        "X_opt": X_opt,
        "F_opt": F_opt,
        "F_opt_nat": F_opt_nat,
        "CV_opt": CV_opt,
        "ref_points": state["ref_points"],
        "n_evals": state["n_evals"],
        "state": state,
    }
