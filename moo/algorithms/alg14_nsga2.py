"""
alg14_nsga2.py: NSGA-II (Fast and Elitist Multiobjective Genetic Algorithm)

Lecture Family: Modern Elitist Multiobjective Evolutionary Algorithms (Part 3 & Bonus Module 5E, Chapter 5)
Original Citation: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
"A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

Algorithmic Logic:
1. Population Initialization:
       Generate random population P_0 of size N respecting problem domain and variable types.
       Evaluate all N individuals against evaluation budget.
2. Fast Non-Dominated Sorting & Crowding Distance:
       - Partition population into Pareto fronts F_1, F_2, ... using Deb's constrained-dominance.
       - Compute crowding distance within each front:
             d_i = sum_{m=1}^M (f_m(i+1) - f_m(i-1)) / (f_m^max - f_m^min)
             with boundary extrema assigned d = 1e14 (infinity).
3. Mating Selection:
       Crowded tournament selection (binary tournament using crowded comparison operator <_n):
       Individual i beats individual j if:
           - rank(i) < rank(j) (i belongs to a better front), OR
           - rank(i) == rank(j) and crowd_dist(i) > crowd_dist(j) (i is less crowded).
4. Genetic Variation:
       - Simulated Binary Crossover (SBX) with distribution index eta_c = 15.0, prob_c = 0.9.
       - Polynomial Mutation (PM) with distribution index eta_m = 20.0, prob_m = 1/n_var.
       - Permutations use Order Crossover (OX) and Swap Mutation.
       - Offspring population Q_t of size N is generated and evaluated.
5. (N + N) Elitist Environmental Selection:
       - Combine parent P_t and offspring Q_t: R_t = P_t U Q_t (size 2N).
       - Sort R_t into non-dominated fronts.
       - Construct next generation P_{t+1}:
             Include full fronts F_1, F_2, ... while |P_{t+1}| + |F_k| <= N.
             If |P_{t+1}| + |F_k| > N, sort F_k descending by crowding distance,
             and accept the best N - |P_{t+1}| solutions.
6. Archive:
       Maintain external non-dominated archive of all evaluated solutions.

Distinction from Algorithm 12 (Original NSGA 1994):
- NSGA-II eliminates the sharing parameter sigma_share, replacing fitness sharing with parameter-free crowding distance.
- NSGA-II introduces strict (N + N) elitist environmental selection where parents compete directly with offspring,
  guaranteeing that non-dominated solutions are never lost across generations.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, crowding_distance
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def assign_ranks_and_crowding(
    F: np.ndarray,
    CV: np.ndarray | None,
    feas_tol: float = 1e-4,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Computes fast non-dominated sorting and crowding distances for all individuals.

    Parameters:
        F: Matrix of objective values of shape (N, M).
        CV: Array of constraint violations of shape (N,).
        feas_tol: Feasibility tolerance for Deb's constrained dominance.

    Returns:
        fronts: list of front index lists [F_0, F_1, ...]
        ranks: np.ndarray of shape (N,) containing front rank (0-indexed)
        crowd_dist: np.ndarray of shape (N,) containing crowding distance
    """
    N = len(F)
    ranks = np.zeros(N, dtype=int)
    crowd_dist = np.zeros(N, dtype=float)

    if N == 0:
        return [], ranks, crowd_dist

    fronts = fast_non_dominated_sort(F, CV=CV, use_deb=True, feas_tol=feas_tol)

    for r, front in enumerate(fronts):
        if len(front) == 0:
            continue
        for idx in front:
            ranks[idx] = r
        dists = crowding_distance(F, front)
        for pos, idx in enumerate(front):
            crowd_dist[idx] = dists[pos]

    return fronts, ranks, crowd_dist


def crowded_tournament_select(
    i: int,
    j: int,
    ranks: np.ndarray,
    crowd_dist: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Crowded comparison operator (Deb et al. 2002):

    Individual i defeats j if:
    1. rank(i) < rank(j) (lower front rank is better), OR
    2. rank(i) == rank(j) and crowd_dist(i) > crowd_dist(j) (less crowded is better).
    Ties are broken uniformly at random.
    """
    if ranks[i] < ranks[j]:
        return i
    if ranks[j] < ranks[i]:
        return j

    # Ranks are identical: compare crowding distances
    cd_i = crowd_dist[i]
    cd_j = crowd_dist[j]
    if cd_i > cd_j:
        return i
    if cd_j > cd_i:
        return j

    # Both rank and crowding distance tied: random selection
    return i if rng.random() < 0.5 else j


def elitist_environmental_selection(
    X_comb: np.ndarray,
    F_comb: np.ndarray,
    CV_comb: np.ndarray,
    target_size: int,
    feas_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    """Canonical NSGA-II (N + N) elitist environmental selection.

    Selects target_size individuals front by front, truncating the final borderline front
    by descending crowding distance.
    """
    M_comb = len(X_comb)
    if M_comb <= target_size:
        fronts, ranks, cd = assign_ranks_and_crowding(F_comb, CV_comb, feas_tol=feas_tol)
        return X_comb, F_comb, CV_comb, fronts, ranks, cd

    fronts = fast_non_dominated_sort(F_comb, CV=CV_comb, use_deb=True, feas_tol=feas_tol)
    chosen_indices: list[int] = []

    for front in fronts:
        if len(chosen_indices) + len(front) <= target_size:
            chosen_indices.extend(front)
        else:
            needed = target_size - len(chosen_indices)
            if needed > 0:
                dists = crowding_distance(F_comb, front)
                # Sort descending by crowding distance (extrema and least crowded first)
                order = np.argsort(-dists)
                selected_from_front = [front[i] for i in order[:needed]]
                chosen_indices.extend(selected_from_front)
            break

    chosen_indices = chosen_indices[:target_size]
    X_new = X_comb[chosen_indices]
    F_new = F_comb[chosen_indices]
    CV_new = CV_comb[chosen_indices]

    # Recompute ranks and crowding distances for the new surviving population
    new_fronts, new_ranks, new_cd = assign_ranks_and_crowding(F_new, CV_new, feas_tol=feas_tol)
    return X_new, F_new, CV_new, new_fronts, new_ranks, new_cd


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 100,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes NSGA-II population, ranks, crowding distances, and state."""
    rng = np.random.default_rng(seed)
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

    fronts, ranks, crowd_dist = assign_ranks_and_crowding(F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop_size,
        "X": pop,
        "F": F,
        "CV": CV,
        "ranks": ranks,
        "crowd_dist": crowd_dist,
        "fronts": fronts,
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
    """Executes one generation of NSGA-II with (N + N) elitist environmental selection."""
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
    ranks = state["ranks"]
    crowd_dist = state["crowd_dist"]

    # 1. Binary tournament selection using crowded comparison operator
    mating_pool_indices = []
    for _ in range(N):
        i, j = rng.choice(N, size=2, replace=False)
        winner = crowded_tournament_select(i, j, ranks, crowd_dist, rng)
        mating_pool_indices.append(winner)

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

    # 4. Elitist (N + N) environmental selection
    X_comb = np.vstack((X, X_off))
    F_comb = np.vstack((F, F_off))
    CV_comb = np.concatenate((CV, CV_off))

    X_new, F_new, CV_new, new_fronts, new_ranks, new_cd = elitist_environmental_selection(
        X_comb, F_comb, CV_comb, target_size=N, feas_tol=feas_tol
    )

    state["X"] = X_new
    state["F"] = F_new
    state["CV"] = CV_new
    state["fronts"] = new_fronts
    state["ranks"] = new_ranks
    state["crowd_dist"] = new_cd

    # 5. Update non-dominated external archive
    combined_arch_X = np.vstack((state["archive_X"], X_off))
    combined_arch_F = np.vstack((state["archive_F"], F_off))
    combined_arch_CV = np.concatenate((state["archive_CV"], CV_off))

    arch_X, arch_F, arch_CV = extract_non_dominated(
        combined_arch_X, combined_arch_F, combined_arch_CV, feas_tol=feas_tol
    )
    if len(arch_X) > 200:
        # Prune archive using crowding distance to maintain maximum diversity
        arch_fronts, _, arch_cd = assign_ranks_and_crowding(arch_F, arch_CV, feas_tol=feas_tol)
        order = np.argsort(-arch_cd)
        keep = order[:200]
        arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    pop_size: int = 100,
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    **kwargs,
) -> dict:
    """Executes the full NSGA-II optimization run up to the evaluation budget."""
    state = initialize(
        problem,
        seed=seed,
        pop_size=pop_size,
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
        "n_evals": state["n_evals"],
        "state": state,
    }
