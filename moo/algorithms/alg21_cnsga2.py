"""
alg21_cnsga2.py: Constrained NSGA-II (Deb's Feasibility-First Elitist GA)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
"A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

Algorithmic Logic:
1. Population Initialization:
       Generate N individuals respecting bounds and variable types; evaluate objectives and CV.
2. Constrained-Dominance Comparison (Deb's Feasibility-First Rule):
       Solution A constrained-dominates Solution B (A <_con B) if:
       - A is feasible and B is infeasible, OR
       - Both are infeasible, but CV(A) < CV(B), OR
       - Both are feasible, and A Pareto-dominates B on all objectives.
3. Fast Constrained Non-Dominated Sorting:
       Partition combined population R_t = P_t U Q_t (size 2N) into ranked fronts F_1, F_2, ...
       Feasible solutions occupy lower (better) fronts; infeasible solutions are ranked by CV.
4. Crowding Distance & Binary Tournament:
       Within feasible fronts, calculate crowding distance.
       Tournament selection prefers lower constrained rank, breaking ties with crowding distance.
5. Elitist Environmental Selection:
       Fill P_{t+1} front-by-front until N individuals are retained, truncating the borderline front
       by crowding distance (or lower CV if infeasible).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, crowding_distance, constrained_dominates
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def assign_constrained_ranks_and_crowding(
    F: np.ndarray,
    CV: np.ndarray,
    feas_tol: float = 1e-4,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """
    Partitions population into non-dominated fronts using Deb's constrained dominance,
    and assigns Pareto rank and crowding distance to each individual.
    """
    N = len(F)
    fronts = fast_non_dominated_sort(F, CV=CV, use_deb=True, feas_tol=feas_tol)

    ranks = np.zeros(N, dtype=int)
    crowd_dist = np.zeros(N, dtype=float)

    for rank_idx, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank_idx

        # Calculate crowding distance within this front
        if len(front) > 0:
            cd_front = crowding_distance(F, front)
            for i, idx in enumerate(front):
                crowd_dist[idx] = cd_front[i]

    return fronts, ranks, crowd_dist


def constrained_tournament_select(
    i: int,
    j: int,
    ranks: np.ndarray,
    crowd_dist: np.ndarray,
    CV: np.ndarray,
    feas_tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Binary tournament selection based on constrained rank and crowding distance.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Rule 1: Lower constrained rank wins
    if ranks[i] < ranks[j]:
        return i
    if ranks[j] < ranks[i]:
        return j

    # Rule 2: Same rank - if both infeasible, smaller CV wins
    is_feas_i = (CV[i] <= feas_tol)
    is_feas_j = (CV[j] <= feas_tol)

    if not is_feas_i and not is_feas_j:
        if CV[i] < CV[j] - 1e-9:
            return i
        if CV[j] < CV[i] - 1e-9:
            return j
        return i if rng.random() < 0.5 else j

    # Rule 3: Same rank and both feasible -> larger crowding distance wins
    if crowd_dist[i] > crowd_dist[j]:
        return i
    if crowd_dist[j] > crowd_dist[i]:
        return j

    return i if rng.random() < 0.5 else j


def elitist_constrained_selection(
    X_pool: np.ndarray,
    F_pool: np.ndarray,
    CV_pool: np.ndarray,
    target_size: int,
    feas_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    """
    (N + N) Elitist survival preserving feasible and low-CV non-dominated solutions.
    """
    N_pool = len(X_pool)
    if N_pool <= target_size:
        fronts, ranks, crowd_dist = assign_constrained_ranks_and_crowding(F_pool, CV_pool, feas_tol=feas_tol)
        return X_pool, F_pool, CV_pool, fronts, ranks, crowd_dist

    fronts = fast_non_dominated_sort(F_pool, CV=CV_pool, use_deb=True, feas_tol=feas_tol)

    selected_indices: list[int] = []
    for front in fronts:
        if len(selected_indices) + len(front) <= target_size:
            selected_indices.extend(front)
        else:
            needed = target_size - len(selected_indices)
            cd_front = crowding_distance(F_pool, front)
            # Check feasibility of individuals in borderline front
            is_feas = np.array([CV_pool[idx] <= feas_tol for idx in front])
            if np.all(is_feas):
                # All feasible: sort descending by crowding distance
                order = np.argsort(-cd_front)
            else:
                # Infeasible or mixed: sort by CV ascending, then -crowd_dist
                cv_vals = CV_pool[front]
                order = np.lexsort((-cd_front, cv_vals))

            chosen = [front[i] for i in order[:needed]]
            selected_indices.extend(chosen)
            break

    sel = np.array(selected_indices, dtype=int)
    X_new = X_pool[sel]
    F_new = F_pool[sel]
    CV_new = CV_pool[sel]

    new_fronts, new_ranks, new_cd = assign_constrained_ranks_and_crowding(F_new, CV_new, feas_tol=feas_tol)
    return X_new, F_new, CV_new, new_fronts, new_ranks, new_cd


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes Constrained NSGA-II population and evaluations within budget."""
    rng = np.random.default_rng(seed)
    pop_size = kwargs.get("pop_size", 50)
    budget = kwargs.get("budget", 10000)

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

    fronts, ranks, crowd_dist = assign_constrained_ranks_and_crowding(F, CV, feas_tol=problem.feas_tol)

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
    """Executes one generation of Constrained NSGA-II."""
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

    # 1. Binary tournament selection using constrained tournament operator
    mating_pool_indices = []
    for _ in range(N):
        i, j = rng.choice(N, size=2, replace=False)
        winner = constrained_tournament_select(i, j, ranks, crowd_dist, CV, feas_tol=feas_tol, rng=rng)
        mating_pool_indices.append(winner)

    mating_pool = X[mating_pool_indices]

    # 2. Variation (SBX + Polynomial Mutation or OX + Swap for permutations)
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

    # 4. Elitist (N + N) constrained environmental selection
    X_comb = np.vstack((X, X_off))
    F_comb = np.vstack((F, F_off))
    CV_comb = np.concatenate((CV, CV_off))

    X_new, F_new, CV_new, new_fronts, new_ranks, new_cd = elitist_constrained_selection(
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
        arch_fronts, _, arch_cd = assign_constrained_ranks_and_crowding(arch_F, arch_CV, feas_tol=feas_tol)
        order = np.argsort(-arch_cd)
        keep = order[:200]
        arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs Constrained NSGA-II until budget is exhausted."""
    state = initialize(problem, seed=seed, budget=budget, **kwargs)

    while state["n_evals"] < budget:
        state = step(state, problem, **kwargs)

    arch_X = state["archive_X"]
    arch_F = state["archive_F"]
    arch_CV = state["archive_CV"]

    # Filter to feasible non-dominated solutions if any feasible solutions exist
    is_feas = (arch_CV <= problem.feas_tol)
    if np.any(is_feas):
        opt_X = arch_X[is_feas]
        opt_F = arch_F[is_feas]
        opt_CV = arch_CV[is_feas]
    else:
        # Otherwise lowest CV
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
