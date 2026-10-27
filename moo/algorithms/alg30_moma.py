"""
alg30_moma.py: MOMA (Multi-Objective Memetic Algorithm)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Knowles, J. D., & Corne, D. W. (2000).
"M-PAES: A Memetic Algorithm for Multiobjective Optimization."
Congress on Evolutionary Computation (CEC 2000), 325-332.
Also: Jaszkiewicz, A. (2002). "On the Performance of Multiple-Objective Genetic Local Search."

Algorithmic Logic:
1. Genetic Backbone (NSGA-II Framework):
       - Fast non-dominated sorting and crowding distance.
       - Crowded binary tournament selection.
       - Variation: SBX crossover and polynomial mutation (or OX/swap for permutations).
       - (N + N) elitist environmental selection.
2. Memetic Phase (Local Search Intensification):
       - Apply individual improvement (local search) to elite (Front 0) solutions.
       - For continuous domains: Pattern search / coordinate descent along random orthogonal directions
         guided by a Tchebycheff scalarization with weight vector matching the solution's objective trade-off.
       - For permutation domains: 2-opt edge-exchange neighborhood search.
       - Replacement occurs if the local variant constrained-dominates or improves scalar fitness.
3. Exact Budget Accounting:
       Every local search probe is evaluated through problem.evaluate() and strictly counted toward the budget.
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
from moo.algorithms.alg14_nsga2 import (
    assign_ranks_and_crowding,
    crowded_tournament_select,
    elitist_environmental_selection,
)


def local_search_continuous(
    x: np.ndarray,
    f: np.ndarray,
    cv: float,
    problem: Problem,
    weights: np.ndarray,
    ideal_point: np.ndarray,
    budget_rem: int,
    rng: np.random.Generator,
    max_steps: int = 5,
    step_size: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Local search for continuous variables using coordinate pattern probes on Tchebycheff scalar.
    Returns (x_best, f_best, cv_best, n_evals_used).
    """
    n_var = len(x)
    span = problem.xu - problem.xl
    span = np.where(span <= 0, 1.0, span)

    w = np.where(weights < 1e-12, 1e-12, weights)

    def _scalar(f_vec: np.ndarray, cv_val: float) -> float:
        if cv_val > problem.feas_tol:
            return 1e6 + cv_val
        return float(np.max(w * np.abs(f_vec - ideal_point)))

    best_x = x.copy()
    best_f = f.copy()
    best_cv = cv
    best_score = _scalar(best_f, best_cv)
    evals_used = 0

    for _ in range(max_steps):
        if evals_used >= budget_rem:
            break

        # Probe a random direction
        j = rng.integers(0, n_var)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        delta = direction * step_size * span[j]

        x_probe = best_x.copy()
        x_probe[j] = np.clip(best_x[j] + delta, problem.xl[j], problem.xu[j])
        if problem.var_type == "integer":
            x_probe[j] = np.round(x_probe[j])

        f_probe, cv_probe, _ = problem.evaluate(x_probe)
        evals_used += 1

        score_probe = _scalar(f_probe, cv_probe)
        if score_probe < best_score:
            best_x = x_probe
            best_f = f_probe
            best_cv = cv_probe
            best_score = score_probe

    return best_x, best_f, best_cv, evals_used


def local_search_permutation(
    tour: np.ndarray,
    f: np.ndarray,
    cv: float,
    problem: Problem,
    budget_rem: int,
    rng: np.random.Generator,
    max_steps: int = 5,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    2-opt local search for permutation variables.
    Returns (tour_best, f_best, cv_best, n_evals_used).
    """
    n = len(tour)
    best_tour = tour.copy()
    best_f = f.copy()
    best_cv = cv
    evals_used = 0

    for _ in range(max_steps):
        if evals_used >= budget_rem:
            break

        i, j = sorted(rng.choice(n, size=2, replace=False))
        # 2-opt reverse segment between i and j
        new_tour = best_tour.copy()
        new_tour[i : j + 1] = new_tour[i : j + 1][::-1]

        f_cand, cv_cand, _ = problem.evaluate(new_tour)
        evals_used += 1

        if constrained_dominates(f_cand, cv_cand, best_f, best_cv, feas_tol=problem.feas_tol):
            best_tour = new_tour
            best_f = f_cand
            best_cv = cv_cand

    return best_tour, best_f, best_cv, evals_used


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MOMA population and evaluations within budget."""
    rng = np.random.default_rng(seed)
    pop_size = kwargs.get("pop_size", 40)
    budget = kwargs.get("budget", 10000)

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

    fronts, ranks, crowd_dist = assign_ranks_and_crowding(F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop,
        "budget": budget,
        "X": pop,
        "F": F,
        "CV": CV,
        "ranks": ranks,
        "crowd_dist": crowd_dist,
        "fronts": fronts,
        "archive_X": pop.copy(),
        "archive_F": F.copy(),
        "archive_CV": CV.copy(),
        "n_evals": effective_pop,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one generation of MOMA: Genetic variation + Elitist survival + Local search."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    ranks = state["ranks"]
    crowd_dist = state["crowd_dist"]

    # 1. Crowded tournament selection
    mating_indices = []
    for _ in range(N):
        i, j = rng.choice(N, size=2, replace=False)
        mating_indices.append(crowded_tournament_select(i, j, ranks, crowd_dist, rng))

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

    # 3. Elitist (N + N) selection
    X_comb = np.vstack((X, X_off))
    F_comb = np.vstack((F, F_off))
    CV_comb = np.concatenate((CV, CV_off))

    X_new, F_new, CV_new, fronts_new, ranks_new, cd_new = elitist_environmental_selection(
        X_comb, F_comb, CV_comb, target_size=N, feas_tol=feas_tol
    )

    # 4. Memetic Local Search on Front 0 solutions
    rem_budget = budget - state["n_evals"]
    if rem_budget > 0 and len(fronts_new) > 0:
        front0 = fronts_new[0]
        # Pick 2 elite individuals for local intensification
        n_elite = min(2, len(front0))
        elite_picks = rng.choice(front0, size=n_elite, replace=False)

        ideal = np.min(F_new, axis=0)

        for e_idx in elite_picks:
            if rem_budget <= 0:
                break

            if is_perm:
                b_x, b_f, b_cv, evals = local_search_permutation(
                    X_new[e_idx], F_new[e_idx], CV_new[e_idx], problem, rem_budget, rng
                )
            else:
                # Weights inverse to distance to ideal
                diff = np.abs(F_new[e_idx] - ideal) + 1e-6
                weights = 1.0 / diff
                weights = weights / np.sum(weights)

                b_x, b_f, b_cv, evals = local_search_continuous(
                    X_new[e_idx], F_new[e_idx], CV_new[e_idx], problem, weights, ideal, rem_budget, rng
                )

            state["n_evals"] += evals
            rem_budget -= evals
            X_new[e_idx] = b_x
            F_new[e_idx] = b_f
            CV_new[e_idx] = b_cv

        fronts_new, ranks_new, cd_new = assign_ranks_and_crowding(F_new, CV_new, feas_tol=feas_tol)

    # 5. Update Archive
    comb_arch_X = np.vstack((state["archive_X"], X_new))
    comb_arch_F = np.vstack((state["archive_F"], F_new))
    comb_arch_CV = np.concatenate((state["archive_CV"], CV_new))

    arch_X, arch_F, arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )
    if len(arch_X) > 200:
        cd = crowding_distance(arch_F, list(range(len(arch_F))))
        order = np.argsort(-cd)
        keep = order[:200]
        arch_X = arch_X[keep]
        arch_F = arch_F[keep]
        arch_CV = arch_CV[keep]

    state["X"] = X_new
    state["F"] = F_new
    state["CV"] = CV_new
    state["fronts"] = fronts_new
    state["ranks"] = ranks_new
    state["crowd_dist"] = cd_new
    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MOMA until budget is exhausted."""
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
