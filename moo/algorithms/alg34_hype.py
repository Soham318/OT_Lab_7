"""
alg34_hype.py: HypE (Hypervolume Estimation Algorithm for Many-Objective Optimization)

Lecture Family: Many-Objective, Indicator, Preference & Gradient Methods (Bonus Module 5E, Chapter 5)
Original Citation: Bader, J., & Zitzler, E. (2011).
"HypE: An Algorithm for Fast Hypervolume-Based Many-Objective Optimization."
Evolutionary Computation, 19(1), 45-76.

Algorithmic Logic:
1. Monte Carlo Hypervolume Approximation:
       Sample P uniform random points within a bounding reference box [L, U] encompassing
       the current population's objective values.
2. Shared Sample Partitioning & Weighting:
       For each sample point s dominated by k solutions in the front, assign weight:
           w(k) = (1 / k) * prod_{j=1}^{k-1} ((needed - j) / (|F| - j))
       Each dominating individual receives credit w(k) for that sample.
       HypE fitness F_H(x_i) equals the sum of credits over all dominated samples.
3. Fast Non-Dominated Sorting:
       Partition combined pool R_t = P_t U Q_t into fronts F_1, F_2, ...
       Fill new population front-by-front.
4. HypE Environmental Truncation:
       Within the borderline front F_L, iteratively remove the solution with minimum HypE fitness
       until exactly N solutions remain.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def compute_hype_fitness_mc(
    F_front: np.ndarray,
    target_k: int,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Computes HypE fitness using Monte Carlo sampling for individuals in F_front.
    target_k: number of individuals to be selected from this front.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, M = F_front.shape
    if N <= 1:
        return np.ones(N)

    # Bounding box
    L = np.min(F_front, axis=0)
    U = np.max(F_front, axis=0) + 1.0

    # Draw Monte Carlo points
    samples = rng.uniform(L, U, size=(n_samples, M))

    # Check dominance: solution i dominates sample s if F_front[i, m] <= sample[s, m] for all m
    # diff: (N, n_samples, M)
    dom_mask = np.all(F_front[:, None, :] <= samples[None, :, :], axis=-1)  # shape (N, n_samples)
    dom_count = np.sum(dom_mask, axis=0)  # shape (n_samples,): how many solutions dominate each sample

    # HypE weight vector
    weights = np.zeros(N + 1)
    for k in range(1, N + 1):
        if k <= target_k:
            prod_val = 1.0
            for j in range(1, k):
                denom = max(1, N - j)
                prod_val *= float(target_k - j) / float(denom)
            weights[k] = prod_val / float(k)

    fitness = np.zeros(N)
    for k in range(1, N + 1):
        if weights[k] > 0:
            # Samples dominated by exactly k solutions
            k_samples = (dom_count == k)
            if np.any(k_samples):
                # Count for each solution how many of these samples it dominates
                sol_counts = np.sum(dom_mask[:, k_samples], axis=1)
                fitness += sol_counts * weights[k]

    return fitness


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes HypE population and evaluations within budget."""
    rng = np.random.default_rng(seed)
    pop_size = kwargs.get("pop_size", 50)
    budget = kwargs.get("budget", 10000)
    n_samples = kwargs.get("n_samples", 10000)

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

    arch_X, arch_F, arch_CV = extract_non_dominated(pop, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop,
        "budget": budget,
        "n_samples": n_samples,
        "X": pop,
        "F": F,
        "CV": CV,
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
    """Executes one generation of HypE with Monte Carlo hypervolume selection."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    feas_tol = problem.feas_tol
    n_samples = state["n_samples"]

    X = state["X"]
    F = state["F"]
    CV = state["CV"]

    # 1. Binary tournament selection using random comparisons
    mating_indices = rng.choice(N, size=N, replace=True)
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

    # 4. Non-dominated sort
    fronts = fast_non_dominated_sort(F_comb, CV=CV_comb, use_deb=True, feas_tol=feas_tol)

    selected_indices = []
    for front in fronts:
        if len(selected_indices) + len(front) <= N:
            selected_indices.extend(front)
        else:
            needed = N - len(selected_indices)
            F_border = F_comb[front]
            # Compute HypE fitness
            fit = compute_hype_fitness_mc(F_border, target_k=needed, n_samples=n_samples, rng=rng)
            order = np.argsort(-fit)
            chosen = [front[idx] for idx in order[:needed]]
            selected_indices.extend(chosen)
            break

    sel = np.array(selected_indices, dtype=int)
    X_new = X_comb[sel]
    F_new = F_comb[sel]
    CV_new = CV_comb[sel]

    # 5. Update Archive
    comb_arch_X = np.vstack((state["archive_X"], X_off))
    comb_arch_F = np.vstack((state["archive_F"], F_off))
    comb_arch_CV = np.concatenate((state["archive_CV"], CV_off))

    arch_X, arch_F, arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )
    if len(arch_X) > 200:
        from moo.moo_utils import crowding_distance
        cd = crowding_distance(arch_F, list(range(len(arch_F))))
        order = np.argsort(-cd)
        keep = order[:200]
        arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

    state["X"] = X_new
    state["F"] = F_new
    state["CV"] = CV_new
    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs HypE until budget is exhausted."""
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
