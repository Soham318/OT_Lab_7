"""
alg12_nsga.py: NSGA (Non-Dominated Sorting Genetic Algorithm — 1994 Original)

Lecture Family: Non-Elitist Evolutionary Algorithms (Part 2, Chapter 5)
Original Citation: Srinivas, N., & Deb, K. (1994).
"Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms."
Evolutionary Computation, 2(3), 221-248.

Algorithmic Logic:
1. Non-Dominated Sorting:
       Partitions the current population of size N into distinct hierarchical fronts:
       Front_1, Front_2, ..., Front_K
       using Deb's constrained non-dominated sorting.
2. Dummy Fitness Assignment & Sharing within Each Front:
       - Front 1 is assigned a large initial dummy fitness: f_dummy^(1) = N.
       - For each solution i in Front 1, calculate niche count against other members of Front 1:
             nc_i = sum_{j in Front_1} Sh( ||f_i^norm - f_j^norm||_2 )
             where Sh(d) = 1 - (d / sigma_share) if d < sigma_share else 0.
       - Shared dummy fitness for solution i:
             f_i' = f_dummy^(1) / nc_i
       - Find the minimum shared fitness in Front 1: f_min^(1) = min_{i in Front_1} f_i'.
3. Progressive Degradation for Subsequent Fronts:
       - To ensure solutions in Front k have strictly higher fitness than Front k+1,
         Front k is initialized with dummy fitness slightly lower than Front k-1's minimum:
             f_dummy^(k) = f_min^(k-1) * 0.95
       - Sharing is computed strictly within Front k:
             f_i' = f_dummy^(k) / nc_i
       - This process repeats until all fronts have been assigned shared dummy fitness.
4. Reproduction:
       - Roulette wheel selection proportional to shared dummy fitness.
       - Simulated Binary Crossover (SBX) and Polynomial Mutation (PM).
       - Non-elitist generational replacement.
5. Archive non-dominated solutions across generations.

Critical Instruction & Viva Note:
- DO NOT confuse with NSGA-II!
- Original NSGA (1994) uses dummy fitness degradation and explicit niche-count fitness sharing within each front.
  It does NOT use crowding distance, nor does it have (N + N) elitist survival (which were introduced later in NSGA-II in 2002).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, normalize_objectives
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    roulette_wheel_selection,
    extract_non_dominated,
)


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 100,
    sigma_share: float = 0.1,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes NSGA population and parameters."""
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

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop_size,
        "sigma_share": float(sigma_share),
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
    """Executes one NSGA generation."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    if N < 2:
        return state

    sigma_share = state["sigma_share"]
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]

    # 1. Non-dominated sort into successive fronts
    fronts = fast_non_dominated_sort(F, CV=CV, use_deb=True, feas_tol=feas_tol)

    # 2. Assign dummy fitness with progressive degradation and sharing inside each front
    F_norm = normalize_objectives(F, ideal=problem.ideal_point, nadir=problem.nadir_point)
    shared_fitness = np.zeros(N, dtype=float)

    current_dummy = float(N)

    for front in fronts:
        L = len(front)
        if L == 0:
            continue

        if L == 1:
            idx = front[0]
            shared_fitness[idx] = current_dummy
            current_dummy = max(0.1, current_dummy * 0.95)
            continue

        # Niche counts inside this front
        F_sub = F_norm[front]
        diff = F_sub[:, None, :] - F_sub[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        sh_vals = np.where(dist < sigma_share, 1.0 - (dist / sigma_share), 0.0)
        niche_counts = np.sum(sh_vals, axis=1)
        niche_counts = np.maximum(1.0, niche_counts)

        # Shared dummy fitness for this front
        front_shared = current_dummy / niche_counts
        for k, idx in enumerate(front):
            shared_fitness[idx] = front_shared[k]

        # Minimum shared fitness in this front defines the start of next front
        min_front_fit = float(np.min(front_shared))
        current_dummy = max(0.01, min_front_fit * 0.95)

    # 3. Roulette wheel selection across whole population
    selected_indices = roulette_wheel_selection(shared_fitness, N, rng)
    mating_pool = X[selected_indices]

    # 4. Genetic variation
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

    # 5. Evaluate offspring
    F_off_list = []
    CV_off_list = []
    for ind in X_off:
        f, cv, _ = problem.evaluate(ind)
        F_off_list.append(f)
        CV_off_list.append(cv)

    F_off = np.array(F_off_list, dtype=float)
    CV_off = np.array(CV_off_list, dtype=float)

    # 6. Generational replacement
    if len(X_off) == N:
        state["X"] = X_off
        state["F"] = F_off
        state["CV"] = CV_off
    else:
        state["X"][:n_to_eval] = X_off
        state["F"][:n_to_eval] = F_off
        state["CV"][:n_to_eval] = CV_off

    state["n_evals"] += n_to_eval
    state["iter"] += 1

    # Update non-dominated archive
    combined_X = np.vstack((state["archive_X"], X_off))
    combined_F = np.vstack((state["archive_F"], F_off))
    combined_CV = np.concatenate((state["archive_CV"], CV_off))

    arch_X, arch_F, arch_CV = extract_non_dominated(combined_X, combined_F, combined_CV, feas_tol=feas_tol)
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
    pop_size: int = 100,
    sigma_share: float = 0.1,
    **kwargs,
) -> dict:
    """Runs NSGA algorithm up to the evaluation budget."""
    state = initialize(problem, seed=seed, pop_size=pop_size, sigma_share=sigma_share, budget=budget, **kwargs)
    state["budget"] = budget

    while state["n_evals"] < budget and state["pop_size"] >= 2:
        prev_evals = state["n_evals"]
        step(state, problem)
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
