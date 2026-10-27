"""
alg11_moga.py: MOGA (Multi-Objective Genetic Algorithm)

Lecture Family: Non-Elitist Evolutionary Algorithms (Part 2, Chapter 5)
Original Citation: Fonseca, C. M., & Fleming, P. J. (1993).
"Genetic Algorithms for Multiobjective Optimization: Formulation, Discussion and Generalization."
Proceedings of the 5th International Conference on Genetic Algorithms (ICGA-93), pp. 416-423.

Algorithmic Logic:
1. Dominance Ranking:
       For every individual i in population of size N:
       Rank r_i = 1 + count( j in Pop such that j dominates i )
       - Non-dominated individuals receive rank 1.
       - Dominated individuals receive rank proportional to how many individuals dominate them.
2. Raw Fitness Assignment:
       - Sort population by rank r_i ascending.
       - Assign decreasing raw fitness from rank 1 down to rank N (e.g. linear ranking F_i = N - rank_order).
       - For all individuals sharing the exact same rank r_i, average their raw fitness values so they start equal.
3. Fitness Sharing in Objective Space:
       - Linearly normalize objectives to [0, 1].
       - For individuals sharing the same rank r, compute the niche count:
             nc_i = sum_{j in same_rank} Sh( ||f_i^norm - f_j^norm||_2 )
             Sh(d) = 1 - (d / sigma_share)^alpha  if d < sigma_share else 0
       - Calculate shared fitness:
             F_i' = F_i / nc_i
4. Reproduction:
       - Roulette wheel selection proportional to shared fitness F_i'.
       - Simulated Binary Crossover (SBX) and Polynomial Mutation (PM).
       - Non-elitist generational replacement.
5. Archive best non-dominated solutions across generations.

Key Theoretical Distinction (Viva Note):
- In MOGA, an individual's rank is not its front number (as in NSGA), but rather 1 + the exact count
  of individuals dominating it.
- Fitness sharing is performed *only among individuals of the same rank*, preventing arbitrary dilution across ranks.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import constrained_dominates, normalize_objectives
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
    """Initializes MOGA population and parameters."""
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
    """Executes one MOGA generation."""
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

    # 1. Compute MOGA Dominance Rank: r_i = 1 + count(j dominates i)
    ranks = np.ones(N, dtype=int)
    for i in range(N):
        count_dominating = 0
        for j in range(N):
            if i != j and constrained_dominates(F[j], CV[j], F[i], CV[i], feas_tol=feas_tol):
                count_dominating += 1
        ranks[i] = 1 + count_dominating

    # 2. Assign raw fitness based on sorted rank order
    sorted_order = np.argsort(ranks)  # best ranks first
    raw_fitness = np.zeros(N, dtype=float)
    for pos, idx in enumerate(sorted_order):
        raw_fitness[idx] = float(N - pos)

    # Average raw fitness for individuals with identical ranks
    unique_ranks = np.unique(ranks)
    for r in unique_ranks:
        members = np.where(ranks == r)[0]
        avg_fit = float(np.mean(raw_fitness[members]))
        raw_fitness[members] = avg_fit

    # 3. Fitness Sharing within each rank group in normalized objective space
    F_norm = normalize_objectives(F, ideal=problem.ideal_point, nadir=problem.nadir_point)
    shared_fitness = raw_fitness.copy()

    for r in unique_ranks:
        members = np.where(ranks == r)[0]
        if len(members) <= 1:
            continue

        F_sub = F_norm[members]
        diff = F_sub[:, None, :] - F_sub[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Sharing function Sh(d)
        sh_vals = np.where(dist < sigma_share, 1.0 - (dist / sigma_share), 0.0)
        niche_counts = np.sum(sh_vals, axis=1)
        niche_counts = np.maximum(1.0, niche_counts)

        shared_fitness[members] = raw_fitness[members] / niche_counts

    # 4. Roulette wheel selection on shared fitness
    selected_indices = roulette_wheel_selection(shared_fitness, N, rng)
    mating_pool = X[selected_indices]

    # 5. Crossover and mutation
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

    # 6. Evaluate offspring
    F_off_list = []
    CV_off_list = []
    for ind in X_off:
        f, cv, _ = problem.evaluate(ind)
        F_off_list.append(f)
        CV_off_list.append(cv)

    F_off = np.array(F_off_list, dtype=float)
    CV_off = np.array(CV_off_list, dtype=float)

    # 7. Generational replacement
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
    """Runs MOGA algorithm up to the evaluation budget."""
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
