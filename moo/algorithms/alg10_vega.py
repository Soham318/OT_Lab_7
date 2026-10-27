"""
alg10_vega.py: VEGA (Vector Evaluated Genetic Algorithm)

Lecture Family: Non-Elitist Evolutionary Algorithms (Part 2, Chapter 5)
Original Citation: Schaffer, J. D. (1985).
"Multiple objective optimization with vector evaluated genetic algorithms."
Proceedings of the First International Conference on Genetic Algorithms (ICGA), pp. 93-100.

Algorithmic Logic:
1. Maintains a population of size N (must be divisible by M; e.g. N = 100).
2. At each generation, divides the selection phase into M sub-populations, each of size q = N / M.
3. For sub-population m in {1, ..., M}:
       Selects q parent individuals solely based on their performance in objective f_m^internal
       (using tournament selection with Deb's constraint violation handling).
4. Shuffles and merges all M sub-populations into a single unified mating pool of size N.
5. Applies genetic variation:
       - Simulated Binary Crossover (SBX, prob_c = 0.9)
       - Polynomial Mutation (PM, prob_m = 1 / n_var)
6. Non-elitist generational replacement: the offspring population replaces the parent population.
7. Maintains an external archive of non-dominated feasible solutions found across all generations.

Key Theoretical Behavior & Viva Question:
- "Why did VEGA collapse towards the extremes in your run?"
- Because each selection step evaluates individuals on only ONE objective at a time, individuals that are
  strong in objective 1 are selected into sub-population 1, and those strong in objective 2 into sub-population 2.
- There is NO explicit selection pressure for intermediate compromise solutions!
- Consequently, VEGA suffers from speciation and tends to converge almost exclusively towards individual
  objective extrema (the boundary points of the Pareto front).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 100,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes VEGA population and evaluation buffers."""
    rng = np.random.default_rng(seed)
    M = problem.n_obj
    effective_pop_size = max(1, min(pop_size, budget))
    if effective_pop_size >= M:
        pop_size_clamped = max(M, (effective_pop_size // M) * M)
    else:
        pop_size_clamped = effective_pop_size

    pop = random_population(problem, pop_size_clamped, rng)
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
        "pop_size": pop_size_clamped,
        "M": M,
        "X": pop,
        "F": F,
        "CV": CV,
        "archive_X": pop.copy(),
        "archive_F": F.copy(),
        "archive_CV": CV.copy(),
        "n_evals": pop_size_clamped,
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
    """Executes one VEGA generation."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    M = state["M"]
    if N < M:
        return state
    q = max(1, N // M)

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    feas_tol = problem.feas_tol

    # 1. Sub-population selection: select q parents for each objective m
    mating_pool_indices = []

    for m in range(M):
        # 2-tournament selection on objective m with Deb's constraint rule
        sub_selected = []
        for _ in range(q):
            if N == 1:
                sub_selected.append(0)
                continue
            i1, i2 = rng.choice(N, size=2, replace=False)
            cv1, cv2 = CV[i1], CV[i2]
            f1, f2 = F[i1, m], F[i2, m]

            # Deb's feasibility rule on single objective
            feas1 = (cv1 <= feas_tol)
            feas2 = (cv2 <= feas_tol)

            if feas1 and not feas2:
                winner = i1
            elif not feas1 and feas2:
                winner = i2
            elif not feas1 and not feas2:
                winner = i1 if cv1 < cv2 else i2
            else:
                winner = i1 if f1 < f2 else i2

            sub_selected.append(winner)

        mating_pool_indices.extend(sub_selected)

    # 2. Shuffle mating pool to enable cross-objective recombination
    rng.shuffle(mating_pool_indices)
    mating_pool = X[mating_pool_indices]

    # 3. Crossover and mutation to produce offspring
    offspring_list = []
    is_perm = (problem.var_type == "permutation")
    pool_len = len(mating_pool)

    for i in range(0, pool_len, 2):
        p1 = mating_pool[i]
        p2 = mating_pool[(i + 1) % pool_len]

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

    # 4. Evaluate offspring
    F_off_list = []
    CV_off_list = []
    for ind in X_off:
        f, cv, _ = problem.evaluate(ind)
        F_off_list.append(f)
        CV_off_list.append(cv)

    F_off = np.array(F_off_list, dtype=float)
    CV_off = np.array(CV_off_list, dtype=float)

    # 5. Generational update
    if len(X_off) == N:
        state["X"] = X_off
        state["F"] = F_off
        state["CV"] = CV_off
    else:
        # Partial replacement if budget stopped generation early
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
    **kwargs,
) -> dict:
    """Runs the VEGA algorithm up to the evaluation budget."""
    state = initialize(problem, seed=seed, pop_size=pop_size, budget=budget, **kwargs)
    state["budget"] = budget

    while state["n_evals"] < budget and state["pop_size"] >= problem.n_obj:
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
