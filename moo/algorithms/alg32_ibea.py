"""
alg32_ibea.py: IBEA (Indicator-Based Evolutionary Algorithm)

Lecture Family: Many-Objective, Indicator, Preference & Gradient Methods (Bonus Module 5E, Chapter 5)
Original Citation: Zitzler, E., & Künzli, S. (2004).
"Indicator-Based Selection in Multiobjective Search."
Parallel Problem Solving from Nature (PPSN VIII), LNCS 3242, 832-842.

Algorithmic Logic:
1. Binary Additive epsilon-Indicator:
       I_{eps+}(x_1, x_2) = max_{m=1..M} (f_m(x_1) - f_m(x_2))
       evaluated on normalized objective coordinates.
2. Fitness Assignment:
       F(x_i) = sum_{x_j in P \\ {x_i}} -exp( -I_{eps+}(x_j, x_i) / kappa )
       Higher fitness indicates higher Pareto quality (fewer individuals dominating x_i).
3. Mating Selection:
       Binary tournament selection based on indicator fitness F(x).
4. Variation:
       Offspring generated via SBX crossover and polynomial mutation (or OX/swap).
5. Environmental Selection (Iterative Worst-Solution Removal):
       Combine parents and offspring R_t = P_t U Q_t (size 2N).
       Iteratively remove individual x* with minimum fitness F(x*).
       Update fitness of remaining individuals:
           F(x_i) += exp( -I_{eps+}(x*, x_i) / kappa )
       Repeat until exactly N individuals remain.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, normalize_objectives
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def compute_epsilon_indicator_matrix(F_norm: np.ndarray) -> np.ndarray:
    """
    Computes pairwise additive epsilon indicator matrix:
    I[i, j] = I_{eps+}(x_i, x_j) = max_m (f_m(x_i) - f_m(x_j)).
    """
    N, M = F_norm.shape
    # diff[i, j, m] = F_norm[i, m] - F_norm[j, m]
    diff = F_norm[:, None, :] - F_norm[None, :, :]
    I_mat = np.max(diff, axis=-1)
    return I_mat


def compute_ibea_fitness(
    I_mat: np.ndarray,
    kappa: float = 0.05,
) -> np.ndarray:
    """Computes IBEA fitness vector F(x_i) = sum_{j != i} -exp(-I[j, i] / kappa)."""
    N = len(I_mat)
    # Scaled exponential matrix
    exp_mat = np.exp(-I_mat / kappa)
    np.fill_diagonal(exp_mat, 0.0)
    # Sum over incoming j: exp(-I[j, i] / kappa)
    fitness = -np.sum(exp_mat, axis=0)
    return fitness


def environmental_selection_ibea(
    X_comb: np.ndarray,
    F_comb: np.ndarray,
    CV_comb: np.ndarray,
    target_size: int,
    kappa: float = 0.05,
    feas_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative environmental selection removing the worst individual one-by-one.
    Returns (X_surv, F_surv, CV_surv, fitness_surv).
    """
    N = len(X_comb)
    if N <= target_size:
        ideal = np.min(F_comb, axis=0)
        nadir = np.max(F_comb, axis=0)
        F_norm = normalize_objectives(F_comb, ideal, nadir)
        I_mat = compute_epsilon_indicator_matrix(F_norm)
        fit = compute_ibea_fitness(I_mat, kappa=kappa)
        return X_comb, F_comb, CV_comb, fit

    # Normalize objectives into [0, 1]
    ideal = np.min(F_comb, axis=0)
    nadir = np.max(F_comb, axis=0)
    F_norm = normalize_objectives(F_comb, ideal, nadir)

    # For infeasible solutions, add CV penalty to normalized objectives
    is_infeas = (CV_comb > feas_tol)
    if np.any(is_infeas):
        F_norm = F_norm + (CV_comb[:, None] * 10.0)

    I_mat = compute_epsilon_indicator_matrix(F_norm)
    exp_mat = np.exp(-I_mat / kappa)
    np.fill_diagonal(exp_mat, 0.0)

    active_indices = list(range(N))
    fitness = -np.sum(exp_mat, axis=0)

    while len(active_indices) > target_size:
        # Find individual with minimum fitness in active set
        active_fit = fitness[active_indices]
        worst_local = int(np.argmin(active_fit))
        worst_global = active_indices[worst_local]

        # Remove worst_global
        active_indices.pop(worst_local)

        # Update remaining fitness: add exp(-I[worst, i] / kappa)
        for idx in active_indices:
            fitness[idx] += exp_mat[worst_global, idx]

    surv = np.array(active_indices, dtype=int)
    return X_comb[surv], F_comb[surv], CV_comb[surv], fitness[surv]


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes IBEA population, fitness, and evaluations within budget."""
    rng = np.random.default_rng(seed)
    pop_size = kwargs.get("pop_size", 50)
    budget = kwargs.get("budget", 10000)
    kappa = kwargs.get("kappa", 0.05)

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
    F_norm = normalize_objectives(F, ideal, nadir)
    I_mat = compute_epsilon_indicator_matrix(F_norm)
    fitness = compute_ibea_fitness(I_mat, kappa=kappa)

    arch_X, arch_F, arch_CV = extract_non_dominated(pop, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop,
        "budget": budget,
        "kappa": kappa,
        "X": pop,
        "F": F,
        "CV": CV,
        "fitness": fitness,
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
    """Executes one generation of IBEA."""
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
    fitness = state["fitness"]
    kappa = state["kappa"]

    # 1. Binary tournament selection based on indicator fitness
    mating_pool_indices = []
    for _ in range(N):
        i, j = rng.choice(N, size=2, replace=False)
        winner = i if fitness[i] >= fitness[j] else j
        mating_pool_indices.append(winner)

    mating_pool = X[mating_pool_indices]

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

    # 4. Environmental Selection
    X_new, F_new, CV_new, fit_new = environmental_selection_ibea(
        X_comb, F_comb, CV_comb, target_size=N, kappa=kappa, feas_tol=feas_tol
    )

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
    state["fitness"] = fit_new
    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs IBEA until budget is exhausted."""
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
