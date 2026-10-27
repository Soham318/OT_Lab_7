"""
alg19_moead_de.py: MOEA/D-DE (MOEA/D with Differential Evolution Operators)

Lecture Family: Modern Elitist Multiobjective Evolutionary Algorithms (Part 3 & Bonus Module 5E, Chapter 5)
Original Citation: Li, H., & Zhang, Q. (2009).
"Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II."
IEEE Transactions on Evolutionary Computation, 13(2), 284-302.

Algorithmic Logic:
1. Differential Evolution Variation:
       - Selection pool: with probability delta (default 0.9), select parents from neighborhood B(i);
         with probability 1 - delta, select from the whole population.
       - Select two distinct random vectors r1, r2 from the selection pool.
       - Donor vector: v = x_i + F_de * (x_r1 - x_r2).
       - Binomial crossover with x_i using crossover rate CR.
       - Polynomial mutation with probability 1/n_var and distribution index eta_m.
2. Maximal Number of Replacements (n_r):
       - To preserve diversity and prevent an exceptional offspring from overtaking the entire neighborhood,
         MOEA/D-DE limits the maximum number of replaced neighbors to n_r (default n_r = 2).
3. Ideal Point and External Archive Updates:
       - Update reference point z* and non-dominated archive EP with each evaluated solution.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import das_dennis
from moo.algorithms.common import (
    random_population,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)
from moo.algorithms.alg18_moead import tchebycheff_scalar


def de_crossover_mutation(
    x_i: np.ndarray,
    x_r1: np.ndarray,
    x_r2: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    F_de: float = 0.5,
    CR: float = 1.0,
    eta_m: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Differential evolution operator (DE/current-to-rand/1 or DE/rand/1/bin) + Polynomial mutation."""
    if rng is None:
        rng = np.random.default_rng()

    n_var = len(x_i)
    v = x_i + F_de * (x_r1 - x_r2)

    # Binomial crossover
    j_rand = rng.integers(0, n_var)
    child = x_i.copy()
    for j in range(n_var):
        if rng.random() <= CR or j == j_rand:
            child[j] = v[j]

    # Bound clipping
    child = np.clip(child, xl, xu)

    # Polynomial mutation
    child = polynomial_mutation(child, xl, xu, eta_m=eta_m, rng=rng)
    return child


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 100,
    T_neighbor: int = 20,
    p_ref: int = 12,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes MOEA/D-DE state."""
    rng = np.random.default_rng(seed)
    M = problem.n_obj

    weights = das_dennis(p=p_ref, M=M)
    N = len(weights)
    if pop_size is not None and pop_size > 0 and len(weights) != pop_size:
        weights = das_dennis(p=max(1, pop_size - 1 if M == 2 else 6), M=M)
        N = len(weights)

    effective_pop_size = max(1, min(N, budget))
    weights = weights[:effective_pop_size]
    N = effective_pop_size

    T = min(max(2, T_neighbor), N)
    diff = weights[:, None, :] - weights[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    neighborhoods = np.argsort(dist_matrix, axis=1)[:, :T]

    pop = random_population(problem, N, rng)
    F_list, CV_list = [], []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)
    z_star = np.min(F, axis=0) if len(F) > 0 else problem.ideal_point.copy()
    arch_X, arch_F, arch_CV = extract_non_dominated(pop, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": N,
        "T": T,
        "weights": weights,
        "neighborhoods": neighborhoods,
        "z_star": z_star,
        "X": pop,
        "F": F,
        "CV": CV,
        "archive_X": arch_X,
        "archive_F": arch_F,
        "archive_CV": arch_CV,
        "n_evals": N,
        "budget": budget,
        "iter": 0,
    }
    return state


def step(
    state: dict,
    problem: Problem,
    F_de: float = 0.5,
    CR: float = 1.0,
    eta_m: float = 20.0,
    delta: float = 0.9,
    n_r: int = 2,
) -> dict:
    """Executes one generation of MOEA/D-DE."""
    budget = state.get("budget", 1000000)
    if state["n_evals"] >= budget:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    weights = state["weights"]
    neighborhoods = state["neighborhoods"]
    z_star = state["z_star"]
    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    feas_tol = problem.feas_tol
    is_perm = (problem.var_type == "permutation")

    for i in range(N):
        if state["n_evals"] >= budget:
            break

        # Select mating pool: neighborhood B(i) with probability delta, else entire population
        if rng.random() <= delta:
            pool = neighborhoods[i]
        else:
            pool = np.arange(N)

        if len(pool) < 3:
            pool = np.arange(N)

        # Select two distinct indices r1, r2 from pool
        valid_indices = [idx for idx in pool if idx != i]
        if len(valid_indices) < 2:
            valid_indices = [idx for idx in range(N) if idx != i]
        if len(valid_indices) < 2:
            continue

        r1, r2 = rng.choice(valid_indices, size=2, replace=False)

        if is_perm:
            c1, _ = order_crossover(X[r1], X[r2], rng)
            child = swap_mutation(c1, rng)
        else:
            child = de_crossover_mutation(
                X[i], X[r1], X[r2], problem.xl, problem.xu,
                F_de=F_de, CR=CR, eta_m=eta_m, rng=rng
            )
            if problem.var_type == "integer":
                child = np.round(child)
            elif problem.var_type == "binary":
                child = (child >= 0.5).astype(float)

        # Evaluate offspring
        f_child, cv_child, _ = problem.evaluate(child)
        state["n_evals"] += 1

        # Update ideal point
        z_star = np.minimum(z_star, f_child)
        state["z_star"] = z_star

        # Replaced count limited to n_r
        replace_candidates = rng.permutation(pool)
        c_replaced = 0
        for j in replace_candidates:
            if c_replaced >= n_r:
                break
            g_old = tchebycheff_scalar(F[j], CV[j], weights[j], z_star)
            g_new = tchebycheff_scalar(f_child, cv_child, weights[j], z_star)
            if g_new <= g_old:
                X[j] = child.copy()
                F[j] = f_child.copy()
                CV[j] = cv_child
                c_replaced += 1

        # Update archive
        combined_arch_X = np.vstack((state["archive_X"], [child]))
        combined_arch_F = np.vstack((state["archive_F"], [f_child]))
        combined_arch_CV = np.concatenate((state["archive_CV"], [cv_child]))

        arch_X, arch_F, arch_CV = extract_non_dominated(
            combined_arch_X, combined_arch_F, combined_arch_CV, feas_tol=feas_tol
        )
        if len(arch_X) > 200:
            keep = rng.choice(len(arch_X), size=200, replace=False)
            arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

        state["archive_X"] = arch_X
        state["archive_F"] = arch_F
        state["archive_CV"] = arch_CV

    state["iter"] += 1
    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    pop_size: int = 100,
    T_neighbor: int = 20,
    p_ref: int = 12,
    F_de: float = 0.5,
    CR: float = 1.0,
    eta_m: float = 20.0,
    delta: float = 0.9,
    n_r: int = 2,
    **kwargs,
) -> dict:
    """Executes the full MOEA/D-DE optimization run up to the evaluation budget."""
    state = initialize(
        problem,
        seed=seed,
        pop_size=pop_size,
        T_neighbor=T_neighbor,
        p_ref=p_ref,
        budget=budget,
        **kwargs,
    )
    state["budget"] = budget

    while state["n_evals"] < budget and state["pop_size"] >= 2:
        prev_evals = state["n_evals"]
        step(state, problem, F_de=F_de, CR=CR, eta_m=eta_m, delta=delta, n_r=n_r)
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
        "weights": state["weights"],
        "n_evals": state["n_evals"],
        "state": state,
    }
