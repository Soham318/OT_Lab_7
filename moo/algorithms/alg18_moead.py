"""
alg18_moead.py: MOEA/D (Multiobjective Evolutionary Algorithm Based on Decomposition)

Lecture Family: Modern Elitist Multiobjective Evolutionary Algorithms (Part 3 & Bonus Module 5E, Chapter 5)
Original Citation: Zhang, Q., & Li, H. (2007).
"MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition."
IEEE Transactions on Evolutionary Computation, 11(6), 712-731.

Algorithmic Logic:
1. Decomposition Setup:
       - Generate N uniform weight vectors lambda^1, ..., lambda^N on the unit simplex (Das-Dennis).
       - Compute Euclidean distances between weight vectors to identify neighborhood B(i) of size T for each subproblem i.
2. Scalarization (Tchebycheff Approach):
       g^te(x | lambda, z*) = max_{m=1}^M lambda_m * |f_m(x) - z_m*| + 1e4 * CV(x).
3. Population Initialization:
       - Random population x_1, ..., x_N evaluated within budget.
       - Initialize ideal reference point: z_m* = min_i f_m(x_i).
       - Initialize external non-dominated archive EP.
4. Collaborative Subproblem Updates:
       For each subproblem i in 1..N:
       - Select two mating parents k, l from neighborhood B(i).
       - Apply crossover (SBX) and mutation (PM) to generate offspring y.
       - Evaluate offspring y.
       - Update ideal point: z_m* = min(z_m*, f_m(y)).
       - For each neighbor j in B(i):
             If g^te(y | lambda^j, z*) <= g^te(x_j | lambda^j, z*), replace x_j with y.
       - Update external Pareto archive with y.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import das_dennis
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def tchebycheff_scalar(
    f: np.ndarray,
    cv: float,
    weight: np.ndarray,
    z_star: np.ndarray,
) -> float:
    """Computes Tchebycheff scalarization with constraint penalty."""
    w = np.where(weight < 1e-12, 1e-12, weight)
    diff = np.abs(f - z_star)
    return float(np.max(w * diff) + 1e4 * cv)


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 100,
    T_neighbor: int = 20,
    p_ref: int = 12,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes MOEA/D weight vectors, neighborhoods, population, and ideal point."""
    rng = np.random.default_rng(seed)
    M = problem.n_obj

    # 1. Uniform weight vectors on simplex
    weights = das_dennis(p=p_ref, M=M)
    N = len(weights)
    if pop_size is not None and pop_size > 0:
        # Match pop_size if provided
        if len(weights) != pop_size:
            weights = das_dennis(p=max(1, pop_size - 1 if M == 2 else 6), M=M)
            N = len(weights)

    effective_pop_size = max(1, min(N, budget))
    weights = weights[:effective_pop_size]
    N = effective_pop_size

    # 2. Neighborhood of size T for each subproblem
    T = min(max(2, T_neighbor), N)
    diff = weights[:, None, :] - weights[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    neighborhoods = np.argsort(dist_matrix, axis=1)[:, :T]

    # 3. Initial population
    pop = random_population(problem, N, rng)
    F_list, CV_list = [], []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    # 4. Ideal reference point
    z_star = np.min(F, axis=0) if len(F) > 0 else problem.ideal_point.copy()

    # External archive
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
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
) -> dict:
    """Executes one generation of MOEA/D."""
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

    children_X = []
    children_F = []
    children_CV = []

    for i in range(N):
        if state["n_evals"] >= budget:
            break

        # Select two parents from neighborhood of subproblem i
        nb = neighborhoods[i]
        k, l = rng.choice(nb, size=2, replace=False)
        p1 = X[k]
        p2 = X[l]

        # Generate offspring
        if is_perm:
            c1, _ = order_crossover(p1, p2, rng)
            child = swap_mutation(c1, rng)
        else:
            c1, _ = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, prob_c=prob_c, rng=rng)
            child = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m, rng=rng)
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

        # Collaborative replacement in neighborhood B(i)
        for j in nb:
            g_old = tchebycheff_scalar(F[j], CV[j], weights[j], z_star)
            g_new = tchebycheff_scalar(f_child, cv_child, weights[j], z_star)
            if g_new <= g_old:
                X[j] = child.copy()
                F[j] = f_child.copy()
                CV[j] = cv_child

        children_X.append(child)
        children_F.append(f_child)
        children_CV.append(cv_child)

    # Generational update of external archive
    if len(children_X) > 0:
        combined_arch_X = np.vstack((state["archive_X"], children_X))
        combined_arch_F = np.vstack((state["archive_F"], children_F))
        combined_arch_CV = np.concatenate((state["archive_CV"], children_CV))

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
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    **kwargs,
) -> dict:
    """Executes the full MOEA/D optimization run up to the evaluation budget."""
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
        "weights": state["weights"],
        "n_evals": state["n_evals"],
        "state": state,
    }
