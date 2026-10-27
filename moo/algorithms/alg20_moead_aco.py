"""
alg20_moead_aco.py: MOEA/D-ACO (Decomposition-Based Ant Colony Optimization)

Lecture Family: Modern Elitist Multiobjective Evolutionary Algorithms (Part 3 & Bonus Module 5E, Chapter 5)
Original Citation: Ke, L., Zhang, Q., & Cook, R. (2010).
"MOEA/D-ACO: A Multiobjective Evolutionary Algorithm Using Decomposition and Ant Colony."
IEEE Transactions on Evolutionary Computation, 14(4), 529-541.

Algorithmic Logic:
1. Subproblem Decomposition:
       - Decomposes the multiobjective problem into N scalar subproblems with weight vectors lambda^1, ..., lambda^N.
       - Each ant k is assigned to a specific subproblem k with weight vector lambda^k.
2. Heuristic Aggregation:
       - Objective-specific heuristics: eta_m(i, j) (e.g. 1/d_ij for distance, 1/t_ij for travel time).
       - Aggregated heuristic for subproblem k:
             eta_k(i, j) = prod_{m=1}^M [eta_m(i, j)]^(lambda_m^k).
3. Tour Construction (for Permutations / TSP):
       - Ant k constructs solution step-by-step using transition probability:
             P_{ij}^k = [tau_{ij}]^alpha * [eta_k(i, j)]^beta / sum_l [tau_{il}]^alpha * [eta_k(i, l)]^beta.
4. Tchebycheff Evaluation & Collaborative Subproblem Updates:
       - Evaluate tour, update subproblem solutions, and update ideal point z*.
5. Pheromone Evaporation & Global Elitist Deposit:
       - Pheromone evaporates: tau_{ij} = (1 - rho) * tau_{ij}.
       - Solutions in the non-dominated archive deposit pheromone: delta_tau = 1.0 / scalar_cost.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import das_dennis
from moo.algorithms.common import (
    random_population,
    order_crossover,
    swap_mutation,
    sbx_crossover,
    polynomial_mutation,
    extract_non_dominated,
)
from moo.algorithms.alg18_moead import tchebycheff_scalar


def initialize(
    problem: Problem,
    seed: int = 42,
    n_ants: int = 30,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes MOEA/D-ACO state, weight vectors, and pheromone matrix."""
    rng = np.random.default_rng(seed)
    M = problem.n_obj

    # 1. Weight vectors
    weights = das_dennis(p=max(1, n_ants - 1 if M == 2 else 6), M=M)
    effective_n_ants = max(1, min(len(weights), budget))
    weights = weights[:effective_n_ants]
    N = effective_n_ants

    # 2. Pheromone matrix (for permutations) or initial population
    n_var = problem.n_var
    tau = np.ones((n_var, n_var), dtype=float)

    # Initial random solutions
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
        "n_ants": N,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "weights": weights,
        "tau": tau,
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


def construct_ant_solution(
    problem: Problem,
    weight: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Constructs a solution guided by pheromones and subproblem weight vector."""
    n_var = problem.n_var

    if problem.var_type == "permutation":
        # Permutation tour construction (e.g. TSP)
        unvisited = list(range(n_var))
        start_city = int(rng.choice(unvisited))
        tour = [start_city]
        unvisited.remove(start_city)

        curr = start_city
        while unvisited:
            # Heuristic visibility from problem if available
            has_p8 = hasattr(problem, "D") and hasattr(problem, "T")
            probs = np.zeros(len(unvisited), dtype=float)

            for idx, nxt in enumerate(unvisited):
                t_val = tau[curr, nxt] ** alpha
                if has_p8:
                    d_val = max(1e-4, float(problem.D[curr, nxt]))
                    time_val = max(1e-4, float(problem.T[curr, nxt]))
                    eta = ((1.0 / d_val) ** weight[0]) * ((1.0 / time_val) ** weight[1])
                else:
                    eta = 1.0
                probs[idx] = t_val * (eta ** beta)

            total_p = np.sum(probs)
            if total_p <= 1e-12:
                next_city = int(rng.choice(unvisited))
            else:
                probs /= total_p
                next_city = int(rng.choice(unvisited, p=probs))

            tour.append(next_city)
            unvisited.remove(next_city)
            curr = next_city

        return np.array(tour, dtype=int)

    # Continuous domain: sample around problem bounds with Gaussian perturbation
    return rng.uniform(problem.xl, problem.xu)


def step(
    state: dict,
    problem: Problem,
) -> dict:
    """Executes one iteration of MOEA/D-ACO."""
    budget = state.get("budget", 1000000)
    if state["n_evals"] >= budget:
        return state

    rng = state["rng"]
    N = state["n_ants"]
    weights = state["weights"]
    tau = state["tau"]
    alpha = state["alpha"]
    beta = state["beta"]
    rho = state["rho"]
    z_star = state["z_star"]
    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    feas_tol = problem.feas_tol

    new_solutions = []
    new_F = []
    new_CV = []

    for k in range(N):
        if state["n_evals"] >= budget:
            break

        w_k = weights[k]
        child = construct_ant_solution(problem, w_k, tau, alpha, beta, rng)
        f_child, cv_child, _ = problem.evaluate(child)
        state["n_evals"] += 1

        new_solutions.append(child)
        new_F.append(f_child)
        new_CV.append(cv_child)

        # Update ideal point
        z_star = np.minimum(z_star, f_child)
        state["z_star"] = z_star

        # Subproblem update
        g_old = tchebycheff_scalar(F[k], CV[k], w_k, z_star)
        g_new = tchebycheff_scalar(f_child, cv_child, w_k, z_star)
        if g_new <= g_old:
            X[k] = child.copy()
            F[k] = f_child.copy()
            CV[k] = cv_child

    # Update external archive
    if len(new_solutions) > 0:
        combined_arch_X = np.vstack((state["archive_X"], new_solutions))
        combined_arch_F = np.vstack((state["archive_F"], new_F))
        combined_arch_CV = np.concatenate((state["archive_CV"], new_CV))

        arch_X, arch_F, arch_CV = extract_non_dominated(
            combined_arch_X, combined_arch_F, combined_arch_CV, feas_tol=feas_tol
        )
        if len(arch_X) > 200:
            keep = rng.choice(len(arch_X), size=200, replace=False)
            arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

        state["archive_X"] = arch_X
        state["archive_F"] = arch_F
        state["archive_CV"] = arch_CV

        # Global Pheromone Update (Evaporation + Elitist Deposit)
        tau = (1.0 - rho) * tau
        if problem.var_type == "permutation":
            n_cities = problem.n_var
            for sol_idx in range(len(arch_X)):
                tour = arch_X[sol_idx]
                f_cost = max(1e-4, float(np.sum(arch_F[sol_idx])))
                deposit = 1.0 / f_cost
                for step_idx in range(n_cities):
                    u = int(tour[step_idx])
                    v = int(tour[(step_idx + 1) % n_cities])
                    tau[u, v] += deposit
                    tau[v, u] += deposit

            state["tau"] = np.clip(tau, 1e-4, 100.0)

    state["iter"] += 1
    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    n_ants: int = 30,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    **kwargs,
) -> dict:
    """Executes the full MOEA/D-ACO optimization run up to the evaluation budget."""
    state = initialize(
        problem,
        seed=seed,
        n_ants=n_ants,
        alpha=alpha,
        beta=beta,
        rho=rho,
        budget=budget,
        **kwargs,
    )
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
        "weights": state["weights"],
        "n_evals": state["n_evals"],
        "state": state,
    }
