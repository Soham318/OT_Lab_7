"""
alg25_moaco.py: MOACO (Multi-Objective Ant Colony Optimization) — Task C Deep Dive #2

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Student Roll No. Ending: 13 -> Assigned Algorithm 25: MOACO
Original Citation: Barán, B., & Schaerer, M. (2003).
"A Multiobjective Ant Colony System for Vehicle Routing Problem with Time Windows."
IEEE International Conference on Systems, Man and Cybernetics.
Also: Alaya, I., Solnon, C., & Ghedira, K. (2007). "Ant Colony Optimization for Multi-Objective Optimization Problems."

Algorithmic Logic:
1. Dual Pheromone & Dual Heuristic Matrices:
       - Distance Pheromone Matrix: tau^1_{ij}
       - Time / Cost Pheromone Matrix: tau^2_{ij}
       - Visibility Heuristics: eta^1_{ij} = 1 / d_{ij}, eta^2_{ij} = 1 / t_{ij}
2. Ant Diversity Weight Assignment:
       Colony of K ants where each ant k is assigned trade-off weights:
           lambda_{k,1} = (k - 1) / max(1, K - 1),  lambda_{k,2} = 1.0 - lambda_{k,1}
3. Probabilistic Tour Construction:
       Ant k at node i chooses unvisited node j with probability:
           P_{ij}^k = [tau^1_{ij}]^(alpha * lambda_{k,1}) * [tau^2_{ij}]^(alpha * lambda_{k,2}) *
                      [eta^1_{ij}]^(beta * lambda_{k,1})  * [eta^2_{ij}]^(beta * lambda_{k,2}) / denominator
4. Pheromone Evaporation:
       tau^m_{ij} = max(tau_min, (1 - rho) * tau^m_{ij}) for m in {1, 2}
5. Pareto Elitist Pheromone Deposit:
       Only ants belonging to the non-dominated Pareto front of the colony deposit pheromone:
           Delta tau^1_{ij} = Q / f_1(tour)
           Delta tau^2_{ij} = Q / f_2(tour)
       along every edge (i, j) of the non-dominated tour.
6. Non-Dominated External Archive:
       Stores all Pareto-optimal solutions across generations up to archive capacity.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import random_population, extract_non_dominated, swap_mutation


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MOACO pheromones, colony, and archive within budget."""
    rng = np.random.default_rng(seed)
    n_ants = kwargs.get("n_ants", kwargs.get("pop_size", 30))
    budget = kwargs.get("budget", 10000)
    archive_size = kwargs.get("archive_size", 100)
    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 2.0)
    rho = kwargs.get("rho", 0.1)
    q0 = kwargs.get("q0", 1.0)
    tau0 = kwargs.get("tau0", 1.0)

    n_nodes = problem.n_var
    is_perm = (problem.var_type == "permutation")

    # Dual pheromone matrices
    tau1 = np.full((n_nodes, n_nodes), tau0, dtype=float)
    tau2 = np.full((n_nodes, n_nodes), tau0, dtype=float)

    # Dual heuristic matrices
    if hasattr(problem, "D") and hasattr(problem, "T"):
        D_mat = getattr(problem, "D")
        T_mat = getattr(problem, "T")
        eta1 = 1.0 / np.where(D_mat <= 1e-12, 1e-6, D_mat)
        eta2 = 1.0 / np.where(T_mat <= 1e-12, 1e-6, T_mat)
    else:
        # Default heuristics for general problems
        eta1 = np.ones((n_nodes, n_nodes), dtype=float)
        eta2 = np.ones((n_nodes, n_nodes), dtype=float)

    np.fill_diagonal(eta1, 0.0)
    np.fill_diagonal(eta2, 0.0)

    # Initial random tours to seed evaluations
    effective_ants = max(1, min(n_ants, budget))
    init_pop = random_population(problem, effective_ants, rng)

    F_list = []
    CV_list = []
    for ind in init_pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    arch_X, arch_F, arch_CV = extract_non_dominated(init_pop, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "n_ants": effective_ants,
        "budget": budget,
        "archive_size": archive_size,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "q0": q0,
        "tau0": tau0,
        "tau_min": 1e-4,
        "tau1": tau1,
        "tau2": tau2,
        "eta1": eta1,
        "eta2": eta2,
        "X": init_pop,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_ants,
        "iter": 0,
    }
    return state


def construct_tour_moaco(
    k: int,
    K: int,
    n_nodes: int,
    tau1: np.ndarray,
    tau2: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Constructs a permutation tour for ant k using dual pheromone and dual heuristic aggregation."""
    # Trade-off weights for ant k
    lam1 = float(k) / max(1.0, float(K - 1)) if K > 1 else 0.5
    lam2 = 1.0 - lam1

    start_node = rng.integers(0, n_nodes)
    tour = [start_node]
    unvisited = set(range(n_nodes))
    unvisited.remove(start_node)

    current = start_node
    while unvisited:
        allowed = list(unvisited)
        t1 = tau1[current, allowed]
        t2 = tau2[current, allowed]
        e1 = eta1[current, allowed]
        e2 = eta2[current, allowed]

        # Product of powered pheromones and heuristics
        weights = (
            (t1 ** (alpha * lam1))
            * (t2 ** (alpha * lam2))
            * (e1 ** (beta * lam1))
            * (e2 ** (beta * lam2))
        )
        total_w = np.sum(weights)

        if total_w <= 1e-14 or not np.isfinite(total_w):
            next_node = rng.choice(allowed)
        else:
            probs = weights / total_w
            next_node = rng.choice(allowed, p=probs)

        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return np.array(tour, dtype=int)


def step(state: dict, problem: Problem) -> dict:
    """Executes one colony cycle of MOACO."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    K = state["n_ants"]
    n_nodes = problem.n_var
    feas_tol = problem.feas_tol

    tau1 = state["tau1"]
    tau2 = state["tau2"]
    eta1 = state["eta1"]
    eta2 = state["eta2"]
    alpha = state["alpha"]
    beta = state["beta"]
    rho = state["rho"]
    q0 = state["q0"]
    tau_min = state["tau_min"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    archive_size = state["archive_size"]

    is_perm = (problem.var_type == "permutation")
    colony_tours = []

    for k in range(K):
        if is_perm:
            tour = construct_tour_moaco(k, K, n_nodes, tau1, tau2, eta1, eta2, alpha, beta, rng)
        else:
            # Continuous problem: ant walks across bins or uses continuous perturbation around best
            base = state["arch_X"][rng.choice(len(state["arch_X"]))] if len(arch_X) > 0 else state["X"][k]
            noise = rng.normal(0, 0.1 * (problem.xu - problem.xl))
            tour = np.clip(base + noise, problem.xl, problem.xu)
            if problem.var_type == "integer":
                tour = np.round(tour)
            elif problem.var_type == "binary":
                tour = (tour >= 0.5).astype(float)
        colony_tours.append(tour)

    n_to_eval = min(K, rem_budget)
    X_eval = np.array(colony_tours[:n_to_eval])

    F_eval_list = []
    CV_eval_list = []
    for ind in X_eval:
        f, cv, _ = problem.evaluate(ind)
        F_eval_list.append(f)
        CV_eval_list.append(cv)

    F_eval = np.array(F_eval_list, dtype=float)
    CV_eval = np.array(CV_eval_list, dtype=float)

    state["n_evals"] += n_to_eval
    state["iter"] += 1

    # 1. Pheromone Evaporation
    tau1 = (1.0 - rho) * tau1
    tau2 = (1.0 - rho) * tau2
    tau1 = np.maximum(tau_min, tau1)
    tau2 = np.maximum(tau_min, tau2)

    # 2. Identify Non-Dominated Ants of the Current Iteration
    fronts = fast_non_dominated_sort(F_eval, CV=CV_eval, use_deb=True, feas_tol=feas_tol)
    non_dom_indices = fronts[0] if len(fronts) > 0 else list(range(n_to_eval))

    # 3. Elitist Pheromone Deposit by Non-Dominated Ants
    if is_perm:
        for idx in non_dom_indices:
            tour = X_eval[idx]
            f = F_eval[idx]
            d_tau1 = q0 / max(1e-6, f[0])
            d_tau2 = q0 / max(1e-6, f[1])

            for step_i in range(n_nodes):
                u = tour[step_i]
                v = tour[(step_i + 1) % n_nodes]
                tau1[u, v] += d_tau1
                tau1[v, u] += d_tau1
                tau2[u, v] += d_tau2
                tau2[v, u] += d_tau2

    # 4. Update External Archive
    comb_arch_X = np.vstack((arch_X, X_eval)) if len(arch_X) > 0 else X_eval
    comb_arch_F = np.vstack((arch_F, F_eval)) if len(arch_F) > 0 else F_eval
    comb_arch_CV = np.concatenate((arch_CV, CV_eval)) if len(arch_CV) > 0 else CV_eval

    new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )

    if len(new_arch_X) > archive_size:
        fronts, _, cd = fast_non_dominated_sort(new_arch_F, CV=new_arch_CV, use_deb=True, feas_tol=feas_tol), None, None
        from moo.moo_utils import crowding_distance
        cd = crowding_distance(new_arch_F, list(range(len(new_arch_F))))
        order = np.argsort(-cd)
        keep = order[:archive_size]
        new_arch_X = new_arch_X[keep]
        new_arch_F = new_arch_F[keep]
        new_arch_CV = new_arch_CV[keep]

    state["tau1"] = tau1
    state["tau2"] = tau2
    state["X"] = X_eval
    state["F"] = F_eval
    state["CV"] = CV_eval
    state["arch_X"] = new_arch_X
    state["arch_F"] = new_arch_F
    state["arch_CV"] = new_arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MOACO until budget is exhausted."""
    state = initialize(problem, seed=seed, budget=budget, **kwargs)

    while state["n_evals"] < budget:
        state = step(state, problem)

    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]

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
