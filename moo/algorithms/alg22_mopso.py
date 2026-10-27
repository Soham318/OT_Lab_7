"""
alg22_mopso.py: MOPSO (Multi-Objective Particle Swarm Optimization)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004).
"Handling Multiple Objectives with Particle Swarm Optimization."
IEEE Transactions on Evolutionary Computation, 8(3), 256-279.

Algorithmic Logic:
1. Swarm Initialization:
       Initialize particle positions X_i and velocities V_i within variable bounds.
       Evaluate initial particles and set pBest_i = X_i.
       Store non-dominated solutions into an External Repository (archive).
2. Adaptive Hypergrid Decomposition:
       Divide the objective space occupied by the repository into n_div equal segments per objective.
       Each repository member is assigned to a grid cell.
       Grid cell fitness is computed as fit_k = 10.0 / N_k (inversely proportional to crowding).
3. Global Best (gBest) Leader Selection:
       Roulette-wheel selection over grid cell fitness chooses a hypercube.
       A particle from that hypercube is selected at random as gBest for particle i.
4. Velocity & Position Update:
       V_i(t+1) = w * V_i(t) + c1 * r1 * (pBest_i - X_i(t)) + c2 * r2 * (gBest - X_i(t))
       X_i(t+1) = clip(X_i(t) + V_i(t+1), xl, xu)
       Apply boundary reflection/clipping and turbulence/mutation.
5. pBest and Repository Maintenance:
       Update pBest_i using Pareto constrained dominance.
       Update External Repository with non-dominated particles, pruning overcrowded grid cells when full.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import random_population, extract_non_dominated, polynomial_mutation


def compute_grid_indices(
    F: np.ndarray,
    n_div: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes grid cell IDs for an objective matrix F in an adaptive hypercube mesh.
    Returns: (cell_ids, f_min, cell_widths).
    """
    N, M = F.shape
    if N == 0:
        return np.empty(0, dtype=int), np.empty(0), np.empty(0)

    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)

    # Pad slightly to prevent border solutions falling out
    f_min_pad = f_min - 1e-6
    f_max_pad = f_max + 1e-6

    cell_widths = (f_max_pad - f_min_pad) / max(1, n_div)
    cell_widths = np.where(cell_widths < 1e-12, 1.0, cell_widths)

    coords = np.floor((F - f_min_pad) / cell_widths).astype(int)
    coords = np.clip(coords, 0, n_div - 1)

    # Map M-dimensional coordinate into 1D cell ID
    multipliers = np.array([n_div ** m for m in range(M)], dtype=int)
    cell_ids = np.dot(coords, multipliers)

    return cell_ids, f_min_pad, cell_widths


def select_gbest_from_repository(
    rep_X: np.ndarray,
    rep_F: np.ndarray,
    n_div: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Selects a leader gBest from the external repository using adaptive grid roulette wheel."""
    N_rep = len(rep_X)
    if N_rep == 0:
        raise ValueError("Repository cannot be empty when selecting gBest")
    if N_rep == 1:
        return rep_X[0]

    cell_ids, _, _ = compute_grid_indices(rep_F, n_div=n_div)
    unique_cells, counts = np.unique(cell_ids, return_counts=True)

    # Fitness inversely proportional to density (fit = 10.0 / count)
    cell_fitness = 10.0 / counts
    probs = cell_fitness / np.sum(cell_fitness)

    chosen_cell = rng.choice(unique_cells, p=probs)
    candidates = np.where(cell_ids == chosen_cell)[0]
    chosen_idx = rng.choice(candidates)
    return rep_X[chosen_idx]


def prune_repository(
    rep_X: np.ndarray,
    rep_F: np.ndarray,
    rep_CV: np.ndarray,
    max_size: int,
    n_div: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prunes external repository down to max_size by removing particles from the most crowded cells."""
    while len(rep_X) > max_size:
        cell_ids, _, _ = compute_grid_indices(rep_F, n_div=n_div)
        unique_cells, counts = np.unique(cell_ids, return_counts=True)

        # Deletion probability proportional to count
        del_probs = counts / np.sum(counts)
        chosen_cell = rng.choice(unique_cells, p=del_probs)
        candidates = np.where(cell_ids == chosen_cell)[0]
        drop_idx = rng.choice(candidates)

        rep_X = np.delete(rep_X, drop_idx, axis=0)
        rep_F = np.delete(rep_F, drop_idx, axis=0)
        rep_CV = np.delete(rep_CV, drop_idx, axis=0)

    return rep_X, rep_F, rep_CV


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MOPSO swarm, velocities, pBest, and repository within budget."""
    rng = np.random.default_rng(seed)
    swarm_size = kwargs.get("swarm_size", kwargs.get("pop_size", 50))
    budget = kwargs.get("budget", 10000)
    rep_size = kwargs.get("rep_size", 100)
    n_div = kwargs.get("n_div", 10)

    effective_size = max(1, min(swarm_size, budget))
    X = random_population(problem, effective_size, rng)

    F_list = []
    CV_list = []
    for ind in X:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    # Initial velocities in [-0.1*(xu-xl), 0.1*(xu-xl)]
    span = problem.xu - problem.xl
    span = np.where(span <= 0, 1.0, span)
    V = rng.uniform(-0.1 * span, 0.1 * span, size=(effective_size, problem.n_var))

    pBest_X = X.copy()
    pBest_F = F.copy()
    pBest_CV = CV.copy()

    # Initial repository contains non-dominated particles
    rep_X, rep_F, rep_CV = extract_non_dominated(X, F, CV, feas_tol=problem.feas_tol)
    if len(rep_X) > rep_size:
        rep_X, rep_F, rep_CV = prune_repository(rep_X, rep_F, rep_CV, rep_size, n_div, rng)

    state = {
        "seed": seed,
        "rng": rng,
        "swarm_size": effective_size,
        "rep_size": rep_size,
        "n_div": n_div,
        "X": X,
        "V": V,
        "F": F,
        "CV": CV,
        "pBest_X": pBest_X,
        "pBest_F": pBest_F,
        "pBest_CV": pBest_CV,
        "rep_X": rep_X,
        "rep_F": rep_F,
        "rep_CV": rep_CV,
        "n_evals": effective_size,
        "budget": budget,
        "iter": 0,
        "w": kwargs.get("w", 0.4),
        "c1": kwargs.get("c1", 1.5),
        "c2": kwargs.get("c2", 1.5),
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one generation/cycle of MOPSO."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["swarm_size"]
    n_var = problem.n_var
    feas_tol = problem.feas_tol

    X = state["X"]
    V = state["V"]
    F = state["F"]
    CV = state["CV"]
    pBest_X = state["pBest_X"]
    pBest_F = state["pBest_F"]
    pBest_CV = state["pBest_CV"]
    rep_X = state["rep_X"]
    rep_F = state["rep_F"]
    rep_CV = state["rep_CV"]

    w = state["w"]
    c1 = state["c1"]
    c2 = state["c2"]
    n_div = state["n_div"]
    rep_size = state["rep_size"]

    is_perm = (problem.var_type == "permutation")
    span = problem.xu - problem.xl
    v_max = 0.5 * np.where(span <= 0, 1.0, span)

    new_X_list = []
    new_V_list = []

    for i in range(N):
        # Leader selection from repository
        if len(rep_X) > 0:
            gBest = select_gbest_from_repository(rep_X, rep_F, n_div, rng)
        else:
            gBest = pBest_X[rng.choice(N)]

        if is_perm:
            # Permutation operator: swap towards pBest and gBest
            cur = X[i].copy()
            if rng.random() < 0.5:
                # Crossover-like bias toward gBest
                pt1, pt2 = sorted(rng.choice(n_var, size=2, replace=False))
                sub = gBest[pt1 : pt2 + 1]
                rem = [c for c in cur if c not in sub]
                cur = np.array(rem[:pt1] + list(sub) + rem[pt1:])
            else:
                idx1, idx2 = rng.choice(n_var, size=2, replace=False)
                cur[idx1], cur[idx2] = cur[idx2], cur[idx1]
            new_X_list.append(cur)
            new_V_list.append(V[i])
        else:
            r1 = rng.uniform(0.0, 1.0, size=n_var)
            r2 = rng.uniform(0.0, 1.0, size=n_var)

            v_new = w * V[i] + c1 * r1 * (pBest_X[i] - X[i]) + c2 * r2 * (gBest - X[i])
            v_new = np.clip(v_new, -v_max, v_max)
            x_new = X[i] + v_new

            # Boundary handling with reflection
            below = x_new < problem.xl
            above = x_new > problem.xu
            v_new[below | above] = -0.5 * v_new[below | above]
            x_new = np.clip(x_new, problem.xl, problem.xu)

            # Turbulence / mutation with probability 0.1
            if rng.random() < 0.1:
                x_new = polynomial_mutation(x_new, problem.xl, problem.xu, eta_m=20.0, rng=rng)

            if problem.var_type == "integer":
                x_new = np.round(x_new)
            elif problem.var_type == "binary":
                x_new = (x_new >= 0.5).astype(float)

            new_X_list.append(x_new)
            new_V_list.append(v_new)

    n_to_eval = min(N, rem_budget)
    X_eval = np.array(new_X_list[:n_to_eval])

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

    # Update swarm state
    X[:n_to_eval] = X_eval
    F[:n_to_eval] = F_eval
    CV[:n_to_eval] = CV_eval
    V[:n_to_eval] = np.array(new_V_list[:n_to_eval])

    # Update pBest
    for i in range(n_to_eval):
        # Deb's constrained dominance comparison: does new position dominate pBest?
        new_dom_pbest = constrained_dominates(F_eval[i], CV_eval[i], pBest_F[i], pBest_CV[i], feas_tol=feas_tol)
        pbest_dom_new = constrained_dominates(pBest_F[i], pBest_CV[i], F_eval[i], CV_eval[i], feas_tol=feas_tol)

        if new_dom_pbest:
            pBest_X[i] = X_eval[i].copy()
            pBest_F[i] = F_eval[i].copy()
            pBest_CV[i] = CV_eval[i]
        elif not pbest_dom_new:
            # Mutually non-dominating: choose randomly
            if rng.random() < 0.5:
                pBest_X[i] = X_eval[i].copy()
                pBest_F[i] = F_eval[i].copy()
                pBest_CV[i] = CV_eval[i]

    # Update Repository
    comb_rep_X = np.vstack((rep_X, X_eval)) if len(rep_X) > 0 else X_eval
    comb_rep_F = np.vstack((rep_F, F_eval)) if len(rep_F) > 0 else F_eval
    comb_rep_CV = np.concatenate((rep_CV, CV_eval)) if len(rep_CV) > 0 else CV_eval

    new_rep_X, new_rep_F, new_rep_CV = extract_non_dominated(
        comb_rep_X, comb_rep_F, comb_rep_CV, feas_tol=feas_tol
    )

    if len(new_rep_X) > rep_size:
        new_rep_X, new_rep_F, new_rep_CV = prune_repository(
            new_rep_X, new_rep_F, new_rep_CV, rep_size, n_div, rng
        )

    state["X"] = X
    state["V"] = V
    state["F"] = F
    state["CV"] = CV
    state["pBest_X"] = pBest_X
    state["pBest_F"] = pBest_F
    state["pBest_CV"] = pBest_CV
    state["rep_X"] = new_rep_X
    state["rep_F"] = new_rep_F
    state["rep_CV"] = new_rep_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MOPSO until budget is exhausted."""
    state = initialize(problem, seed=seed, budget=budget, **kwargs)

    while state["n_evals"] < budget:
        state = step(state, problem)

    rep_X = state["rep_X"]
    rep_F = state["rep_F"]
    rep_CV = state["rep_CV"]

    # Filter feasible if available
    is_feas = (rep_CV <= problem.feas_tol)
    if np.any(is_feas):
        opt_X = rep_X[is_feas]
        opt_F = rep_F[is_feas]
        opt_CV = rep_CV[is_feas]
    else:
        opt_X = rep_X
        opt_F = rep_F
        opt_CV = rep_CV

    return {
        "X_opt": opt_X,
        "F_opt": opt_F,
        "CV_opt": opt_CV,
        "n_evals": state["n_evals"],
        "history": [],
        "state": state,
    }
