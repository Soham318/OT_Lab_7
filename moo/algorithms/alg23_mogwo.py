"""
alg23_mogwo.py: MOGWO (Multi-Objective Grey Wolf Optimizer)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Mirjalili, S., Saremi, S., Mirjalili, S. M., & Coelho, L. D. S. (2016).
"Multi-Objective Grey Wolf Optimizer: A Novel Algorithm for Multi-Criterion Optimization."
Expert Systems with Applications, 47, 106-119.

Algorithmic Logic:
1. Grey Wolf Pack Initialization:
       Initialize pack positions X_i within variable bounds [xl, xu].
       Evaluate wolves and populate an External Non-Dominated Archive (repository).
2. Adaptive Hypergrid Partitioning:
       Partition objective space into n_div hypercubes.
       Grid fitness fit_k = 10.0 / N_k prioritizes under-populated regions for diversity.
3. Alpha, Beta, Delta Leader Selection:
       Select three guide wolves (alpha, beta, delta) from the external archive via roulette wheel
       over hypercube fitness.
4. Encircling & Hunting Update:
       a decays linearly from 2 to 0 over iterations.
       For leader L in {alpha, beta, delta}:
           A = 2 * a * r1 - a,  C = 2 * r2
           D_L = |C * X_L - X_i|
           X_step = X_L - A * D_L
       X_new = (X_1 + X_2 + X_3) / 3.0, clipped to [xl, xu].
5. Archive Pruning & Maintenance:
       Incorporate newly evaluated wolves into the archive, filter dominated points,
       and prune overcrowded cells when archive capacity is reached.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import random_population, extract_non_dominated, order_crossover, swap_mutation
from moo.algorithms.alg22_mopso import compute_grid_indices, prune_repository


def select_leader_from_archive(
    arch_X: np.ndarray,
    arch_F: np.ndarray,
    n_div: int,
    rng: np.random.Generator,
    exclude_indices: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    """Selects a leader (alpha, beta, or delta) from the archive using roulette wheel over grid fitness."""
    N_arch = len(arch_X)
    if N_arch == 0:
        raise ValueError("Archive cannot be empty when selecting leaders")
    if N_arch == 1:
        return arch_X[0], 0

    if exclude_indices is None:
        exclude_indices = []

    available_indices = [idx for idx in range(N_arch) if idx not in exclude_indices]
    if len(available_indices) == 0:
        available_indices = list(range(N_arch))

    avail_X = arch_X[available_indices]
    avail_F = arch_F[available_indices]

    cell_ids, _, _ = compute_grid_indices(avail_F, n_div=n_div)
    unique_cells, counts = np.unique(cell_ids, return_counts=True)

    cell_fitness = 10.0 / counts
    probs = cell_fitness / np.sum(cell_fitness)

    chosen_cell = rng.choice(unique_cells, p=probs)
    candidates = np.where(cell_ids == chosen_cell)[0]
    local_idx = rng.choice(candidates)
    orig_idx = available_indices[local_idx]
    return arch_X[orig_idx], orig_idx


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MOGWO pack, archive, and evaluations within budget."""
    rng = np.random.default_rng(seed)
    pack_size = kwargs.get("pack_size", kwargs.get("pop_size", 50))
    budget = kwargs.get("budget", 10000)
    archive_size = kwargs.get("archive_size", 100)
    n_div = kwargs.get("n_div", 10)

    effective_size = max(1, min(pack_size, budget))
    X = random_population(problem, effective_size, rng)

    F_list = []
    CV_list = []
    for ind in X:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    arch_X, arch_F, arch_CV = extract_non_dominated(X, F, CV, feas_tol=problem.feas_tol)
    if len(arch_X) > archive_size:
        arch_X, arch_F, arch_CV = prune_repository(arch_X, arch_F, arch_CV, archive_size, n_div, rng)

    state = {
        "seed": seed,
        "rng": rng,
        "pack_size": effective_size,
        "archive_size": archive_size,
        "n_div": n_div,
        "X": X,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_size,
        "budget": budget,
        "iter": 0,
        "max_iter": max(1, budget // effective_size),
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one hunting cycle of MOGWO."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pack_size"]
    n_var = problem.n_var
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    archive_size = state["archive_size"]
    n_div = state["n_div"]

    # Parameter a linearly decreases from 2 to 0
    t = state["iter"]
    max_iter = state["max_iter"]
    a = 2.0 - 2.0 * min(1.0, t / max_iter)

    # 1. Select Alpha, Beta, Delta leaders from archive
    if len(arch_X) >= 3:
        leader_alpha, idx_a = select_leader_from_archive(arch_X, arch_F, n_div, rng)
        leader_beta, idx_b = select_leader_from_archive(arch_X, arch_F, n_div, rng, exclude_indices=[idx_a])
        leader_delta, _ = select_leader_from_archive(arch_X, arch_F, n_div, rng, exclude_indices=[idx_a, idx_b])
    elif len(arch_X) == 2:
        leader_alpha = arch_X[0]
        leader_beta = arch_X[1]
        leader_delta = arch_X[rng.choice(2)]
    elif len(arch_X) == 1:
        leader_alpha = leader_beta = leader_delta = arch_X[0]
    else:
        leader_alpha = leader_beta = leader_delta = X[0]

    is_perm = (problem.var_type == "permutation")
    new_X_list = []

    for i in range(N):
        if is_perm:
            # Combinatorial permutation update: OX with leaders
            child, _ = order_crossover(X[i], leader_alpha, rng)
            if rng.random() < 0.5:
                child, _ = order_crossover(child, leader_beta, rng)
            child = swap_mutation(child, rng, prob_m=0.2)
            new_X_list.append(child)
        else:
            # Continuous hunting equations
            # Alpha influence
            r1 = rng.uniform(0.0, 1.0, size=n_var)
            r2 = rng.uniform(0.0, 1.0, size=n_var)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * leader_alpha - X[i])
            X1 = leader_alpha - A1 * D_alpha

            # Beta influence
            r1 = rng.uniform(0.0, 1.0, size=n_var)
            r2 = rng.uniform(0.0, 1.0, size=n_var)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * leader_beta - X[i])
            X2 = leader_beta - A2 * D_beta

            # Delta influence
            r1 = rng.uniform(0.0, 1.0, size=n_var)
            r2 = rng.uniform(0.0, 1.0, size=n_var)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * leader_delta - X[i])
            X3 = leader_delta - A3 * D_delta

            x_new = (X1 + X2 + X3) / 3.0
            x_new = np.clip(x_new, problem.xl, problem.xu)

            if problem.var_type == "integer":
                x_new = np.round(x_new)
            elif problem.var_type == "binary":
                x_new = (x_new >= 0.5).astype(float)

            new_X_list.append(x_new)

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

    X[:n_to_eval] = X_eval
    F[:n_to_eval] = F_eval
    CV[:n_to_eval] = CV_eval

    # Update Archive
    comb_arch_X = np.vstack((arch_X, X_eval)) if len(arch_X) > 0 else X_eval
    comb_arch_F = np.vstack((arch_F, F_eval)) if len(arch_F) > 0 else F_eval
    comb_arch_CV = np.concatenate((arch_CV, CV_eval)) if len(arch_CV) > 0 else CV_eval

    new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )

    if len(new_arch_X) > archive_size:
        new_arch_X, new_arch_F, new_arch_CV = prune_repository(
            new_arch_X, new_arch_F, new_arch_CV, archive_size, n_div, rng
        )

    state["X"] = X
    state["F"] = F
    state["CV"] = CV
    state["arch_X"] = new_arch_X
    state["arch_F"] = new_arch_F
    state["arch_CV"] = new_arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MOGWO until budget is exhausted."""
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
