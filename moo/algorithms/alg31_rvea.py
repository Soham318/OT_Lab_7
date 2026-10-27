"""
alg31_rvea.py: RVEA (Reference Vector Guided Evolutionary Algorithm)

Lecture Family: Many-Objective, Indicator, Preference & Gradient Methods (Bonus Module 5E, Chapter 5)
Original Citation: Cheng, R., Jin, Y., Olhofer, M., & Sendhoff, B. (2016).
"A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization."
IEEE Transactions on Evolutionary Computation, 20(5), 773-791.

Algorithmic Logic:
1. Reference Vector Initialization:
       Generate uniform reference vectors V = {v_1, ..., v_K} via the Das-Dennis method,
       normalized to unit length: v_k = v_k / ||v_k||_2.
       Compute minimum angle between reference vectors: gamma_k = min_{j != k} angle(v_k, v_j).
2. Objective Translation:
       Translate objectives using the current ideal point z*:
           f'_m(x) = f_m(x) - z*_m
3. Reference Vector Association:
       Associate each individual i with the reference vector v_k that minimizes the acute angle:
           theta_{i, k} = arccos( (f'_i . v_k) / (||f'_i||_2 * ||v_k||_2) )
4. Angle-Penalized Distance (APD):
       APD(x_i) = (1 + M * (t / t_max)^alpha * (theta_{i, k} / gamma_k)) * ||f'_i||_2
       - Early generations (t -> 0): APD is dominated by ||f'||_2 (convergence).
       - Late generations (t -> t_max): APD heavily penalizes angle deviation (diversity).
5. Environmental Selection:
       From each reference vector's associated candidates, select the one with minimum APD.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import das_dennis, fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def compute_gamma(V: np.ndarray) -> np.ndarray:
    """Computes minimum angle between each reference vector and all others."""
    K, M = V.shape
    gamma = np.zeros(K)
    for k in range(K):
        angles = []
        for j in range(K):
            if j != k:
                cos_val = np.clip(np.dot(V[k], V[j]), -1.0, 1.0)
                angles.append(np.arccos(cos_val))
        gamma[k] = min(angles) if len(angles) > 0 else 1.0
    return np.where(gamma <= 1e-6, 1e-6, gamma)


def associate_and_compute_apd(
    F: np.ndarray,
    CV: np.ndarray,
    V: np.ndarray,
    gamma: np.ndarray,
    ideal: np.ndarray,
    progress: float,
    alpha: float = 2.0,
    feas_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Translates objectives, associates each candidate with its closest reference vector,
    and calculates Angle-Penalized Distance (APD).
    Returns (associations, apd_values).
    """
    N, M = F.shape
    K = len(V)

    # 1. Translate objectives
    F_trans = F - ideal.reshape(1, -1)
    F_trans = np.maximum(0.0, F_trans)

    norms = np.linalg.norm(F_trans, axis=1)
    norms_clean = np.where(norms < 1e-12, 1e-12, norms)

    # Cosine angles with all reference vectors: (N, K)
    cos_angles = np.dot(F_trans / norms_clean[:, None], V.T)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    associations = np.argmin(angles, axis=1)

    # Calculate APD
    apd = np.zeros(N)
    time_factor = M * (progress ** alpha)

    for i in range(N):
        k = associations[i]
        theta = angles[i, k]
        penalty = 1.0 + time_factor * (theta / gamma[k])
        # If infeasible, add huge constraint penalty
        if CV[i] > feas_tol:
            apd[i] = 1e6 + CV[i] * 1000.0
        else:
            apd[i] = penalty * norms[i]

    return associations, apd


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes RVEA reference vectors, population, and evaluations within budget."""
    rng = np.random.default_rng(seed)
    M = problem.n_obj
    budget = kwargs.get("budget", 10000)

    # Generate Das-Dennis reference vectors
    p = 12 if M == 2 else (6 if M == 3 else 3)
    V_raw = das_dennis(p, M)
    V_norms = np.linalg.norm(V_raw, axis=1, keepdims=True)
    V = V_raw / np.where(V_norms < 1e-12, 1.0, V_norms)
    K = len(V)

    gamma = compute_gamma(V)

    pop_size = kwargs.get("pop_size", K)
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

    arch_X, arch_F, arch_CV = extract_non_dominated(pop, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop,
        "budget": budget,
        "V": V,
        "gamma": gamma,
        "ideal": ideal,
        "alpha": kwargs.get("alpha", 2.0),
        "X": pop,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "archive_X": arch_X,
        "archive_F": arch_F,
        "archive_CV": arch_CV,
        "n_evals": effective_pop,
        "iter": 0,
        "max_iter": max(1, budget // effective_pop),
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one generation of RVEA with reference vector guided selection."""
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
    V = state["V"]
    gamma = state["gamma"]
    ideal = state["ideal"]
    alpha = state["alpha"]

    # 1. Random mating selection & variation
    mating_indices = rng.choice(N, size=N, replace=True)
    mating_pool = X[mating_indices]

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

    # 2. Combine pool
    X_comb = np.vstack((X, X_off))
    F_comb = np.vstack((F, F_off))
    CV_comb = np.concatenate((CV, CV_off))

    # Update ideal point
    ideal = np.minimum(ideal, np.min(F_comb, axis=0))
    state["ideal"] = ideal

    # 3. Association and APD selection
    progress = min(1.0, float(state["iter"]) / max(1, state["max_iter"]))
    assoc, apd = associate_and_compute_apd(
        F_comb, CV_comb, V, gamma, ideal, progress, alpha=alpha, feas_tol=feas_tol
    )

    selected_indices = []
    K = len(V)
    for k in range(K):
        candidates = np.where(assoc == k)[0]
        if len(candidates) > 0:
            best_in_cell = candidates[np.argmin(apd[candidates])]
            selected_indices.append(best_in_cell)

    # If fewer than N selected, fill with remaining smallest APD
    if len(selected_indices) < N:
        remaining = [idx for idx in range(len(X_comb)) if idx not in selected_indices]
        order = np.argsort(apd[remaining])
        needed = N - len(selected_indices)
        selected_indices.extend([remaining[i] for i in order[:needed]])
    elif len(selected_indices) > N:
        # If more than N, pick top N by APD
        order = np.argsort(apd[selected_indices])
        selected_indices = [selected_indices[i] for i in order[:N]]

    sel = np.array(selected_indices, dtype=int)
    X_new = X_comb[sel]
    F_new = F_comb[sel]
    CV_new = CV_comb[sel]

    # 4. Update Archive
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
    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs RVEA until budget is exhausted."""
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
