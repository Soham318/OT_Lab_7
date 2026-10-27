"""
alg16_spea2.py: SPEA2 (Strength Pareto Evolutionary Algorithm 2)

Lecture Family: Modern Elitist Multiobjective Evolutionary Algorithms (Part 3 & Bonus Module 5E, Chapter 5)
Original Citation: Zitzler, E., Laumanns, M., & Thiele, L. (2001).
"SPEA2: Improving the Strength Pareto Evolutionary Algorithm."
TIK-Report 103, Computer Engineering and Networks Laboratory (TIK), Swiss Federal Institute of Technology (ETH) Zurich.

Algorithmic Logic:
1. Population and Archive:
       Maintains an active population P_t of size N and an external archive P_bar_t of fixed size N_bar.
2. Fitness Assignment on P_t U P_bar_t:
       - Strength S(i): Number of solutions that individual i dominates.
       - Raw Fitness R(i): Sum of strengths of all solutions that dominate individual i:
             R(i) = sum_{j in P_t U P_bar_t, j dominates i} S(j).
             (Non-dominated solutions have R(i) = 0).
       - Density Estimation D(i):
             Computes pairwise Euclidean distances in objective space.
             Identifies k-th nearest neighbor distance sigma_i^k, where k = floor(sqrt(N + N_bar)).
             D(i) = 1.0 / (sigma_i^k + 2.0).
       - Total Fitness F(i) = R(i) + D(i) (to be minimized; F(i) < 1 implies non-dominated).
3. Environmental Selection into Next Archive P_bar_{t+1}:
       - Copy all non-dominated solutions (F(i) < 1) to P_bar_{t+1}.
       - Underfill (|P_bar| < N_bar): Select best dominated individuals with smallest F(i) >= 1.
       - Overfill (|P_bar| > N_bar): Iteratively truncate the solution that has the minimum distance
         to its nearest neighbor (tie-breaking with 2nd nearest, etc.) until size equals N_bar.
4. Mating Selection & Variation:
       - Binary tournament selection on P_bar_{t+1} using total fitness F(i).
       - SBX crossover and polynomial mutation produce N offspring P_{t+1}.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import constrained_dominates, dominates, normalize_objectives
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def compute_spea2_fitness(
    F: np.ndarray,
    CV: np.ndarray | None,
    k_nn: int,
    feas_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes SPEA2 Strength S, Raw Fitness R, Density D, and Total Fitness F = R + D."""
    M_total = len(F)
    if M_total == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    apply_deb = (CV is not None)
    cv_vec = np.maximum(0.0, CV) if apply_deb else np.zeros(M_total)

    # 1. Pairwise dominance matrix: dom_matrix[i, j] is True if i dominates j
    dom_matrix = np.zeros((M_total, M_total), dtype=bool)
    for i in range(M_total):
        for j in range(i + 1, M_total):
            if apply_deb:
                i_dom_j = constrained_dominates(F[i], cv_vec[i], F[j], cv_vec[j], feas_tol=feas_tol)
                j_dom_i = constrained_dominates(F[j], cv_vec[j], F[i], cv_vec[i], feas_tol=feas_tol)
            else:
                i_dom_j = dominates(F[i], F[j])
                j_dom_i = dominates(F[j], F[i])
            dom_matrix[i, j] = i_dom_j
            dom_matrix[j, i] = j_dom_i

    # Strength S(i): count of solutions dominated by i
    S = np.sum(dom_matrix, axis=1)

    # Raw fitness R(i): sum of strengths of solutions dominating i
    R = np.zeros(M_total, dtype=float)
    for i in range(M_total):
        dominators = np.where(dom_matrix[:, i])[0]
        R[i] = float(np.sum(S[dominators]))

    # 2. Density estimation D(i) using k-th nearest neighbor distance in normalized objective space
    F_norm = normalize_objectives(F)
    diff = F_norm[:, None, :] - F_norm[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Sort each row to get distances to nearest neighbors (dist_matrix[i, 0] is 0.0 self-distance)
    sorted_dists = np.sort(dist_matrix, axis=1)
    k_idx = min(max(1, k_nn), M_total - 1)
    sigma_k = sorted_dists[:, k_idx]
    D = 1.0 / (sigma_k + 2.0)

    total_fitness = R + D
    return total_fitness, R, D, dist_matrix


def spea2_environmental_selection(
    X_comb: np.ndarray,
    F_comb: np.ndarray,
    CV_comb: np.ndarray,
    archive_size: int,
    k_nn: int,
    feas_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Environmental selection into next archive of size archive_size."""
    total_fit, R, D, dist_matrix = compute_spea2_fitness(F_comb, CV_comb, k_nn, feas_tol=feas_tol)
    M_total = len(F_comb)

    if M_total <= archive_size:
        return X_comb, F_comb, CV_comb, total_fit

    # Identify non-dominated solutions (R == 0)
    non_dom_indices = [i for i in range(M_total) if R[i] < 1e-12]

    if len(non_dom_indices) == archive_size:
        chosen = non_dom_indices
    elif len(non_dom_indices) < archive_size:
        # Underfill: fill remaining slots with best dominated solutions (smallest total fitness)
        sorted_all = np.argsort(total_fit)
        chosen = list(sorted_all[:archive_size])
    else:
        # Overfill: iterative truncation of non-dominated set
        active = list(non_dom_indices)
        sub_dist = dist_matrix[np.ix_(active, active)].copy()

        while len(active) > archive_size:
            # For each active solution, sort its distances to other active solutions
            L_act = len(active)
            np.fill_diagonal(sub_dist, np.inf)

            # Find individual with minimal distance to its neighbor (lexicographically)
            sorted_rows = np.sort(sub_dist, axis=1)

            # Lexicographical comparison across nearest neighbor distances
            worst_local_idx = 0
            for cand in range(1, L_act):
                for col in range(L_act - 1):
                    if sorted_rows[cand, col] < sorted_rows[worst_local_idx, col]:
                        worst_local_idx = cand
                        break
                    elif sorted_rows[cand, col] > sorted_rows[worst_local_idx, col]:
                        break

            active.pop(worst_local_idx)
            sub_dist = np.delete(sub_dist, worst_local_idx, axis=0)
            sub_dist = np.delete(sub_dist, worst_local_idx, axis=1)

        chosen = active

    X_arch = X_comb[chosen]
    F_arch = F_comb[chosen]
    CV_arch = CV_comb[chosen]
    fit_arch = total_fit[chosen]
    return X_arch, F_arch, CV_arch, fit_arch


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 80,
    archive_size: int = 80,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes SPEA2 population and empty archive."""
    rng = np.random.default_rng(seed)
    effective_pop_size = max(1, min(pop_size, budget))
    effective_arch_size = max(1, min(archive_size, budget))

    pop = random_population(problem, effective_pop_size, rng)
    F_list, CV_list = [], []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    # Initialize archive with non-dominated solutions from initial population
    k_nn = int(np.floor(np.sqrt(effective_pop_size + effective_arch_size)))
    arch_X, arch_F, arch_CV, arch_fit = spea2_environmental_selection(
        pop, F, CV, archive_size=effective_arch_size, k_nn=k_nn, feas_tol=problem.feas_tol
    )

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop_size,
        "archive_size": effective_arch_size,
        "k_nn": k_nn,
        "X": pop,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "arch_fit": arch_fit,
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
    """Executes one generation of SPEA2."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    N_bar = state["archive_size"]
    if len(state["arch_X"]) < 2:
        return state

    arch_X = state["arch_X"]
    arch_fit = state["arch_fit"]
    L_arch = len(arch_X)

    # 1. Binary tournament selection from archive (smaller fitness is better)
    mating_pool_indices = []
    for _ in range(N):
        i, j = rng.choice(L_arch, size=2, replace=False)
        winner = i if arch_fit[i] <= arch_fit[j] else j
        mating_pool_indices.append(winner)

    mating_pool = arch_X[mating_pool_indices]

    # 2. Genetic variation (SBX + PM)
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

    # 3. Evaluate offspring
    F_off_list, CV_off_list = [], []
    for ind in X_off:
        f, cv, _ = problem.evaluate(ind)
        F_off_list.append(f)
        CV_off_list.append(cv)

    F_off = np.array(F_off_list, dtype=float)
    CV_off = np.array(CV_off_list, dtype=float)

    state["n_evals"] += n_to_eval
    state["iter"] += 1

    # 4. Environmental selection into next archive
    X_comb = np.vstack((arch_X, X_off))
    F_comb = np.vstack((state["arch_F"], F_off))
    CV_comb = np.concatenate((state["arch_CV"], CV_off))

    k_nn = int(np.floor(np.sqrt(len(X_comb))))
    new_arch_X, new_arch_F, new_arch_CV, new_arch_fit = spea2_environmental_selection(
        X_comb, F_comb, CV_comb, archive_size=N_bar, k_nn=k_nn, feas_tol=problem.feas_tol
    )

    state["X"] = X_off
    state["F"] = F_off
    state["CV"] = CV_off
    state["arch_X"] = new_arch_X
    state["arch_F"] = new_arch_F
    state["arch_CV"] = new_arch_CV
    state["arch_fit"] = new_arch_fit

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    pop_size: int = 80,
    archive_size: int = 80,
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    **kwargs,
) -> dict:
    """Executes the full SPEA2 optimization run up to the evaluation budget."""
    state = initialize(
        problem,
        seed=seed,
        pop_size=pop_size,
        archive_size=archive_size,
        budget=budget,
        **kwargs,
    )
    state["budget"] = budget

    while state["n_evals"] < budget and len(state["arch_X"]) >= 2:
        prev_evals = state["n_evals"]
        step(state, problem, prob_c=prob_c, eta_c=eta_c, eta_m=eta_m)
        if state["n_evals"] == prev_evals:
            break

    X_opt, F_opt, CV_opt = extract_non_dominated(
        state["arch_X"], state["arch_F"], state["arch_CV"], feas_tol=problem.feas_tol
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
