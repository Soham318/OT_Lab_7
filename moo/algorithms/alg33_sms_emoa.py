"""
alg33_sms_emoa.py: SMS-EMOA (S-Metric Selection Evolutionary Multi-Objective Algorithm)

Lecture Family: Many-Objective, Indicator, Preference & Gradient Methods (Bonus Module 5E, Chapter 5)
Original Citation: Beume, N., Naujoks, B., & Emmerich, M. (2007).
"SMS-EMOA: Multiobjective Selection Based on Dominated Hypervolume."
European Journal of Operational Research, 181(3), 1653-1669.

Algorithmic Logic:
1. Steady-State (mu + 1) Framework:
       Maintain a population P of size mu.
       Each generation produces exactly ONE offspring q_t through crossover and mutation.
2. Fast Non-Dominated Sorting:
       Partition combined pool R_t = P U {q_t} (size mu + 1) into fronts F_1, F_2, ..., F_L.
3. Hypervolume Contribution Truncation:
       If the worst front F_L contains only one individual, discard it.
       If |F_L| > 1:
           Compute the exclusive hypervolume contribution Delta S(s, F_L) for each s in F_L:
               Delta S(s, F_L) = S(F_L) - S(F_L \\ {s})
           with reference point y_ref = max_{s in F_L} f_m(s) + 1.0.
           Discard the individual s* with minimal Delta S.
4. Non-Dominated External Archive:
       Maintains non-dominated solutions discovered throughout the run.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, hypervolume_2d, hypervolume_monte_carlo
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    extract_non_dominated,
)


def compute_exclusive_hv_contribution(
    F_front: np.ndarray,
    ref_point: np.ndarray,
) -> np.ndarray:
    """
    Computes exclusive hypervolume contribution Delta S(s, F_front) for each solution in the front.
    """
    N, M = F_front.shape
    if N <= 1:
        return np.ones(N)

    # In 2D, exact analytical rectangular contribution
    if M == 2:
        # Sort by f1 ascending
        order = np.argsort(F_front[:, 0])
        F_sorted = F_front[order]
        contrib = np.zeros(N)

        for i in range(N):
            # Left bound (next smaller f1) or ref
            f1_curr = F_sorted[i, 0]
            f2_curr = F_sorted[i, 1]

            # Right neighbor in f1 gives width
            right_f1 = F_sorted[i + 1, 0] if i + 1 < N else ref_point[0]
            # Left neighbor in f2 (which is above i) gives height
            above_f2 = F_sorted[i - 1, 1] if i > 0 else ref_point[1]

            w = max(0.0, right_f1 - f1_curr)
            h = max(0.0, above_f2 - f2_curr)
            contrib[i] = w * h

        # Map back to original order
        orig_contrib = np.zeros(N)
        orig_contrib[order] = contrib
        return orig_contrib

    # For M >= 3: Monte Carlo or subset difference
    total_hv = hypervolume_monte_carlo(F_front, ref_point, n_samples=10000, seed=42)
    contrib = np.zeros(N)
    for i in range(N):
        mask = [j for j in range(N) if j != i]
        sub_hv = hypervolume_monte_carlo(F_front[mask], ref_point, n_samples=10000, seed=42)
        contrib[i] = max(0.0, total_hv - sub_hv)

    return contrib


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes SMS-EMOA population (mu) and evaluations within budget."""
    rng = np.random.default_rng(seed)
    mu = kwargs.get("mu", kwargs.get("pop_size", 40))
    budget = kwargs.get("budget", 10000)

    effective_mu = max(2, min(mu, budget))
    pop = random_population(problem, effective_mu, rng)

    F_list = []
    CV_list = []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    arch_X, arch_F, arch_CV = extract_non_dominated(pop, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "mu": effective_mu,
        "budget": budget,
        "X": pop,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_mu,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one steady-state (mu + 1) step of SMS-EMOA."""
    budget = state.get("budget", 1000000)
    if state["n_evals"] >= budget:
        return state

    rng = state["rng"]
    mu = state["mu"]
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]

    # 1. Select two random parents
    i, j = rng.choice(mu, size=2, replace=False)
    p1, p2 = X[i], X[j]

    is_perm = (problem.var_type == "permutation")
    if is_perm:
        c1, _ = order_crossover(p1, p2, rng)
        c1 = swap_mutation(c1, rng)
    else:
        c1, _ = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=15.0, prob_c=0.9, rng=rng)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=20.0, rng=rng)
        if problem.var_type == "integer":
            c1 = np.round(c1)
        elif problem.var_type == "binary":
            c1 = (c1 >= 0.5).astype(float)

    f_off, cv_off, _ = problem.evaluate(c1)
    state["n_evals"] += 1
    state["iter"] += 1

    # 2. Combine to size (mu + 1)
    X_comb = np.vstack((X, c1.reshape(1, -1)))
    F_comb = np.vstack((F, f_off.reshape(1, -1)))
    CV_comb = np.concatenate((CV, [cv_off]))

    # 3. Non-dominated sort
    fronts = fast_non_dominated_sort(F_comb, CV=CV_comb, use_deb=True, feas_tol=feas_tol)
    worst_front = fronts[-1]

    if len(worst_front) == 1:
        drop_idx = worst_front[0]
    else:
        # Calculate exclusive hypervolume contribution on worst front
        F_last = F_comb[worst_front]
        ref = np.max(F_last, axis=0) + 1.0
        contrib = compute_exclusive_hv_contribution(F_last, ref)
        min_local = int(np.argmin(contrib))
        drop_idx = worst_front[min_local]

    # Discard drop_idx
    keep = [idx for idx in range(len(X_comb)) if idx != drop_idx]
    state["X"] = X_comb[keep]
    state["F"] = F_comb[keep]
    state["CV"] = CV_comb[keep]

    # 4. Update Archive
    comb_arch_X = np.vstack((arch_X, c1.reshape(1, -1)))
    comb_arch_F = np.vstack((arch_F, f_off.reshape(1, -1)))
    comb_arch_CV = np.concatenate((arch_CV, [cv_off]))

    new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )

    if len(new_arch_X) > 200:
        from moo.moo_utils import crowding_distance
        cd = crowding_distance(new_arch_F, list(range(len(new_arch_F))))
        order = np.argsort(-cd)
        keep_arch = order[:200]
        new_arch_X = new_arch_X[keep_arch]
        new_arch_F = new_arch_F[keep_arch]
        new_arch_CV = new_arch_CV[keep_arch]

    state["arch_X"] = new_arch_X
    state["arch_F"] = new_arch_F
    state["arch_CV"] = new_arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs SMS-EMOA until budget is exhausted."""
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
