"""
alg29_moba.py: MOBA (Multi-Objective Bat Algorithm)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Yang, X. S. (2011).
"Bat Algorithm for Multi-Objective Optimisation."
International Journal of Bio-Inspired Computation, 3(5), 267-274.

Algorithmic Logic:
1. Bat Population Initialization:
       Initialize bat positions X_i and velocities V_i in [xl, xu].
       Frequencies f_i in [f_min, f_max].
       Initial loudness A_i = A_0 (e.g. 1.0), pulse rate r_i = r_0 (e.g. 0.5).
       Maintain an External Non-Dominated Archive.
2. Global Best Guide Selection:
       Select best guide bat X_* from the non-dominated archive via crowding-distance roulette wheel.
3. Frequency & Velocity Update:
       f_i = f_min + (f_max - f_min) * beta,  beta in [0, 1]
       V_i = V_i + (X_i - X_*) * f_i
       X_cand = clip(X_i + V_i, xl, xu)
4. Local Flight around Best Solution:
       If rand() > r_i:
           X_cand = clip(X_* + epsilon * mean(A), xl, xu)
5. Acceptance & Acoustic Parameter Schedules:
       If X_cand constrained-dominates X_i and rand() < A_i:
           X_i = X_cand
           A_i = alpha * A_i  (loudness decay)
           r_i = r_0 * (1 - exp(-gamma * t))  (pulse emission rate increase)
6. Non-Dominated Archive Maintenance:
       Updated with all accepted bats; pruned when capacity is exceeded.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, crowding_distance
from moo.algorithms.common import random_population, extract_non_dominated, swap_mutation, order_crossover


def select_best_bat(
    arch_X: np.ndarray,
    arch_F: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Selects guide bat X_* from archive using crowding distance."""
    N_arch = len(arch_X)
    if N_arch <= 2:
        return arch_X[rng.choice(N_arch)]

    cd = crowding_distance(arch_F, list(range(N_arch)))
    cd_clean = np.where(cd >= 1e10, 2.0, cd)
    total_cd = np.sum(cd_clean)
    if total_cd <= 1e-12:
        return arch_X[rng.choice(N_arch)]

    probs = cd_clean / total_cd
    chosen_idx = rng.choice(N_arch, p=probs)
    return arch_X[chosen_idx]


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MOBA bats, acoustic schedules, and archive within budget."""
    rng = np.random.default_rng(seed)
    n_bats = kwargs.get("n_bats", kwargs.get("pop_size", 40))
    budget = kwargs.get("budget", 10000)
    archive_size = kwargs.get("archive_size", 100)
    f_min = kwargs.get("f_min", 0.0)
    f_max = kwargs.get("f_max", 2.0)
    A0 = kwargs.get("A0", 1.0)
    r0 = kwargs.get("r0", 0.5)
    alpha = kwargs.get("alpha", 0.95)
    gamma = kwargs.get("gamma", 0.05)

    effective_bats = max(2, min(n_bats, budget))
    X = random_population(problem, effective_bats, rng)

    F_list = []
    CV_list = []
    for ind in X:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    V = np.zeros((effective_bats, problem.n_var), dtype=float)
    frequencies = np.zeros(effective_bats, dtype=float)
    loudness = np.full(effective_bats, A0, dtype=float)
    pulse_rates = np.full(effective_bats, r0, dtype=float)

    arch_X, arch_F, arch_CV = extract_non_dominated(X, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "n_bats": effective_bats,
        "budget": budget,
        "archive_size": archive_size,
        "f_min": f_min,
        "f_max": f_max,
        "A0": A0,
        "r0": r0,
        "alpha": alpha,
        "gamma": gamma,
        "X": X,
        "V": V,
        "F": F,
        "CV": CV,
        "frequencies": frequencies,
        "loudness": loudness,
        "pulse_rates": pulse_rates,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_bats,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one echolocation cycle of MOBA."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["n_bats"]
    n_var = problem.n_var
    feas_tol = problem.feas_tol

    X = state["X"]
    V = state["V"]
    F = state["F"]
    CV = state["CV"]
    loudness = state["loudness"]
    pulse_rates = state["pulse_rates"]
    frequencies = state["frequencies"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    archive_size = state["archive_size"]

    f_min = state["f_min"]
    f_max = state["f_max"]
    alpha = state["alpha"]
    gamma = state["gamma"]
    r0 = state["r0"]
    t = state["iter"]

    is_perm = (problem.var_type == "permutation")
    mean_A = float(np.mean(loudness))

    if len(arch_X) > 0:
        x_best = select_best_bat(arch_X, arch_F, rng)
    else:
        x_best = X[0]

    cand_X_list = []
    for i in range(N):
        if is_perm:
            cand = order_crossover(X[i], x_best, rng)[0]
            cand = swap_mutation(cand, rng, prob_m=0.2)
        else:
            beta = rng.uniform(0.0, 1.0)
            frequencies[i] = f_min + (f_max - f_min) * beta
            V[i] = V[i] + (X[i] - x_best) * frequencies[i]
            cand = X[i] + V[i]

            # Random local walk around best solution
            if rng.random() > pulse_rates[i]:
                span = problem.xu - problem.xl
                epsilon = rng.uniform(-1.0, 1.0, size=n_var)
                cand = x_best + epsilon * (0.01 * span) * mean_A

            cand = np.clip(cand, problem.xl, problem.xu)
            if problem.var_type == "integer":
                cand = np.round(cand)
            elif problem.var_type == "binary":
                cand = (cand >= 0.5).astype(float)

        cand_X_list.append(cand)

    n_to_eval = min(N, rem_budget)
    X_trial = np.array(cand_X_list[:n_to_eval])

    F_trial_list = []
    CV_trial_list = []
    for ind in X_trial:
        f, cv, _ = problem.evaluate(ind)
        F_trial_list.append(f)
        CV_trial_list.append(cv)

    F_trial = np.array(F_trial_list, dtype=float)
    CV_trial = np.array(CV_trial_list, dtype=float)

    state["n_evals"] += n_to_eval
    state["iter"] += 1

    # Acceptance check
    for i in range(n_to_eval):
        cand_dom_cur = constrained_dominates(F_trial[i], CV_trial[i], F[i], CV[i], feas_tol=feas_tol)
        cur_dom_cand = constrained_dominates(F[i], CV[i], F_trial[i], CV_trial[i], feas_tol=feas_tol)

        # Accept if candidate dominates current and loudness check passes
        if (cand_dom_cur or not cur_dom_cand) and (rng.random() < loudness[i]):
            X[i] = X_trial[i]
            F[i] = F_trial[i]
            CV[i] = CV_trial[i]

            # Update loudness and pulse rate
            loudness[i] = alpha * loudness[i]
            pulse_rates[i] = r0 * (1.0 - np.exp(-gamma * t))

    # Update Archive
    comb_arch_X = np.vstack((arch_X, X_trial)) if len(arch_X) > 0 else X_trial
    comb_arch_F = np.vstack((arch_F, F_trial)) if len(arch_F) > 0 else F_trial
    comb_arch_CV = np.concatenate((arch_CV, CV_trial)) if len(arch_CV) > 0 else CV_trial

    new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )

    if len(new_arch_X) > archive_size:
        cd = crowding_distance(new_arch_F, list(range(len(new_arch_F))))
        order = np.argsort(-cd)
        keep = order[:archive_size]
        new_arch_X = new_arch_X[keep]
        new_arch_F = new_arch_F[keep]
        new_arch_CV = new_arch_CV[keep]

    state["X"] = X
    state["V"] = V
    state["F"] = F
    state["CV"] = CV
    state["loudness"] = loudness
    state["pulse_rates"] = pulse_rates
    state["frequencies"] = frequencies
    state["arch_X"] = new_arch_X
    state["arch_F"] = new_arch_F
    state["arch_CV"] = new_arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MOBA until budget is exhausted."""
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
