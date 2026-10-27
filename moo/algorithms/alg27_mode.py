"""
alg27_mode.py: MODE (Multi-Objective Differential Evolution)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Babu, B. V., & Gujarathi, P. K. (2007).
"Multi-Objective Differential Evolution (MODE) for Optimization of Adiabatic Styrene Reactor."
Chemical Engineering Science, 62(24), 7147-7157.

Algorithmic Logic:
1. Population Initialization:
       Generate population X_i (size N) in [xl, xu] and evaluate objectives and constraints.
2. Differential Evolution Mutation (DE/rand/1):
       For each target vector X_i, select distinct random vectors r1, r2, r3 != i:
           v_i = X_{r1} + F_scale * (X_{r2} - X_{r3})
3. Binomial Crossover:
       Form trial vector u_i:
           u_{ij} = v_{ij} if (rand <= CR or j == j_rand) else X_{ij}
4. Selection via Constrained Pareto Dominance:
       - If u_i constrained-dominates X_i -> X_i = u_i.
       - If X_i constrained-dominates u_i -> discard u_i.
       - If mutually non-dominating -> both are pooled into an intermediate population R_t
         and pruned back to size N via fast non-dominated sorting and crowding distance.
5. External Non-Dominated Archive:
       Stores all Pareto-optimal solutions across generations.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, crowding_distance
from moo.algorithms.common import random_population, extract_non_dominated, order_crossover, swap_mutation
from moo.algorithms.alg14_nsga2 import elitist_environmental_selection


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MODE population, archive, and evaluations within budget."""
    rng = np.random.default_rng(seed)
    pop_size = kwargs.get("pop_size", 50)
    budget = kwargs.get("budget", 10000)
    archive_size = kwargs.get("archive_size", 100)
    F_scale = kwargs.get("F_scale", 0.5)
    CR = kwargs.get("CR", 0.7)

    effective_pop = max(4, min(pop_size, budget))
    X = random_population(problem, effective_pop, rng)

    F_list = []
    CV_list = []
    for ind in X:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    arch_X, arch_F, arch_CV = extract_non_dominated(X, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop,
        "archive_size": archive_size,
        "budget": budget,
        "F_scale": F_scale,
        "CR": CR,
        "X": X,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_pop,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one generation of MODE."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    n_var = problem.n_var
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    F_scale = state["F_scale"]
    CR = state["CR"]

    is_perm = (problem.var_type == "permutation")
    trial_X_list = []

    for i in range(N):
        # Choose 3 distinct vectors != i
        candidates = [idx for idx in range(N) if idx != i]
        r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

        if is_perm:
            # Permutation crossover
            u_i, _ = order_crossover(X[r1], X[r2], rng)
            u_i = swap_mutation(u_i, rng, prob_m=0.2)
        else:
            # DE mutation & crossover
            v_i = X[r1] + F_scale * (X[r2] - X[r3])
            j_rand = rng.integers(0, n_var)
            u_i = X[i].copy()
            for j in range(n_var):
                if rng.random() <= CR or j == j_rand:
                    u_i[j] = v_i[j]

            u_i = np.clip(u_i, problem.xl, problem.xu)
            if problem.var_type == "integer":
                u_i = np.round(u_i)
            elif problem.var_type == "binary":
                u_i = (u_i >= 0.5).astype(float)

        trial_X_list.append(u_i)

    n_to_eval = min(N, rem_budget)
    X_trial = np.array(trial_X_list[:n_to_eval])

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

    # Selection: compare trial against target
    pool_X = []
    pool_F = []
    pool_CV = []

    for i in range(N):
        if i >= n_to_eval:
            pool_X.append(X[i])
            pool_F.append(F[i])
            pool_CV.append(CV[i])
            continue

        u_dom_x = constrained_dominates(F_trial[i], CV_trial[i], F[i], CV[i], feas_tol=feas_tol)
        x_dom_u = constrained_dominates(F[i], CV[i], F_trial[i], CV_trial[i], feas_tol=feas_tol)

        if u_dom_x:
            pool_X.append(X_trial[i])
            pool_F.append(F_trial[i])
            pool_CV.append(CV_trial[i])
        elif x_dom_u:
            pool_X.append(X[i])
            pool_F.append(F[i])
            pool_CV.append(CV[i])
        else:
            # Mutually non-dominating: retain both into environmental pool
            pool_X.append(X[i])
            pool_F.append(F[i])
            pool_CV.append(CV[i])
            pool_X.append(X_trial[i])
            pool_F.append(F_trial[i])
            pool_CV.append(CV_trial[i])

    pool_X_arr = np.array(pool_X)
    pool_F_arr = np.array(pool_F)
    pool_CV_arr = np.array(pool_CV)

    # Prune back to N using non-dominated sorting and crowding distance
    X_new, F_new, CV_new, _, _, _ = elitist_environmental_selection(
        pool_X_arr, pool_F_arr, pool_CV_arr, target_size=N, feas_tol=feas_tol
    )

    # Update Archive
    comb_arch_X = np.vstack((arch_X, X_trial)) if len(arch_X) > 0 else X_trial
    comb_arch_F = np.vstack((arch_F, F_trial)) if len(arch_F) > 0 else F_trial
    comb_arch_CV = np.concatenate((arch_CV, CV_trial)) if len(arch_CV) > 0 else CV_trial

    new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
        comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
    )

    if len(new_arch_X) > state["archive_size"]:
        cd = crowding_distance(new_arch_F, list(range(len(new_arch_F))))
        order = np.argsort(-cd)
        keep = order[: state["archive_size"]]
        new_arch_X = new_arch_X[keep]
        new_arch_F = new_arch_F[keep]
        new_arch_CV = new_arch_CV[keep]

    state["X"] = X_new
    state["F"] = F_new
    state["CV"] = CV_new
    state["arch_X"] = new_arch_X
    state["arch_F"] = new_arch_F
    state["arch_CV"] = new_arch_CV

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MODE until budget is exhausted."""
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
