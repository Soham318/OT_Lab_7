"""
alg26_mosa.py: MOSA (Multi-Objective Simulated Annealing)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Suppapitnarm, A., Seffen, K. A., Parks, G. T., & Clarkson, P. J. (2000).
"A Simulated Annealing Algorithm for Multiobjective Optimization."
Engineering Optimization, 33(1), 59-85.

Algorithmic Logic:
1. Temperature Schedule & Initial State:
       Initial temperature T_0, cooling rate alpha in (0, 1), and minimum temperature T_min.
       Initialize current state x_cur and evaluate objectives and CV.
       Maintain an External Non-Dominated Archive.
2. Neighborhood Perturbation:
       Generate candidate neighbor x_cand via Gaussian perturbation (or swap for permutation).
3. Multiobjective Acceptance Probability:
       - Case 1: If x_cand constrained-dominates x_cur -> ACCEPT unconditionally.
       - Case 2: If x_cand and x_cur are mutually non-dominating:
                 If x_cand is non-dominated with respect to the archive -> ACCEPT.
                 Else accept with probability exp(-Delta_E_arch / T).
       - Case 3: If x_cur constrained-dominates x_cand:
                 Accept with probability P = prod_{m=1}^M exp(-max(0, f_m(cand) - f_m(cur)) / (R_m * T)).
4. Cooling & Archive Maintenance:
       T = alpha * T. Periodically re-heat or return from archive if trapped.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, dominates
from moo.algorithms.common import random_population, extract_non_dominated, swap_mutation


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MOSA state, temperature, and archive within budget."""
    rng = np.random.default_rng(seed)
    budget = kwargs.get("budget", 10000)
    archive_size = kwargs.get("archive_size", 100)
    T0 = kwargs.get("T0", 100.0)
    alpha = kwargs.get("alpha", 0.98)
    T_min = kwargs.get("T_min", 1e-4)

    # Initial state
    x_cur = random_population(problem, 1, rng)[0]
    f_cur, cv_cur, _ = problem.evaluate(x_cur)

    arch_X = np.array([x_cur])
    arch_F = np.array([f_cur])
    arch_CV = np.array([cv_cur])

    # Objective ranges for scaling
    R_m = np.ones(problem.n_obj, dtype=float)

    state = {
        "seed": seed,
        "rng": rng,
        "budget": budget,
        "archive_size": archive_size,
        "T0": T0,
        "T": T0,
        "alpha": alpha,
        "T_min": T_min,
        "x_cur": x_cur,
        "f_cur": f_cur,
        "cv_cur": cv_cur,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "R_m": R_m,
        "n_evals": 1,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one cooling step of MOSA."""
    budget = state.get("budget", 1000000)
    if state["n_evals"] >= budget:
        return state

    rng = state["rng"]
    n_var = problem.n_var
    feas_tol = problem.feas_tol

    x_cur = state["x_cur"]
    f_cur = state["f_cur"]
    cv_cur = state["cv_cur"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    T = state["T"]
    R_m = state["R_m"]
    is_perm = (problem.var_type == "permutation")

    # 1. Generate neighbor
    if is_perm:
        x_cand = swap_mutation(x_cur, rng, prob_m=1.0)
    else:
        span = problem.xu - problem.xl
        step_size = 0.05 * span * (T / state["T0"] + 0.1)
        x_cand = x_cur + rng.normal(0.0, step_size, size=n_var)
        x_cand = np.clip(x_cand, problem.xl, problem.xu)
        if problem.var_type == "integer":
            x_cand = np.round(x_cand)
        elif problem.var_type == "binary":
            x_cand = (x_cand >= 0.5).astype(float)

    f_cand, cv_cand, _ = problem.evaluate(x_cand)
    state["n_evals"] += 1
    state["iter"] += 1

    # Update ranges if needed
    if len(arch_F) > 1:
        f_span = np.max(arch_F, axis=0) - np.min(arch_F, axis=0)
        R_m = np.where(f_span <= 1e-12, 1.0, f_span)
        state["R_m"] = R_m

    # 2. Acceptance Decision
    cand_dom_cur = constrained_dominates(f_cand, cv_cand, f_cur, cv_cur, feas_tol=feas_tol)
    cur_dom_cand = constrained_dominates(f_cur, cv_cur, f_cand, cv_cand, feas_tol=feas_tol)

    accept = False
    if cand_dom_cur:
        # Candidate strictly dominates current -> accept unconditionally
        accept = True
    elif not cur_dom_cand:
        # Mutually non-dominating: check if candidate is non-dominated by archive
        is_dominated_by_arch = False
        for a_idx in range(len(arch_F)):
            if constrained_dominates(arch_F[a_idx], arch_CV[a_idx], f_cand, cv_cand, feas_tol=feas_tol):
                is_dominated_by_arch = True
                break
        if not is_dominated_by_arch:
            accept = True
        else:
            # Acceptance probability based on archive distance
            diffs = np.maximum(0.0, f_cand - f_cur) / R_m
            delta_E = float(np.sum(diffs))
            prob = np.exp(-delta_E / max(1e-6, T))
            accept = (rng.random() < prob)
    else:
        # Current dominates candidate: standard Boltzmann probability
        diffs = np.maximum(0.0, f_cand - f_cur) / R_m
        delta_E = float(np.sum(diffs))
        prob = np.exp(-delta_E / max(1e-6, T))
        accept = (rng.random() < prob)

    if accept:
        state["x_cur"] = x_cand
        state["f_cur"] = f_cand
        state["cv_cur"] = cv_cand

    # 3. Update Archive
    comb_X = np.vstack((arch_X, x_cand.reshape(1, -1)))
    comb_F = np.vstack((arch_F, f_cand.reshape(1, -1)))
    comb_CV = np.concatenate((arch_CV, [cv_cand]))

    new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
        comb_X, comb_F, comb_CV, feas_tol=feas_tol
    )

    if len(new_arch_X) > state["archive_size"]:
        from moo.moo_utils import crowding_distance
        cd = crowding_distance(new_arch_F, list(range(len(new_arch_F))))
        order = np.argsort(-cd)
        keep = order[: state["archive_size"]]
        new_arch_X = new_arch_X[keep]
        new_arch_F = new_arch_F[keep]
        new_arch_CV = new_arch_CV[keep]

    state["arch_X"] = new_arch_X
    state["arch_F"] = new_arch_F
    state["arch_CV"] = new_arch_CV

    # 4. Temperature Decay and Reheating
    T = state["alpha"] * T
    if T < state["T_min"]:
        # Reheat and re-center at random archive solution
        T = state["T0"] * 0.5
        pick = rng.choice(len(new_arch_X))
        state["x_cur"] = new_arch_X[pick].copy()
        state["f_cur"] = new_arch_F[pick].copy()
        state["cv_cur"] = new_arch_CV[pick]

    state["T"] = T
    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MOSA until budget is exhausted."""
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
