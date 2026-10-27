"""
alg02_epsilon_constraint.py: epsilon-Constraint Method (Classical Scalarization)

Lecture Family: Classical Methods (Part 1, Chapter 5)
Original Citation: Haimes, Y. Y., Lasdon, L. S., & Wismer, D. A. (1971).
"On a bicriterion formulation of the problems of integrated system identification and system optimization."
IEEE Transactions on Systems, Man, and Cybernetics.

Algorithmic Logic:
1. Selects one primary objective f_k to minimize.
2. Converts all other M - 1 objectives into inequality constraints bounded by epsilon:
       min_x  f_k^internal(x)
       subject to  f_m^internal(x) <= epsilon_m,  for all m != k
                   g_j(x) <= 0,  h_k(x) = 0,  x_l <= x <= x_u
3. Solves anchor single-objective problems to determine the valid range [eps_min, eps_max] for each non-primary objective.
4. Sweeps epsilon across this range in discrete steps (at least 10 steps).
5. For each epsilon, solves the constrained subproblem using scipy.optimize.minimize.

Key Theoretical Advantage (Viva Note):
- Can reach non-convex (dented/inward-curving) regions of the Pareto front (e.g. P3 ZDT2).
- Does not suffer from the convexity limitation of the Weighted Sum method because non-primary objectives
  act as boundary constraints rather than linear slope combinations.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from moo.problems.base import Problem
from moo.algorithms.common import extract_non_dominated


def initialize(
    problem: Problem,
    seed: int = 42,
    primary_idx: int = 0,
    n_eps: int = 21,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes epsilon-Constraint state and estimates bounds for non-primary objectives."""
    rng = np.random.default_rng(seed)
    n_obj = problem.n_obj
    primary_idx = int(primary_idx) % n_obj
    secondary_indices = [m for m in range(n_obj) if m != primary_idx]

    bounds = list(zip(problem.xl, problem.xu))
    init_evals = [0]
    X_anchors = []
    F_anchors = []

    # If budget is very small, use problem's analytical/known ideal/nadir points without costly SLSQP
    if budget < 30:
        sec_min = np.array([problem.ideal_point[m] for m in secondary_indices])
        sec_max = np.array([problem.nadir_point[m] for m in secondary_indices])
        if budget > 0:
            x0 = rng.uniform(problem.xl, problem.xu)
            f0, _, _ = problem.evaluate(x0)
            init_evals[0] += 1
            X_anchors.append(x0)
            F_anchors.append(f0)
    else:
        # Quick anchor solves to find bounds for secondary objectives
        x0 = rng.uniform(problem.xl, problem.xu)
        max_init_evals = min(50, budget // 4)
        last_val = [1e9]

        # Anchor solve for primary objective
        def _anchor_primary(x: np.ndarray) -> float:
            if init_evals[0] >= max_init_evals:
                return last_val[0]
            f_int, cv, _ = problem.evaluate(np.clip(x, problem.xl, problem.xu))
            init_evals[0] += 1
            val = float(f_int[primary_idx] + 1e4 * cv)
            last_val[0] = val
            return val

        res_k = minimize(
            _anchor_primary,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 20},
        )
        if init_evals[0] < max_init_evals:
            f_at_k, _, _ = problem.evaluate(np.clip(res_k.x, problem.xl, problem.xu))
            init_evals[0] += 1
        else:
            f_at_k = problem.ideal_point.copy()

        X_anchors.append(res_k.x)
        F_anchors.append(f_at_k)

        sec_min = np.zeros(len(secondary_indices))
        sec_max = np.zeros(len(secondary_indices))

        for idx_s, s_m in enumerate(secondary_indices):
            last_s_val = [1e9]

            def _anchor_sec(x: np.ndarray) -> float:
                if init_evals[0] >= max_init_evals:
                    return last_s_val[0]
                f_int, cv, _ = problem.evaluate(np.clip(x, problem.xl, problem.xu))
                init_evals[0] += 1
                val = float(f_int[s_m] + 1e4 * cv)
                last_s_val[0] = val
                return val

            res_s = minimize(
                _anchor_sec,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 20},
            )
            if init_evals[0] < max_init_evals:
                f_at_s, _, _ = problem.evaluate(np.clip(res_s.x, problem.xl, problem.xu))
                init_evals[0] += 1
            else:
                f_at_s = problem.nadir_point.copy()
            X_anchors.append(res_s.x)
            F_anchors.append(f_at_s)

            s_min_val = min(f_at_s[s_m], f_at_k[s_m], problem.ideal_point[s_m])
            s_max_val = max(f_at_s[s_m], f_at_k[s_m], problem.nadir_point[s_m])
            sec_min[idx_s] = s_min_val
            sec_max[idx_s] = s_max_val

    # Epsilon grid across secondary objectives
    if len(secondary_indices) == 1:
        eps_grid = np.linspace(sec_min[0], sec_max[0], n_eps).reshape(-1, 1)
    else:
        alphas = np.linspace(0.0, 1.0, n_eps)
        eps_grid = np.zeros((n_eps, len(secondary_indices)))
        for i, alpha in enumerate(alphas):
            eps_grid[i] = (1.0 - alpha) * sec_min + alpha * sec_max

    state = {
        "seed": seed,
        "rng": rng,
        "primary_idx": primary_idx,
        "secondary_indices": secondary_indices,
        "eps_grid": eps_grid,
        "n_eps": len(eps_grid),
        "current_idx": 0,
        "X_all": X_anchors,
        "F_all": F_anchors,
        "CV_all": [0.0] * len(F_anchors),
        "n_evals": init_evals[0],
        "budget": budget,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem, max_evals_per_solve: int = 200) -> dict:
    """Performs one epsilon subproblem optimization."""
    idx = state["current_idx"]
    if idx >= state["n_eps"] or state["n_evals"] >= state.get("budget", 1000000):
        return state

    eps_targets = state["eps_grid"][idx]
    p_idx = state["primary_idx"]
    s_indices = state["secondary_indices"]
    rng = state["rng"]

    x0 = state["X_all"][-1].copy() if len(state["X_all"]) > 0 else rng.uniform(problem.xl, problem.xu)
    eval_count = [0]
    last_val = [1e9]
    last_f = [None]
    last_cv = [0.0]

    def _eps_objective(x: np.ndarray) -> float:
        if state["n_evals"] + eval_count[0] >= state.get("budget", 1000000):
            return last_val[0]
        x_clamped = np.clip(x, problem.xl, problem.xu)
        f_int, cv, _ = problem.evaluate(x_clamped)
        eval_count[0] += 1

        obj = f_int[p_idx]
        sec_viol = 0.0
        for s_idx, target in zip(s_indices, eps_targets):
            sec_viol += max(0.0, f_int[s_idx] - target)

        penalty = 1e3 * sec_viol + 1e4 * cv
        cost = float(obj + penalty)
        last_val[0] = cost
        last_f[0] = f_int
        last_cv[0] = cv
        return cost

    bounds = list(zip(problem.xl, problem.xu))
    res = minimize(
        _eps_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max(5, max_evals_per_solve // 3), "ftol": 1e-6},
    )

    x_opt = np.clip(res.x, problem.xl, problem.xu)
    if problem.var_type == "integer":
        x_opt = np.round(x_opt)
    elif problem.var_type == "binary":
        x_opt = (x_opt >= 0.5).astype(float)

    if state["n_evals"] + eval_count[0] < state.get("budget", 1000000):
        f_opt, cv_opt, _ = problem.evaluate(x_opt)
        eval_count[0] += 1
    else:
        f_opt = last_f[0] if last_f[0] is not None else problem.ideal_point.copy()
        cv_opt = last_cv[0]

    state["X_all"].append(x_opt)
    state["F_all"].append(f_opt)
    state["CV_all"].append(cv_opt)
    state["n_evals"] += eval_count[0]
    state["current_idx"] += 1
    state["iter"] += 1

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    primary_idx: int = 0,
    n_eps: int = 21,
    **kwargs,
) -> dict:
    """Executes the full epsilon-Constraint sweep."""
    state = initialize(problem, seed=seed, primary_idx=primary_idx, n_eps=n_eps, budget=budget, **kwargs)
    state["budget"] = budget
    evals_per_solve = max(10, (budget - state["n_evals"]) // max(1, state["n_eps"]))

    while state["current_idx"] < state["n_eps"] and state["n_evals"] < budget:
        step(state, problem, max_evals_per_solve=evals_per_solve)

    X_all = np.array(state["X_all"])
    F_all = np.array(state["F_all"])
    CV_all = np.array(state["CV_all"])

    X_opt, F_opt, CV_opt = extract_non_dominated(X_all, F_all, CV_all, feas_tol=problem.feas_tol)
    F_opt_nat = problem.to_natural(F_opt)

    return {
        "X_opt": X_opt,
        "F_opt": F_opt,
        "F_opt_nat": F_opt_nat,
        "CV_opt": CV_opt,
        "n_evals": state["n_evals"],
        "state": state,
    }
