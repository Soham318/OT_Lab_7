"""
alg04_weight_metric.py: Weight Metric Method (p-norm & Tchebycheff Scalarization)

Lecture Family: Classical Methods (Part 1, Chapter 5)
Original Citation: Bowman, V. J. (1976).
"On the relationship of the Tchebycheff norm and the efficient frontier of multiple-criteria objectives."
Multiple Criteria Decision Making, Lecture Notes in Economics and Mathematical Systems, 130.

Algorithmic Logic:
1. Measures the weighted distance from the objective vector to the ideal point z*:
       - For p = 1 (Manhattan metric):
             L_1(x) = sum_{m=1}^M w_m * |f_m^internal(x) - z_m*|
       - For p = 2 (Euclidean metric):
             L_2(x) = sqrt( sum_{m=1}^M w_m * (f_m^internal(x) - z_m*)^2 )
       - For p = infinity (Augmented Tchebycheff metric):
             L_inf(x) = max_m [ w_m * |f_m^internal(x) - z_m*| ] + rho * sum_m [ w_m * |f_m^internal(x) - z_m*| ]
         where rho = 1e-4 prevents weak Pareto solutions.
2. Sweeps weight vectors w on the unit simplex.
3. For each weight setting, solves the distance minimization subproblem with constraint penalty.

Key Theoretical Advantage (Viva Note):
- For p = infinity (Tchebycheff), ANY Pareto-optimal point (including points in non-convex or dented regions)
  can be obtained by a suitable choice of weight vector w.
- For p = 1, it is identical to Weighted Sum and cannot reach non-convex regions.
- For p = 2, it forms circular compromise contours.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from moo.problems.base import Problem
from moo.algorithms.common import extract_non_dominated
from moo.algorithms.alg01_weighted_sum import generate_weight_vectors


def initialize(
    problem: Problem,
    seed: int = 42,
    p: int | float | str = "inf",
    n_weights: int = 21,
    rho: float = 1e-4,
    normalize: bool = False,
    weights: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """Initializes Weight Metric state."""
    rng = np.random.default_rng(seed)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.ndim == 1:
            w_arr = w_arr.reshape(1, -1)
        row_sums = np.sum(w_arr, axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        weights_to_use = np.maximum(0.0, w_arr) / row_sums
    else:
        weights_to_use = generate_weight_vectors(problem.n_obj, n_weights=n_weights)
    
    # Ideal point z*
    z_star = problem.ideal_point.copy()

    state = {
        "seed": seed,
        "rng": rng,
        "p": p,
        "rho": float(rho),
        "weights": weights_to_use,
        "z_star": z_star,
        "normalize": bool(normalize),
        "n_weights": len(weights_to_use),
        "current_idx": 0,
        "X_all": [],
        "F_all": [],
        "CV_all": [],
        "n_evals": 0,
        "budget": kwargs.get("budget", 10000),
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem, max_evals_per_solve: int = 200) -> dict:
    """Solves one weight metric subproblem."""
    idx = state["current_idx"]
    if idx >= state["n_weights"] or state["n_evals"] >= state.get("budget", 1000000):
        return state

    w = state["weights"][idx]
    z_star = state["z_star"]
    p_norm = state["p"]
    rho = state["rho"]
    norm_flag = state.get("normalize", False)
    rng = state["rng"]

    x0 = state["X_all"][-1].copy() if len(state["X_all"]) > 0 else rng.uniform(problem.xl, problem.xu)
    eval_count = [0]
    denom = np.where(np.abs(problem.nadir_point - problem.ideal_point) < 1e-12, 1.0, np.abs(problem.nadir_point - problem.ideal_point))
    last_val = [1e9]
    last_f = [None]
    last_cv = [0.0]

    def _metric_objective(x: np.ndarray) -> float:
        if state["n_evals"] + eval_count[0] >= state.get("budget", 1000000):
            return last_val[0]
        x_clamped = np.clip(x, problem.xl, problem.xu)
        f_int, cv, _ = problem.evaluate(x_clamped)
        eval_count[0] += 1

        diff = np.abs(f_int - z_star)
        if norm_flag:
            diff = diff / denom

        if str(p_norm).lower() in ("inf", "infinity", "tchebycheff"):
            # Augmented Tchebycheff
            weighted_diff = w * diff
            dist = float(np.max(weighted_diff) + rho * np.sum(weighted_diff))
        elif float(p_norm) == 1.0:
            dist = float(np.sum(w * diff))
        elif float(p_norm) == 2.0:
            dist = float(np.sqrt(np.sum(w * (diff ** 2))))
        else:
            p_val = float(p_norm)
            dist = float((np.sum(w * (diff ** p_val))) ** (1.0 / p_val))

        penalty = 1e4 * cv
        cost = dist + penalty
        last_val[0] = cost
        last_f[0] = f_int
        last_cv[0] = cv
        return cost

    bounds = list(zip(problem.xl, problem.xu))
    res = minimize(
        _metric_objective,
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
    p: int | float | str = "inf",
    n_weights: int = 21,
    rho: float = 1e-4,
    normalize: bool = False,
    weights: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """Executes the full Weight Metric sweep."""
    state = initialize(problem, seed=seed, p=p, n_weights=n_weights, rho=rho, normalize=normalize, weights=weights, budget=budget, **kwargs)
    state["budget"] = budget
    evals_per_solve = max(10, (budget - state["n_evals"]) // max(1, state["n_weights"]))

    while state["current_idx"] < state["n_weights"] and state["n_evals"] < budget:
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
