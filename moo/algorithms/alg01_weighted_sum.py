"""
alg01_weighted_sum.py: Weighted Sum Method (Classical Scalarization)

Lecture Family: Classical Methods (Part 1, Chapter 5)
Original Citation: Zadeh, L. (1963). "Optimality and non-scalar-valued performance criteria." IEEE Trans. Autom. Control.

Algorithmic Logic:
1. Converts a multiobjective problem into a single scalar objective using a convex combination of weights:
       min_x  F_scalar(x) = sum_{m=1}^M w_m * f_m^internal(x) + penalty * CV(x)
   where sum(w_m) = 1, w_m >= 0.
2. Sweeps across multiple weight vectors w on the unit simplex to approximate the Pareto front.
3. For each weight vector, solves the scalarized single-objective problem using scipy.optimize.minimize
   (permitted scoped exception for classical subproblems).
4. Accumulates all solutions and extracts the non-dominated subset.

Key Theoretical Property (Viva Note):
- Weighted Sum can ONLY find solutions on the convex hull of the Pareto front.
- If the Pareto front has non-convex (dented/inward-curving) regions (e.g. P3 ZDT2),
  any weight vector tangent to the dented curve will achieve a lower value at the boundary extrema
  than at the interior non-convex points, entirely missing the interior trade-off!
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from moo.problems.base import Problem
from moo.moo_utils import das_dennis
from moo.algorithms.common import extract_non_dominated


def generate_weight_vectors(n_obj: int, n_weights: int = 21) -> np.ndarray:
    """Generates uniform weight vectors on the unit simplex sum(w_m) = 1."""
    if n_obj == 2:
        w1 = np.linspace(0.0, 1.0, n_weights)
        w2 = 1.0 - w1
        return np.column_stack((w1, w2))
    elif n_obj == 3:
        # Use Das-Dennis for M >= 3
        # find p divisions giving close to n_weights
        p = 5
        weights = das_dennis(p, n_obj)
        return weights
    else:
        # Default dirichlet sampling + corner points
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(n_obj), size=n_weights)
        # Ensure pure single-objective corners are present
        corners = np.eye(n_obj)
        return np.vstack((corners, weights))


def initialize(
    problem: Problem,
    seed: int = 42,
    n_weights: int = 21,
    weights: np.ndarray | None = None,
    normalize: bool = False,
    **kwargs,
) -> dict:
    """Initializes the Weighted Sum algorithm state."""
    rng = np.random.default_rng(seed)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.ndim == 1:
            w_arr = w_arr.reshape(1, -1)
        # Normalize each weight vector to sum to 1
        row_sums = np.sum(w_arr, axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        weights_to_use = np.maximum(0.0, w_arr) / row_sums
    else:
        weights_to_use = generate_weight_vectors(problem.n_obj, n_weights=n_weights)
    
    state = {
        "seed": seed,
        "rng": rng,
        "weights": weights_to_use,
        "normalize": bool(normalize),
        "current_idx": 0,
        "n_weights": len(weights_to_use),
        "X_all": [],
        "F_all": [],
        "CV_all": [],
        "n_evals": 0,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem, max_evals_per_solve: int = 200) -> dict:
    """Performs one weight optimization step."""
    idx = state["current_idx"]
    if idx >= state["n_weights"]:
        return state

    w = state["weights"][idx]
    norm_flag = state.get("normalize", False)
    rng = state["rng"]
    n_var = problem.n_var

    # Warm start from previous good solution if available, else random
    if len(state["X_all"]) > 0:
        x0 = state["X_all"][-1].copy()
    else:
        x0 = rng.uniform(problem.xl, problem.xu)

    eval_count = [0]
    ideal = problem.ideal_point
    nadir = problem.nadir_point
    denom = np.where(np.abs(nadir - ideal) < 1e-12, 1.0, np.abs(nadir - ideal))

    last_val = [1e9]
    last_f = [None]
    last_cv = [0.0]

    def _scalar_objective(x: np.ndarray) -> float:
        if state["n_evals"] + eval_count[0] >= state.get("budget", 1000000):
            return last_val[0]
        x_clamped = np.clip(x, problem.xl, problem.xu)
        f_int, cv, _ = problem.evaluate(x_clamped)
        eval_count[0] += 1
        
        if norm_flag:
            f_eval = (f_int - ideal) / denom
        else:
            f_eval = f_int

        # Scalarized cost + penalty for constraint violation
        penalty = 1e4 * cv
        cost = float(np.dot(w, f_eval) + penalty)
        last_val[0] = cost
        last_f[0] = f_int
        last_cv[0] = cv
        return cost

    bounds = list(zip(problem.xl, problem.xu))
    res = minimize(
        _scalar_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max(10, max_evals_per_solve // 3), "ftol": 1e-6},
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
        f_opt = last_f[0] if last_f[0] is not None else problem.evaluate(x_opt)[0]
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
    n_weights: int = 21,
    weights: np.ndarray | None = None,
    normalize: bool = False,
    **kwargs,
) -> dict:
    """Executes the full Weighted Sum sweep across weights."""
    state = initialize(problem, seed=seed, n_weights=n_weights, weights=weights, normalize=normalize, **kwargs)
    state["budget"] = budget
    evals_per_weight = max(20, budget // max(1, state["n_weights"]))

    while state["current_idx"] < state["n_weights"] and state["n_evals"] < budget:
        step(state, problem, max_evals_per_solve=evals_per_weight)

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
