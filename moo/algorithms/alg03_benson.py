"""
alg03_benson.py: Benson's Method (Classical Scalarization)

Lecture Family: Classical Methods (Part 1, Chapter 5)
Original Citation: Benson, H. P. (1978).
"Existence of efficient solutions for vector maximization problems in Euclidean space."
Journal of Optimization Theory and Applications, 26(4), 569-580.

Algorithmic Logic:
1. Given a reference point z0 (such as an anchor point or dominated starting point):
2. Formulates the shortfall / distance maximization problem:
       max_{x, l}  sum_{m=1}^M w_m * l_m
       subject to  f_m^internal(x) + l_m <= z0_m,  l_m >= 0,  x in Feasible
   which seeks an efficient point that dominates z0 by the maximum weighted margin l.
3. In minimization form, this is equivalent to:
       min_x  sum_{m=1}^M w_m * (f_m^internal(x) - z0_m) + penalty * (constraint violations)
       subject to  f_m^internal(x) <= z0_m
4. By sweeping the weight vector w or sweeping the reference point z0,
   Benson's method generates points on the Pareto-optimal frontier.

Key Property (Viva Note):
- Benson's method originated as a test for Pareto efficiency: if the optimal objective value
  sum(w_m * l_m) is 0, the reference point z0 is already Pareto optimal;
  if > 0, the method finds an efficient point that strictly improves upon z0.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from moo.problems.base import Problem
from moo.algorithms.common import extract_non_dominated
from moo.algorithms.alg01_weighted_sum import generate_weight_vectors


def generate_reference_sweep(problem: Problem, n_sweeps: int = 21) -> np.ndarray:
    """Generates a sweep of reference points z0 across the objective trade-off space."""
    ideal = problem.ideal_point
    nadir = problem.nadir_point
    n_obj = problem.n_obj

    if n_obj == 2:
        # Sweep along the trade-off line between (ideal_1, nadir_2) and (nadir_1, ideal_2)
        # with an outward buffer so reference points dominate or bound sections of the front
        t = np.linspace(0.0, 1.0, n_sweeps)
        p_a = np.array([ideal[0], nadir[1]])
        p_b = np.array([nadir[0], ideal[1]])
        # Buffer slightly outside the convex hull
        buffer = 0.1 * np.abs(nadir - ideal)
        sweep = np.zeros((n_sweeps, 2))
        for i, ti in enumerate(t):
            sweep[i] = (1.0 - ti) * p_a + ti * p_b + buffer
        return sweep
    else:
        # Linear sweep from nadir down towards ideal with buffer
        alphas = np.linspace(0.1, 1.0, n_sweeps)
        sweep = np.zeros((n_sweeps, n_obj))
        for i, a in enumerate(alphas):
            sweep[i] = a * nadir + (1.0 - a) * ideal + 0.05 * np.abs(nadir - ideal)
        return sweep


def initialize(
    problem: Problem,
    seed: int = 42,
    n_sweeps: int = 21,
    z0: np.ndarray | None = None,
    ref_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes Benson's method state."""
    rng = np.random.default_rng(seed)

    if ref_points is not None:
        z0_arr = np.asarray(ref_points, dtype=float)
        if z0_arr.ndim == 1:
            z0_arr = z0_arr.reshape(1, -1)
    elif z0 is not None:
        z0_arr = np.asarray(z0, dtype=float).reshape(1, -1)
    else:
        z0_arr = generate_reference_sweep(problem, n_sweeps=n_sweeps)

    n_pts = len(z0_arr)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.ndim == 1:
            w_arr = np.tile(w_arr, (n_pts, 1))
    else:
        # Default equal weighting across objectives
        w_arr = np.ones((n_pts, problem.n_obj)) / problem.n_obj

    state = {
        "seed": seed,
        "rng": rng,
        "weights": w_arr,
        "ref_points": z0_arr,
        "n_sweeps": n_pts,
        "current_idx": 0,
        "X_all": [],
        "F_all": [],
        "CV_all": [],
        "shortfalls": [],
        "n_evals": 0,
        "budget": budget,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem, max_evals_per_solve: int = 200) -> dict:
    """Solves one Benson projection step."""
    idx = state["current_idx"]
    if idx >= state["n_sweeps"] or state["n_evals"] >= state.get("budget", 1000000):
        return state

    w = state["weights"][idx]
    z0 = state["ref_points"][idx]
    rng = state["rng"]

    x0 = state["X_all"][-1].copy() if len(state["X_all"]) > 0 else rng.uniform(problem.xl, problem.xu)
    eval_count = [0]
    last_val = [1e9]
    last_f = [None]
    last_cv = [0.0]

    def _benson_objective(x: np.ndarray) -> float:
        if state["n_evals"] + eval_count[0] >= state.get("budget", 1000000):
            return last_val[0]
        x_clamped = np.clip(x, problem.xl, problem.xu)
        f_int, cv, _ = problem.evaluate(x_clamped)
        eval_count[0] += 1

        l_m = np.maximum(0.0, z0 - f_int)
        shortfall_reward = - np.sum(w * l_m)

        over_z0 = np.sum(np.maximum(0.0, f_int - z0))
        penalty = 1e3 * over_z0 + 1e4 * cv
        cost = float(shortfall_reward + penalty)
        last_val[0] = cost
        last_f[0] = f_int
        last_cv[0] = cv
        return cost

    bounds = list(zip(problem.xl, problem.xu))
    res = minimize(
        _benson_objective,
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

    l_opt = np.maximum(0.0, z0 - f_opt)
    state["X_all"].append(x_opt)
    state["F_all"].append(f_opt)
    state["CV_all"].append(cv_opt)
    state["shortfalls"].append(l_opt)
    state["n_evals"] += eval_count[0]
    state["current_idx"] += 1
    state["iter"] += 1

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    n_sweeps: int = 21,
    z0: np.ndarray | None = None,
    ref_points: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """Executes Benson's method sweep."""
    state = initialize(
        problem,
        seed=seed,
        n_sweeps=n_sweeps,
        z0=z0,
        ref_points=ref_points,
        weights=weights,
        budget=budget,
        **kwargs,
    )
    state["budget"] = budget
    evals_per_solve = max(10, (budget - state["n_evals"]) // max(1, state["n_sweeps"]))

    while state["current_idx"] < state["n_sweeps"] and state["n_evals"] < budget:
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
        "ref_points": state["ref_points"],
        "shortfalls": np.array(state["shortfalls"]),
        "n_evals": state["n_evals"],
        "state": state,
    }
