"""
alg05_value_function.py: Value Function Method (Decision-Maker Preference Optimization)

Lecture Family: Classical Methods (Part 1, Chapter 5)
Original Citation: Keeney, R. L., & Raiffa, H. (1976).
"Decisions with Multiple Objectives: Preferences and Value Trade-Offs."
John Wiley & Sons, New York.

Algorithmic Logic:
1. Elicits or specifies an explicit mathematical utility / value function U(f_1^natural, ..., f_M^natural)
   representing the decision-maker's (DM's) complete preference structure.
2. Directly solves the single-objective problem of maximizing utility:
       max_x  U(f_1^natural(x), ..., f_M^natural(x))
       subject to  g_j(x) <= 0,  h_k(x) = 0,  x_l <= x <= x_u
   Equivalently in minimization format:
       min_x  -U(f^natural(x)) + penalty * CV(x)
3. Returns the single most preferred solution and the explicit preference model that produced it.

Supported Preference Models:
- Target-based Quadratic Loss:
      U(f) = - sum_{m=1}^M w_m * (f_m^natural - Target_m)^2
      (Used for P14 Laptop Design with DM target 1.8 kg, 12.0 h).
- Mean-Variance Risk-Aversion Utility:
      U(w) = Return - (gamma / 2) * Risk
      (Used for P4 Investment Portfolio with risk-aversion coefficient gamma).
- Custom callable utility function U(f_natural).

Key Property (Viva Note):
- Does not generate an entire Pareto front; finds the single point on the front that maximizes DM utility.
- If the utility model is inaccurate or misspecified, the resulting solution will not match real DM preferences.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.optimize import minimize
from moo.problems.base import Problem


def default_utility_for_problem(problem: Problem) -> tuple[Callable[[np.ndarray], float], str]:
    """Provides a tailored decision-maker utility function based on problem scenario."""
    if problem.name.startswith("P14"):
        # P14 Laptop Design: Target is 1.8 kg, 12.0 hours
        target = np.array([1.8, 12.0])
        weights = np.array([1.0, 1.0])
        desc = f"Quadratic Aspirational Loss to target={target} with weights={weights}"

        def _u_p14(f_nat: np.ndarray) -> float:
            return float(-np.sum(weights * ((f_nat - target) ** 2)))

        return _u_p14, desc

    if problem.name.startswith("P4"):
        # P4 Portfolio: Risk aversion gamma = 2.0 -> Return - gamma * Risk
        gamma = 2.0
        desc = f"Mean-Variance Utility: Return - ({gamma} / 2) * Risk"

        def _u_p4(f_nat: np.ndarray) -> float:
            risk = f_nat[0]
            ret = f_nat[1]
            return float(ret - (gamma / 2.0) * risk)

        return _u_p4, desc

    # Generic default: normalized distance to ideal point
    desc = "Generic Weighted Linear Utility"
    weights = np.ones(problem.n_obj) / problem.n_obj

    def _u_generic(f_nat: np.ndarray) -> float:
        # Default minimize natural sum
        return float(-np.sum(weights * f_nat))

    return _u_generic, desc


def initialize(
    problem: Problem,
    seed: int = 42,
    utility_fn: Callable[[np.ndarray], float] | None = None,
    utility_desc: str | None = None,
    **kwargs,
) -> dict:
    """Initializes the Value Function algorithm state."""
    rng = np.random.default_rng(seed)

    if utility_fn is None:
        utility_fn, auto_desc = default_utility_for_problem(problem)
        if utility_desc is None:
            utility_desc = auto_desc
    elif utility_desc is None:
        utility_desc = "User-Supplied Custom Utility Function"

    state = {
        "seed": seed,
        "rng": rng,
        "utility_fn": utility_fn,
        "utility_desc": utility_desc,
        "x_opt": None,
        "f_opt": None,
        "f_opt_nat": None,
        "cv_opt": None,
        "n_evals": 0,
        "budget": kwargs.get("budget", 10000),
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem, max_evals: int = 500) -> dict:
    """Solves the utility maximization problem."""
    if state["n_evals"] >= state.get("budget", 1000000):
        return state

    rng = state["rng"]
    u_fn = state["utility_fn"]

    x0 = rng.uniform(problem.xl, problem.xu)
    eval_count = [0]
    last_val = [1e9]
    last_f = [None]
    last_cv = [0.0]

    def _objective(x: np.ndarray) -> float:
        if state["n_evals"] + eval_count[0] >= state.get("budget", 1000000):
            return last_val[0]
        x_clamped = np.clip(x, problem.xl, problem.xu)
        f_int, cv, _ = problem.evaluate(x_clamped)
        eval_count[0] += 1

        f_nat = problem.to_natural(f_int)
        u_val = u_fn(f_nat)
        penalty = 1e4 * cv
        cost = float(-u_val + penalty)
        last_val[0] = cost
        last_f[0] = f_int
        last_cv[0] = cv
        return cost

    bounds = list(zip(problem.xl, problem.xu))
    res = minimize(
        _objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max(5, max_evals // 3), "ftol": 1e-6},
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

    f_opt_nat = problem.to_natural(f_opt)

    state["x_opt"] = x_opt
    state["f_opt"] = f_opt
    state["f_opt_nat"] = f_opt_nat
    state["cv_opt"] = cv_opt
    state["n_evals"] += eval_count[0]
    state["iter"] += 1

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    utility_fn: Callable[[np.ndarray], float] | None = None,
    utility_desc: str | None = None,
    **kwargs,
) -> dict:
    """Executes the Value Function optimization."""
    state = initialize(problem, seed=seed, utility_fn=utility_fn, utility_desc=utility_desc, budget=budget, **kwargs)
    state["budget"] = budget
    step(state, problem, max_evals=budget)

    X_opt = np.array([state["x_opt"]]) if state["x_opt"] is not None else np.empty((0, problem.n_var))
    F_opt = np.array([state["f_opt"]]) if state["f_opt"] is not None else np.empty((0, problem.n_obj))
    F_opt_nat = np.array([state["f_opt_nat"]]) if state["f_opt_nat"] is not None else np.empty((0, problem.n_obj))
    CV_opt = np.array([state["cv_opt"]]) if state["cv_opt"] is not None else np.empty(0)

    return {
        "X_opt": X_opt,
        "F_opt": F_opt,
        "F_opt_nat": F_opt_nat,
        "CV_opt": CV_opt,
        "utility_desc": state["utility_desc"],
        "n_evals": state["n_evals"],
        "state": state,
    }
