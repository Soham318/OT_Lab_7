"""
alg06_goal_programming.py: Goal Programming (Targets and Deviation Variables)

Lecture Family: Classical Methods (Part 1 & Bonus Module 5E, Chapter 5)
Original Citation: Charnes, A., & Cooper, W. W. (1961).
"Management Models and Industrial Applications of Linear Programming."
John Wiley & Sons, New York.

Algorithmic Logic:
1. Decision maker specifies aspirational targets T_m for each objective m.
2. Expresses actual achievement using positive and negative deviational variables:
       f_m^natural(x) - d_m^+ + d_m^- = T_m,   d_m^+, d_m^- >= 0
   - d_m^+: overachievement of target T_m
   - d_m^-: underachievement of target T_m
3. Formulates the unwanted deviation minimization problem:
       - If goal is f_m >= T_m (e.g. profit >= 2400, volume >= 600 in P11):
             unwanted deviation is underachievement d_m^- = max(0, T_m - f_m^natural)
       - If goal is f_m <= T_m (e.g. cost <= T_m, delay <= T_m):
             unwanted deviation is overachievement d_m^+ = max(0, f_m^natural - T_m)
       - If goal is f_m = T_m:
             unwanted deviation is total absolute deviation |f_m^natural - T_m|
4. Weighted Archimedean Goal Programming minimizes:
       min_x  sum_{m=1}^M w_m * d_m^unwanted(x) + penalty * CV(x)
   where w_m allows weighting different goals (or normalizing deviations by target size).

Key Sanity Check on P11 (Viva Note):
- Targets: Profit >= 2400, Units >= 600.
- Feasible region limits units to at most 500 (at xA=300, xB=200).
- Goal programming achieves Profit = 2400 (shortfall = 0) and Units = 500 (shortfall = 100).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, differential_evolution
from moo.problems.base import Problem


def default_goals_for_problem(problem: Problem) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Returns default targets, goal types ('ge', 'le', 'eq'), and weights for problem."""
    if problem.name.startswith("P11"):
        # P11 Production Planning: Profit >= 2400, Units >= 600
        targets = np.array([2400.0, 600.0])
        goal_types = ["ge", "ge"]
        weights = np.array([1.0, 1.0])
        return targets, goal_types, weights

    if problem.name.startswith("P7"):
        # P7 Water Allocation: Agriculture yield >= 40, Industry output >= 50
        targets = np.array([40.0, 50.0])
        goal_types = ["ge", "ge"]
        weights = np.array([1.0, 1.0])
        return targets, goal_types, weights

    # Generic: aspirational target at midpoint between ideal and nadir
    targets = (problem.ideal_point + problem.nadir_point) / 2.0
    goal_types = ["le"] * problem.n_obj
    weights = np.ones(problem.n_obj)
    return targets, goal_types, weights


def initialize(
    problem: Problem,
    seed: int = 42,
    targets: np.ndarray | None = None,
    goal_types: list[str] | None = None,
    weights: np.ndarray | None = None,
    normalized_weights: bool = False,
    method: str = "weighted",
    priorities: list[int] | None = None,
    **kwargs,
) -> dict:
    """Initializes Goal Programming state."""
    rng = np.random.default_rng(seed)

    def_targets, def_types, def_weights = default_goals_for_problem(problem)
    t = np.asarray(targets, dtype=float) if targets is not None else def_targets
    g_types = goal_types if goal_types is not None else def_types
    w = np.asarray(weights, dtype=float) if weights is not None else def_weights

    if normalized_weights:
        # Scale deviations by target magnitude
        denom = np.where(np.abs(t) < 1e-12, 1.0, np.abs(t))
        w = w / denom

    pri = priorities if priorities is not None else list(range(problem.n_obj))

    state = {
        "seed": seed,
        "rng": rng,
        "targets": t,
        "goal_types": g_types,
        "weights": w,
        "method": method.lower(),
        "priorities": pri,
        "x_opt": None,
        "f_opt": None,
        "f_opt_nat": None,
        "deviations": None,
        "d_minus": None,
        "d_plus": None,
        "achieved": None,
        "cv_opt": None,
        "n_evals": 0,
        "iter": 0,
    }
    return state


def _compute_deviations(f_nat: np.ndarray, targets: np.ndarray, goal_types: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Calculates underachievement (d-), overachievement (d+), unwanted deviation, and achievement status."""
    M = len(targets)
    d_minus = np.maximum(0.0, targets - f_nat)
    d_plus = np.maximum(0.0, f_nat - targets)
    unwanted = np.zeros(M, dtype=float)
    achieved = []

    for m in range(M):
        gtype = goal_types[m].lower()
        if gtype in ("ge", ">="):
            dev = d_minus[m]
            is_met = bool(dev <= 1e-4)
        elif gtype in ("le", "<="):
            dev = d_plus[m]
            is_met = bool(dev <= 1e-4)
        else:  # equality 'eq' or '=='
            dev = d_minus[m] + d_plus[m]
            is_met = bool(dev <= 1e-4)
        unwanted[m] = dev
        achieved.append(is_met)

    return d_minus, d_plus, unwanted, achieved


def step(state: dict, problem: Problem, max_evals: int = 500) -> dict:
    """Solves the Goal Programming deviation minimization problem."""
    if state["n_evals"] >= state.get("budget", 1000000):
        return state

    rng = state["rng"]
    targets = state["targets"]
    goal_types = state["goal_types"]
    weights = state["weights"]
    gp_method = state.get("method", "weighted")
    priorities = state.get("priorities", list(range(problem.n_obj)))
    budget = state.get("budget", 1000000)

    bounds = list(zip(problem.xl, problem.xu))
    eval_count = [0]
    last_val = [1e9]
    last_f = [None]
    last_cv = [0.0]

    if gp_method == "preemptive":
        # Pre-emptive (Lexicographic) Goal Programming: solve priority tiers sequentially
        x_curr = rng.uniform(problem.xl, problem.xu)
        prior_tier_optima: list[tuple[int, float]] = []

        for p_m in priorities:
            if state["n_evals"] + eval_count[0] >= budget:
                break

            def _preemptive_obj(x: np.ndarray) -> float:
                if state["n_evals"] + eval_count[0] >= budget:
                    return last_val[0]
                x_clamped = np.clip(x, problem.xl, problem.xu)
                f_int, cv, _ = problem.evaluate(x_clamped)
                eval_count[0] += 1
                f_nat = problem.to_natural(f_int)

                _, _, unw, _ = _compute_deviations(f_nat, targets, goal_types)
                tier_cost = unw[p_m]

                prior_pen = 0.0
                for prev_m, best_val in prior_tier_optima:
                    viol = max(0.0, unw[prev_m] - (best_val + 1e-4))
                    prior_pen += 1e4 * viol

                penalty = 1e4 * cv
                cost = float(tier_cost + prior_pen + penalty)
                last_val[0] = cost
                last_f[0] = f_int
                last_cv[0] = cv
                return cost

            if budget < 30:
                res = minimize(
                    _preemptive_obj,
                    x_curr,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={"maxiter": max(2, (budget - state["n_evals"]) // (2 * len(priorities)))},
                )
                x_curr = res.x
            else:
                max_iter_de = min(25, max(3, max_evals // (25 * max(1, problem.n_var) * len(priorities))))
                res_de = differential_evolution(
                    _preemptive_obj,
                    bounds,
                    maxiter=max_iter_de,
                    popsize=10,
                    seed=int(rng.integers(0, 1000000)),
                )
                res = minimize(
                    _preemptive_obj,
                    res_de.x,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={"maxiter": 80},
                )
                x_curr = res.x

            if state["n_evals"] + eval_count[0] < budget:
                f_stage, _, _ = problem.evaluate(np.clip(x_curr, problem.xl, problem.xu))
                eval_count[0] += 1
                last_f[0] = f_stage
            else:
                f_stage = last_f[0] if last_f[0] is not None else problem.ideal_point.copy()

            f_nat_stage = problem.to_natural(f_stage)
            _, _, unw_stage, _ = _compute_deviations(f_nat_stage, targets, goal_types)
            prior_tier_optima.append((p_m, unw_stage[p_m]))

        x_opt = np.clip(x_curr, problem.xl, problem.xu)
    else:
        # Weighted Archimedean Goal Programming
        def _objective(x: np.ndarray) -> float:
            if state["n_evals"] + eval_count[0] >= budget:
                return last_val[0]
            x_clamped = np.clip(x, problem.xl, problem.xu)
            f_int, cv, _ = problem.evaluate(x_clamped)
            eval_count[0] += 1
            f_nat = problem.to_natural(f_int)

            _, _, unwanted, _ = _compute_deviations(f_nat, targets, goal_types)
            total_dev = np.dot(weights, unwanted)
            penalty = 1e4 * cv
            cost = float(total_dev + penalty)
            last_val[0] = cost
            last_f[0] = f_int
            last_cv[0] = cv
            return cost

        x0 = rng.uniform(problem.xl, problem.xu)
        if budget < 30:
            res = minimize(
                _objective,
                x0,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": max(2, budget - state["n_evals"])},
            )
            x_opt = np.clip(res.x, problem.xl, problem.xu)
        else:
            max_iter_de = min(25, max(3, max_evals // (25 * max(1, problem.n_var))))
            res_de = differential_evolution(
                _objective,
                bounds,
                maxiter=max_iter_de,
                popsize=10,
                seed=int(rng.integers(0, 1000000)),
            )
            res = minimize(
                _objective,
                res_de.x,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": 80},
            )
            x_opt = np.clip(res.x, problem.xl, problem.xu)

    if problem.var_type == "integer":
        x_opt = np.round(x_opt)
    elif problem.var_type == "binary":
        x_opt = (x_opt >= 0.5).astype(float)

    if state["n_evals"] + eval_count[0] < budget:
        f_opt, cv_opt, _ = problem.evaluate(x_opt)
        eval_count[0] += 1
    else:
        f_opt = last_f[0] if last_f[0] is not None else problem.ideal_point.copy()
        cv_opt = last_cv[0]

    f_opt_nat = problem.to_natural(f_opt)
    d_minus, d_plus, unwanted, achieved = _compute_deviations(f_opt_nat, targets, goal_types)

    state["x_opt"] = x_opt
    state["f_opt"] = f_opt
    state["f_opt_nat"] = f_opt_nat
    state["deviations"] = unwanted
    state["d_minus"] = d_minus
    state["d_plus"] = d_plus
    state["achieved"] = achieved
    state["cv_opt"] = cv_opt
    state["n_evals"] += eval_count[0]
    state["iter"] += 1

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    targets: np.ndarray | None = None,
    goal_types: list[str] | None = None,
    weights: np.ndarray | None = None,
    normalized_weights: bool = False,
    method: str = "weighted",
    priorities: list[int] | None = None,
    **kwargs,
) -> dict:
    """Executes the Goal Programming solve."""
    state = initialize(
        problem,
        seed=seed,
        targets=targets,
        goal_types=goal_types,
        weights=weights,
        normalized_weights=normalized_weights,
        method=method,
        priorities=priorities,
        budget=budget,
        **kwargs,
    )
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
        "targets": state["targets"],
        "deviations": state["deviations"],
        "d_minus": state["d_minus"],
        "d_plus": state["d_plus"],
        "achieved": state["achieved"],
        "method": state["method"],
        "n_evals": state["n_evals"],
        "state": state,
    }
