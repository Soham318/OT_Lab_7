"""
alg07_lexicographic.py: Lexicographic Method (Strict Priority Sequential Optimization)

Lecture Family: Classical Methods (Part 1 & Bonus Module 5E, Chapter 5)
Original Citation: Stadler, W. (1988).
"Multicriteria Optimization in Engineering and in the Sciences."
Mathematical Concepts and Methods in Science and Engineering, Vol. 37. Plenum Press, New York.

Algorithmic Logic:
1. Decision maker orders the M objectives in a strict hierarchical priority:
       Priority 1 >> Priority 2 >> ... >> Priority M
   where higher priority objectives cannot be traded off against lower priority ones.
2. Sequential optimization with tolerance:
       - Stage 1: Solve for Priority 1:
             min_x  f_pi(1)^internal(x) + penalty * CV(x)
             giving optimal value f_1* and solution x^(1).
       - Stage 2: Solve for Priority 2, subject to preserving Priority 1 within tolerance tau_1:
             min_x  f_pi(2)^internal(x) + penalty * CV(x)
             subject to  f_pi(1)^internal(x) <= f_1* + tau_1
       - Stage k: Solve for Priority k, subject to:
             f_pi(m)^internal(x) <= f_m* + tau_m   for all m < k.
3. If at any stage the solution is unique, subsequent stages cannot alter the decision.
   If multiple solutions tie for the higher priority, the lower priority breaks the tie.

Penalty Approximation & Limitations (Important Methodological Note):
- In pure mathematics, lexicographic optimization enforces strict non-compensation.
- In continuous numerical optimization, exact equality level sets {x: f_1(x) = f_1*} have measure zero,
  requiring a numerical tolerance tau (default 1e-4) to permit exploration of ties.
- Continuous unconstrained solvers (e.g. Nelder-Mead, DE) approximate hard level set constraints via
  penalty scalarization: penalty = 1e4 * max(0, f_pi(m)(x) - (f_m* + tau_m)).
- Limitation: A penalty multiplier alone can theoretically permit a minute violation of earlier-stage
  optima if the gradient of the current objective balances the penalty gradient.
- Mathematical Guarantee in this Implementation:
  To eliminate this limitation and guarantee that earlier-stage optima are strictly preserved,
  this implementation:
  1. Warm-starts subsequent stages directly from the verified optimum of the previous stage.
  2. Tracks the best candidate solution evaluated during the stage that strictly satisfies all
     prior-stage tolerances (f_pi(m) <= f_m* + tau_m) and problem feasibility (CV <= 1e-4).
  3. Verifies the stage termination point: if any prior stage deteriorates beyond tau or CV > 1e-4,
     it automatically falls back to the best verified candidate, strictly guaranteeing non-deterioration.

Key Viva Question & Sanity Check on P11:
- "Explain why the profit tie must be broken by the second priority."
- In P11 (Production Planning), Priority 1 is Profit, Priority 2 is Units.
- Maximum profit is 2400, which is achieved by infinitely many points along the entire boundary segment
  between (0, 400) and (300, 200).
- Stage 2 then maximizes Units along this segment, breaking the tie in favor of (300, 200) where Units = 500!
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, differential_evolution
from moo.problems.base import Problem


def default_priorities_for_problem(problem: Problem) -> list[int]:
    """Returns default priority order of objectives (0-indexed)."""
    if problem.name.startswith("P11"):
        # P11 Production Planning: Profit first (idx 0), Units second (idx 1)
        return [0, 1]

    if problem.name.startswith("P7"):
        # P7 Water Allocation: Agriculture yield first (idx 0), Industry second (idx 1)
        return [0, 1]

    # Default: natural index order
    return list(range(problem.n_obj))


def initialize(
    problem: Problem,
    seed: int = 42,
    priorities: list[int] | None = None,
    tolerances: list[float] | None = None,
    **kwargs,
) -> dict:
    """Initializes the Lexicographic optimization state."""
    rng = np.random.default_rng(seed)
    pri = priorities if priorities is not None else default_priorities_for_problem(problem)
    tols = tolerances if tolerances is not None else [1e-4] * len(pri)

    state = {
        "seed": seed,
        "rng": rng,
        "priorities": pri,
        "tolerances": tols,
        "current_stage": 0,
        "stage_optima": [],  # stores (opt_val, x_opt) for each stage
        "x_opt": None,
        "f_opt": None,
        "f_opt_nat": None,
        "cv_opt": None,
        "n_evals": 0,
        "budget": kwargs.get("budget", 10000),
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem, max_evals_per_stage: int = 200) -> dict:
    """Executes one priority stage of lexicographic optimization."""
    stage = state["current_stage"]
    if stage >= len(state["priorities"]) or state["n_evals"] >= state.get("budget", 1000000):
        return state

    target_obj_idx = state["priorities"][stage]
    rng = state["rng"]
    bounds = list(zip(problem.xl, problem.xu))
    budget = state.get("budget", 1000000)

    # Warm start from previous stage solution if available
    if len(state["stage_optima"]) > 0:
        x0 = state["stage_optima"][-1][1].copy()
    else:
        x0 = rng.uniform(problem.xl, problem.xu)

    eval_count = [0]
    prior_stages = list(state["stage_optima"])
    priorities = state["priorities"]
    tolerances = state["tolerances"]
    last_val = [1e9]
    last_f = [None]
    last_cv = [0.0]

    best_stage_solution = [x0.copy()]
    best_stage_cost = [1e9]

    def _stage_objective(x: np.ndarray) -> float:
        if state["n_evals"] + eval_count[0] >= budget:
            return last_val[0]
        x_clamped = np.clip(x, problem.xl, problem.xu)
        f_int, cv, _ = problem.evaluate(x_clamped)
        eval_count[0] += 1

        obj_val = f_int[target_obj_idx]

        # Enforce preservation of higher-priority objectives
        prior_violation = 0.0
        all_prior_satisfied = True
        for s_idx, (best_val, _) in enumerate(prior_stages):
            p_obj = priorities[s_idx]
            tau = tolerances[s_idx]
            viol = max(0.0, f_int[p_obj] - (best_val + tau))
            if viol > 0.0:
                all_prior_satisfied = False
            prior_violation += viol

        penalty = 1e4 * cv + 1e4 * prior_violation
        cost = float(obj_val + penalty)
        last_val[0] = cost
        last_f[0] = f_int
        last_cv[0] = cv

        # Safeguard: track best point that strictly satisfies all prior stages and feasibility
        if cv <= 1e-4 and all_prior_satisfied and obj_val < best_stage_cost[0]:
            best_stage_cost[0] = obj_val
            best_stage_solution[0] = x_clamped.copy()

        return cost

    if budget < 30 or stage > 0:
        max_iter_nm = min(150, max(5, (budget - state["n_evals"]) // (2 * max(1, len(priorities) - stage))))
        res = minimize(
            _stage_objective,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter_nm},
        )
        x_stage = np.clip(res.x, problem.xl, problem.xu)
    else:
        max_iter_de = min(30, max(5, max_evals_per_stage // (20 * max(1, problem.n_var))))
        res_de = differential_evolution(
            _stage_objective,
            bounds,
            maxiter=max_iter_de,
            popsize=10,
            seed=int(rng.integers(0, 1000000)),
        )
        res = minimize(
            _stage_objective,
            res_de.x,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 80},
        )
        x_stage = np.clip(res.x, problem.xl, problem.xu)

    if problem.var_type == "integer":
        x_stage = np.round(x_stage)
    elif problem.var_type == "binary":
        x_stage = (x_stage >= 0.5).astype(float)

    if state["n_evals"] + eval_count[0] < budget:
        f_stage, cv_stage, _ = problem.evaluate(x_stage)
        eval_count[0] += 1
        last_f[0] = f_stage
        last_cv[0] = cv_stage
    else:
        f_stage = last_f[0] if last_f[0] is not None else problem.ideal_point.copy()
        cv_stage = last_cv[0]

    # Non-deterioration verification: ensure earlier stages did not deteriorate beyond tolerance
    deteriorated = any(
        f_stage[priorities[s_idx]] > prior_stages[s_idx][0] + tolerances[s_idx]
        for s_idx in range(len(prior_stages))
    ) or (cv_stage > 1e-4)

    if deteriorated and best_stage_solution[0] is not None:
        x_opt = best_stage_solution[0].copy()
        if state["n_evals"] + eval_count[0] < budget:
            f_stage, cv_stage, _ = problem.evaluate(x_opt)
            eval_count[0] += 1
            last_f[0] = f_stage
            last_cv[0] = cv_stage
    else:
        x_opt = x_stage

    f_opt = f_stage
    cv_opt = cv_stage
    f_opt_nat = problem.to_natural(f_opt)
    stage_best_val = f_opt[target_obj_idx]

    state["stage_optima"].append((stage_best_val, x_opt))
    state["x_opt"] = x_opt
    state["f_opt"] = f_opt
    state["f_opt_nat"] = f_opt_nat
    state["cv_opt"] = cv_opt
    state["n_evals"] += eval_count[0]
    state["current_stage"] += 1
    state["iter"] += 1

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    priorities: list[int] | None = None,
    tolerances: list[float] | None = None,
    **kwargs,
) -> dict:
    """Executes the complete Lexicographic sequential optimization."""
    state = initialize(problem, seed=seed, priorities=priorities, tolerances=tolerances, budget=budget, **kwargs)
    state["budget"] = budget
    evals_per_stage = max(10, budget // len(state["priorities"]))

    while state["current_stage"] < len(state["priorities"]) and state["n_evals"] < budget:
        step(state, problem, max_evals_per_stage=evals_per_stage)

    X_opt = np.array([state["x_opt"]]) if state["x_opt"] is not None else np.empty((0, problem.n_var))
    F_opt = np.array([state["f_opt"]]) if state["f_opt"] is not None else np.empty((0, problem.n_obj))
    F_opt_nat = np.array([state["f_opt_nat"]]) if state["f_opt_nat"] is not None else np.empty((0, problem.n_obj))
    CV_opt = np.array([state["cv_opt"]]) if state["cv_opt"] is not None else np.empty(0)

    return {
        "X_opt": X_opt,
        "F_opt": F_opt,
        "F_opt_nat": F_opt_nat,
        "CV_opt": CV_opt,
        "priorities": state["priorities"],
        "stage_optima": [s[0] for s in state["stage_optima"]],
        "n_evals": state["n_evals"],
        "state": state,
    }
