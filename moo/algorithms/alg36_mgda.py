"""
alg36_mgda.py: MGDA (Multiple Gradient Descent Algorithm)

Lecture Family: Many-Objective, Indicator, Preference & Gradient Methods (Bonus Module 5E, Chapter 5)
Original Citation: Désidéri, J. A. (2012).
"Multiple-Gradient Descent Algorithm (MGDA) for Multiobjective Optimization."
Comptes Rendus Mathematique, 350(5-6), 313-318.
Also: Sener, O., & Koltun, V. (2018). "Multi-Task Learning as Multi-Objective Optimization." NeurIPS 2018.

Algorithmic Logic:
1. Problem Type & Gradient Computation:
       Designed for continuous, differentiable multiobjective problems (e.g. P1, P4, P13).
       Evaluates objective gradients grad f_m(x) analytically if available, or via central
       finite differences: (f_m(x + eps*e_j) - f_m(x - eps*e_j)) / (2*eps).
2. Minimum-Norm Point in Convex Hull of Gradients:
       Find convex weights alpha in Delta_M minimizing:
           min_{alpha} || sum_{m=1}^M alpha_m * grad f_m(x) ||_2^2
           s.t. sum alpha_m = 1, alpha_m >= 0.
       - For M = 2: Solved analytically via line segment projection.
       - For M >= 3: Solved via Frank-Wolfe quadratic solver on the unit simplex.
3. Pareto Criticality & Common Descent Direction:
       If ||omega*|| <= eps_crit (Pareto stationary): point is Pareto optimal; restart at new seed.
       Otherwise, descent direction d = -omega* ensures:
           <grad f_m(x), d> <= -||omega*||^2 < 0  for all m=1..M.
4. Line Search & Trajectory Updates:
       x_{t+1} = clip(x_t + lr * d, xl, xu).
5. Multi-Start Diversity:
       Executes multi-start trajectories across [xl, xu] to discover diverse Pareto front points.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import random_population, extract_non_dominated


def min_norm_2obj(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Computes exact analytical minimum-norm convex combination for 2 objectives."""
    diff = g1 - g2
    denom = float(np.dot(diff, diff))
    if denom <= 1e-14:
        return 0.5 * (g1 + g2)

    alpha = float(np.dot(g2, g2 - g1)) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha * g1 + (1.0 - alpha) * g2


def min_norm_frank_wolfe(G: np.ndarray, max_iter: int = 25) -> np.ndarray:
    """
    Solves min_{alpha in Delta_M} ||G^T alpha||^2 via Frank-Wolfe algorithm on unit simplex.
    G is (M, D) matrix of objective gradients.
    """
    M, D = G.shape
    if M == 1:
        return G[0]
    if M == 2:
        return min_norm_2obj(G[0], G[1])

    # Initial uniform weights
    alpha = np.full(M, 1.0 / M)
    w = np.dot(alpha, G)

    for _ in range(max_iter):
        # Linear subproblem: argmin_{m} <G[m], w>
        grad_dots = np.dot(G, w)
        t = int(np.argmin(grad_dots))
        e_t = G[t]

        # Line search for step size gamma in [0, 1]
        diff = e_t - w
        denom = float(np.dot(diff, diff))
        if denom <= 1e-14:
            break

        gamma = -float(np.dot(w, diff)) / denom
        gamma = np.clip(gamma, 0.0, 1.0)

        # Update
        alpha = (1.0 - gamma) * alpha
        alpha[t] += gamma
        w = (1.0 - gamma) * w + gamma * e_t

        if gamma < 1e-6:
            break

    return w


def compute_numerical_gradients(
    x: np.ndarray,
    problem: Problem,
    budget_rem: int,
    eps: float = 1e-5,
) -> tuple[np.ndarray, int]:
    """
    Computes numerical gradient matrix G of shape (M, D) via central finite differences.
    Returns (G, evals_used).
    """
    M = problem.n_obj
    D = len(x)
    G = np.zeros((M, D), dtype=float)
    evals_used = 0

    # If problem provides an analytical gradient function (e.g. P13)
    if hasattr(problem, "compute_gradients"):
        g_analytic = getattr(problem, "compute_gradients")(x)
        return np.asarray(g_analytic, dtype=float), 0

    for j in range(D):
        if evals_used + 2 > budget_rem:
            break

        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] = np.clip(x[j] + eps, problem.xl[j], problem.xu[j])
        x_minus[j] = np.clip(x[j] - eps, problem.xl[j], problem.xu[j])

        dx = x_plus[j] - x_minus[j]
        if abs(dx) < 1e-12:
            continue

        f_plus, _, _ = problem.evaluate(x_plus)
        f_minus, _, _ = problem.evaluate(x_minus)
        evals_used += 2

        G[:, j] = (f_plus - f_minus) / dx

    return G, evals_used


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MGDA state, learning rate, and archive within budget."""
    rng = np.random.default_rng(seed)
    budget = kwargs.get("budget", 10000)
    lr = kwargs.get("lr", 0.05)
    eps_crit = kwargs.get("eps_crit", 1e-4)

    # Initial starting point
    x_cur = random_population(problem, 1, rng)[0]
    f_cur, cv_cur, _ = problem.evaluate(x_cur)

    arch_X = np.array([x_cur])
    arch_F = np.array([f_cur])
    arch_CV = np.array([cv_cur])

    state = {
        "seed": seed,
        "rng": rng,
        "budget": budget,
        "lr": lr,
        "eps_crit": eps_crit,
        "x_cur": x_cur,
        "f_cur": f_cur,
        "cv_cur": cv_cur,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": 1,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one descent step of MGDA."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    feas_tol = problem.feas_tol
    lr = state["lr"]
    eps_crit = state["eps_crit"]
    x_cur = state["x_cur"]
    f_cur = state["f_cur"]
    cv_cur = state["cv_cur"]

    # 1. Compute gradients for all objectives
    G, evals_used = compute_numerical_gradients(x_cur, problem, rem_budget)
    state["n_evals"] += evals_used
    rem_budget -= evals_used

    # 2. Minimum-norm vector in convex hull
    omega = min_norm_frank_wolfe(G)
    norm_sq = float(np.dot(omega, omega))

    if norm_sq < eps_crit or rem_budget <= 0:
        # Pareto stationary or out of evals -> restart at new random location
        x_new = random_population(problem, 1, rng)[0]
    else:
        # Common descent step: d = -omega / ||omega||
        norm_val = np.sqrt(norm_sq)
        d = -omega / max(1e-12, norm_val)
        x_new = np.clip(x_cur + lr * d, problem.xl, problem.xu)

    if rem_budget > 0:
        f_new, cv_new, _ = problem.evaluate(x_new)
        state["n_evals"] += 1
        state["x_cur"] = x_new
        state["f_cur"] = f_new
        state["cv_cur"] = cv_new

        # Update Archive
        comb_X = np.vstack((state["arch_X"], x_new.reshape(1, -1)))
        comb_F = np.vstack((state["arch_F"], f_new.reshape(1, -1)))
        comb_CV = np.concatenate((state["arch_CV"], [cv_new]))

        new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
            comb_X, comb_F, comb_CV, feas_tol=feas_tol
        )
        if len(new_arch_X) > 200:
            from moo.moo_utils import crowding_distance
            cd = crowding_distance(new_arch_F, list(range(len(new_arch_F))))
            order = np.argsort(-cd)
            keep = order[:200]
            new_arch_X = new_arch_X[keep]
            new_arch_F = new_arch_F[keep]
            new_arch_CV = new_arch_CV[keep]

        state["arch_X"] = new_arch_X
        state["arch_F"] = new_arch_F
        state["arch_CV"] = new_arch_CV

    state["iter"] += 1
    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MGDA until budget is exhausted."""
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
