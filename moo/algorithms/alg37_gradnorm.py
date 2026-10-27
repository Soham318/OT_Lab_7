"""
alg37_gradnorm.py: GradNorm (Gradient Normalization for Multi-Task Learning) — Task C Deep Dive #3

Lecture Family: Many-Objective, Indicator, Preference & Gradient Methods (Bonus Module 5E, Chapter 5)
Student Roll No. Ending: 13 -> Assigned Algorithm 37: GradNorm
Original Citation: Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018).
"GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks."
International Conference on Machine Learning (ICML 2018), PMLR 80, 794-803.

Algorithmic Logic:
1. Dynamic Task Weighting & Total Loss:
       L_total(t) = sum_{m=1}^M w_m(t) * L_m(t)
       where weights are initialized to w_m(0) = 1.0 (summing to M).
2. Gradient Norm Computation:
       For shared parameters W:
           G_W^{(m)}(t) = || grad_W (w_m(t) * L_m(t)) ||_2 = w_m(t) * || grad_W L_m(t) ||_2
       Average norm across all tasks:
           bar{G}_W(t) = (1 / M) * sum_{m=1}^M G_W^{(m)}(t)
3. Inverse Training Rate Balancing:
       Loss ratio: tilde{L}_m(t) = L_m(t) / L_m(0)
       Relative inverse training rate:
           r_m(t) = tilde{L}_m(t) / ( (1 / M) * sum_{j=1}^M tilde{L}_j(t) )
       Target gradient norm:
           Target_m(t) = bar{G}_W(t) * [r_m(t)]^alpha
       where alpha > 0 is the asymmetry restoring force parameter.
4. GradNorm Weight Loss & Update:
       L_grad(w_1, ..., w_M) = sum_{m=1}^M |G_W^{(m)}(t) - Target_m(t)|
       Gradient: d(L_grad)/d(w_m) = sgn(G_W^{(m)}(t) - Target_m(t)) * || grad_W L_m(t) ||_2
       Update: w_m <- max(0.01, w_m - eta_grad * d(L_grad)/d(w_m))
       Renormalize: w_m <- (w_m / sum_j w_j) * M
5. Shared Model Parameter Update:
       W <- W - eta_model * sum_{m=1}^M w_m * grad_W L_m
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import random_population, extract_non_dominated


def compute_task_gradients(
    x: np.ndarray,
    problem: Problem,
    budget_rem: int,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Evaluates current objectives/losses and computes task gradient matrix G of shape (M, D).
    Returns (f_cur, cv_cur, G, evals_used).
    """
    f_cur, cv_cur, _ = problem.evaluate(x)
    evals_used = 1

    M = problem.n_obj
    D = len(x)
    G = np.zeros((M, D), dtype=float)

    if hasattr(problem, "compute_gradients"):
        G_analytic = getattr(problem, "compute_gradients")(x)
        return f_cur, cv_cur, np.asarray(G_analytic, dtype=float), evals_used

    # Central finite differences
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

    return f_cur, cv_cur, G, evals_used


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes GradNorm weights, learning rates, and archive within budget."""
    rng = np.random.default_rng(seed)
    budget = kwargs.get("budget", 10000)
    alpha = kwargs.get("alpha", 1.5)
    eta_model = kwargs.get("eta_model", kwargs.get("lr", 0.05))
    eta_grad = kwargs.get("eta_grad", 0.025)

    M = problem.n_obj
    weights = np.ones(M, dtype=float)

    # Initial parameter vector
    x_cur = random_population(problem, 1, rng)[0]
    f_init, cv_init, _ = problem.evaluate(x_cur)

    # Initial losses L_m(0) for tracking relative training rates
    L0 = np.maximum(1e-6, np.abs(f_init))

    arch_X = np.array([x_cur])
    arch_F = np.array([f_init])
    arch_CV = np.array([cv_init])

    state = {
        "seed": seed,
        "rng": rng,
        "budget": budget,
        "alpha": alpha,
        "eta_model": eta_model,
        "eta_grad": eta_grad,
        "M": M,
        "weights": weights,
        "L0": L0,
        "x_cur": x_cur,
        "f_cur": f_init,
        "cv_cur": cv_init,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": 1,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one optimization step of GradNorm."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    feas_tol = problem.feas_tol
    M = state["M"]
    weights = state["weights"]
    L0 = state["L0"]
    alpha = state["alpha"]
    eta_model = state["eta_model"]
    eta_grad = state["eta_grad"]
    x_cur = state["x_cur"]

    # 1. Compute task losses and gradients
    f_cur, cv_cur, G, evals_used = compute_task_gradients(x_cur, problem, rem_budget)
    state["n_evals"] += evals_used
    state["f_cur"] = f_cur
    state["cv_cur"] = cv_cur
    rem_budget -= evals_used

    # 2. Gradient norms: G_norm_m = || grad_W (w_m * L_m) ||_2 = w_m * || grad_W L_m ||_2
    task_grad_norms = np.linalg.norm(G, axis=1)
    G_weighted_norms = weights * task_grad_norms
    mean_G_norm = float(np.mean(G_weighted_norms))

    # 3. Relative inverse training rates
    losses = np.maximum(1e-6, np.abs(f_cur))
    loss_ratios = losses / L0
    mean_loss_ratio = float(np.mean(loss_ratios))
    r_task = loss_ratios / max(1e-8, mean_loss_ratio)

    # 4. Target gradient norms
    targets = mean_G_norm * (r_task ** alpha)

    # 5. GradNorm weight gradients and updates
    diffs = G_weighted_norms - targets
    grad_L_weights = np.sign(diffs) * task_grad_norms

    # Gradient descent update on weights
    weights = weights - eta_grad * grad_L_weights
    weights = np.maximum(0.01, weights)

    # Renormalize weights so that sum(weights) = M
    weights = (weights / np.sum(weights)) * M
    state["weights"] = weights

    # 6. Model parameter update
    total_grad = np.dot(weights, G)
    x_next = np.clip(x_cur - eta_model * total_grad, problem.xl, problem.xu)

    state["x_cur"] = x_next
    state["iter"] += 1

    # 7. Update Archive
    comb_X = np.vstack((state["arch_X"], x_cur.reshape(1, -1)))
    comb_F = np.vstack((state["arch_F"], f_cur.reshape(1, -1)))
    comb_CV = np.concatenate((state["arch_CV"], [cv_cur]))

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

    # Periodically restart if stationary or reached bounds
    if np.linalg.norm(total_grad) < 1e-5 and rem_budget > 10:
        state["x_cur"] = random_population(problem, 1, rng)[0]

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs GradNorm until budget is exhausted."""
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
