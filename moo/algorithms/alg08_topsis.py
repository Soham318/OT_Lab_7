"""
alg08_topsis.py: TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

Lecture Family: Ranking / MCDM Methods (Part 1 & Bonus Module 5E, Chapter 5)
Original Citation: Hwang, C. L., & Yoon, K. (1981).
"Multiple Attribute Decision Making: Methods and Applications."
Lecture Notes in Economics and Mathematical Systems, Vol. 186. Springer-Verlag, New York.

Algorithmic Logic:
1. Input: A finite set of N discrete candidate alternatives evaluated on M criteria (N x M decision matrix X),
   with criterion weights w = (w_1, ..., w_M) and types ('min' for cost, 'max' for benefit).
2. Vector Normalization:
       r_{ij} = x_{ij} / sqrt( sum_{k=1}^N x_{kj}^2 )
3. Weighted Normalized Matrix:
       v_{ij} = w_j * r_{ij}
4. Determine Ideal Solution A* (best) and Anti-Ideal Solution A' (worst):
       - If criterion j is benefit ('max'):  A_j* = max_i v_{ij},  A_j' = min_i v_{ij}
       - If criterion j is cost ('min'):     A_j* = min_i v_{ij},  A_j' = max_i v_{ij}
5. Calculate Euclidean Separation Measures:
       S_i* = sqrt( sum_{j=1}^M (v_{ij} - A_j*)^2 )   [Distance to positive ideal]
       S_i' = sqrt( sum_{j=1}^M (v_{ij} - A_j')^2 )   [Distance to negative ideal]
6. Compute Relative Closeness to Ideal Solution:
       C_i = S_i' / (S_i* + S_i') in [0, 1]
7. Rank alternatives in descending order of C_i (C_i = 1 implies identity with ideal).

Crucial Distinction (Viva Note):
- "Why do TOPSIS and VIKOR not count as optimizers?"
  They do NOT search or generate new candidate solutions in the decision space.
  They are Multi-Criteria Decision Making (MCDM) ranking methods that take an already-existing,
  discrete set of alternatives (e.g. P12 supplier choices or an obtained Pareto front from an EA)
  and sort them based on geometric compromise criteria.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from moo.problems.base import Problem


def topsis_rank(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    criteria_types: Sequence[str],
) -> dict:
    """
    Ranks a discrete decision matrix using TOPSIS.

    Parameters:
        decision_matrix (np.ndarray): Matrix of shape (N, M).
        weights (np.ndarray): Weights of shape (M,), summing to 1.
        criteria_types (Sequence[str]): List of 'min' or 'max' for each criterion.

    Returns:
        dict:
            'scores': Closeness coefficients C_i of shape (N,)
            'rankings': Indices sorted from best (rank 1) to worst
            'ideal_pos': Positive ideal coordinates A*
            'ideal_neg': Negative ideal coordinates A'
            's_pos': Separation distances to positive ideal S*
            's_neg': Separation distances to negative ideal S'
    """
    X = np.asarray(decision_matrix, dtype=float)
    N, M = X.shape
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    # 1. Vector normalization: r_ij = x_ij / sqrt(sum(x_kj^2))
    col_norms = np.sqrt(np.sum(X ** 2, axis=0))
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    R = X / col_norms

    # 2. Weighted normalized matrix: v_ij = w_j * r_ij
    V = R * w

    # 3. Determine positive ideal (A*) and negative ideal (A')
    A_star = np.zeros(M, dtype=float)
    A_prime = np.zeros(M, dtype=float)

    for j in range(M):
        ctype = criteria_types[j].lower()
        if ctype in ("max", "benefit"):
            A_star[j] = np.max(V[:, j])
            A_prime[j] = np.min(V[:, j])
        else:  # 'min' or 'cost'
            A_star[j] = np.min(V[:, j])
            A_prime[j] = np.max(V[:, j])

    # 4. Euclidean separation
    S_star = np.sqrt(np.sum((V - A_star) ** 2, axis=1))
    S_prime = np.sqrt(np.sum((V - A_prime) ** 2, axis=1))

    # 5. Relative closeness: C_i = S' / (S* + S')
    denom = S_star + S_prime
    denom = np.where(denom < 1e-12, 1.0, denom)
    C = S_prime / denom

    # 6. Ranking (descending order of C_i)
    ranking = np.argsort(-C)

    return {
        "scores": C,
        "rankings": ranking,
        "ideal_pos": A_star,
        "ideal_neg": A_prime,
        "s_pos": S_star,
        "s_neg": S_prime,
    }


def initialize(
    problem: Problem,
    seed: int = 42,
    weights: np.ndarray | None = None,
    criteria_types: Sequence[str] | None = None,
    alternatives_matrix: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """Initializes TOPSIS state."""
    rng = np.random.default_rng(seed)

    n_evals = 0
    sample_X = None
    sample_F_int = None
    sample_F_nat = None
    sample_CV = None

    if alternatives_matrix is not None:
        X = np.asarray(alternatives_matrix, dtype=float)
    elif hasattr(problem, "get_decision_matrix"):
        X = problem.get_decision_matrix()
    else:
        # Sample points across decision space if not a discrete problem
        sample_size = max(1, min(20, kwargs.get("budget", 10000)))
        sample_X = rng.uniform(problem.xl, problem.xu, size=(sample_size, problem.n_var))
        f_int_list = []
        f_nat_list = []
        cv_list = []
        for x in sample_X:
            f_i, cv_i, _ = problem.evaluate(x)
            f_int_list.append(f_i)
            f_nat_list.append(problem.to_natural(f_i))
            cv_list.append(cv_i)
        sample_F_int = np.array(f_int_list, dtype=float)
        sample_F_nat = np.array(f_nat_list, dtype=float)
        sample_CV = np.array(cv_list, dtype=float)
        X = sample_F_nat
        n_evals = sample_size

    if weights is not None:
        w = np.asarray(weights, dtype=float)
    elif hasattr(problem, "weights"):
        w = problem.weights.copy()
    else:
        w = np.ones(X.shape[1]) / X.shape[1]

    if criteria_types is not None:
        c_types = list(criteria_types)
    elif hasattr(problem, "criteria_min_max"):
        c_types = problem.criteria_min_max
    else:
        c_types = ["min"] * X.shape[1]

    state = {
        "seed": seed,
        "rng": rng,
        "decision_matrix": X,
        "sample_X": sample_X,
        "sample_F_int": sample_F_int,
        "sample_F_nat": sample_F_nat,
        "sample_CV": sample_CV,
        "weights": w,
        "criteria_types": c_types,
        "results": None,
        "n_evals": n_evals,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Performs TOPSIS ranking."""
    res = topsis_rank(
        decision_matrix=state["decision_matrix"],
        weights=state["weights"],
        criteria_types=state["criteria_types"],
    )
    state["results"] = res
    state["iter"] += 1
    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    weights: np.ndarray | None = None,
    criteria_types: Sequence[str] | None = None,
    alternatives_matrix: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """Executes TOPSIS ranking on the alternatives."""
    state = initialize(
        problem,
        seed=seed,
        weights=weights,
        criteria_types=criteria_types,
        alternatives_matrix=alternatives_matrix,
        budget=budget,
        **kwargs,
    )
    step(state, problem)
    res = state["results"]

    best_idx = int(res["rankings"][0])
    X_mat = state["decision_matrix"]

    if hasattr(problem, "evaluate") and problem.name.startswith("P12"):
        # For P12 Supplier Selection, decision variable is the integer supplier index [0..4]
        x_sol = np.array([float(best_idx)])
        f_int, cv_val, _ = problem.evaluate(x_sol)
        f_nat = problem.to_natural(f_int)
        state["n_evals"] += 1
    elif state.get("sample_X") is not None and state.get("sample_F_int") is not None:
        x_sol = state["sample_X"][best_idx]
        f_int = state["sample_F_int"][best_idx]
        f_nat = state["sample_F_nat"][best_idx]
        cv_val = state["sample_CV"][best_idx]
    else:
        # Fallback for arbitrary decision matrix input
        row_nat = X_mat[best_idx].copy()
        f_nat = row_nat
        mult = np.array([-1.0 if str(d).lower() == "max" else 1.0 for d in problem.obj_directions], dtype=float)
        f_int = f_nat * mult
        cv_val = 0.0
        x_sol = np.array([float(best_idx)]) if problem.n_var == 1 else row_nat[:problem.n_var]

    return {
        "X_opt": np.array([x_sol]),
        "F_opt": np.array([f_int]),
        "F_opt_nat": np.array([f_nat]),
        "CV_opt": np.array([cv_val]),
        "rankings": res["rankings"],
        "scores": res["scores"],
        "best_idx": int(best_idx),
        "n_evals": state["n_evals"],
        "state": state,
    }
