"""
alg09_vikor.py: VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)

Lecture Family: Ranking / MCDM Methods (Part 1 & Bonus Module 5E, Chapter 5)
Original Citation: Opricovic, S. (1998).
"Multicriteria Optimization of Civil Engineering Systems." Faculty of Civil Engineering, Belgrade.
Opricovic, S., & Tzeng, G. H. (2004).
"Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS."
European Journal of Operational Research, 156(2), 445-455.

Algorithmic Logic:
1. Input: N discrete alternatives evaluated on M criteria (N x M matrix X), criteria weights w,
   types ('min' or 'max'), and weight of maximum group utility strategy v in [0, 1] (default v = 0.5).
2. Determine the best f_j* and worst f_j- values for each criterion:
       - Benefit ('max'): f_j* = max_i x_{ij},  f_j- = min_i x_{ij}
       - Cost ('min'):    f_j* = min_i x_{ij},  f_j- = max_i x_{ij}
3. Compute Group Utility (S_i) and Individual Regret (R_i):
       S_i = sum_{j=1}^M w_j * |f_j* - x_{ij}| / |f_j* - f_j-|
       R_i = max_{j=1...M} [ w_j * |f_j* - x_{ij}| / |f_j* - f_j-| ]
4. Compute the VIKOR Compromise Index (Q_i):
       Q_i = v * (S_i - S*) / (S- - S*) + (1 - v) * (R_i - R*) / (R- - R*)
   where S* = min S_i, S- = max S_i, R* = min R_i, R- = max R_i.
   - v > 0.5: voting by majority rule
   - v = 0.5: consensus
   - v < 0.5: veto / individual regret dominance
5. Rank alternatives by ascending order of S, R, and Q (smaller is better).
6. Verify Compromise Conditions:
   - Condition 1 (Acceptable Advantage):
         Q(A^(2)) - Q(A^(1)) >= DQ,  where DQ = 1 / (N - 1)
   - Condition 2 (Acceptable Stability in Decision Making):
         A^(1) must also be best ranked by S or by R.
   If conditions are not both met, a compromise set of solutions is returned.

Key Distinction vs TOPSIS (Viva Note):
- TOPSIS selects an alternative closest to the ideal solution and farthest from the negative ideal,
  without considering the relative importance of group utility vs individual regret.
- VIKOR focuses on reaching a compromise solution with an acceptable advantage that balances
  majority benefit (S) against the maximum individual complaint/regret (R).
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from moo.problems.base import Problem


def vikor_rank(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    criteria_types: Sequence[str],
    v: float = 0.5,
) -> dict:
    """
    Ranks a discrete decision matrix using VIKOR.

    Parameters:
        decision_matrix (np.ndarray): Decision matrix of shape (N, M).
        weights (np.ndarray): Weights of shape (M,), summing to 1.
        criteria_types (Sequence[str]): 'min' or 'max' for each criterion.
        v (float): Weight of the strategy of maximum group utility (default: 0.5).

    Returns:
        dict:
            'S': Group utility array of shape (N,)
            'R': Individual regret array of shape (N,)
            'Q': VIKOR index array of shape (N,)
            'rank_S': Ranking by S (ascending)
            'rank_R': Ranking by R (ascending)
            'rank_Q': Ranking by Q (ascending)
            'c1_passed': bool (Acceptable Advantage test)
            'c2_passed': bool (Acceptable Stability test)
            'compromise_set': list of indices belonging to the compromise set
    """
    X = np.asarray(decision_matrix, dtype=float)
    N, M = X.shape
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    f_star = np.zeros(M, dtype=float)
    f_minus = np.zeros(M, dtype=float)

    for j in range(M):
        ctype = criteria_types[j].lower()
        if ctype in ("max", "benefit"):
            f_star[j] = np.max(X[:, j])
            f_minus[j] = np.min(X[:, j])
        else:  # 'min' or 'cost'
            f_star[j] = np.min(X[:, j])
            f_minus[j] = np.max(X[:, j])

    # Range per criterion
    f_diff = np.abs(f_star - f_minus)
    f_diff = np.where(f_diff < 1e-12, 1.0, f_diff)

    # Compute S_i and R_i
    # Normalized regret per criterion: w_j * |f_j* - x_ij| / |f_j* - f_j-|
    normalized_regret = w * np.abs(f_star - X) / f_diff  # shape: (N, M)
    S = np.sum(normalized_regret, axis=1)                # shape: (N,)
    R = np.max(normalized_regret, axis=1)                # shape: (N,)

    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    S_range = S_minus - S_star if abs(S_minus - S_star) > 1e-12 else 1.0
    R_range = R_minus - R_star if abs(R_minus - R_star) > 1e-12 else 1.0

    # Compute Q_i
    Q = v * ((S - S_star) / S_range) + (1.0 - v) * ((R - R_star) / R_range)

    # Rankings (ascending: smaller Q, S, R is better)
    rank_Q = np.argsort(Q)
    rank_S = np.argsort(S)
    rank_R = np.argsort(R)

    # Check compromise conditions
    c1_passed = False
    c2_passed = False
    compromise_set = [int(rank_Q[0])]

    if N > 1:
        DQ = 1.0 / (N - 1)
        best_Q = Q[rank_Q[0]]
        second_Q = Q[rank_Q[1]]

        # C1: Acceptable Advantage: Q(second) - Q(best) >= DQ
        c1_passed = bool((second_Q - best_Q) >= (DQ - 1e-9))

        # C2: Acceptable Stability: best in Q must also be best in S or R
        c2_passed = bool(rank_Q[0] == rank_S[0] or rank_Q[0] == rank_R[0])

        if c1_passed and c2_passed:
            compromise_set = [int(rank_Q[0])]
        elif not c1_passed:
            # All alternatives A^(k) such that Q(A^(k)) - Q(A^(1)) < DQ
            compromise_set = [int(idx) for idx in rank_Q if (Q[idx] - best_Q) < (DQ + 1e-9)]
        else:
            # C1 passed but C2 failed: compromise set is {A^(1), A^(2)}
            compromise_set = [int(rank_Q[0]), int(rank_Q[1])]

    return {
        "S": S,
        "R": R,
        "Q": Q,
        "rank_S": rank_S,
        "rank_R": rank_R,
        "rank_Q": rank_Q,
        "c1_passed": c1_passed,
        "c2_passed": c2_passed,
        "compromise_set": compromise_set,
    }


def initialize(
    problem: Problem,
    seed: int = 42,
    weights: np.ndarray | None = None,
    criteria_types: Sequence[str] | None = None,
    alternatives_matrix: np.ndarray | None = None,
    v: float = 0.5,
    **kwargs,
) -> dict:
    """Initializes VIKOR state."""
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
        "v": float(v),
        "results": None,
        "n_evals": n_evals,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Performs VIKOR ranking."""
    res = vikor_rank(
        decision_matrix=state["decision_matrix"],
        weights=state["weights"],
        criteria_types=state["criteria_types"],
        v=state["v"],
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
    v: float = 0.5,
    **kwargs,
) -> dict:
    """Executes VIKOR compromise ranking."""
    state = initialize(
        problem,
        seed=seed,
        weights=weights,
        criteria_types=criteria_types,
        alternatives_matrix=alternatives_matrix,
        v=v,
        budget=budget,
        **kwargs,
    )
    step(state, problem)
    res = state["results"]

    best_idx = int(res["rank_Q"][0])
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
        "rankings": res["rank_Q"],
        "S": res["S"],
        "R": res["R"],
        "Q": res["Q"],
        "compromise_set": res["compromise_set"],
        "c1_passed": res["c1_passed"],
        "c2_passed": res["c2_passed"],
        "best_idx": int(best_idx),
        "n_evals": state["n_evals"],
        "state": state,
    }
