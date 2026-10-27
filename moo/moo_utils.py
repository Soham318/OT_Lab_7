"""
moo_utils.py: Shared Multi-Objective Optimization Toolkit

This module implements core utilities for multiobjective optimization from scratch:
- Pareto dominance test
- Deb's constrained-dominance test (using documented CV & feasibility tolerance 1e-4)
- Fast non-dominated sorting (Deb's O(M N^2) algorithm)
- Crowding distance assignment
- Exact 2D Hypervolume computation (rectangle decomposition)
- Monte Carlo Hypervolume computation (for M >= 3 objectives)
- Inverted Generational Distance (IGD)
- Schott's Spacing metric
- Objective normalization
- Das-Dennis reference point generation on the unit simplex

Rules:
- Uses ONLY NumPy and the Python standard library.
- Constraint violation CV is strictly non-negative (CV >= 0.0).
- Feasibility is determined as CV <= feas_tol (default 1e-4).
- Internal objectives operate strictly as MINIMIZATION.
"""

from __future__ import annotations
import math
from typing import Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Dominance & Feasibility Checking
# ---------------------------------------------------------------------------

def dominates(f1: np.ndarray, f2: np.ndarray) -> bool:
    """
    Standard Pareto dominance test for minimization objectives.
    
    Solution 1 dominates Solution 2 (f1 <_p f2) if:
    1. f1 is no worse than f2 in all objectives (f1 <= f2 element-wise).
    2. f1 is strictly better than f2 in at least one objective (f1 < f2 for some m).

    Parameters:
        f1 (np.ndarray): Objective vector of candidate 1 (shape: (M,)).
        f2 (np.ndarray): Objective vector of candidate 2 (shape: (M,)).

    Returns:
        bool: True if f1 dominates f2, False otherwise.
    """
    f1 = np.asarray(f1, dtype=float).ravel()
    f2 = np.asarray(f2, dtype=float).ravel()
    if len(f1) != len(f2):
        raise ValueError(f"Objective vectors must have same length, got {len(f1)} and {len(f2)}")
    if len(f1) == 0:
        return False
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def constrained_dominates(
    f1: np.ndarray,
    cv1: float,
    f2: np.ndarray,
    cv2: float,
    feas_tol: float = 1e-4,
) -> bool:
    """
    Deb's feasibility-first constrained-dominance test.
    
    Solution 1 constrained-dominates Solution 2 (1 <_con 2) iff:
    - Case 1: Solution 1 is feasible (cv1 <= feas_tol) and Solution 2 is infeasible (cv2 > feas_tol).
    - Case 2: Both solutions are infeasible (cv1 > feas_tol and cv2 > feas_tol),
              and Solution 1 has a strictly smaller constraint violation (cv1 < cv2).
    - Case 3: Both solutions are feasible (cv1 <= feas_tol and cv2 <= feas_tol),
              and Solution 1 Pareto-dominates Solution 2 on objectives (dominates(f1, f2)).

    Parameters:
        f1 (np.ndarray): Objective vector of candidate 1 (minimization).
        cv1 (float): Constraint violation of candidate 1 (must be >= 0.0).
        f2 (np.ndarray): Objective vector of candidate 2 (minimization).
        cv2 (float): Constraint violation of candidate 2 (must be >= 0.0).
        feas_tol (float): Numerical tolerance for feasibility (default: 1e-4).

    Returns:
        bool: True if candidate 1 constrained-dominates candidate 2, False otherwise.
    """
    cv1 = max(0.0, float(cv1))
    cv2 = max(0.0, float(cv2))
    
    feas1 = (cv1 <= feas_tol)
    feas2 = (cv2 <= feas_tol)

    if feas1 and not feas2:
        return True
    if not feas1 and feas2:
        return False
    if not feas1 and not feas2:
        # Both infeasible: strictly smaller CV wins (equal CV neither dominates)
        return bool(cv1 < cv2)
    # Both feasible: Pareto dominance on objectives
    return dominates(f1, f2)


# ---------------------------------------------------------------------------
# Fast Non-Dominated Sorting
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(
    F: np.ndarray,
    CV: np.ndarray | None = None,
    use_deb: bool = True,
    feas_tol: float = 1e-4,
) -> list[list[int]]:
    """
    Deb's O(M N^2) Fast Non-Dominated Sorting algorithm.
    
    Partitions population into successive fronts [Front_0, Front_1, ...],
    where Front_0 is the non-dominated set.

    Parameters:
        F (np.ndarray): Matrix of objective values of shape (N, M).
        CV (np.ndarray | None): Array of constraint violations of shape (N,).
                                If provided and use_deb=True, constrained dominance is applied.
        use_deb (bool): Whether to enforce Deb's feasibility rules (default: True).
        feas_tol (float): Feasibility numerical tolerance (default: 1e-4).

    Returns:
        list[list[int]]: List of fronts, where each front is a list of solution indices.
    """
    F = np.asarray(F, dtype=float)
    N = F.shape[0]
    if N == 0:
        return []

    if CV is not None and use_deb:
        CV = np.maximum(0.0, np.asarray(CV, dtype=float).ravel())
        apply_deb = True
    else:
        apply_deb = False

    # S[p] = list of solutions that solution p dominates
    S: list[list[int]] = [[] for _ in range(N)]
    # n[p] = number of solutions that dominate solution p
    n = np.zeros(N, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(N):
        f_p = F[p]
        cv_p = CV[p] if apply_deb else 0.0

        for q in range(p + 1, N):
            f_q = F[q]
            cv_q = CV[q] if apply_deb else 0.0

            if apply_deb:
                p_dom_q = constrained_dominates(f_p, cv_p, f_q, cv_q, feas_tol=feas_tol)
                q_dom_p = constrained_dominates(f_q, cv_q, f_p, cv_p, feas_tol=feas_tol)
            else:
                p_dom_q = dominates(f_p, f_q)
                q_dom_p = dominates(f_q, f_p)

            if p_dom_q:
                S[p].append(q)
                n[q] += 1
            elif q_dom_p:
                S[q].append(p)
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    # Iteratively peel away subsequent fronts
    i = 0
    while len(fronts[i]) > 0:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        if len(next_front) == 0:
            break
        i += 1
        fronts.append(next_front)

    return fronts


# ---------------------------------------------------------------------------
# Crowding Distance
# ---------------------------------------------------------------------------

def crowding_distance(F: np.ndarray, front: Sequence[int]) -> np.ndarray:
    """
    Computes crowding distance for solutions in a given front.
    
    Assigns infinity (1e14) to the boundary extrema of each objective
    and normalized Manhattan span to intermediate points.

    Parameters:
        F (np.ndarray): Matrix of objective values of shape (N, M).
        front (Sequence[int]): Indices of solutions belonging to the front.

    Returns:
        np.ndarray: Array of crowding distances for solutions in the front (shape: (len(front),)).
    """
    front_indices = list(front)
    L = len(front_indices)
    if L == 0:
        return np.array([], dtype=float)
    if L <= 2:
        return np.full(L, 1e14, dtype=float)

    F_sub = F[front_indices]
    M = F_sub.shape[1]
    distances = np.zeros(L, dtype=float)

    for m in range(M):
        # Sort solutions in front by m-th objective
        sorted_order = np.argsort(F_sub[:, m])
        f_min = F_sub[sorted_order[0], m]
        f_max = F_sub[sorted_order[-1], m]

        diff = f_max - f_min
        if diff < 1e-12:
            continue

        # Extrema get infinite distance
        distances[sorted_order[0]] = 1e14
        distances[sorted_order[-1]] = 1e14

        for i in range(1, L - 1):
            if distances[sorted_order[i]] < 1e14:
                dist_contrib = (F_sub[sorted_order[i + 1], m] - F_sub[sorted_order[i - 1], m]) / diff
                distances[sorted_order[i]] += dist_contrib

    return distances


# ---------------------------------------------------------------------------
# Hypervolume Computation (Exact 2D & Monte Carlo M >= 3)
# ---------------------------------------------------------------------------

def hypervolume_2d(F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Computes exact 2D hypervolume dominated by point set F and bounded by ref_point.
    
    Uses coordinate sorting and rectangular decomposition:
    1. Filters out points that do not strictly dominate the reference point.
    2. Keeps only non-dominated points among filtered candidates.
    3. Sorts by f1 ascending (which orders f2 strictly descending).
    4. Computes area as the sum of non-overlapping rectangular vertical slices.

    Parameters:
        F (np.ndarray): Points in objective space (shape: (N, 2), minimization).
        ref_point (np.ndarray): Reference point coordinates [r1, r2].

    Returns:
        float: Exact dominated hypervolume (>= 0.0).
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    ref = np.asarray(ref_point, dtype=float).ravel()
    if F.size == 0 or F.shape[1] != 2 or len(ref) != 2:
        return 0.0

    # Filter points bounded by ref_point (must be strictly better than ref_point)
    valid = np.all(F < ref, axis=1)
    pts = F[valid]
    if len(pts) == 0:
        return 0.0

    # Retain only non-dominated points
    fronts = fast_non_dominated_sort(pts, use_deb=False)
    pts = pts[fronts[0]]

    # Sort primarily by f1 ascending, break ties by f2 ascending
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    # Remove duplicates on f1
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - unique_pts[-1][0]) > 1e-12:
            unique_pts.append(p)
    pts = np.array(unique_pts)

    area = 0.0
    K = len(pts)
    for i in range(K):
        x_left = pts[i, 0]
        x_right = pts[i + 1, 0] if i + 1 < K else ref[0]
        height = ref[1] - pts[i, 1]
        width = x_right - x_left
        if width > 0.0 and height > 0.0:
            area += width * height

    return float(area)


def hypervolume_monte_carlo(
    F: np.ndarray,
    ref_point: np.ndarray,
    n_samples: int = 100000,
    seed: int = 42,
) -> float:
    """
    Estimates dominated hypervolume for M >= 3 objectives using Monte Carlo sampling.

    Parameters:
        F (np.ndarray): Points in objective space (shape: (N, M), minimization).
        ref_point (np.ndarray): Reference point coordinates (shape: (M,)).
        n_samples (int): Number of uniform Monte Carlo points (default: 100,000).
        seed (int): Deterministic RNG seed for sampling.

    Returns:
        float: Estimated hypervolume (>= 0.0).
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    ref = np.asarray(ref_point, dtype=float).ravel()
    if F.size == 0 or F.shape[1] != len(ref):
        return 0.0

    M = F.shape[1]
    # Filter points bounded by ref_point
    valid = np.all(F < ref, axis=1)
    pts = F[valid]
    if len(pts) == 0:
        return 0.0

    # Bounding box [L, ref]
    L = np.min(pts, axis=0)
    box_widths = ref - L
    if np.any(box_widths <= 0.0):
        return 0.0

    total_volume = float(np.prod(box_widths))

    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=L, high=ref, size=(n_samples, M))

    # Vectorized dominance check of samples
    # sample s is dominated if exists point p in pts such that all p_m <= s_m
    # To conserve memory with large n_samples, evaluate in chunks of 10,000
    chunk_size = 10000
    dominated_count = 0

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = samples[start:end]  # (B, M)
        # Check if any point in pts dominates chunk: (pts[:, None, :] <= chunk[None, :, :]).all(axis=2)
        # Shape: (P, B)
        dom_matrix = np.all(pts[:, None, :] <= chunk[None, :, :], axis=2)
        dominated_chunk = np.any(dom_matrix, axis=0)
        dominated_count += int(np.sum(dominated_chunk))

    return float(total_volume * (dominated_count / n_samples))


def hypervolume(F: np.ndarray, ref_point: np.ndarray, n_samples: int = 100000, seed: int = 42) -> float:
    """
    Unified hypervolume dispatcher routing to 2D exact or Monte Carlo for M >= 3.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    ref = np.asarray(ref_point, dtype=float).ravel()
    if F.size == 0:
        return 0.0
    if F.shape[1] == 2 and len(ref) == 2:
        return hypervolume_2d(F, ref)
    return hypervolume_monte_carlo(F, ref, n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# Inverted Generational Distance (IGD) & Schott's Spacing (S)
# ---------------------------------------------------------------------------

def igd(obtained_F: np.ndarray, true_F: np.ndarray) -> float:
    """
    Calculates Inverted Generational Distance (IGD).
    
    IGD measures both convergence and diversity by finding the average Euclidean
    distance from each reference point in true_F to its nearest neighbor in obtained_F:
        IGD(P, P*) = (1 / |P*|) * sum_{v in P*} min_{u in P} ||u - v||_2

    Parameters:
        obtained_F (np.ndarray): Obtained non-dominated front (shape: (N, M)).
        true_F (np.ndarray): Uniformly sampled true Pareto front (shape: (K, M)).

    Returns:
        float: IGD metric value (smaller is better).
    """
    P = np.atleast_2d(np.asarray(obtained_F, dtype=float))
    P_star = np.atleast_2d(np.asarray(true_F, dtype=float))

    if P.size == 0 or P_star.size == 0 or P.shape[1] != P_star.shape[1]:
        return float("inf")

    # For each point in P_star, compute minimum distance to any point in P
    # Pairwise distances: (K, N)
    diff = P_star[:, None, :] - P[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    min_dist_per_star = np.min(dist, axis=1)

    return float(np.mean(min_dist_per_star))


def spacing(obtained_F: np.ndarray) -> float:
    """
    Calculates Schott's Spacing metric (S).
    
    Measures the uniformity of spread among solutions along the obtained front:
        d_i = min_{j != i} sum_{m=1}^M |f_m^i - f_m^j|  (Manhattan distance)
        S = sqrt( (1 / (N - 1)) * sum_{i=1}^N (d_i - d_bar)^2 )

    Parameters:
        obtained_F (np.ndarray): Obtained objective vectors (shape: (N, M)).

    Returns:
        float: Spacing value (0.0 implies perfectly equidistant spacing).
    """
    F = np.atleast_2d(np.asarray(obtained_F, dtype=float))
    if F.size == 0:
        return 0.0
    N = F.shape[0]
    if N <= 1:
        return 0.0

    # Filter to non-dominated front
    fronts = fast_non_dominated_sort(F, use_deb=False)
    F_nd = F[fronts[0]]
    K = F_nd.shape[0]
    if K <= 1:
        return 0.0

    # Pairwise Manhattan distance
    manhattan = np.sum(np.abs(F_nd[:, None, :] - F_nd[None, :, :]), axis=-1)
    # Mask diagonal
    np.fill_diagonal(manhattan, np.inf)

    d = np.min(manhattan, axis=1)
    d_bar = np.mean(d)

    return float(np.sqrt(np.sum((d - d_bar)**2) / (K - 1)))


# ---------------------------------------------------------------------------
# Objective Normalization & Reference Points
# ---------------------------------------------------------------------------

def normalize_objectives(
    F: np.ndarray,
    ideal: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Linearly normalizes objective vectors into the [0, 1] range:
        f_norm = (f - ideal) / (nadir - ideal + eps)
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if F.size == 0:
        return F

    if ideal is None:
        ideal = np.min(F, axis=0)
    else:
        ideal = np.asarray(ideal, dtype=float).ravel()

    if nadir is None:
        nadir = np.max(F, axis=0)
    else:
        nadir = np.asarray(nadir, dtype=float).ravel()

    denom = nadir - ideal
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return (F - ideal) / denom


def das_dennis(p: int, M: int) -> np.ndarray:
    """
    Generates structured uniform reference points on the unit simplex:
        sum_{m=1}^M lambda_m = 1, lambda_m >= 0
    with p divisions along each axis.
    
    Total points = binom(p + M - 1, M - 1).

    Parameters:
        p (int): Number of divisions along each objective.
        M (int): Number of objectives.

    Returns:
        np.ndarray: Matrix of reference points of shape (H, M).
    """
    ref_points: list[list[float]] = []

    def _generate(index: int, current_sum: int, current_vector: list[float]) -> None:
        if index == M - 1:
            val = (p - current_sum) / float(p)
            ref_points.append(current_vector + [val])
            return
        for i in range(p - current_sum + 1):
            val = i / float(p)
            _generate(index + 1, current_sum + i, current_vector + [val])

    _generate(0, 0, [])
    return np.asarray(ref_points, dtype=float)
