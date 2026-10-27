"""
common.py: Shared Genetic Operators and Utilities for Evolutionary & MCDM Algorithms.

This module provides reusable, pure-NumPy genetic operators:
- Random population generation respecting variable types ('real', 'integer', 'binary', 'permutation')
- Simulated Binary Crossover (SBX)
- Polynomial Mutation (PM)
- Discrete permutation crossover (OX) and mutation (swap/inversion)
- Binary uniform crossover and bit-flip mutation
- Fitness sharing in objective space (Goldberg & Richardson 1987)
- Roulette wheel selection
- Non-dominated front filtering and result packaging
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, dominates


def random_population(problem: Problem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generates an initial random population respecting variable types and bounds."""
    n_var = problem.n_var
    vtype = problem.var_type

    if vtype == "permutation":
        # Each row is a random permutation of integers 0 to n_var - 1
        pop = np.zeros((pop_size, n_var), dtype=int)
        for i in range(pop_size):
            pop[i] = rng.permutation(n_var)
        return pop

    if vtype == "binary":
        return rng.integers(0, 2, size=(pop_size, n_var), dtype=int)

    if vtype == "integer":
        low = np.floor(problem.xl).astype(int)
        high = np.ceil(problem.xu).astype(int) + 1
        pop = np.zeros((pop_size, n_var), dtype=float)
        for j in range(n_var):
            pop[:, j] = rng.integers(low[j], high[j], size=pop_size).astype(float)
        return pop

    # Default: continuous real-valued variables in [xl, xu]
    return rng.uniform(low=problem.xl, high=problem.xu, size=(pop_size, n_var))


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta_c: float = 15.0,
    prob_c: float = 0.9,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) by Deb & Agrawal (1995).
    Generates two offspring c1, c2 bounded within [xl, xu].
    """
    if rng is None:
        rng = np.random.default_rng()

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    c1 = p1.copy()
    c2 = p2.copy()

    if rng.random() > prob_c:
        return c1, c2

    n_var = len(p1)
    for i in range(n_var):
        if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
            y1 = min(p1[i], p2[i])
            y2 = max(p1[i], p2[i])
            yl = xl[i]
            yu = xu[i]

            u = rng.random()
            # Beta calculation with boundary adjustment
            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - (beta ** -(eta_c + 1.0))
            if u <= (1.0 / alpha):
                beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

            child1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
            alpha = 2.0 - (beta ** -(eta_c + 1.0))
            if u <= (1.0 / alpha):
                beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

            child2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

            c1[i] = np.clip(child1, yl, yu)
            c2[i] = np.clip(child2, yl, yu)
            if rng.random() <= 0.5:
                c1[i], c2[i] = c2[i], c1[i]

    return c1, c2


def polynomial_mutation(
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta_m: float = 20.0,
    prob_m: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Polynomial Mutation (Deb & Goyal 1996).
    Mutates individual x bounded within [xl, xu].
    """
    if rng is None:
        rng = np.random.default_rng()

    x_mut = np.asarray(x, dtype=float).copy()
    n_var = len(x)
    if prob_m is None:
        prob_m = 1.0 / max(1, n_var)

    for i in range(n_var):
        if rng.random() <= prob_m:
            y = x_mut[i]
            yl = xl[i]
            yu = xu[i]
            if yl == yu:
                continue

            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            u = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)

            if u <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
                delta_q = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
                delta_q = 1.0 - (val ** mut_pow)

            y = y + delta_q * (yu - yl)
            x_mut[i] = np.clip(y, yl, yu)

    return x_mut


def order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Order Crossover (OX) for permutation chromosomes (e.g. TSP tours)."""
    n = len(p1)
    pt1, pt2 = sorted(rng.choice(n, size=2, replace=False))

    def _ox_child(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = np.full(n, -1, dtype=int)
        child[pt1 : pt2 + 1] = parent1[pt1 : pt2 + 1]
        copied_set = set(child[pt1 : pt2 + 1])

        # Fill remaining positions circularly from parent2
        p2_order = [parent2[(pt2 + 1 + k) % n] for k in range(n)]
        p2_remaining = [val for val in p2_order if val not in copied_set]

        idx = (pt2 + 1) % n
        for val in p2_remaining:
            child[idx] = val
            idx = (idx + 1) % n
        return child

    return _ox_child(p1, p2), _ox_child(p2, p1)


def swap_mutation(p: np.ndarray, rng: np.random.Generator, prob_m: float = 0.2) -> np.ndarray:
    """Swap mutation for permutation chromosomes."""
    child = np.asarray(p, dtype=int).copy()
    if rng.random() <= prob_m:
        i, j = rng.choice(len(p), size=2, replace=False)
        child[i], child[j] = child[j], child[i]
    return child


def sharing_function(dist: float, sigma_share: float, alpha: float = 1.0) -> float:
    """Standard sharing function Sh(d) = 1 - (d / sigma_share)^alpha for d < sigma_share, else 0."""
    if dist < sigma_share:
        return float(1.0 - (dist / sigma_share) ** alpha)
    return 0.0


def compute_niche_counts(
    F_norm: np.ndarray,
    sigma_share: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Computes niche count for every individual in normalized objective matrix F_norm.
    nc_i = sum_j Sh(||f_i - f_j||_2).
    """
    N = F_norm.shape[0]
    # Pairwise Euclidean distances
    diff = F_norm[:, None, :] - F_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Vectorized sharing calculation
    sh_matrix = np.where(dist < sigma_share, 1.0 - (dist / sigma_share) ** alpha, 0.0)
    return np.sum(sh_matrix, axis=1)


def compute_single_niche_count(
    f_norm: np.ndarray,
    F_norm: np.ndarray,
    sigma_share: float,
    alpha: float = 1.0,
) -> float:
    """Computes niche count for a single normalized point against a population."""
    diff = F_norm - f_norm.reshape(1, -1)
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    sh_vals = np.where(dist < sigma_share, 1.0 - (dist / sigma_share) ** alpha, 0.0)
    return float(np.sum(sh_vals))


def roulette_wheel_selection(
    fitness: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Roulette wheel selection (fitness proportional selection). Returns selected indices."""
    fit = np.maximum(0.0, np.asarray(fitness, dtype=float))
    total_fit = np.sum(fit)
    if total_fit <= 1e-12:
        return rng.choice(len(fitness), size=n_select, replace=True)

    probs = fit / total_fit
    return rng.choice(len(fitness), size=n_select, replace=True, p=probs)


def extract_non_dominated(
    X: np.ndarray,
    F: np.ndarray,
    CV: np.ndarray | None = None,
    feas_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters population to the non-dominated set using Deb's constrained sorting.
    Returns (X_opt, F_opt, CV_opt).
    """
    X = np.asarray(X)
    F = np.asarray(F, dtype=float)
    if CV is not None:
        CV = np.maximum(0.0, np.asarray(CV, dtype=float).ravel())
    else:
        CV = np.zeros(len(F), dtype=float)

    if len(F) == 0:
        return np.empty((0, X.shape[1] if X.ndim > 1 else 0)), np.empty((0, F.shape[1] if F.ndim > 1 else 0)), np.empty(0)

    fronts = fast_non_dominated_sort(F, CV=CV, use_deb=True, feas_tol=feas_tol)
    if len(fronts) == 0 or len(fronts[0]) == 0:
        return X, F, CV

    front0 = fronts[0]
    return X[front0], F[front0], CV[front0]
