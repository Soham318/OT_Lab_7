"""
base.py: Abstract Base Class for Multi-Objective Optimization Problems

Standard Conventions:
- Variable types: 'real', 'integer', 'binary', 'permutation'.
- Internal objective vectors F_internal are strictly MINIMIZED (natural maximization is negated).
- Natural objective vectors F_natural preserve real-world directions for reporting and plotting.
- Inequality constraints are formulated as g_j(x) <= 0.
- Equality constraints are formulated as h_k(x) == 0.
- Total constraint violation:
      CV(x) = sum_j max(0, g_j(x)) + sum_k max(0, |h_k(x)| - tolerance)
  with tolerance = 1e-4. CV(x) is guaranteed non-negative (CV >= 0.0).
- Feasibility is determined by: CV(x) <= feas_tol (feas_tol = 1e-4).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Problem(ABC):
    """Abstract Base Class for all multi-objective test problems."""

    def __init__(
        self,
        name: str,
        n_var: int,
        n_obj: int,
        n_constr: int,
        xl: np.ndarray,
        xu: np.ndarray,
        var_type: str = "real",
        ref_point: np.ndarray | None = None,
        ideal_point: np.ndarray | None = None,
        nadir_point: np.ndarray | None = None,
        tolerance: float = 1e-4,
        feas_tol: float = 1e-4,
        obj_directions: list[str] | None = None,
    ) -> None:
        self.name = name
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.n_constr = int(n_constr)
        self.xl = np.asarray(xl, dtype=float)
        self.xu = np.asarray(xu, dtype=float)
        self.var_type = var_type  # 'real', 'integer', 'binary', 'permutation'
        self.tolerance = float(tolerance)
        self.feas_tol = float(feas_tol)
        self.obj_directions = (
            list(obj_directions)
            if obj_directions is not None
            else ["min"] * self.n_obj
        )

        # Default reference, ideal, and nadir points if not provided
        self.ideal_point = (
            np.asarray(ideal_point, dtype=float)
            if ideal_point is not None
            else np.zeros(self.n_obj, dtype=float)
        )
        self.nadir_point = (
            np.asarray(nadir_point, dtype=float)
            if nadir_point is not None
            else np.ones(self.n_obj, dtype=float)
        )
        self.ref_point = (
            np.asarray(ref_point, dtype=float)
            if ref_point is not None
            else self.nadir_point * 1.1
        )

    def to_natural(self, F: np.ndarray) -> np.ndarray:
        """Converts internal minimization objectives to natural coordinates without extra evaluations."""
        F_arr = np.asarray(F, dtype=float)
        if F_arr.size == 0:
            return F_arr.copy()
        mult = np.array(
            [-1.0 if str(d).lower() == "max" else 1.0 for d in self.obj_directions],
            dtype=float,
        )
        if F_arr.ndim == 1:
            return F_arr * mult
        return F_arr * mult.reshape(1, -1)

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Evaluates a single decision vector x in INTERNAL minimization format.

        Parameters:
            x (np.ndarray): Decision vector (shape: (n_var,)).

        Returns:
            tuple[np.ndarray, float, np.ndarray]:
                - F_internal (shape: (n_obj,)): All objectives converted to minimization.
                - CV (float): Total constraint violation (must be >= 0.0).
                - G (shape: (n_constr,)): Raw constraint violations.
        """
        pass

    @abstractmethod
    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Evaluates a single decision vector x in NATURAL real-world directions.
        (e.g., maximizing return, profit, accuracy, etc.)

        Parameters:
            x (np.ndarray): Decision vector (shape: (n_var,)).

        Returns:
            tuple[np.ndarray, float, np.ndarray]:
                - F_natural (shape: (n_obj,)): Natural objective values.
                - CV (float): Total constraint violation (>= 0.0).
                - G (shape: (n_constr,)): Raw constraint violations.
        """
        pass

    def evaluate_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch evaluation for population matrix X of shape (N, n_var).

        Returns:
            F_batch (shape: (N, n_obj)): Internal minimization objectives.
            CV_batch (shape: (N,)): Total constraint violation for each individual.
            G_batch (shape: (N, n_constr)): Raw constraint violation matrix.
        """
        X = np.asarray(X)
        N = X.shape[0]
        F_list: list[np.ndarray] = []
        CV_list: list[float] = []
        G_list: list[np.ndarray] = []

        for i in range(N):
            f, cv, g = self.evaluate(X[i])
            F_list.append(f)
            CV_list.append(cv)
            G_list.append(g)

        return (
            np.asarray(F_list, dtype=float),
            np.asarray(CV_list, dtype=float),
            np.asarray(G_list, dtype=float) if self.n_constr > 0 else np.zeros((N, 0), dtype=float),
        )

    def is_feasible(self, cv: float) -> bool:
        """Returns True if constraint violation does not exceed feasibility tolerance."""
        return bool(max(0.0, float(cv)) <= self.feas_tol)

    def get_true_front(self, n_points: int = 100) -> np.ndarray | None:
        """
        Returns sampled true Pareto front points in internal (minimization) coordinates.
        Returns None if no analytical front is available.
        """
        return None

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray | None:
        """
        Returns sampled true Pareto front points in natural coordinates.
        Returns None if no analytical front is available.
        """
        return None
