"""
p04_portfolio.py: P4 — Investment Portfolio (Continuous, equality constraint, medium)

Scenario:
An investor splits capital across four assets and wants every efficient
risk-return combination.

Variables:
- Weights w_1...w_4 >= 0 with sum(w_i) = 1.

Data:
- Expected returns mu = [0.08, 0.12, 0.15, 0.10]
- Covariance matrix Sigma:
  [[0.010, 0.002, 0.001, 0.003],
   [0.002, 0.040, 0.006, 0.004],
   [0.001, 0.006, 0.090, 0.005],
   [0.003, 0.004, 0.005, 0.020]]

Objectives:
- Minimize risk: f1 = w^T Sigma w
- Maximize return: f2_natural = mu^T w  ==>  f2_internal = -mu^T w

Constraint:
- Equality constraint: sum(w_i) - 1 = 0
- Standardized violation: CV = max(0.0, |sum(w_i) - 1| - 1e-4)

Sanity Check:
- w = (0, 0, 1, 0) gives return 0.15 and risk 0.09 (the maximum-return end).
- CV = 0.0 (feasible).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class PortfolioProblem(Problem):
    def __init__(self) -> None:
        self.mu = np.array([0.08, 0.12, 0.15, 0.10], dtype=float)
        self.Sigma = np.array(
            [
                [0.010, 0.002, 0.001, 0.003],
                [0.002, 0.040, 0.006, 0.004],
                [0.001, 0.006, 0.090, 0.005],
                [0.003, 0.004, 0.005, 0.020],
            ],
            dtype=float,
        )
        super().__init__(
            name="P4_Portfolio",
            n_var=4,
            n_obj=2,
            n_constr=1,
            xl=np.zeros(4),
            xu=np.ones(4),
            var_type="real",
            # Internal minimization: risk in [~0.007, 0.09], return in [-0.15, -0.08]
            ideal_point=np.array([0.005, -0.15]),
            nadir_point=np.array([0.090, -0.08]),
            ref_point=np.array([0.100, -0.07]),
            obj_directions=["min", "max"],
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        w = np.asarray(x, dtype=float).ravel()
        risk = float(w @ self.Sigma @ w)
        ret = float(self.mu @ w)

        # Equality constraint violation: sum(w) == 1
        eq_error = abs(float(np.sum(w)) - 1.0)
        cv = max(0.0, eq_error - self.tolerance)
        g = np.array([max(0.0, eq_error - self.tolerance)], dtype=float)

        # Internal minimization: negate return
        f_internal = np.array([risk, -ret], dtype=float)
        return f_internal, cv, g

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        w = np.asarray(x, dtype=float).ravel()
        risk = float(w @ self.Sigma @ w)
        ret = float(self.mu @ w)
        eq_error = abs(float(np.sum(w)) - 1.0)
        cv = max(0.0, eq_error - self.tolerance)
        g = np.array([max(0.0, eq_error - self.tolerance)], dtype=float)

        f_natural = np.array([risk, ret], dtype=float)
        return f_natural, cv, g
