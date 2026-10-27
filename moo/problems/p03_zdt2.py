"""
p03_zdt2.py: P3 — Non-Convex Benchmark (ZDT2, n = 10, medium)

Scenario:
A synthetic test problem whose Pareto front curves inward (non-convex),
used to expose methods that cannot reach dented regions.

Variables:
- x_i in [0, 1], i = 1...10.

Objectives:
- f1 = x1
- g = 1 + sum_{i=2}^{10} x_i  (which equals 1 + 9 * sum_{i=2}^{10} x_i / (n - 1) for n = 10)
- f2 = g * (1 - (f1 / g)^2)
- Minimize both f1 and f2.

Constraints:
- Bounds only: [0, 1]^10. CV = 0.0.

Sanity Check:
- On true front: x_2 = ... = x_10 = 0, so g = 1.
- f2 = 1 - f1^2 for f1 in [0, 1].
- At x1 = 0.5: f1 = 0.5, f2 = 1 - 0.25 = 0.75.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class ZDT2Problem(Problem):
    def __init__(self, n_var: int = 10) -> None:
        super().__init__(
            name="P3_ZDT2",
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
            var_type="real",
            ideal_point=np.array([0.0, 0.0]),
            nadir_point=np.array([1.0, 1.0]),
            ref_point=np.array([1.1, 1.1]),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        f1 = float(x_vec[0])
        if self.n_var > 1:
            g = 1.0 + 9.0 * float(np.sum(x_vec[1:])) / (self.n_var - 1)
        else:
            g = 1.0
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2], dtype=float), 0.0, np.array([], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - f1 ** 2
        return np.column_stack((f1, f2))

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        return self.get_true_front(n_points)
