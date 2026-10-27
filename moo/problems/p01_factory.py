"""
p01_factory.py: P1 — Factory Production Speed (Convex, 1 variable, easy)

Scenario:
A factory runs one line. Faster speed lowers unit cost but hurts quality.
The plant manager wants the whole cost-quality trade-off curve, not one speed.

Variable:
- Production speed x, 1 <= x <= 10.

Objectives:
- Minimize cost: f1 = (x - 3)^2
- Minimize quality loss: f2 = (x - 7)^2

Constraints:
- Bounds only: [1, 10]. CV = 0.0.

Sanity Check:
- Pareto set: 3 <= x <= 7.
- At x = 5: point is (4, 4).
- At x = 3: point is (0, 16).
- At x = 7: point is (16, 0).
- Front is convex: f2 = (4 - sqrt(f1))^2 for f1 in [0, 16].
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class FactorySpeedProblem(Problem):
    def __init__(self) -> None:
        super().__init__(
            name="P1_FactorySpeed",
            n_var=1,
            n_obj=2,
            n_constr=0,
            xl=np.array([1.0]),
            xu=np.array([10.0]),
            var_type="real",
            ideal_point=np.array([0.0, 0.0]),
            nadir_point=np.array([16.0, 16.0]),
            ref_point=np.array([17.0, 17.0]),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_val = float(np.asarray(x).ravel()[0])
        f1 = (x_val - 3.0) ** 2
        f2 = (x_val - 7.0) ** 2
        return np.array([f1, f2], dtype=float), 0.0, np.array([], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        # Both natural objectives are minimization
        return self.evaluate(x)

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        # Pareto set is x in [3, 7]
        x_vals = np.linspace(3.0, 7.0, n_points)
        f1 = (x_vals - 3.0) ** 2
        f2 = (x_vals - 7.0) ** 2
        return np.column_stack((f1, f2))

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        return self.get_true_front(n_points)
