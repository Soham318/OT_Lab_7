"""
p02_fleet.py: P2 — Delivery Fleet Sizing (1 variable, real vs integer)

Scenario:
A courier company decides how many vehicles to deploy. More vehicles cut
cost per delivery but add coordination delay.

Variable:
- Number of vehicles x, 1 <= x <= 10 (evaluated with real x or integer x).

Objectives:
- Minimize delivery cost: f1 = 100 / x
- Minimize delay penalty: f2 = x^2

Constraints:
- Bounds only: [1, 10]. Every x in [1, 10] is Pareto-optimal. CV = 0.0.

Sanity Check:
- x = 2 gives (50, 4).
- x = 10 gives (10, 100).
- Front is f2 = 10000 / (f1^2) for f1 in [10, 100].
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class FleetSizingProblem(Problem):
    def __init__(self, is_integer: bool = False) -> None:
        self.is_integer = is_integer
        super().__init__(
            name="P2_FleetSizing_Integer" if is_integer else "P2_FleetSizing",
            n_var=1,
            n_obj=2,
            n_constr=0,
            xl=np.array([1.0]),
            xu=np.array([10.0]),
            var_type="integer" if is_integer else "real",
            ideal_point=np.array([10.0, 1.0]),
            nadir_point=np.array([100.0, 100.0]),
            ref_point=np.array([110.0, 110.0]),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_val = float(np.asarray(x).ravel()[0])
        if self.is_integer:
            x_val = float(round(x_val))
            x_val = min(max(x_val, 1.0), 10.0)

        f1 = 100.0 / x_val
        f2 = x_val ** 2
        return np.array([f1, f2], dtype=float), 0.0, np.array([], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        if self.is_integer:
            x_vals = np.arange(1.0, 11.0)
        else:
            x_vals = np.linspace(1.0, 10.0, n_points)
        f1 = 100.0 / x_vals
        f2 = x_vals ** 2
        return np.column_stack((f1, f2))

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        return self.get_true_front(n_points)
