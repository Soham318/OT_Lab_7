"""
p07_water.py: P7 — Water Allocation with Environmental Flow (Linear trade-off)

Scenario:
A river basin authority shares 90 million m^3 between farmers and industry
after reserving 10 million m^3 for the environment.

Variables:
- x_1 = agriculture, x_2 = industry (million m^3). Domain: x1 in [10, 70], x2 in [20, 80].

Objectives:
- Maximize agricultural yield: f1_natural = 0.8 * x1  ==>  f1_internal = -0.8 * x1
- Maximize industrial output:   f2_natural = 1.2 * x2  ==>  f2_internal = -1.2 * x2

Constraints:
- g1: x1 + x2 <= 90  ==>  g1(x) = x1 + x2 - 90 <= 0
- g2: x1 >= 10       ==>  g2(x) = 10 - x1 <= 0
- g3: x2 >= 20       ==>  g3(x) = 20 - x2 <= 0
- Standardized violation: CV = sum_{j=1}^3 max(0.0, g_j)

Sanity Check:
- Front is the line x1 + x2 = 90 from (10, 80), giving natural (8, 96) [internal (-8, -96)],
  to (70, 20), giving natural (56, 24) [internal (-56, -24)].
- Both endpoints are feasible (CV = 0.0).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class WaterAllocationProblem(Problem):
    def __init__(self) -> None:
        super().__init__(
            name="P7_WaterAllocation",
            n_var=2,
            n_obj=2,
            n_constr=3,
            xl=np.array([10.0, 20.0]),
            xu=np.array([70.0, 80.0]),
            var_type="real",
            # Internal minimization: [-56, -96] to [-8, -24]
            ideal_point=np.array([-56.0, -96.0]),
            nadir_point=np.array([-8.0, -24.0]),
            ref_point=np.array([0.0, 0.0]),
            obj_directions=["max", "max"],
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        x1, x2 = float(x_vec[0]), float(x_vec[1])

        # Internal minimization (negated natural maximization)
        f1_int = -0.8 * x1
        f2_int = -1.2 * x2

        g1 = (x1 + x2) - 90.0
        g2 = 10.0 - x1
        g3 = 20.0 - x2

        v1 = max(0.0, g1)
        v2 = max(0.0, g2)
        v3 = max(0.0, g3)
        cv = v1 + v2 + v3
        g = np.array([v1, v2, v3], dtype=float)

        return np.array([f1_int, f2_int], dtype=float), cv, g

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        x1, x2 = float(x_vec[0]), float(x_vec[1])

        f1_nat = 0.8 * x1
        f2_nat = 1.2 * x2

        g1 = (x1 + x2) - 90.0
        g2 = 10.0 - x1
        g3 = 20.0 - x2
        cv = max(0.0, g1) + max(0.0, g2) + max(0.0, g3)
        g = np.array([max(0.0, g1), max(0.0, g2), max(0.0, g3)], dtype=float)

        return np.array([f1_nat, f2_nat], dtype=float), cv, g

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        # x1 from 10 to 70, x2 = 90 - x1
        x1_vals = np.linspace(10.0, 70.0, n_points)
        x2_vals = 90.0 - x1_vals
        f1_int = -0.8 * x1_vals
        f2_int = -1.2 * x2_vals
        return np.column_stack((f1_int, f2_int))

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        x1_vals = np.linspace(10.0, 70.0, n_points)
        x2_vals = 90.0 - x1_vals
        f1_nat = 0.8 * x1_vals
        f2_nat = 1.2 * x2_vals
        return np.column_stack((f1_nat, f2_nat))
