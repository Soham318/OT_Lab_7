"""
p14_laptop.py: P14 — Laptop Design with Decision-Maker Preference

Scenario:
A company designs a laptop. The product manager wants a machine near 1.8 kg
that lasts about 12 hours.

Variables:
- Screen size x1 in [13, 17] inches
- Battery capacity x2 in [40, 90] Wh

Objectives:
- Minimize weight: f1_natural = 0.4 + 0.05 * x1 + 0.012 * x2 kg
- Maximize runtime: f2_natural = 0.2 * x2 / (0.5 + 0.03 * x1) hours
  ==> Internal minimization: f2_internal = -0.2 * x2 / (0.5 + 0.03 * x1)

Constraints:
- Bounds only: [13, 17] x [40, 90]. CV = 0.0.

Decision-Maker Aspirational Reference Point:
- (1.8 kg, 12.0 h) in natural coordinates  [internal: (1.8, -12.0)].

Sanity Check:
- (13, 40) gives 1.53 kg and ~9.0 h (8.9888 h).
- (13, 90) gives 2.13 kg and ~20.2 h (20.2247 h).
- Increasing x1 worsens both objectives (weight increases, runtime decreases).
  Therefore, every Pareto-optimal point has x1 = 13.0 inches!
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class LaptopDesignProblem(Problem):
    def __init__(self) -> None:
        super().__init__(
            name="P14_LaptopDesign",
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=np.array([13.0, 40.0]),
            xu=np.array([17.0, 90.0]),
            var_type="real",
            # Internal minimization: weight in [1.53, 2.33], runtime in [-20.22, -7.92]
            ideal_point=np.array([1.53, -20.2247]),
            nadir_point=np.array([2.33, -7.9208]),
            ref_point=np.array([2.50, -7.0]),
            obj_directions=["min", "max"],
        )
        # Decision-maker's preference target in natural and internal space
        self.dm_preference_natural = np.array([1.8, 12.0], dtype=float)
        self.dm_preference_internal = np.array([1.8, -12.0], dtype=float)

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        x1, x2 = float(x_vec[0]), float(x_vec[1])

        weight = 0.4 + 0.05 * x1 + 0.012 * x2
        runtime = (0.2 * x2) / (0.5 + 0.03 * x1)

        f_internal = np.array([weight, -runtime], dtype=float)
        return f_internal, 0.0, np.array([], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        x1, x2 = float(x_vec[0]), float(x_vec[1])

        weight = 0.4 + 0.05 * x1 + 0.012 * x2
        runtime = (0.2 * x2) / (0.5 + 0.03 * x1)

        f_natural = np.array([weight, runtime], dtype=float)
        return f_natural, 0.0, np.array([], dtype=float)

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        # All Pareto points have x1 = 13.0, x2 in [40, 90]
        x1_fixed = 13.0
        x2_vals = np.linspace(40.0, 90.0, n_points)
        weights = 0.4 + 0.05 * x1_fixed + 0.012 * x2_vals
        runtimes = (0.2 * x2_vals) / (0.5 + 0.03 * x1_fixed)
        return np.column_stack((weights, -runtimes))

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        x1_fixed = 13.0
        x2_vals = np.linspace(40.0, 90.0, n_points)
        weights = 0.4 + 0.05 * x1_fixed + 0.012 * x2_vals
        runtimes = (0.2 * x2_vals) / (0.5 + 0.03 * x1_fixed)
        return np.column_stack((weights, runtimes))
