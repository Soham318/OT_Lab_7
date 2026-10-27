"""
p06_binh_korn.py: P6 — Binh-Korn Constrained Benchmark (Hard constraints)

Scenario:
A standard test problem with two nonlinear constraints that cut the feasible
region into an awkward shape.

Variables:
- x_1 in [0, 5], x_2 in [0, 3].

Objectives:
- Minimize f1 = 4 * x1^2 + 4 * x2^2
- Minimize f2 = (x1 - 5)^2 + (x2 - 5)^2

Constraints:
- g1: (x1 - 5)^2 + x2^2 <= 25  ==>  g1(x) = (x1 - 5)^2 + x2^2 - 25 <= 0
- g2: (x1 - 8)^2 + (x2 + 3)^2 >= 7.7  ==>  g2(x) = 7.7 - ((x1 - 8)^2 + (x2 + 3)^2) <= 0
- Standardized violation: CV = max(0.0, g1) + max(0.0, g2)

Sanity Check:
- (0, 0) gives (0, 50) and is feasible (CV = 0.0).
- (5, 3) gives (136, 4) and is feasible (CV = 0.0).
- These are the two ends of the Pareto front.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class BinhKornProblem(Problem):
    def __init__(self) -> None:
        super().__init__(
            name="P6_BinhKorn",
            n_var=2,
            n_obj=2,
            n_constr=2,
            xl=np.array([0.0, 0.0]),
            xu=np.array([5.0, 3.0]),
            var_type="real",
            ideal_point=np.array([0.0, 4.0]),
            nadir_point=np.array([136.0, 50.0]),
            ref_point=np.array([150.0, 55.0]),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        x1, x2 = float(x_vec[0]), float(x_vec[1])

        f1 = 4.0 * (x1 ** 2) + 4.0 * (x2 ** 2)
        f2 = ((x1 - 5.0) ** 2) + ((x2 - 5.0) ** 2)

        g1 = ((x1 - 5.0) ** 2) + (x2 ** 2) - 25.0
        g2 = 7.7 - (((x1 - 8.0) ** 2) + ((x2 + 3.0) ** 2))

        v1 = max(0.0, g1)
        v2 = max(0.0, g2)
        cv = v1 + v2
        g = np.array([v1, v2], dtype=float)

        return np.array([f1, f2], dtype=float), cv, g

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)
