"""
p05_cantilever.py: P5 — Cantilever Beam Design (Constrained, 2 variables, medium)

Scenario:
A rectangular steel beam fixed at one end carries a tip load. The designer
wants it light and stiff.

Data:
- L = 1 m, P = 1000 N, E = 200 GPa (200e9 Pa), density = 7850 kg/m^3

Variables:
- Width b in [0.01, 0.10] m
- Height h in [0.02, 0.20] m

Objectives:
- Minimize mass: f1 = 7850 * L * b * h
- Minimize tip deflection: f2 = 4 * P * L^3 / (E * b * h^3) = 2e-8 / (b * h^3)

Constraint:
- Bending stress: 6 * P * L / (b * h^2) <= 250 MPa (250e6 Pa)
  Equivalent to: b * h^2 >= 2.4e-5 m^3
  In standard form: g(b, h) = 2.4e-5 - b * h^2 <= 0
- Standardized violation: CV = max(0.0, 2.4e-5 - b * h^2)

Sanity Check:
- (b, h) = (0.01, 0.05) gives mass = 3.925 kg, deflection = 0.016 m,
  and bending stress = 240 MPa (feasible, CV = 0.0).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class CantileverBeamProblem(Problem):
    def __init__(self) -> None:
        self.L = 1.0          # m
        self.P = 1000.0       # N
        self.E = 200e9        # Pa
        self.density = 7850.0 # kg/m^3
        self.sigma_max = 250e6 # Pa

        super().__init__(
            name="P5_CantileverBeam",
            n_var=2,
            n_obj=2,
            n_constr=1,
            xl=np.array([0.01, 0.02]),
            xu=np.array([0.10, 0.20]),
            var_type="real",
            ideal_point=np.array([1.57, 1.25e-5]),
            nadir_point=np.array([157.0, 0.025]),
            ref_point=np.array([180.0, 0.030]),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        b, h = float(x_vec[0]), float(x_vec[1])

        # Mass (kg)
        mass = self.density * self.L * b * h
        # Tip deflection (m): 4 * P * L^3 / (E * b * h^3)
        deflection = (4.0 * self.P * (self.L ** 3)) / (self.E * b * (h ** 3))

        # Constraint: b * h^2 >= 2.4e-5  ==>  g = 2.4e-5 - b * h^2 <= 0
        g_val = 2.4e-5 - b * (h ** 2)
        cv = max(0.0, g_val)
        g = np.array([cv], dtype=float)

        return np.array([mass, deflection], dtype=float), cv, g

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)
