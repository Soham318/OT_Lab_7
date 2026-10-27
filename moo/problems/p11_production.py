"""
p11_production.py: P11 — Production Planning (Goals and priorities)

Scenario:
A workshop makes products A and B each week and has targets for profit and volume.

Variables:
- xA, xB >= 0 (units per week). Domain: [0, 500] x [0, 500].

Objectives:
- Maximize profit: f1_natural = 4 * xA + 6 * xB  ==>  f1_internal = -4 * xA - 6 * xB
- Maximize total units: f2_natural = xA + xB     ==>  f2_internal = -xA - xB

Constraints:
- Labour:   g1(x) = 2 * xA + 3 * xB - 1200 <= 0
- Material: g2(x) = 4 * xA + 2 * xB - 1600 <= 0
- Non-negativity: g3(x) = -xA <= 0, g4(x) = -xB <= 0
- Standardized violation: CV = sum_{j=1}^4 max(0.0, g_j)

Goals (Goal Programming):
- Profit >= 2400 and units >= 600.

Priorities (Lexicographic):
- Profit first, units second.

Sanity Check:
- Feasible corners: (0, 0), (400, 0), (300, 200), (0, 400).
- Maximum profit is 2400 along the segment between (300, 200) and (0, 400).
- Maximum units is 500 at (300, 200). Goal of 600 has minimum shortfall of 100.
- All corners are feasible (CV = 0.0).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


class ProductionPlanningProblem(Problem):
    def __init__(self) -> None:
        super().__init__(
            name="P11_ProductionPlanning",
            n_var=2,
            n_obj=2,
            n_constr=4,
            xl=np.array([0.0, 0.0]),
            xu=np.array([500.0, 500.0]),
            var_type="real",
            # Internal minimization: profit in [-2400, 0], units in [-500, 0]
            ideal_point=np.array([-2400.0, -500.0]),
            nadir_point=np.array([0.0, 0.0]),
            ref_point=np.array([100.0, 50.0]),
            obj_directions=["max", "max"],
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        xA, xB = float(x_vec[0]), float(x_vec[1])

        # Internal minimization
        f1_int = -(4.0 * xA + 6.0 * xB)
        f2_int = -(xA + xB)

        g1 = 2.0 * xA + 3.0 * xB - 1200.0
        g2 = 4.0 * xA + 2.0 * xB - 1600.0
        g3 = -xA
        g4 = -xB

        v1 = max(0.0, g1)
        v2 = max(0.0, g2)
        v3 = max(0.0, g3)
        v4 = max(0.0, g4)
        cv = v1 + v2 + v3 + v4
        g = np.array([v1, v2, v3, v4], dtype=float)

        return np.array([f1_int, f2_int], dtype=float), cv, g

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        xA, xB = float(x_vec[0]), float(x_vec[1])

        f1_nat = 4.0 * xA + 6.0 * xB
        f2_nat = xA + xB

        g1 = 2.0 * xA + 3.0 * xB - 1200.0
        g2 = 4.0 * xA + 2.0 * xB - 1600.0
        g3 = -xA
        g4 = -xB
        cv = max(0.0, g1) + max(0.0, g2) + max(0.0, g3) + max(0.0, g4)
        g = np.array([max(0.0, g1), max(0.0, g2), max(0.0, g3), max(0.0, g4)], dtype=float)

        return np.array([f1_nat, f2_nat], dtype=float), cv, g

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        # Pareto front in objective space is the line segment from (0, 400) to (300, 200)
        # Along this line: profit is constantly 2400, units range from 400 to 500
        # In natural: (2400, 400) to (2400, 500)
        # Note: strictly, (2400, 500) dominates (2400, 400) if both are maximized!
        # Thus (300, 200) is the unique non-dominated point on this edge!
        # The full trade-off connects (400, 0) [1600, 400] to (300, 200) [2400, 500] along 4xA + 2xB = 1600
        # Let's check: at (400, 0): profit 1600, units 400.
        # As xA decreases from 400 to 300, xB = 800 - 2xA increases from 0 to 200.
        # Profit: 4xA + 6(800 - 2xA) = 4800 - 8xA increases from 1600 to 2400.
        # Units: xA + (800 - 2xA) = 800 - xA increases from 400 to 500.
        # Both increase! So (300, 200) dominates all points on that edge too!
        # Therefore, the point (300, 200) with profit 2400 and units 500 dominates all other corners!
        return np.array([[-2400.0, -500.0]], dtype=float)

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        return np.array([[2400.0, 500.0]], dtype=float)
