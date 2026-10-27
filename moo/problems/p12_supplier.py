"""
p12_supplier.py: P12 — Supplier Selection (MCDM Ranking)

Scenario:
A purchasing team must rank five suppliers on four criteria with weights (0.3, 0.3, 0.2, 0.2).

Data Table:
Supplier | Cost, $k (min) | Quality, /10 (max) | Delivery, days (min) | Reliability, % (max)
S1       | 120            | 8                  | 10                   | 92
S2       | 100            | 7                  | 14                   | 88
S3       | 140            | 9                  | 7                    | 95
S4       | 90             | 6                  | 18                   | 80
S5       | 130            | 8                  | 9                    | 90

Criteria Weights:
- w = [0.3, 0.3, 0.2, 0.2]

Objectives:
- Criteria 1: Cost ($k) -> Minimize
- Criteria 2: Quality (/10) -> Maximize (internal: -Quality)
- Criteria 3: Delivery (days) -> Minimize
- Criteria 4: Reliability (%) -> Maximize (internal: -Reliability)

Constraints:
- Discrete choices {0, 1, 2, 3, 4}. CV = 0.0.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


SUPPLIER_NAMES = ["S1", "S2", "S3", "S4", "S5"]
DEFAULT_WEIGHTS = np.array([0.3, 0.3, 0.2, 0.2], dtype=float)

# Raw data matrix: rows S1..S5, cols [Cost (min), Quality (max), Delivery (min), Reliability (max)]
RAW_MATRIX = np.array(
    [
        [120.0, 8.0, 10.0, 92.0],  # S1
        [100.0, 7.0, 14.0, 88.0],  # S2
        [140.0, 9.0,  7.0, 95.0],  # S3
        [90.0,  6.0, 18.0, 80.0],  # S4
        [130.0, 8.0,  9.0, 90.0],  # S5
    ],
    dtype=float,
)


class SupplierSelectionProblem(Problem):
    def __init__(self) -> None:
        self.raw_data = RAW_MATRIX
        self.weights = DEFAULT_WEIGHTS
        self.criteria_min_max = ["min", "max", "min", "max"]

        super().__init__(
            name="P12_SupplierSelection",
            n_var=1,
            n_obj=4,
            n_constr=0,
            xl=np.array([0.0]),
            xu=np.array([4.0]),
            var_type="integer",
            ideal_point=np.array([90.0, -9.0, 7.0, -95.0]),
            nadir_point=np.array([140.0, -6.0, 18.0, -80.0]),
            ref_point=np.array([150.0, -5.0, 20.0, -75.0]),
            obj_directions=["min", "max", "min", "max"],
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        idx = int(np.clip(round(float(np.asarray(x).ravel()[0])), 0, 4))
        row = self.raw_data[idx]
        # Internal minimization: negate Quality (col 1) and Reliability (col 3)
        f_int = np.array([row[0], -row[1], row[2], -row[3]], dtype=float)
        return f_int, 0.0, np.array([], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        idx = int(np.clip(round(float(np.asarray(x).ravel()[0])), 0, 4))
        f_nat = self.raw_data[idx].copy()
        return f_nat, 0.0, np.array([], dtype=float)

    def get_decision_matrix(self) -> np.ndarray:
        """Returns the 5x4 raw decision matrix."""
        return self.raw_data.copy()
