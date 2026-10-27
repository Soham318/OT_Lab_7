"""
p10_dtlz2.py: P10 — Many-Objective Benchmark (DTLZ2, M = 3 and M = 5, hard)

Scenario:
A test benchmark where the number of objectives grows (many-objective optimization),
demonstrating why Pareto dominance and crowding distance lose selection pressure.

Variables:
- n = M + 9 variables in [0, 1].

Objectives:
- g = sum_{i=M}^{n} (x_i - 0.5)^2
- f_1 = (1 + g) * prod_{i=1}^{M-1} cos(x_i * pi / 2)
- f_m = (1 + g) * (prod_{i=1}^{M-m} cos(x_i * pi / 2)) * sin(x_{M-m+1} * pi / 2), for m = 2...M
- Minimize all M objectives.

Constraints:
- Bounds only: [0, 1]^n. CV = 0.0.

Sanity Check:
- When g = 0 (all x_i = 0.5 for i >= M), sum_{m=1}^M (f_m)^2 = 1.0 exactly.
- Used to construct the true Pareto front for IGD calculations.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import das_dennis


class DTLZ2Problem(Problem):
    def __init__(self, M: int = 3) -> None:
        self.M = int(M)
        n_var = self.M + 9
        super().__init__(
            name=f"P10_DTLZ2_M{self.M}",
            n_var=n_var,
            n_obj=self.M,
            n_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
            var_type="real",
            ideal_point=np.zeros(self.M),
            nadir_point=np.ones(self.M),
            ref_point=np.full(self.M, 1.1),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        M = self.M
        # 0-indexed: variables for g are indices M-1 to n-1
        g = float(np.sum((x_vec[M - 1 :] - 0.5) ** 2))

        f = np.zeros(M, dtype=float)
        # f_1 (0-indexed: f[0])
        f[0] = (1.0 + g) * float(np.prod(np.cos(x_vec[: M - 1] * np.pi / 2.0)))

        # f_m for m = 2...M (0-indexed: m_idx = 1...M-1)
        for m_idx in range(1, M):
            k = M - (m_idx + 1)
            prod_part = float(np.prod(np.cos(x_vec[:k] * np.pi / 2.0))) if k > 0 else 1.0
            sin_part = float(np.sin(x_vec[k] * np.pi / 2.0))
            f[m_idx] = (1.0 + g) * prod_part * sin_part

        return f, 0.0, np.array([], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        # Sample points on the unit hypersphere octant (sum f_m^2 = 1, f_m >= 0)
        # Using normalized Das-Dennis reference directions
        p_divs = 12 if self.M == 3 else 6
        ref_dirs = das_dennis(p_divs, self.M)
        norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        sphere_pts = ref_dirs / norms
        return sphere_pts

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        return self.get_true_front(n_points)
