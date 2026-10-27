"""
p08_tsp.py: P8 — Bi-Objective Delivery Routing (Discrete, TSP, hard)

Scenario:
A courier visits 15 customers once and returns to the depot, and cares about
both distance and travel time because traffic differs by road.

Data:
- 15 cities with coordinates uniform in [0, 100]^2 using random seed 7.
- d_ij = Euclidean distance between city i and city j.
- Travel time t_ij = d_ij * u_ij with symmetric u_ij ~ U(0.5, 2.0), u_ii = 0.

Solution:
- A permutation of the 15 cities (0 to 14) representing a closed tour.

Objectives:
- Minimize total distance: f1 = sum_{k=0}^{14} d_{pi(k), pi((k+1)%15)}
- Minimize total travel time: f2 = sum_{k=0}^{14} t_{pi(k), pi((k+1)%15)}

Constraint:
- Every city exactly once (valid permutation of {0, ..., 14}).
- Infeasibility CV = count of missing / duplicate cities.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


def generate_p8_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates deterministic coordinates, distance matrix, and travel time matrix for P8.
    Uses np.random.RandomState(seed) to match legacy numpy.random.seed(7) specified in PDF.
    """
    prng = np.random.RandomState(seed)
    n_cities = 15
    coords = prng.uniform(0.0, 100.0, size=(n_cities, 2))

    # Euclidean distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Symmetric traffic speed multiplier u_ij ~ U(0.5, 2.0)
    u_matrix = np.zeros((n_cities, n_cities), dtype=float)
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            u_val = float(prng.uniform(0.5, 2.0))
            u_matrix[i, j] = u_val
            u_matrix[j, i] = u_val

    time_matrix = dist_matrix * u_matrix
    return coords, dist_matrix, time_matrix


class DeliveryRoutingTSPProblem(Problem):
    def __init__(self, seed: int = 7) -> None:
        self.coords, self.dist_matrix, self.time_matrix = generate_p8_data(seed)
        n_cities = 15

        super().__init__(
            name="P8_DeliveryRoutingTSP",
            n_var=n_cities,
            n_obj=2,
            n_constr=1,
            xl=np.zeros(n_cities),
            xu=np.full(n_cities, n_cities - 1),
            var_type="permutation",
            ideal_point=np.array([200.0, 200.0]),
            nadir_point=np.array([1200.0, 1500.0]),
            ref_point=np.array([1500.0, 1800.0]),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        tour = np.asarray(x, dtype=int).ravel()
        n = self.n_var

        # Validate permutation constraint
        unique_cities = np.unique(tour)
        missing_count = n - len(unique_cities)
        out_of_bounds = int(np.sum((tour < 0) | (tour >= n)))
        total_violation = float(missing_count + out_of_bounds)

        if total_violation > 0:
            # Infeasible permutation penalty
            cv = max(0.0, total_violation)
            return np.array([1e6, 1e6], dtype=float), cv, np.array([cv], dtype=float)

        # Compute closed tour distance and travel time
        total_dist = 0.0
        total_time = 0.0
        for k in range(n):
            u = tour[k]
            v = tour[(k + 1) % n]
            total_dist += self.dist_matrix[u, v]
            total_time += self.time_matrix[u, v]

        return np.array([total_dist, total_time], dtype=float), 0.0, np.array([0.0], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)
