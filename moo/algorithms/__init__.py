"""
moo/algorithms: Multiobjective Optimization Algorithm implementations.

Phase 2 exposes Algorithms 1 through 13:
1. Weighted Sum (alg01_weighted_sum)
2. epsilon-Constraint (alg02_epsilon_constraint)
3. Benson's Method (alg03_benson)
4. Weight Metric (alg04_weight_metric)
5. Value Function (alg05_value_function)
6. Goal Programming (alg06_goal_programming)
7. Lexicographic (alg07_lexicographic)
8. TOPSIS (alg08_topsis)
9. VIKOR (alg09_vikor)
10. VEGA (alg10_vega)
11. MOGA (alg11_moga)
12. NSGA (alg12_nsga)
13. NPGA (alg13_npga) — Task C Deep Dive #1
"""

from __future__ import annotations
from typing import Any

from moo.algorithms import (
    alg01_weighted_sum,
    alg02_epsilon_constraint,
    alg03_benson,
    alg04_weight_metric,
    alg05_value_function,
    alg06_goal_programming,
    alg07_lexicographic,
    alg08_topsis,
    alg09_vikor,
    alg10_vega,
    alg11_moga,
    alg12_nsga,
    alg13_npga,
)

ALGORITHMS = {
    1: alg01_weighted_sum,
    2: alg02_epsilon_constraint,
    3: alg03_benson,
    4: alg04_weight_metric,
    5: alg05_value_function,
    6: alg06_goal_programming,
    7: alg07_lexicographic,
    8: alg08_topsis,
    9: alg09_vikor,
    10: alg10_vega,
    11: alg11_moga,
    12: alg12_nsga,
    13: alg13_npga,
}

ALGORITHM_NAMES = {
    "weighted_sum": 1,
    "weightedsum": 1,
    "alg01": 1,
    "epsilon_constraint": 2,
    "eps_constraint": 2,
    "alg02": 2,
    "benson": 3,
    "alg03": 3,
    "weight_metric": 4,
    "tchebycheff": 4,
    "alg04": 4,
    "value_function": 5,
    "utility": 5,
    "alg05": 5,
    "goal_programming": 6,
    "alg06": 6,
    "lexicographic": 7,
    "alg07": 7,
    "topsis": 8,
    "alg08": 8,
    "vikor": 9,
    "alg09": 9,
    "vega": 10,
    "alg10": 10,
    "moga": 11,
    "alg11": 11,
    "nsga": 12,
    "alg12": 12,
    "npga": 13,
    "alg13": 13,
}


def get_algorithm(id_or_name: int | str) -> Any:
    """Retrieves an algorithm module by number (1-13) or string key."""
    if isinstance(id_or_name, int):
        if id_or_name in ALGORITHMS:
            return ALGORITHMS[id_or_name]
        raise ValueError(f"Algorithm #{id_or_name} not available in Phase 2 (available: 1-13)")

    key = str(id_or_name).lower().strip().replace("-", "_")
    if key.isdigit():
        return get_algorithm(int(key))

    if key in ALGORITHM_NAMES:
        return ALGORITHMS[ALGORITHM_NAMES[key]]

    raise ValueError(f"Unknown algorithm '{id_or_name}'. Available in Phase 2: {list(ALGORITHMS.keys())}")


__all__ = [
    "get_algorithm",
    "ALGORITHMS",
    "ALGORITHM_NAMES",
    "alg01_weighted_sum",
    "alg02_epsilon_constraint",
    "alg03_benson",
    "alg04_weight_metric",
    "alg05_value_function",
    "alg06_goal_programming",
    "alg07_lexicographic",
    "alg08_topsis",
    "alg09_vikor",
    "alg10_vega",
    "alg11_moga",
    "alg12_nsga",
    "alg13_npga",
]
