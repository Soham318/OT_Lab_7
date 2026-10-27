"""
moo/algorithms: Multiobjective Optimization Algorithm implementations.

Comprehensive Algorithm Catalogue (Algorithms 1 through 37):
1.  Weighted Sum (alg01_weighted_sum)
2.  epsilon-Constraint (alg02_epsilon_constraint)
3.  Benson's Method (alg03_benson)
4.  Weight Metric (alg04_weight_metric)
5.  Value Function (alg05_value_function)
6.  Goal Programming (alg06_goal_programming)
7.  Lexicographic (alg07_lexicographic)
8.  TOPSIS (alg08_topsis)
9.  VIKOR (alg09_vikor)
10. VEGA (alg10_vega)
11. MOGA (alg11_moga)
12. NSGA (alg12_nsga)
13. NPGA (alg13_npga) — Task C Deep Dive #1
14. NSGA-II (alg14_nsga2)
15. NSGA-III (alg15_nsga3)
16. SPEA2 (alg16_spea2)
17. PESA-II (alg17_pesa2)
18. MOEA/D (alg18_moead)
19. MOEA/D-DE (alg19_moead_de)
20. MOEA/D-ACO (alg20_moead_aco)
21. Constrained NSGA-II (alg21_cnsga2)
22. MOPSO (alg22_mopso)
23. MOGWO (alg23_mogwo)
24. MOABC (alg24_moabc)
25. MOACO (alg25_moaco) — Task C Deep Dive #2
26. MOSA (alg26_mosa)
27. MODE (alg27_mode)
28. MO-TLBO (alg28_motlbo)
29. MOBA (alg29_moba)
30. MOMA (alg30_moma)
31. RVEA (alg31_rvea)
32. IBEA (alg32_ibea)
33. SMS-EMOA (alg33_sms_emoa)
34. HypE (alg34_hype)
35. R-NSGA-II (alg35_r_nsga2)
36. MGDA (alg36_mgda)
37. GradNorm (alg37_gradnorm) — Task C Deep Dive #3
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
    alg14_nsga2,
    alg15_nsga3,
    alg16_spea2,
    alg17_pesa2,
    alg18_moead,
    alg19_moead_de,
    alg20_moead_aco,
    alg21_cnsga2,
    alg22_mopso,
    alg23_mogwo,
    alg24_moabc,
    alg25_moaco,
    alg26_mosa,
    alg27_mode,
    alg28_motlbo,
    alg29_moba,
    alg30_moma,
    alg31_rvea,
    alg32_ibea,
    alg33_sms_emoa,
    alg34_hype,
    alg35_r_nsga2,
    alg36_mgda,
    alg37_gradnorm,
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
    14: alg14_nsga2,
    15: alg15_nsga3,
    16: alg16_spea2,
    17: alg17_pesa2,
    18: alg18_moead,
    19: alg19_moead_de,
    20: alg20_moead_aco,
    21: alg21_cnsga2,
    22: alg22_mopso,
    23: alg23_mogwo,
    24: alg24_moabc,
    25: alg25_moaco,
    26: alg26_mosa,
    27: alg27_mode,
    28: alg28_motlbo,
    29: alg29_moba,
    30: alg30_moma,
    31: alg31_rvea,
    32: alg32_ibea,
    33: alg33_sms_emoa,
    34: alg34_hype,
    35: alg35_r_nsga2,
    36: alg36_mgda,
    37: alg37_gradnorm,
}

ALGORITHM_NAMES = {
    # 1. Weighted Sum
    "weighted_sum": 1,
    "weightedsum": 1,
    "alg01": 1,
    "alg1": 1,
    # 2. epsilon-Constraint
    "epsilon_constraint": 2,
    "eps_constraint": 2,
    "alg02": 2,
    "alg2": 2,
    # 3. Benson's
    "benson": 3,
    "alg03": 3,
    "alg3": 3,
    # 4. Weight Metric
    "weight_metric": 4,
    "tchebycheff": 4,
    "alg04": 4,
    "alg4": 4,
    # 5. Value Function
    "value_function": 5,
    "utility": 5,
    "alg05": 5,
    "alg5": 5,
    # 6. Goal Programming
    "goal_programming": 6,
    "alg06": 6,
    "alg6": 6,
    # 7. Lexicographic
    "lexicographic": 7,
    "alg07": 7,
    "alg7": 7,
    # 8. TOPSIS
    "topsis": 8,
    "alg08": 8,
    "alg8": 8,
    # 9. VIKOR
    "vikor": 9,
    "alg09": 9,
    "alg9": 9,
    # 10. VEGA
    "vega": 10,
    "alg10": 10,
    # 11. MOGA
    "moga": 11,
    "alg11": 11,
    # 12. NSGA
    "nsga": 12,
    "alg12": 12,
    # 13. NPGA
    "npga": 13,
    "alg13": 13,
    # 14. NSGA-II
    "nsga2": 14,
    "nsga_2": 14,
    "nsga_ii": 14,
    "nsgaii": 14,
    "alg14": 14,
    # 15. NSGA-III
    "nsga3": 15,
    "nsga_3": 15,
    "nsga_iii": 15,
    "nsgaiii": 15,
    "alg15": 15,
    # 16. SPEA2
    "spea2": 16,
    "spea_2": 16,
    "alg16": 16,
    # 17. PESA-II
    "pesa2": 17,
    "pesa_2": 17,
    "pesa_ii": 17,
    "pesaii": 17,
    "alg17": 17,
    # 18. MOEA/D
    "moead": 18,
    "moea_d": 18,
    "alg18": 18,
    # 19. MOEA/D-DE
    "moead_de": 19,
    "moea_d_de": 19,
    "alg19": 19,
    # 20. MOEA/D-ACO
    "moead_aco": 20,
    "moea_d_aco": 20,
    "alg20": 20,
    # 21. Constrained NSGA-II
    "cnsga2": 21,
    "c_nsga2": 21,
    "constrained_nsga2": 21,
    "constrained_nsga_ii": 21,
    "alg21": 21,
    # 22. MOPSO
    "mopso": 22,
    "alg22": 22,
    # 23. MOGWO
    "mogwo": 23,
    "alg23": 23,
    # 24. MOABC
    "moabc": 24,
    "alg24": 24,
    # 25. MOACO
    "moaco": 25,
    "alg25": 25,
    # 26. MOSA
    "mosa": 26,
    "alg26": 26,
    # 27. MODE
    "mode": 27,
    "alg27": 27,
    # 28. MO-TLBO
    "motlbo": 28,
    "mo_tlbo": 28,
    "tlbo": 28,
    "alg28": 28,
    # 29. MOBA
    "moba": 29,
    "alg29": 29,
    # 30. MOMA
    "moma": 30,
    "alg30": 30,
    # 31. RVEA
    "rvea": 31,
    "alg31": 31,
    # 32. IBEA
    "ibea": 32,
    "alg32": 32,
    # 33. SMS-EMOA
    "sms_emoa": 33,
    "smsemoa": 33,
    "alg33": 33,
    # 34. HypE
    "hype": 34,
    "alg34": 34,
    # 35. R-NSGA-II
    "r_nsga2": 35,
    "rnsga2": 35,
    "r_nsga_ii": 35,
    "rnsgaii": 35,
    "alg35": 35,
    # 36. MGDA
    "mgda": 36,
    "alg36": 36,
    # 37. GradNorm
    "gradnorm": 37,
    "grad_norm": 37,
    "alg37": 37,
}


def get_algorithm(id_or_name: int | str) -> Any:
    """Retrieves an algorithm module by number (1-37) or string key."""
    if isinstance(id_or_name, int):
        if id_or_name in ALGORITHMS:
            return ALGORITHMS[id_or_name]
        raise ValueError(f"Algorithm #{id_or_name} not available (available: 1-37)")

    key = str(id_or_name).lower().strip().replace("-", "_").replace("/", "_")
    if key.isdigit():
        return get_algorithm(int(key))

    if key in ALGORITHM_NAMES:
        return ALGORITHMS[ALGORITHM_NAMES[key]]

    raise ValueError(f"Unknown algorithm '{id_or_name}'. Available: {list(ALGORITHMS.keys())}")


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
    "alg14_nsga2",
    "alg15_nsga3",
    "alg16_spea2",
    "alg17_pesa2",
    "alg18_moead",
    "alg19_moead_de",
    "alg20_moead_aco",
    "alg21_cnsga2",
    "alg22_mopso",
    "alg23_mogwo",
    "alg24_moabc",
    "alg25_moaco",
    "alg26_mosa",
    "alg27_mode",
    "alg28_motlbo",
    "alg29_moba",
    "alg30_moma",
    "alg31_rvea",
    "alg32_ibea",
    "alg33_sms_emoa",
    "alg34_hype",
    "alg35_r_nsga2",
    "alg36_mgda",
    "alg37_gradnorm",
]
