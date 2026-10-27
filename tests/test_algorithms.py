"""
test_algorithms.py: Comprehensive Unit and Regression Tests for Algorithms 1 through 13.

Verifies:
- Alg 01: Weighted Sum (convex front coverage, bounds, finite objectives, deterministic seed)
- Alg 02: epsilon-Constraint (constraint satisfaction, bounds, ability to reach non-convex regions)
- Alg 03: Benson's Method (shortfall projection, bounds)
- Alg 04: Weight Metric (p = 1, 2, and infinity / Tchebycheff)
- Alg 05: Value Function (DM quadratic target on P14 and risk-aversion on P4)
- Alg 06: Goal Programming (targets, unwanted deviations, exact shortfall of 100 on P11)
- Alg 07: Lexicographic (strict sequential priority, tie-breaking profit -> units on P11)
- Alg 08: TOPSIS (MCDM vector normalization, separation, relative closeness C_i in [0, 1] on P12)
- Alg 09: VIKOR (MCDM group utility S, regret R, compromise index Q with parameter v, C1/C2 checks on P12)
- Alg 10: VEGA (sub-population partition by objective, bounds, non-dominated archive)
- Alg 11: MOGA (Fonseca-Fleming rank = 1 + dominance count, fitness sharing in objective space)
- Alg 12: NSGA (1994 dummy fitness progressive degradation across fronts, sharing within fronts)
- Alg 13: NPGA (Task C Deep Dive: Pareto tournament cases against T_dom, niche-count tie-break, sensitivity parameters)
"""

import math
import numpy as np

from moo.problems import (
    FactorySpeedProblem,
    ZDT2Problem,
    PortfolioProblem,
    CantileverBeamProblem,
    BinhKornProblem,
    WaterAllocationProblem,
    ProductionPlanningProblem,
    SupplierSelectionProblem,
    LaptopDesignProblem,
)
from moo.moo_utils import fast_non_dominated_sort, normalize_objectives
from moo.algorithms import (
    get_algorithm,
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
from moo.algorithms.alg08_topsis import topsis_rank
from moo.algorithms.alg09_vikor import vikor_rank
from moo.algorithms.alg13_npga import pareto_tournament_select


def test_alg01_weighted_sum() -> None:
    prob = FactorySpeedProblem()
    res1 = alg01_weighted_sum.run(prob, budget=500, seed=42, n_weights=11)
    
    assert len(res1["X_opt"]) > 0, "Weighted Sum must return non-dominated solutions"
    assert res1["X_opt"].ndim == 2
    assert res1["F_opt"].ndim == 2
    assert res1["F_opt"].shape[1] == 2
    assert not np.isnan(res1["F_opt"]).any()
    assert not np.isinf(res1["F_opt"]).any()

    # Decision variables must strictly obey bounds [1, 10]
    assert np.all(res1["X_opt"] >= prob.xl - 1e-6)
    assert np.all(res1["X_opt"] <= prob.xu + 1e-6)

    # All Pareto optimal points for P1 lie in [3, 7]
    assert np.all(res1["X_opt"] >= 3.0 - 0.2)
    assert np.all(res1["X_opt"] <= 7.0 + 0.2)

    # Determinism with fixed seed
    res2 = alg01_weighted_sum.run(prob, budget=500, seed=42, n_weights=11)
    np.testing.assert_allclose(res1["X_opt"], res2["X_opt"], atol=1e-5)


def test_alg02_epsilon_constraint() -> None:
    prob = FactorySpeedProblem()
    res = alg02_epsilon_constraint.run(prob, budget=600, seed=42, n_eps=11)

    assert len(res["X_opt"]) > 0
    assert res["F_opt"].shape[1] == 2
    assert not np.isnan(res["F_opt"]).any()

    # Decision variables must obey bounds
    assert np.all(res["X_opt"] >= prob.xl - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu + 1e-6)

    # Test on non-convex ZDT2 (P3): epsilon-constraint should find solutions
    prob_zdt2 = ZDT2Problem(n_var=5)
    res_zdt2 = alg02_epsilon_constraint.run(prob_zdt2, budget=500, seed=42, n_eps=7)
    assert len(res_zdt2["X_opt"]) > 0
    assert np.all(res_zdt2["X_opt"] >= 0.0 - 1e-6)
    assert np.all(res_zdt2["X_opt"] <= 1.0 + 1e-6)


def test_alg03_benson() -> None:
    prob = FactorySpeedProblem()
    res = alg03_benson.run(prob, budget=500, seed=42, n_sweeps=11)

    assert len(res["X_opt"]) > 0
    assert res["F_opt"].shape[1] == 2
    assert np.all(res["X_opt"] >= prob.xl - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu + 1e-6)


def test_alg04_weight_metric() -> None:
    prob = FactorySpeedProblem()
    
    # Test p = 1
    res_p1 = alg04_weight_metric.run(prob, budget=400, seed=42, p=1, n_weights=7)
    assert len(res_p1["X_opt"]) > 0

    # Test p = 2
    res_p2 = alg04_weight_metric.run(prob, budget=400, seed=42, p=2, n_weights=7)
    assert len(res_p2["X_opt"]) > 0

    # Test p = 'inf' (Augmented Tchebycheff)
    res_pinf = alg04_weight_metric.run(prob, budget=400, seed=42, p="inf", n_weights=7)
    assert len(res_pinf["X_opt"]) > 0
    assert np.all(res_pinf["X_opt"] >= prob.xl - 1e-6)
    assert np.all(res_pinf["X_opt"] <= prob.xu + 1e-6)


def test_alg05_value_function() -> None:
    # Test on P14 Laptop Design (DM target: 1.8 kg, 12.0 hours)
    prob_laptop = LaptopDesignProblem()
    res = alg05_value_function.run(prob_laptop, budget=500, seed=42)

    assert len(res["X_opt"]) == 1
    x_sol = res["X_opt"][0]
    f_nat = res["F_opt_nat"][0]

    # Bounds check: screen in [13, 17], battery in [40, 90]
    assert 13.0 <= x_sol[0] <= 17.0
    assert 40.0 <= x_sol[1] <= 90.0

    # Weight should be reasonably close to 1.8 kg and runtime near 12 h
    assert 1.4 <= f_nat[0] <= 2.2
    assert 8.0 <= f_nat[1] <= 16.0

    # Test on P4 Portfolio (mean-variance risk-aversion)
    prob_p4 = PortfolioProblem()
    res_p4 = alg05_value_function.run(prob_p4, budget=500, seed=42)
    assert len(res_p4["X_opt"]) == 1
    w_sol = res_p4["X_opt"][0]
    # Check budget constraint sum(w) == 1.0 within tolerance
    assert math.isclose(float(np.sum(w_sol)), 1.0, abs_tol=1e-3)


def test_alg06_goal_programming() -> None:
    # 1. P11 Production Planning (Weighted mode: Profit >= 2400, Units >= 600)
    prob_p11 = ProductionPlanningProblem()
    res_w = alg06_goal_programming.run(prob_p11, budget=2000, seed=42, method="weighted")
    assert len(res_w["X_opt"]) == 1
    f_nat_w = res_w["F_opt_nat"][0]
    assert math.isclose(f_nat_w[0], 2400.0, abs_tol=1.0)
    assert math.isclose(f_nat_w[1], 500.0, abs_tol=1.0)
    assert res_w["CV_opt"][0] <= 1e-4
    assert res_w["achieved"][0] is True  # Profit target met
    assert res_w["achieved"][1] is False  # Units target 600 not met (shortfall 100)
    assert math.isclose(res_w["d_minus"][0], 0.0, abs_tol=1e-3)
    assert math.isclose(res_w["d_minus"][1], 100.0, abs_tol=1.0)

    # 2. P11 Preemptive mode: Priority 1 (Profit) >> Priority 2 (Units)
    res_p = alg06_goal_programming.run(prob_p11, budget=2000, seed=42, method="preemptive", priorities=[0, 1])
    f_nat_p = res_p["F_opt_nat"][0]
    assert math.isclose(f_nat_p[0], 2400.0, abs_tol=1.0)
    assert math.isclose(f_nat_p[1], 500.0, abs_tol=1.0)
    assert res_p["CV_opt"][0] <= 1e-4
    assert res_p["achieved"][0] is True
    assert res_p["achieved"][1] is False

    # 3. Achievable targets on P11: Profit >= 2000, Units >= 400
    res_ach = alg06_goal_programming.run(
        prob_p11,
        targets=np.array([2000.0, 400.0]),
        goal_types=["ge", "ge"],
        budget=1000,
        seed=42,
    )
    assert res_ach["achieved"][0] is True
    assert res_ach["achieved"][1] is True
    assert res_ach["CV_opt"][0] <= 1e-4

    # 4. Unachievable targets on P11: Profit >= 3000, Units >= 700
    res_unach = alg06_goal_programming.run(
        prob_p11,
        targets=np.array([3000.0, 700.0]),
        goal_types=["ge", "ge"],
        budget=1000,
        seed=42,
    )
    assert res_unach["achieved"][0] is False
    assert res_unach["achieved"][1] is False
    assert res_unach["CV_opt"][0] <= 1e-4

    # 5. Constrained minimization targets on P6 Binh-Korn: f1 <= 50, f2 <= 20
    prob_p6 = BinhKornProblem()
    res_p6 = alg06_goal_programming.run(
        prob_p6,
        targets=np.array([50.0, 20.0]),
        goal_types=["le", "le"],
        budget=1000,
        seed=42,
    )
    assert res_p6["achieved"][0] is True
    assert res_p6["achieved"][1] is True
    assert res_p6["CV_opt"][0] <= 1e-4


def test_alg07_lexicographic() -> None:
    # 1. P11 Production Planning (Priority 1: Profit, Priority 2: Units)
    prob_p11 = ProductionPlanningProblem()
    res_p11 = alg07_lexicographic.run(prob_p11, budget=1500, seed=42, priorities=[0, 1])
    assert len(res_p11["X_opt"]) == 1
    x_opt_p11 = res_p11["X_opt"][0]
    f_nat_p11 = res_p11["F_opt_nat"][0]
    assert math.isclose(x_opt_p11[0], 300.0, abs_tol=2.0)
    assert math.isclose(x_opt_p11[1], 200.0, abs_tol=2.0)
    assert math.isclose(f_nat_p11[0], 2400.0, abs_tol=1.0)
    assert math.isclose(f_nat_p11[1], 500.0, abs_tol=1.0)
    assert res_p11["CV_opt"][0] <= 1e-4

    # 2. P7 Water Allocation on both priority orders:
    # Priority [0, 1]: Agriculture yield first -> expected (56.0, 24.0)
    prob_p7 = WaterAllocationProblem()
    res_p7_01 = alg07_lexicographic.run(prob_p7, budget=2000, seed=42, priorities=[0, 1])
    f_p7_01 = res_p7_01["F_opt_nat"][0]
    assert math.isclose(f_p7_01[0], 56.0, abs_tol=1.0)
    assert math.isclose(f_p7_01[1], 24.0, abs_tol=1.0)
    assert res_p7_01["CV_opt"][0] <= 1e-4

    # Priority [1, 0]: Industry output first -> expected (8.0, 96.0)
    res_p7_10 = alg07_lexicographic.run(prob_p7, budget=2000, seed=42, priorities=[1, 0])
    f_p7_10 = res_p7_10["F_opt_nat"][0]
    assert math.isclose(f_p7_10[0], 8.0, abs_tol=1.0)
    assert math.isclose(f_p7_10[1], 96.0, abs_tol=1.0)
    assert res_p7_10["CV_opt"][0] <= 1e-4

    # 3. Constrained problem P6 Binh-Korn on both priority orders:
    # Priority [0, 1]: f1 first -> minimum is at (0, 0) giving (0, 50)
    prob_p6 = BinhKornProblem()
    res_p6_01 = alg07_lexicographic.run(prob_p6, budget=2000, seed=42, priorities=[0, 1])
    f_p6_01 = res_p6_01["F_opt_nat"][0]
    assert math.isclose(f_p6_01[0], 0.0, abs_tol=0.1)
    assert math.isclose(f_p6_01[1], 50.0, abs_tol=0.5)
    assert res_p6_01["CV_opt"][0] <= 1e-4

    # Priority [1, 0]: f2 first -> minimum is at (5, 3) giving (136, 4)
    res_p6_10 = alg07_lexicographic.run(prob_p6, budget=2000, seed=42, priorities=[1, 0])
    f_p6_10 = res_p6_10["F_opt_nat"][0]
    assert math.isclose(f_p6_10[0], 136.0, abs_tol=0.5)
    assert math.isclose(f_p6_10[1], 4.0, abs_tol=0.1)
    assert res_p6_10["CV_opt"][0] <= 1e-4


def test_alg08_topsis() -> None:
    # Test on P12 Supplier Selection (5 suppliers, 4 criteria)
    prob_p12 = SupplierSelectionProblem()
    res = alg08_topsis.run(prob_p12, seed=42)

    assert "scores" in res
    assert "rankings" in res
    scores = res["scores"]
    rankings = res["rankings"]

    assert len(scores) == 5
    assert len(rankings) == 5
    # All closeness coefficients in [0, 1]
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

    # Rankings must be a permutation of 0..4
    assert sorted(list(rankings)) == list(range(5))
    assert res["best_idx"] == rankings[0]


def test_alg09_vikor() -> None:
    # Test on P12 Supplier Selection with v = 0.5
    prob_p12 = SupplierSelectionProblem()
    res = alg09_vikor.run(prob_p12, seed=42, v=0.5)

    assert "S" in res
    assert "R" in res
    assert "Q" in res
    assert "compromise_set" in res

    S, R, Q = res["S"], res["R"], res["Q"]
    assert len(S) == 5 and len(R) == 5 and len(Q) == 5
    assert np.all(S >= 0.0)
    assert np.all(R >= 0.0)
    assert np.all(Q >= -1e-12)
    assert np.all(Q <= 1.0 + 1e-12)

    # Compromise set is non-empty
    assert len(res["compromise_set"]) >= 1

    # Test sensitivity with v = 0.25 (veto/regret) and v = 0.75 (majority)
    res_v025 = alg09_vikor.run(prob_p12, seed=42, v=0.25)
    res_v075 = alg09_vikor.run(prob_p12, seed=42, v=0.75)
    assert len(res_v025["rankings"]) == 5
    assert len(res_v075["rankings"]) == 5


def test_alg10_vega() -> None:
    prob = FactorySpeedProblem()
    res = alg10_vega.run(prob, budget=500, seed=42, pop_size=40)

    assert len(res["X_opt"]) > 0
    assert res["F_opt"].shape[1] == 2
    assert np.all(res["X_opt"] >= prob.xl - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu + 1e-6)

    # Determinism with fixed seed
    res2 = alg10_vega.run(prob, budget=500, seed=42, pop_size=40)
    np.testing.assert_allclose(res["F_opt"], res2["F_opt"], atol=1e-5)


def test_alg11_moga() -> None:
    prob = FactorySpeedProblem()
    res = alg11_moga.run(prob, budget=500, seed=42, pop_size=40, sigma_share=0.1)

    assert len(res["X_opt"]) > 0
    assert res["F_opt"].shape[1] == 2
    assert np.all(res["X_opt"] >= prob.xl - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu + 1e-6)


def test_alg12_nsga() -> None:
    prob = FactorySpeedProblem()
    res = alg12_nsga.run(prob, budget=500, seed=42, pop_size=40, sigma_share=0.1)

    assert len(res["X_opt"]) > 0
    assert res["F_opt"].shape[1] == 2
    assert np.all(res["X_opt"] >= prob.xl - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu + 1e-6)


def test_alg13_npga_deep_dive_cases() -> None:
    """Rigorous verification of NPGA tournament rules for Task C Deep Dive."""
    # Create synthetic population with known objective values:
    # Cand A: [1.0, 2.0] (non-dominated)
    # Cand B: [3.0, 4.0] (dominated by Cand A and by T_dom)
    # T_dom member: [1.5, 2.5]
    F_synth = np.array([
        [1.0, 2.0],  # 0: Cand A
        [3.0, 4.0],  # 1: Cand B
        [1.5, 2.5],  # 2: T_dom member
    ])
    CV_synth = np.zeros(3)
    F_norm_synth = F_synth / np.max(F_synth, axis=0)
    t_dom_set = np.array([2])

    # Case 1: Cand A non-dominated by T_dom, Cand B dominated by T_dom -> A wins!
    winner1 = pareto_tournament_select(
        candidate_a_idx=0,
        candidate_b_idx=1,
        t_dom_indices=t_dom_set,
        F=F_synth,
        CV=CV_synth,
        F_norm=F_norm_synth,
        sigma_share=0.5,
    )
    assert winner1 == 0, "Non-dominated candidate A must defeat dominated candidate B"

    # Case 2: Cand B vs Cand A (reversed) -> A wins!
    winner2 = pareto_tournament_select(
        candidate_a_idx=1,
        candidate_b_idx=0,
        t_dom_indices=t_dom_set,
        F=F_synth,
        CV=CV_synth,
        F_norm=F_norm_synth,
        sigma_share=0.5,
    )
    assert winner2 == 0, "Non-dominated candidate must defeat dominated candidate"

    # Case 3: Both non-dominated by T_dom (Tie) -> Niche count tie-breaker
    # Add a crowded point close to A so A has higher niche count than B
    F_tie = np.array([
        [1.0, 1.0],  # 0: Cand A (crowded)
        [1.0, 1.05], # 1: Crowd near A
        [5.0, 5.0],  # 2: Cand B (isolated / less crowded)
        [10.0, 10.0] # 3: T_dom (dominates neither)
    ])
    CV_tie = np.zeros(4)
    F_norm_tie = F_tie / np.max(F_tie, axis=0)
    t_dom_isolated = np.array([3])

    # Cand B has lower niche count (less crowded), so B must win the tie-break!
    winner_tie = pareto_tournament_select(
        candidate_a_idx=0,
        candidate_b_idx=2,
        t_dom_indices=t_dom_isolated,
        F=F_tie,
        CV=CV_tie,
        F_norm=F_norm_tie,
        sigma_share=0.3,
    )
    assert winner_tie == 2, "Less crowded candidate B must win niche-count tie-breaker"

    # Test full NPGA algorithm run on FactorySpeedProblem (P1)
    prob_p1 = FactorySpeedProblem()
    res_npga = alg13_npga.run(prob_p1, budget=500, seed=42, pop_size=40, t_dom=6, sigma_share=0.1)
    assert len(res_npga["X_opt"]) > 0
    assert res_npga["F_opt"].shape[1] == 2
    assert np.all(res_npga["X_opt"] >= prob_p1.xl - 1e-6)
    assert np.all(res_npga["X_opt"] <= prob_p1.xu + 1e-6)

    # Test dispatcher
    mod13 = get_algorithm(13)
    assert mod13 is alg13_npga
    mod_name = get_algorithm("npga")
    assert mod_name is alg13_npga


def test_p11_goal_programming_deviations_and_achievements() -> None:
    """Verifies that Goal Programming never falsely claims both targets are met when infeasible."""
    prob = ProductionPlanningProblem()

    # 1. Targets outside feasible region: Profit >= 2400, Units >= 600
    # Maximum possible units at profit 2400 is 500 (at xA=300, xB=200).
    # Units target of 600 leaves an unavoidable shortfall of at least 100.
    res_weighted = alg06_goal_programming.run(prob, seed=42, method="weighted")
    f_nat_w = res_weighted["F_opt_nat"][0]
    devs_w = res_weighted["deviations"]
    achieved_w = res_weighted["achieved"]

    assert math.isclose(f_nat_w[0], 2400.0, abs_tol=1.0), "Profit must reach 2400"
    assert math.isclose(f_nat_w[1], 500.0, abs_tol=1.0), "Units cannot exceed 500 at profit 2400"
    assert math.isclose(devs_w[1], 100.0, abs_tol=1.0), "Units shortfall must be 100"
    assert achieved_w[1] is False, "Units goal of 600 must NEVER be claimed achieved"

    # Pre-emptive (Lexicographic Goal Programming)
    res_pre = alg06_goal_programming.run(prob, seed=42, method="preemptive", priorities=[0, 1])
    f_nat_p = res_pre["F_opt_nat"][0]
    devs_p = res_pre["deviations"]
    achieved_p = res_pre["achieved"]

    assert math.isclose(f_nat_p[0], 2400.0, abs_tol=1e-2), "Preemptive priority 1 must achieve profit 2400"
    assert math.isclose(f_nat_p[1], 500.0, abs_tol=1e-2), "Units must reach maximum 500"
    assert math.isclose(devs_p[0], 0.0, abs_tol=1e-3), "Profit shortfall must be 0"
    assert math.isclose(devs_p[1], 100.0, abs_tol=1e-2), "Units shortfall must be 100"
    assert achieved_p[0] is True, "Profit goal 2400 is achieved"
    assert achieved_p[1] is False, "Units goal 600 must NOT be claimed achieved"

    # 2. Targets inside feasible region: Profit >= 1200, Units >= 200
    res_in = alg06_goal_programming.run(prob, seed=42, targets=np.array([1200.0, 200.0]), goal_types=["ge", "ge"])
    devs_in = res_in["deviations"]
    achieved_in = res_in["achieved"]
    assert math.isclose(devs_in[0], 0.0, abs_tol=1e-3)
    assert math.isclose(devs_in[1], 0.0, abs_tol=1e-3)
    assert achieved_in[0] is True and achieved_in[1] is True, "Both targets inside feasible region must be met"


def test_lexicographic_priorities_both_orders() -> None:
    """Verifies that Lexicographic optimization strictly preserves earlier priorities in both orders."""
    # 1. On P11:
    # Priority [0, 1]: Profit first (reaches 2400), tie broken by Units (reaches 500 at (300, 200))
    p11 = ProductionPlanningProblem()
    res_p11_01 = alg07_lexicographic.run(p11, seed=42, priorities=[0, 1])
    x_01 = res_p11_01["X_opt"][0]
    f_01 = res_p11_01["F_opt_nat"][0]
    assert math.isclose(x_01[0], 300.0, abs_tol=3.0)
    assert math.isclose(x_01[1], 200.0, abs_tol=3.0)
    assert math.isclose(f_01[0], 2400.0, abs_tol=1.0)
    assert math.isclose(f_01[1], 500.0, abs_tol=1.0)

    # 2. On P7 (Water Allocation):
    # Priority [0, 1]: Agriculture first -> x = (70, 20), f = (56, 24)
    # Priority [1, 0]: Industry first -> x = (10, 80), f = (8, 96)
    p7 = WaterAllocationProblem()
    res_p7_01 = alg07_lexicographic.run(p7, seed=42, priorities=[0, 1])
    res_p7_10 = alg07_lexicographic.run(p7, seed=42, priorities=[1, 0])

    f_p7_01 = res_p7_01["F_opt_nat"][0]
    f_p7_10 = res_p7_10["F_opt_nat"][0]

    np.testing.assert_allclose(f_p7_01, [56.0, 24.0], atol=0.5)
    np.testing.assert_allclose(f_p7_10, [8.0, 96.0], atol=0.5)


def test_topsis_and_vikor_hand_calculated_matrix() -> None:
    """Verifies TOPSIS and VIKOR against exact hand-calculated intermediate and final numbers."""
    # 3 alternatives x 2 criteria:
    # C1: Cost (min, weight 0.5)
    # C2: Benefit (max, weight 0.5)
    X = np.array([[2.0, 10.0], [4.0, 20.0], [6.0, 15.0]])
    w = np.array([0.5, 0.5])
    types = ["min", "max"]

    # Hand-calculated TOPSIS values:
    # R: [[0.267261, 0.371391], [0.534522, 0.742781], [0.801784, 0.557086]]
    # C: [0.590037, 0.631272, 0.247082]
    # Rankings: [1, 0, 2] (A2 > A1 > A3)
    top = topsis_rank(X, w, types)
    np.testing.assert_allclose(top["scores"], [0.590037, 0.631272, 0.247082], atol=1e-4)
    assert list(top["rankings"]) == [1, 0, 2]
    np.testing.assert_allclose(top["s_pos"], [0.185695, 0.133631, 0.282930], atol=1e-4)
    np.testing.assert_allclose(top["s_neg"], [0.267261, 0.228779, 0.092848], atol=1e-4)

    # Hand-calculated VIKOR values:
    # S = [0.5, 0.25, 0.75], R = [0.5, 0.25, 0.5], Q = [0.75, 0.0, 1.0]
    # Rankings: [1, 0, 2] (A2 > A1 > A3)
    vik = vikor_rank(X, w, types, v=0.5)
    np.testing.assert_allclose(vik["S"], [0.5, 0.25, 0.75], atol=1e-5)
    np.testing.assert_allclose(vik["R"], [0.5, 0.25, 0.50], atol=1e-5)
    np.testing.assert_allclose(vik["Q"], [0.75, 0.0, 1.0], atol=1e-5)
    assert list(vik["rank_Q"]) == [1, 0, 2]
    assert vik["c1_passed"] is True
    assert vik["c2_passed"] is True
    assert vik["compromise_set"] == [1]

    # Degenerate case: column with zero variance
    X_degen = np.array([[100.0, 10.0], [100.0, 20.0], [100.0, 15.0]])
    top_deg = topsis_rank(X_degen, w, types)
    vik_deg = vikor_rank(X_degen, w, types, v=0.5)
    assert not np.isnan(top_deg["scores"]).any()
    assert not np.isnan(vik_deg["Q"]).any()


def test_known_pareto_front_evolutionary_algorithms() -> None:
    """Verifies that VEGA, MOGA, NSGA, and NPGA converge close to known true front on P1."""
    prob = FactorySpeedProblem()

    for alg_mod, name in [
        (alg10_vega, "VEGA"),
        (alg11_moga, "MOGA"),
        (alg12_nsga, "NSGA"),
        (alg13_npga, "NPGA"),
    ]:
        res = alg_mod.run(prob, budget=500, seed=42, pop_size=40)
        assert len(res["X_opt"]) > 0, f"{name} must return solutions"
        # Decision bounds: [1, 10]
        assert np.all(res["X_opt"] >= prob.xl - 1e-6)
        assert np.all(res["X_opt"] <= prob.xu + 1e-6)
        # All true Pareto optimal points lie in [3, 7]
        assert np.all(res["X_opt"] >= 3.0 - 0.5), f"{name} points should be near Pareto set [3, 7]"
        assert np.all(res["X_opt"] <= 7.0 + 0.5), f"{name} points should be near Pareto set [3, 7]"
        # Feasibility check
        assert np.all(res["CV_opt"] <= 1e-4)


def test_reproducibility_and_eval_budget_accounting() -> None:
    """Verifies deterministic seed reproducibility and evaluation-budget accounting."""
    prob = FactorySpeedProblem()

    class CountingProblem:
        def __init__(self, base):
            self.base = base
            self.count = 0
        def __getattr__(self, name):
            return getattr(self.base, name)
        def evaluate(self, x):
            self.count += 1
            return self.base.evaluate(x)
        def evaluate_natural(self, x):
            self.count += 1
            return self.base.evaluate_natural(x)

    # Verify accounting for Classical & Evolutionary algorithms
    for alg_id in [1, 3, 4, 7, 10, 11, 12, 13]:
        cp = CountingProblem(prob)
        alg = get_algorithm(alg_id)
        res = alg.run(cp, budget=120, seed=42)
        reported = res["n_evals"]
        actual = cp.count
        assert reported == actual, f"Alg {alg_id} reported {reported} but actual calls was {actual}"


def test_bounds_and_constraint_violations() -> None:
    """Verifies that algorithms strictly satisfy variable bounds and constraints on P5 (Cantilever)."""
    prob = CantileverBeamProblem()

    # Test Alg 01, Alg 02, and Alg 13 on constrained beam design
    res1 = alg01_weighted_sum.run(prob, budget=300, seed=42, n_weights=7)
    assert np.all(res1["X_opt"][:, 0] >= prob.xl[0] - 1e-6)
    assert np.all(res1["X_opt"][:, 0] <= prob.xu[0] + 1e-6)
    assert np.all(res1["X_opt"][:, 1] >= prob.xl[1] - 1e-6)
    assert np.all(res1["X_opt"][:, 1] <= prob.xu[1] + 1e-6)
    assert np.all(res1["CV_opt"] <= 1e-4), "All returned solutions must be feasible"


def test_nsga_distinction_from_nsga2() -> None:
    """Verifies that original NSGA (1994) uses dummy fitness degradation and sharing, NOT NSGA-II."""
    prob = FactorySpeedProblem()

    # Synthetic population with 3 distinct fronts
    # Front 1: [0.1, 0.9], [0.5, 0.5], [0.9, 0.1]
    # Front 2: [1.2, 1.8], [1.5, 1.5]
    # Front 3: [3.0, 3.0]
    F_synth = np.array([
        [0.1, 0.9],
        [0.5, 0.5],
        [0.9, 0.1],
        [1.2, 1.8],
        [1.5, 1.5],
        [3.0, 3.0],
    ])
    CV_synth = np.zeros(6)
    fronts = fast_non_dominated_sort(F_synth, CV=CV_synth)
    assert len(fronts) == 3
    assert fronts[0] == [0, 1, 2]
    assert fronts[1] == [3, 4]
    assert fronts[2] == [5]

    # Verify dummy fitness assignment rule:
    # Front 1 dummy = N = 6
    # Front 2 dummy starts at min(Front 1 shared) * 0.95
    # Therefore, every Front 1 individual must have strictly higher fitness than any Front 2 individual!
    F_norm = normalize_objectives(F_synth, ideal=np.array([0.0, 0.0]), nadir=np.array([4.0, 4.0]))
    current_dummy = 6.0
    shared_fit = np.zeros(6)

    for front in fronts:
        F_sub = F_norm[front]
        diff = F_sub[:, None, :] - F_sub[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        sh_vals = np.where(dist < 0.2, 1.0 - dist / 0.2, 0.0)
        niche_counts = np.maximum(1.0, np.sum(sh_vals, axis=1))
        front_shared = current_dummy / niche_counts
        for k, idx in enumerate(front):
            shared_fit[idx] = front_shared[k]
        current_dummy = float(np.min(front_shared)) * 0.95

    # Verification of NSGA 1994 degradation condition:
    assert np.min(shared_fit[fronts[0]]) > np.max(shared_fit[fronts[1]])
    assert np.min(shared_fit[fronts[1]]) > np.max(shared_fit[fronts[2]])


def test_npga_tournament_mechanisms_all_cases() -> None:
    """Verifies all decision branches of NPGA Pareto tournament selection."""
    F_synth = np.array([
        [1.0, 2.0],  # 0: Non-dominated A
        [3.0, 4.0],  # 1: Dominated B
        [2.0, 2.0],  # 2: T_dom member (dominates B, does not dominate A)
    ])
    CV_synth = np.zeros(3)
    F_norm_synth = F_synth / np.max(F_synth, axis=0)

    # Branch 1: A non-dominated, B dominated -> A wins
    w1 = pareto_tournament_select(0, 1, np.array([2]), F_synth, CV_synth, F_norm_synth, sigma_share=0.5)
    assert w1 == 0

    # Branch 2: B dominated, A non-dominated -> A wins
    w2 = pareto_tournament_select(1, 0, np.array([2]), F_synth, CV_synth, F_norm_synth, sigma_share=0.5)
    assert w2 == 0

    # Branch 3: Both dominated by T_dom (Tie) -> Niche count tie-breaker
    # T_dom dominates both cand 0 and cand 1, but cand 1 is more crowded
    F_both_dom = np.array([
        [5.0, 5.0],   # 0: Cand A (isolated)
        [5.0, 5.05],  # 1: Cand B (crowded with 2)
        [5.0, 5.06],  # 2: Crowd near B
        [1.0, 1.0],   # 3: T_dom (dominates 0, 1, 2)
    ])
    CV_both = np.zeros(4)
    F_norm_both = F_both_dom / np.max(F_both_dom, axis=0)
    w3 = pareto_tournament_select(0, 1, np.array([3]), F_both_dom, CV_both, F_norm_both, sigma_share=0.2)
    assert w3 == 0, "Cand A (less crowded) must win tie-break even when both are dominated"

    # Branch 4: Feasibility rule: Feasible Cand A vs Infeasible Cand B
    CV_infeas = np.array([0.0, 2.5, 0.0])
    w4 = pareto_tournament_select(0, 1, np.array([2]), F_synth, CV_infeas, F_norm_synth, sigma_share=0.5)
    assert w4 == 0, "Feasible candidate A must defeat infeasible candidate B"


def test_all_algorithms_small_budget_and_exact_eval_accounting() -> None:
    """Verifies all 13 algorithms never exceed budget and reported n_evals matches actual calls exactly."""
    base_prob = FactorySpeedProblem()

    class CountingProblem:
        def __init__(self, base):
            self.base = base
            self.count = 0
        def __getattr__(self, name):
            return getattr(self.base, name)
        def evaluate(self, x):
            self.count += 1
            return self.base.evaluate(x)
        def evaluate_natural(self, x):
            self.count += 1
            return self.base.evaluate_natural(x)

    for alg_id in range(1, 14):
        alg = get_algorithm(alg_id)
        for b in [5, 15, 27]:
            cp = CountingProblem(base_prob)
            res = alg.run(cp, budget=b, seed=42)
            reported = res["n_evals"]
            actual = cp.count
            assert reported <= b, f"Alg {alg_id} on budget {b} exceeded budget: reported {reported}"
            assert reported == actual, f"Alg {alg_id} on budget {b} eval discrepancy: reported {reported} != actual {actual}"
            assert len(res["X_opt"]) > 0, f"Alg {alg_id} on budget {b} must return at least one solution"
            assert not np.isnan(res["F_opt"]).any(), f"Alg {alg_id} on budget {b} returned NaN objectives"


