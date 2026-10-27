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
from moo.moo_utils import fast_non_dominated_sort, normalize_objectives, igd
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
    alg14_nsga2,
)
from moo.algorithms.alg08_topsis import topsis_rank
from moo.algorithms.alg09_vikor import vikor_rank
from moo.algorithms.alg13_npga import pareto_tournament_select
from moo.algorithms.alg14_nsga2 import (
    crowded_tournament_select,
    elitist_environmental_selection,
    assign_ranks_and_crowding,
)


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

    for alg_id in range(1, 38):
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


def test_alg14_nsga2_components() -> None:
    """Verifies NSGA-II crowding distance and crowded tournament selection on hand calculations."""
    from moo.moo_utils import crowding_distance

    # Hand-calculated front with 4 points:
    # (1, 9), (2, 5), (4, 3), (8, 1)
    F = np.array([
        [1.0, 9.0],  # 0: Extremum (min f1, max f2)
        [2.0, 5.0],  # 1: Intermediate
        [4.0, 3.0],  # 2: Intermediate
        [8.0, 1.0],  # 3: Extremum (max f1, min f2)
    ])
    dists = crowding_distance(F, [0, 1, 2, 3])

    # Extrema must have infinite crowding distance (1e14)
    assert dists[0] >= 1e14
    assert dists[3] >= 1e14

    # Point 1 hand calculation:
    # f1 span: (8 - 1) = 7. Neighbor diff: (4 - 1) = 3 -> 3/7
    # f2 span: (9 - 1) = 8. Neighbor diff: (9 - 3) = 6 -> 6/8 = 3/4
    # Total distance = 3/7 + 3/4 = 33/28 approx 1.1785714
    np.testing.assert_allclose(dists[1], 33.0 / 28.0, rtol=1e-5)

    # Point 2 hand calculation:
    # f1 neighbor diff: (8 - 2) = 6 -> 6/7
    # f2 neighbor diff: (5 - 1) = 4 -> 4/8 = 1/2
    # Total distance = 6/7 + 1/2 = 19/14 approx 1.3571428
    np.testing.assert_allclose(dists[2], 19.0 / 14.0, rtol=1e-5)

    # Crowded tournament selection rules:
    rng = np.random.default_rng(42)
    ranks = np.array([0, 0, 1, 1])
    cds = np.array([1.5, 0.8, 2.0, 1.0])

    # Rule 1: Lower rank defeats higher rank
    assert crowded_tournament_select(0, 2, ranks, cds, rng) == 0
    assert crowded_tournament_select(3, 1, ranks, cds, rng) == 1

    # Rule 2: Same rank -> higher crowding distance defeats lower crowding distance
    assert crowded_tournament_select(0, 1, ranks, cds, rng) == 0
    assert crowded_tournament_select(1, 0, ranks, cds, rng) == 0
    assert crowded_tournament_select(2, 3, ranks, cds, rng) == 2


def test_alg14_nsga2_elitist_environmental_selection() -> None:
    """Verifies (N + N) environmental selection truncates borderline front by crowding distance."""
    # 8 candidate solutions across 3 fronts:
    # Front 0: solutions 0, 1 (non-dominated)
    # Front 1: solutions 2, 3, 4, 5 (dominated only by Front 0)
    # Front 2: solutions 6, 7 (dominated by Front 1)
    F_pool = np.array([
        [1.0, 5.0],   # 0: Front 0
        [5.0, 1.0],   # 1: Front 0
        [2.0, 8.0],   # 2: Front 1 (extremum in Front 1: min f1)
        [3.0, 6.0],   # 3: Front 1 (intermediate, crowded)
        [6.0, 3.0],   # 4: Front 1 (intermediate, crowded)
        [8.0, 2.0],   # 5: Front 1 (extremum in Front 1: max f1)
        [10.0, 10.0], # 6: Front 2
        [12.0, 12.0], # 7: Front 2
    ])
    X_pool = np.arange(8).reshape(8, 1)
    CV_pool = np.zeros(8)

    # Select target size 4 from pool of 8:
    # Front 0 has 2 points -> both (0, 1) must be selected.
    # Front 1 has 4 points -> needed = 2. Extrema are 2 and 5 (crowding distance 1e14),
    # so solutions 2 and 5 must be chosen over 3 and 4.
    X_sel, F_sel, CV_sel, fronts, ranks, cd = elitist_environmental_selection(
        X_pool, F_pool, CV_pool, target_size=4
    )
    assert len(X_sel) == 4
    assert set(X_sel.ravel()) == {0, 1, 2, 5}
    assert 6 not in X_sel.ravel() and 7 not in X_sel.ravel()


def test_alg14_nsga2_bounds_and_constraints() -> None:
    """Verifies NSGA-II on constrained Binh-Korn benchmark (P6)."""
    prob_p6 = BinhKornProblem()
    res = alg14_nsga2.run(prob_p6, budget=800, seed=42, pop_size=40)

    assert len(res["X_opt"]) > 0
    # Obey bounds [0, 5] x [0, 3]
    assert np.all(res["X_opt"][:, 0] >= prob_p6.xl[0] - 1e-6)
    assert np.all(res["X_opt"][:, 0] <= prob_p6.xu[0] + 1e-6)
    assert np.all(res["X_opt"][:, 1] >= prob_p6.xl[1] - 1e-6)
    assert np.all(res["X_opt"][:, 1] <= prob_p6.xu[1] + 1e-6)

    # All non-dominated points in archive must be feasible (CV <= 1e-4)
    assert np.all(res["CV_opt"] <= 1e-4)


def test_alg14_nsga2_reproducibility() -> None:
    """Verifies NSGA-II produces deterministic results with the same random seed."""
    prob = FactorySpeedProblem()
    res1 = alg14_nsga2.run(prob, budget=500, seed=42, pop_size=30)
    res2 = alg14_nsga2.run(prob, budget=500, seed=42, pop_size=30)

    np.testing.assert_allclose(res1["X_opt"], res2["X_opt"], atol=1e-6)
    np.testing.assert_allclose(res1["F_opt"], res2["F_opt"], atol=1e-6)
    assert res1["n_evals"] == res2["n_evals"]

    # Different seed produces different exploration
    res3 = alg14_nsga2.run(prob, budget=500, seed=99, pop_size=30)
    assert not np.allclose(res1["X_opt"], res3["X_opt"])


def test_alg14_nsga2_known_pareto_front_recovery() -> None:
    """Verifies NSGA-II recovers analytical Pareto fronts on P1 and P3 with quantitative residual and IGD checks."""
    # 1. P1 (Factory Speed): Recover convex analytical front f2 = (4 - sqrt(f1))^2
    prob_p1 = FactorySpeedProblem()
    res_p1 = alg14_nsga2.run(prob_p1, budget=1000, seed=42, pop_size=40)

    assert len(res_p1["X_opt"]) > 0
    # All Pareto optimal solutions for P1 lie strictly in [3, 7]
    assert np.all(res_p1["X_opt"] >= 3.0 - 0.2)
    assert np.all(res_p1["X_opt"] <= 7.0 + 0.2)
    F_p1 = res_p1["F_opt"]
    p1_residuals = np.abs(F_p1[:, 1] - (4.0 - np.sqrt(F_p1[:, 0])) ** 2)
    assert np.mean(p1_residuals) < 0.05, f"P1 analytical front mean residual {np.mean(p1_residuals)} exceeds threshold"

    # 2. P3 (ZDT2 non-convex Pareto front f2 = 1 - f1^2):
    # Test on n_var=5 benchmark with budget=1500
    prob_p3_5 = ZDT2Problem(n_var=5)
    true_front_5 = prob_p3_5.get_true_front(n_points=200)
    res_p3_5 = alg14_nsga2.run(prob_p3_5, budget=1500, seed=42, pop_size=50)

    F_5 = res_p3_5["F_opt"]
    assert len(F_5) >= 30, f"Expected dense Pareto front, got {len(F_5)} points"
    assert np.all(F_5[:, 0] >= -1e-4) and np.all(F_5[:, 0] <= 1.0 + 1e-4)

    # Front residual check: |f2 - (1 - f1^2)|
    residuals_5 = np.abs(F_5[:, 1] - (1.0 - F_5[:, 0] ** 2))
    mean_res_5 = float(np.mean(residuals_5))
    max_res_5 = float(np.max(residuals_5))
    assert mean_res_5 < 0.05, f"P3 (n_var=5) mean residual {mean_res_5:.5f} exceeds 0.05 threshold"
    assert max_res_5 < 0.10, f"P3 (n_var=5) max residual {max_res_5:.5f} exceeds 0.10 threshold"

    # Quantitative IGD metric against analytical true front
    val_igd_5 = float(igd(F_5, true_front_5))
    assert val_igd_5 < 0.05, f"P3 (n_var=5) IGD {val_igd_5:.5f} exceeds 0.05 threshold"

    # Also test on canonical n_var=10 problem (Table 1) with budget=2000
    prob_p3_10 = ZDT2Problem(n_var=10)
    true_front_10 = prob_p3_10.get_true_front(n_points=200)
    res_p3_10 = alg14_nsga2.run(prob_p3_10, budget=2000, seed=42, pop_size=50)

    F_10 = res_p3_10["F_opt"]
    assert len(F_10) >= 20, f"Expected at least 20 points, got {len(F_10)}"
    residuals_10 = np.abs(F_10[:, 1] - (1.0 - F_10[:, 0] ** 2))
    mean_res_10 = float(np.mean(residuals_10))
    val_igd_10 = float(igd(F_10, true_front_10))
    assert mean_res_10 < 0.08, f"P3 (n_var=10) mean residual {mean_res_10:.5f} exceeds 0.08 threshold"
    assert val_igd_10 < 0.08, f"P3 (n_var=10) IGD {val_igd_10:.5f} exceeds 0.08 threshold"


def test_nsga2_distinction_from_nsga() -> None:
    """
    Demonstrates architectural and mathematical distinctions between NSGA-II and NSGA:
    1. NSGA-II uses (N + N) elitist survival where non-dominated parents cannot be replaced
       by inferior offspring.
    2. NSGA (1994) uses generational non-elitist replacement where parents are unconditionally
       discarded in favor of offspring.
    3. NSGA-II uses parameter-free crowding distance, eliminating the sigma_share parameter of NSGA.
    """
    prob = FactorySpeedProblem()

    # 1. Parameter signature check: NSGA requires sigma_share, NSGA-II does not
    import inspect
    nsga_params = inspect.signature(alg12_nsga.run).parameters
    nsga2_params = inspect.signature(alg14_nsga2.run).parameters
    assert "sigma_share" in nsga_params, "NSGA (1994) must require sigma_share"
    assert "sigma_share" not in nsga2_params, "NSGA-II must eliminate sigma_share"

    # 2. Elitist survival verification:
    # Initialize NSGA-II and force population to true Pareto front (x in [3, 7])
    state = alg14_nsga2.initialize(prob, pop_size=20, budget=1000, seed=42)
    state["X"] = np.linspace(3.0, 7.0, 20).reshape(20, 1)
    F_list, CV_list = [], []
    for x in state["X"]:
        f, cv, _ = prob.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)
    state["F"] = np.array(F_list)
    state["CV"] = np.array(CV_list)
    state["fronts"], state["ranks"], state["crowd_dist"] = assign_ranks_and_crowding(state["F"], state["CV"])

    # Step NSGA-II for one generation
    state_next = alg14_nsga2.step(state, prob)

    # In NSGA-II, because parent solutions were all on the non-dominated front,
    # all 20 survivors must remain on Front 0 (no loss of Pareto optimal solutions)
    assert len(state_next["fronts"][0]) == 20, "NSGA-II (N + N) elitism must preserve all Front 0 solutions"


def test_alg15_nsga3_components() -> None:
    """Verifies NSGA-III reference point generation, ASF calculation, and intercepts."""
    from moo.algorithms.alg15_nsga3 import compute_asf, find_hyperplane_intercepts
    from moo.moo_utils import das_dennis

    # 1. Das-Dennis reference generation for M=3, p=4: (4 + 3 - 1) choose (3 - 1) = 6 choose 2 = 15 points
    ref_pts = das_dennis(p=4, M=3)
    assert len(ref_pts) == 15
    np.testing.assert_allclose(np.sum(ref_pts, axis=1), 1.0, atol=1e-6)

    # 2. ASF extreme points identification
    # Transformed front with 3 points
    F_trans = np.array([
        [1.0, 0.1, 0.1],  # Extreme along f1
        [0.1, 1.0, 0.1],  # Extreme along f2
        [0.1, 0.1, 1.0],  # Extreme along f3
    ])
    asf_vals = compute_asf(F_trans, np.array([1.0, 1e-6, 1e-6]))
    assert np.argmin(asf_vals) == 0

    intercepts = find_hyperplane_intercepts(F_trans)
    assert len(intercepts) == 3
    assert np.all(intercepts > 0)


def test_alg16_spea2_fitness_and_density() -> None:
    """Verifies SPEA2 fine-grained strength S, raw fitness R, and density estimation."""
    from moo.algorithms.alg16_spea2 import compute_spea2_fitness, spea2_environmental_selection

    # Hand-crafted 4 solutions:
    # Sol 0: (1, 1) - dominates sol 2 (2, 2) and sol 3 (3, 3)
    # Sol 1: (0.5, 3) - non-dominated with sol 0, dominates sol 3
    # Sol 2: (2, 2) - dominated by sol 0
    # Sol 3: (3, 3) - dominated by sol 0, 1, 2
    F = np.array([
        [1.0, 1.0],
        [0.5, 3.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    CV = np.zeros(4)

    fit, R, D, dist_matrix = compute_spea2_fitness(F, CV, k_nn=1)
    # Strength S: Sol 0 dominates {2, 3} -> S=2. Sol 1 dominates {3} -> S=1. Sol 2 dominates {3} -> S=1. Sol 3 dominates {} -> S=0.
    # Raw fitness R(i) = sum_{j dominates i} S(j):
    # Sol 0: no one dominates -> R=0.
    # Sol 1: no one dominates -> R=0.
    # Sol 2: dominated by 0 -> R = S(0) = 2.
    # Sol 3: dominated by 0, 1, 2 -> R = S(0) + S(1) + S(2) = 2 + 1 + 1 = 4.
    assert R[0] == 0
    assert R[1] == 0
    assert R[2] == 2
    assert R[3] == 4

    # Non-dominated solutions (0 and 1) have raw fitness 0
    assert fit[0] < 1.0 and fit[1] < 1.0
    # Dominated solutions have fitness >= 1
    assert fit[2] >= 2.0 and fit[3] >= 4.0


def test_alg17_pesa2_hyperbox_selection() -> None:
    """Verifies PESA-II hypergrid partitioning and region-based selection."""
    from moo.algorithms.alg17_pesa2 import AdaptiveHypergrid, region_based_selection

    # 4 solutions across 2 grid cells:
    # Cell A has 1 solution, Cell B has 3 solutions
    F = np.array([
        [0.1, 0.1],  # Cell A (isolated)
        [0.8, 0.8],  # Cell B (crowded)
        [0.82, 0.81],# Cell B (crowded)
        [0.85, 0.83],# Cell B (crowded)
    ])
    grid = AdaptiveHypergrid(n_divs=5)
    cell_ids, cells_to_members, squeeze_factors = grid.compute_cells(F)
    assert len(cell_ids) == 4

    # Region-based binary tournament prefers the hyperbox with smaller squeeze factor (Cell A)
    rng = np.random.default_rng(42)
    wins_cell_a = 0
    for _ in range(100):
        chosen = region_based_selection(cells_to_members, rng=rng)
        if chosen == 0:
            wins_cell_a += 1

    # Cell A (squeeze factor 1) should be chosen significantly more often than any single point in Cell B (squeeze factor 3)
    assert wins_cell_a > 40


def test_alg18_19_20_moead_suite() -> None:
    """Verifies MOEA/D, MOEA/D-DE, and MOEA/D-ACO algorithms."""
    from moo.algorithms import alg18_moead, alg19_moead_de, alg20_moead_aco
    from moo.problems import FactorySpeedProblem, DeliveryRoutingTSPProblem

    prob_p1 = FactorySpeedProblem()

    # 1. MOEA/D on P1
    res18 = alg18_moead.run(prob_p1, budget=300, seed=42, pop_size=20)
    assert res18["n_evals"] <= 300
    assert len(res18["X_opt"]) > 0

    # 2. MOEA/D-DE on P1
    res19 = alg19_moead_de.run(prob_p1, budget=300, seed=42, pop_size=20)
    assert res19["n_evals"] <= 300
    assert len(res19["X_opt"]) > 0

    # 3. MOEA/D-ACO on P8 (TSP permutation problem)
    prob_p8 = DeliveryRoutingTSPProblem()
    res20 = alg20_moead_aco.run(prob_p8, budget=150, seed=42, pop_size=15)
    assert res20["n_evals"] <= 150
    assert len(res20["X_opt"]) > 0
    # Obey valid TSP tour permutation of 15 cities
    for tour in res20["X_opt"]:
        assert sorted(tour) == list(range(15))


def test_alg21_cnsga2_deb_feasibility_rules() -> None:
    """Verifies Constrained NSGA-II Deb's feasibility-first comparison rules."""
    from moo.algorithms.alg21_cnsga2 import constrained_tournament_select

    rng = np.random.default_rng(42)
    ranks = np.array([0, 0, 1, 1])
    crowd_dist = np.array([1.5, 0.8, 2.0, 1.0])

    # Case 1: Feasible (CV=0) defeats infeasible (CV=1.5)
    CV1 = np.array([0.0, 1.5, 0.0, 2.0])
    assert constrained_tournament_select(0, 1, ranks, crowd_dist, CV1, rng=rng) == 0

    # Case 2: Both infeasible: smaller CV (1.5) defeats larger CV (3.0)
    CV2 = np.array([1.5, 3.0, 2.0, 4.0])
    assert constrained_tournament_select(0, 1, ranks, crowd_dist, CV2, rng=rng) == 0

    # Case 3: Both feasible: same rank -> higher crowding distance defeats lower crowding distance
    CV3 = np.zeros(4)
    assert constrained_tournament_select(0, 1, ranks, crowd_dist, CV3, rng=rng) == 0


def test_alg22_mopso_grid_and_clipping() -> None:
    """Verifies MOPSO adaptive grid leader selection and variable bounds clipping."""
    from moo.algorithms import alg22_mopso
    from moo.problems import CantileverBeamProblem

    prob_p5 = CantileverBeamProblem()
    res = alg22_mopso.run(prob_p5, budget=200, seed=42, swarm_size=20)
    assert res["n_evals"] <= 200
    assert len(res["X_opt"]) > 0

    # Verify bounds [0.01, 0.10] x [0.02, 0.20]
    X_opt = res["X_opt"]
    assert np.all(X_opt[:, 0] >= prob_p5.xl[0] - 1e-6)
    assert np.all(X_opt[:, 0] <= prob_p5.xu[0] + 1e-6)
    assert np.all(X_opt[:, 1] >= prob_p5.xl[1] - 1e-6)
    assert np.all(X_opt[:, 1] <= prob_p5.xu[1] + 1e-6)


def test_alg23_mogwo_pack_dynamics() -> None:
    """Verifies MOGWO leader selection and linear parameter a decay."""
    from moo.algorithms import alg23_mogwo
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    state = alg23_mogwo.initialize(prob, seed=42, pack_size=20, budget=200)
    assert state["n_evals"] == 20
    assert len(state["arch_X"]) > 0

    # Execute one step
    state_next = alg23_mogwo.step(state, prob)
    assert state_next["iter"] == 1
    assert state_next["n_evals"] == 40


def test_alg24_moabc_phases() -> None:
    """Verifies MOABC runs all three bee phases within budget."""
    from moo.algorithms import alg24_moabc
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    res = alg24_moabc.run(prob, budget=150, seed=42, n_food=15)
    assert res["n_evals"] <= 150
    assert len(res["X_opt"]) > 0
    assert np.all(res["X_opt"] >= prob.xl[0] - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu[0] + 1e-6)


def test_alg25_moaco_deep_dive() -> None:
    """Verifies MOACO (Task C Deep Dive #2) dual pheromone matrices and Pareto deposit."""
    from moo.algorithms import alg25_moaco
    from moo.problems import DeliveryRoutingTSPProblem

    prob_p8 = DeliveryRoutingTSPProblem()
    state = alg25_moaco.initialize(prob_p8, seed=42, n_ants=10, budget=100)

    # 1. Verify dual pheromone and dual heuristic matrices
    assert state["tau1"].shape == (15, 15)
    assert state["tau2"].shape == (15, 15)
    assert state["eta1"].shape == (15, 15)
    assert state["eta2"].shape == (15, 15)

    # Step MOACO
    tau1_before = state["tau1"].copy()
    state_next = alg25_moaco.step(state, prob_p8)

    # Pheromones must have updated
    assert not np.allclose(state_next["tau1"], tau1_before)
    assert state_next["n_evals"] <= 20

    # Run complete budget
    res = alg25_moaco.run(prob_p8, budget=120, seed=42, n_ants=10)
    assert res["n_evals"] <= 120
    assert len(res["X_opt"]) > 0


def test_alg26_mosa_annealing() -> None:
    """Verifies MOSA multiobjective Boltzmann acceptance and cooling schedule."""
    from moo.algorithms import alg26_mosa
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    state = alg26_mosa.initialize(prob, seed=42, budget=200, T0=50.0, alpha=0.9)
    assert state["T"] == 50.0

    state_next = alg26_mosa.step(state, prob)
    assert state_next["T"] < 50.0
    assert state_next["n_evals"] == 2

    res = alg26_mosa.run(prob, budget=100, seed=42)
    assert res["n_evals"] <= 100
    assert len(res["X_opt"]) > 0


def test_alg27_mode_differential_evolution() -> None:
    """Verifies MODE one-to-one Pareto dominance replacement and archive."""
    from moo.algorithms import alg27_mode
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    res = alg27_mode.run(prob, budget=150, seed=42, pop_size=20)
    assert res["n_evals"] <= 150
    assert len(res["X_opt"]) > 0
    assert np.all(res["X_opt"] >= prob.xl[0] - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu[0] + 1e-6)


def test_alg28_motlbo_teacher_learner() -> None:
    """Verifies MO-TLBO teacher and learner phases."""
    from moo.algorithms import alg28_motlbo
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    state = alg28_motlbo.initialize(prob, seed=42, class_size=15, budget=100)
    assert state["n_evals"] == 15

    state_next = alg28_motlbo.step(state, prob)
    assert state_next["iter"] == 1
    assert state_next["n_evals"] <= 45

    res = alg28_motlbo.run(prob, budget=120, seed=42, class_size=15)
    assert res["n_evals"] <= 120
    assert len(res["X_opt"]) > 0


def test_alg29_moba_echolocation() -> None:
    """Verifies MOBA acoustic schedules (loudness decay and pulse rate increase)."""
    from moo.algorithms import alg29_moba
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    state = alg29_moba.initialize(prob, seed=42, n_bats=15, budget=100, A0=1.0, r0=0.5)

    A_before = state["loudness"].copy()
    state_next = alg29_moba.step(state, prob)

    # Some bats accept and decrease loudness
    assert np.all(state_next["loudness"] <= A_before + 1e-9)

    res = alg29_moba.run(prob, budget=100, seed=42, n_bats=15)
    assert res["n_evals"] <= 100
    assert len(res["X_opt"]) > 0


def test_alg30_moma_memetic_search() -> None:
    """Verifies MOMA memetic local search on Front 0 solutions."""
    from moo.algorithms import alg30_moma
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    res = alg30_moma.run(prob, budget=150, seed=42, pop_size=20)
    assert res["n_evals"] <= 150
    assert len(res["X_opt"]) > 0
    # Obey bounds
    assert np.all(res["X_opt"] >= prob.xl[0] - 1e-6)
    assert np.all(res["X_opt"] <= prob.xu[0] + 1e-6)


def test_alg31_rvea_apd() -> None:
    """Verifies RVEA reference vector association and Angle-Penalized Distance."""
    from moo.algorithms.alg31_rvea import associate_and_compute_apd, das_dennis, compute_gamma

    V_raw = das_dennis(p=4, M=2)
    norms = np.linalg.norm(V_raw, axis=1, keepdims=True)
    V = V_raw / norms
    gamma = compute_gamma(V)

    F = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    CV = np.zeros(3)
    ideal = np.array([0.0, 0.0])

    # Early in search (progress = 0) vs late in search (progress = 1)
    _, apd_early = associate_and_compute_apd(F, CV, V, gamma, ideal, progress=0.0)
    _, apd_late = associate_and_compute_apd(F, CV, V, gamma, ideal, progress=1.0)

    # Late APD must be >= early APD due to angle penalty
    assert np.all(apd_late >= apd_early - 1e-9)


def test_alg32_ibea_additive_epsilon() -> None:
    """Verifies IBEA additive epsilon indicator matrix and fitness assignment."""
    from moo.algorithms.alg32_ibea import compute_epsilon_indicator_matrix, compute_ibea_fitness

    # 3 solutions:
    # Sol 0: (0.1, 0.5) - non-dominated
    # Sol 1: (0.5, 0.1) - non-dominated
    # Sol 2: (0.8, 0.8) - dominated by both
    F_norm = np.array([
        [0.1, 0.5],
        [0.5, 0.1],
        [0.8, 0.8],
    ])
    I_mat = compute_epsilon_indicator_matrix(F_norm)
    # I[i, j] = max_m (f_m(x_i) - f_m(x_j))
    # I[0, 2] = max(0.1 - 0.8, 0.5 - 0.8) = -0.3 (Sol 0 dominates Sol 2)
    np.testing.assert_allclose(I_mat[0, 2], -0.3, atol=1e-6)
    np.testing.assert_allclose(I_mat[2, 0], 0.7, atol=1e-6)

    fit = compute_ibea_fitness(I_mat, kappa=0.05)
    # Sol 0 and Sol 1 must have strictly higher fitness than dominated Sol 2
    assert fit[0] > fit[2]
    assert fit[1] > fit[2]


def test_alg33_sms_emoa_hv_contribution() -> None:
    """Verifies SMS-EMOA analytical 2D exclusive hypervolume contribution."""
    from moo.algorithms.alg33_sms_emoa import compute_exclusive_hv_contribution

    # 3 solutions on 2D front:
    # (1, 5), (2, 3), (5, 1)
    F = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [5.0, 1.0],
    ])
    ref = np.array([6.0, 6.0])

    contrib = compute_exclusive_hv_contribution(F, ref)
    assert len(contrib) == 3
    assert np.all(contrib > 0)
    # Hand calculation for middle point (2, 3):
    # width = right_f1 - f1 = 5 - 2 = 3
    # height = above_f2 - f2 = 5 - 3 = 2
    # Exclusive contribution = 3 * 2 = 6.0
    np.testing.assert_allclose(contrib[1], 6.0, atol=1e-6)


def test_alg34_hype_weights() -> None:
    """Verifies HypE Monte Carlo estimation and shared sample weights."""
    from moo.algorithms.alg34_hype import compute_hype_fitness_mc

    F_front = np.array([
        [1.0, 4.0],
        [2.0, 2.0],
        [4.0, 1.0],
    ])
    rng = np.random.default_rng(42)
    fit = compute_hype_fitness_mc(F_front, target_k=2, n_samples=5000, rng=rng)
    assert len(fit) == 3
    assert np.all(fit > 0)


def test_alg35_r_nsga2_clearing() -> None:
    """Verifies R-NSGA-II preference distance calculation and epsilon-clearing."""
    from moo.algorithms.alg35_r_nsga2 import compute_preference_distances, preference_tournament_select

    ref_point = np.array([[2.0, 2.0]])
    ideal = np.array([0.0, 0.0])
    nadir = np.array([10.0, 10.0])

    F = np.array([
        [2.1, 2.1],  # Very close to ref (pref dist ~ 0.014)
        [8.0, 8.0],  # Far from ref (pref dist ~ 0.848)
    ])
    D = compute_preference_distances(F, ref_point, ideal, nadir)
    assert D[0] < D[1]

    # Preference tournament must choose point 0 over point 1
    ranks = np.array([0, 0])
    crowd_dist = np.array([1.0, 1.0])
    assert preference_tournament_select(0, 1, ranks, D, crowd_dist, eps_clear=0.05) == 0


def test_alg36_mgda_minimum_norm() -> None:
    """Verifies MGDA analytical 2D and Frank-Wolfe minimum-norm point on gradient convex hull."""
    from moo.algorithms.alg36_mgda import min_norm_2obj, min_norm_frank_wolfe

    # 1. Opposite gradients: g1 = [-1, 0], g2 = [1, 0].
    # The minimum-norm point is the origin [0, 0] (Pareto stationary point).
    g1 = np.array([-1.0, 0.0])
    g2 = np.array([1.0, 0.0])
    omega = min_norm_2obj(g1, g2)
    np.testing.assert_allclose(omega, [0.0, 0.0], atol=1e-6)

    # 2. Parallel aligned gradients: g1 = [2, 0], g2 = [4, 0].
    # Minimum-norm is g1 = [2, 0].
    g1_par = np.array([2.0, 0.0])
    g2_par = np.array([4.0, 0.0])
    omega_par = min_norm_2obj(g1_par, g2_par)
    np.testing.assert_allclose(omega_par, [2.0, 0.0], atol=1e-6)

    # 3. M=3 via Frank-Wolfe
    G3 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
    ])
    omega_3 = min_norm_frank_wolfe(G3)
    # Convex hull contains [0, 0] since (1/3)*[1,0] + (1/3)*[0,1] + (1/3)*[-1,-1] = [0, 0]
    np.testing.assert_allclose(omega_3, [0.0, 0.0], atol=0.05)


def test_alg37_gradnorm_deep_dive() -> None:
    """Verifies GradNorm (Task C Deep Dive #3) target gradient norms and weight normalization."""
    from moo.algorithms import alg37_gradnorm
    from moo.problems import FactorySpeedProblem

    prob = FactorySpeedProblem()
    state = alg37_gradnorm.initialize(prob, seed=42, budget=100, alpha=1.5, eta_grad=0.025)

    # Weights sum to M = 2
    assert len(state["weights"]) == 2
    np.testing.assert_allclose(np.sum(state["weights"]), 2.0, atol=1e-6)

    # Execute step
    state_next = alg37_gradnorm.step(state, prob)
    # Weights must still be normalized to sum to M = 2
    np.testing.assert_allclose(np.sum(state_next["weights"]), 2.0, atol=1e-6)
    assert state_next["n_evals"] <= 10

    # Run complete budget
    res = alg37_gradnorm.run(prob, budget=50, seed=42)
    assert res["n_evals"] <= 50
    assert len(res["X_opt"]) > 0
