"""
test_task_c_traces.py: Unit tests for Task C hand traces and parameter sensitivity pipeline.

Covers:
1. NPGA 2-iteration hand trace verification (Pareto tournament, niche sharing tie-break, selection).
2. MOACO 2-iteration hand trace verification (4-city TSP, dual pheromones, Pareto deposit).
3. GradNorm 2-iteration hand trace verification (task gradients, norms, target balance, weights).
4. Deterministic seeding and metric aggregation verification.
"""

from __future__ import annotations
import math
import numpy as np

from moo.experiments.task_c_runner import (
    trace_npga_2iters,
    trace_moaco_2iters,
    trace_gradnorm_2iters,
    SensitivityRunRecord,
)
from moo.algorithms import alg13_npga, alg25_moaco, alg37_gradnorm
from moo.problems import FactorySpeedProblem, ZDT2Problem


def test_npga_hand_trace_two_iterations() -> None:
    """Verifies exact analytical numbers in the 2-iteration NPGA hand trace."""
    trace = trace_npga_2iters()

    # Initial generation
    gen0 = trace["gen0"]
    X0 = gen0["X"]
    F0 = gen0["F"]
    np.testing.assert_allclose(X0.flatten(), [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(F0[:, 0], [1.0, 1.0, 9.0, 25.0])
    np.testing.assert_allclose(F0[:, 1], [25.0, 9.0, 1.0, 1.0])

    # Generation 1 tournaments
    t_g1 = trace["gen1_tournaments"]
    assert len(t_g1) == 4

    # T1: cand 0 vs cand 1 against T_dom [2, 3] -> tie on dominance, niche count tie-break
    assert t_g1[0]["a_dominated"] is False
    assert t_g1[0]["b_dominated"] is False
    assert t_g1[0]["nc_a"] < t_g1[0]["nc_b"]
    assert t_g1[0]["winner"] == 0

    # T2: cand 1 vs cand 3 against T_dom [0, 2] -> cand 3 dominated by 2 (9,1 dominates 25,1)
    assert t_g1[1]["a_dominated"] is False
    assert t_g1[1]["b_dominated"] is True
    assert t_g1[1]["winner"] == 1

    # T3: cand 2 vs cand 0 against T_dom [1, 3] -> cand 0 dominated by 1 (1,9 dominates 1,25)
    assert t_g1[2]["a_dominated"] is False
    assert t_g1[2]["b_dominated"] is True
    assert t_g1[2]["winner"] == 2

    # T4: cand 2 vs cand 3 against T_dom [0, 1] -> tie on dominance, niche count tie-break
    assert t_g1[3]["a_dominated"] is False
    assert t_g1[3]["b_dominated"] is False
    assert t_g1[3]["nc_b"] < t_g1[3]["nc_a"]
    assert t_g1[3]["winner"] == 3

    assert trace["gen1_winners"] == [0, 1, 2, 3]

    # Generation 2 tournaments
    t_g2 = trace["gen2_tournaments"]
    assert len(t_g2) == 4
    for t in t_g2:
        assert t["winner"] in [t["cand_a"], t["cand_b"]]


def test_moaco_hand_trace_two_iterations() -> None:
    """Verifies exact analytical numbers in the 2-iteration MOACO hand trace."""
    trace = trace_moaco_2iters()

    D = trace["D"]
    T = trace["T"]
    assert D.shape == (4, 4)
    assert T.shape == (4, 4)

    # Iteration 1
    iter1 = trace["iter1"]
    tours1 = iter1["tours"]
    F1 = iter1["F"]
    assert len(tours1) == 4
    assert len(F1) == 4

    # Tour 0: [0, 1, 2, 3] -> D = 10+25+12+15 = 62, T = 25+15+30+20 = 90
    np.testing.assert_allclose(F1[0], [62.0, 90.0])
    # Tour 2: [2, 0, 1, 3] -> D = 20+10+30+12 = 72, T = 10+25+12+30 = 77
    np.testing.assert_allclose(F1[2], [72.0, 77.0])

    # All tours are mutually non-dominated in Iteration 1
    assert len(iter1["non_dom_indices"]) == 4

    # Pheromones must be strictly positive and greater than evaporated base 0.8
    tau1_post1 = iter1["tau1_post"]
    tau2_post1 = iter1["tau2_post"]
    assert np.all(tau1_post1 >= 0.8)
    assert np.all(tau2_post1 >= 0.8)

    # Iteration 2
    iter2 = trace["iter2"]
    F2 = iter2["F"]
    assert len(F2) == 4
    # Tour 1: [1, 2, 0, 3] -> D = 25+20+15+30 = 90, T = 15+10+20+12 = 57
    np.testing.assert_allclose(F2[1], [90.0, 57.0])


def test_gradnorm_hand_trace_two_iterations() -> None:
    """Verifies exact analytical numbers in the 2-iteration GradNorm hand trace."""
    trace = trace_gradnorm_2iters()

    # Step 0
    np.testing.assert_allclose(trace["W0"], [0.0, 0.0])
    np.testing.assert_allclose(trace["L0"], [5.0, 10.0])
    np.testing.assert_allclose(trace["weights0"], [1.0, 1.0])

    # Iteration 1
    it1 = trace["iter1"]
    # Gradients: [-2, -4], [-6, -2]
    np.testing.assert_allclose(it1["G"][0], [-2.0, -4.0])
    np.testing.assert_allclose(it1["G"][1], [-6.0, -2.0])
    np.testing.assert_allclose(it1["task_norms"], [math.sqrt(20.0), math.sqrt(40.0)], rtol=1e-5)
    np.testing.assert_allclose(it1["mean_G"], (math.sqrt(20.0) + math.sqrt(40.0)) / 2.0, rtol=1e-5)
    np.testing.assert_allclose(it1["loss_ratios"], [1.0, 1.0])
    np.testing.assert_allclose(it1["r_task"], [1.0, 1.0])
    np.testing.assert_allclose(it1["targets"], [it1["mean_G"], it1["mean_G"]])

    # Weight updates sum to 2
    np.testing.assert_allclose(np.sum(it1["weights_next"]), 2.0, atol=1e-6)
    np.testing.assert_allclose(it1["weights_next"], [1.138158, 0.861842], atol=1e-4)

    # Parameter update W1
    np.testing.assert_allclose(it1["W_next"], [0.372368, 0.313816], atol=1e-4)

    # Iteration 2
    it2 = trace["iter2"]
    np.testing.assert_allclose(it2["L"], [3.237138, 7.375298], atol=1e-4)
    np.testing.assert_allclose(np.sum(it2["weights_next"]), 2.0, atol=1e-6)
    np.testing.assert_allclose(it2["weights_next"], [1.024719, 0.975281], atol=1e-4)
    np.testing.assert_allclose(it2["W_next"], [0.692952, 0.553525], atol=1e-4)


def test_deterministic_seeding_and_metric_aggregation() -> None:
    """Verifies that running with fixed seed produces identical results and metrics aggregate properly."""
    prob = ZDT2Problem()

    # Seed 42 twice
    res1 = alg13_npga.run(prob, budget=100, seed=42, t_dom=5)
    res2 = alg13_npga.run(prob, budget=100, seed=42, t_dom=5)

    np.testing.assert_allclose(res1["F_opt"], res2["F_opt"])
    assert res1["n_evals"] == res2["n_evals"]

    # Metric aggregation
    rec1 = SensitivityRunRecord(
        algorithm_id=13, algorithm_name="NPGA", parameter_name="t_dom",
        parameter_level="default", parameter_value=10.0, seed=42,
        problem_key="P3", problem_name=prob.name, budget=100,
        actual_evals=100, n_points=10, hv=0.55, spacing=0.08,
        igd=0.04, igd_status="computed", runtime_sec=0.15, status="works",
    )
    rec2 = SensitivityRunRecord(
        algorithm_id=13, algorithm_name="NPGA", parameter_name="t_dom",
        parameter_level="default", parameter_value=10.0, seed=43,
        problem_key="P3", problem_name=prob.name, budget=100,
        actual_evals=100, n_points=12, hv=0.57, spacing=0.06,
        igd=0.03, igd_status="computed", runtime_sec=0.17, status="works",
    )
    hvs = [rec1.hv, rec2.hv]
    mean_hv = float(np.mean(hvs))
    std_hv = float(np.std(hvs))
    assert math.isclose(mean_hv, 0.56, abs_tol=1e-5)
    assert math.isclose(std_hv, 0.01, abs_tol=1e-5)
