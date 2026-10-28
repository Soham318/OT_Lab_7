"""
task_c_runner.py: Task C Deep Dive Analysis and Parameter Sensitivity Runner

Assigned Algorithms for Roll Number 13:
- Algorithm 13: NPGA (Niched Pareto Genetic Algorithm)
- Algorithm 25: MOACO (Multi-Objective Ant Colony Optimization)
- Algorithm 37: GradNorm (Gradient Normalization for Multi-Task Learning)

Features:
1. Two-iteration numerical hand traces for NPGA, MOACO, and GradNorm matching implementation logic.
2. Parameter sensitivity study across 3 values (low, default, high) x 10 seeds (42..51) = 90 runs.
3. Summary statistical aggregation (mean +- std) for HV, Spacing, IGD, runtime, and evals.
4. Export of raw CSVs, summary CSV, and publication-quality sensitivity visualization plots.
"""

from __future__ import annotations
import csv
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moo.algorithms import get_algorithm
from moo.algorithms import alg13_npga, alg25_moaco, alg37_gradnorm
from moo.problems import (
    get_problem,
    FactorySpeedProblem,
    ZDT2Problem,
    DeliveryRoutingTSPProblem,
    MultiTaskToyProblem,
    Problem,
)
from moo.moo_utils import (
    constrained_dominates,
    normalize_objectives,
    fast_non_dominated_sort,
    hypervolume,
    spacing,
    igd,
)
from moo.algorithms.common import compute_single_niche_count


# ==============================================================================
# 1. TWO-ITERATION HAND TRACE IMPLEMENTATIONS
# ==============================================================================

def trace_npga_2iters() -> dict[str, Any]:
    """
    Executes a transparent, fully reproducible 2-iteration hand trace of NPGA on P1 Factory Speed.
    Population N = 4, t_dom = 2, sigma_share = 0.5.
    
    Initial candidates on x in [1, 10]:
        x = [2.0, 4.0, 6.0, 8.0]
        f1 = (x - 3)^2,  f2 = (x - 7)^2
    """
    prob = FactorySpeedProblem()
    N = 4
    t_dom = 2
    sigma_share = 0.5

    # 1. Initial population evaluation
    X0 = np.array([[2.0], [4.0], [6.0], [8.0]], dtype=float)
    F0 = np.array([prob.evaluate(x)[0] for x in X0], dtype=float)
    CV0 = np.zeros(N, dtype=float)
    # Natural bounds: ideal = [0, 0], nadir = [25, 25]
    F_norm0 = normalize_objectives(F0, ideal=np.array([0.0, 0.0]), nadir=np.array([25.0, 25.0]))

    # Tournament setups for Generation 1:
    # 4 tournaments to select 4 parents:
    # T1: cand A=0 vs cand B=1, T_dom = [2, 3]
    # T2: cand A=1 vs cand B=3, T_dom = [0, 2]
    # T3: cand A=2 vs cand B=0, T_dom = [1, 3]
    # T4: cand A=2 vs cand B=3, T_dom = [0, 1]
    tournaments_g1 = [
        {"cand_a": 0, "cand_b": 1, "t_dom": [2, 3]},
        {"cand_a": 1, "cand_b": 3, "t_dom": [0, 2]},
        {"cand_a": 2, "cand_b": 0, "t_dom": [1, 3]},
        {"cand_a": 2, "cand_b": 3, "t_dom": [0, 1]},
    ]

    winners_g1 = []
    trace_t_g1 = []
    for t_info in tournaments_g1:
        a_idx = t_info["cand_a"]
        b_idx = t_info["cand_b"]
        t_set = np.array(t_info["t_dom"])

        a_dom = any(constrained_dominates(F0[t], CV0[t], F0[a_idx], CV0[a_idx]) for t in t_set)
        b_dom = any(constrained_dominates(F0[t], CV0[t], F0[b_idx], CV0[b_idx]) for t in t_set)

        nc_a = compute_single_niche_count(F_norm0[a_idx], F_norm0, sigma_share=sigma_share)
        nc_b = compute_single_niche_count(F_norm0[b_idx], F_norm0, sigma_share=sigma_share)

        winner = alg13_npga.pareto_tournament_select(
            a_idx, b_idx, t_set, F0, CV0, F_norm0, sigma_share=sigma_share
        )
        winners_g1.append(winner)
        trace_t_g1.append({
            "cand_a": a_idx, "cand_b": b_idx, "t_dom": list(t_set),
            "a_dominated": a_dom, "b_dominated": b_dom,
            "nc_a": float(nc_a), "nc_b": float(nc_b), "winner": winner,
        })

    # Mating pool for Gen 1: X0[winners_g1]
    # Pair 1: (winner 0, winner 1) -> SBX blend
    # Pair 2: (winner 2, winner 3) -> SBX blend
    # Deterministic offspring blend for trace:
    # Offspring 0: 0.5 * (P1 + P2) = 0.5 * (2.0 + 4.0) = 3.0
    # Offspring 1: 0.5 * (3*P2 - P1) = 0.5 * (12.0 - 2.0) = 5.0
    # Offspring 2: 0.5 * (P3 + P4) = 0.5 * (6.0 + 8.0) = 7.0
    # Offspring 3: 0.5 * (3*P3 - P4) = 0.5 * (18.0 - 8.0) = 5.0 -> perturb to 4.5
    X1 = np.array([[3.0], [5.0], [7.0], [4.5]], dtype=float)
    F1 = np.array([prob.evaluate(x)[0] for x in X1], dtype=float)
    CV1 = np.zeros(N, dtype=float)
    F_norm1 = normalize_objectives(F1, ideal=np.array([0.0, 0.0]), nadir=np.array([25.0, 25.0]))

    # Tournament setups for Generation 2:
    tournaments_g2 = [
        {"cand_a": 0, "cand_b": 1, "t_dom": [2, 3]},
        {"cand_a": 1, "cand_b": 2, "t_dom": [0, 3]},
        {"cand_a": 2, "cand_b": 3, "t_dom": [0, 1]},
        {"cand_a": 0, "cand_b": 3, "t_dom": [1, 2]},
    ]
    winners_g2 = []
    trace_t_g2 = []
    for t_info in tournaments_g2:
        a_idx = t_info["cand_a"]
        b_idx = t_info["cand_b"]
        t_set = np.array(t_info["t_dom"])

        a_dom = any(constrained_dominates(F1[t], CV1[t], F1[a_idx], CV1[a_idx]) for t in t_set)
        b_dom = any(constrained_dominates(F1[t], CV1[t], F1[b_idx], CV1[b_idx]) for t in t_set)

        nc_a = compute_single_niche_count(F_norm1[a_idx], F_norm1, sigma_share=sigma_share)
        nc_b = compute_single_niche_count(F_norm1[b_idx], F_norm1, sigma_share=sigma_share)

        winner = alg13_npga.pareto_tournament_select(
            a_idx, b_idx, t_set, F1, CV1, F_norm1, sigma_share=sigma_share
        )
        winners_g2.append(winner)
        trace_t_g2.append({
            "cand_a": a_idx, "cand_b": b_idx, "t_dom": list(t_set),
            "a_dominated": a_dom, "b_dominated": b_dom,
            "nc_a": float(nc_a), "nc_b": float(nc_b), "winner": winner,
        })

    # Offspring for Gen 2
    X2 = np.array([[3.2], [4.8], [6.5], [3.8]], dtype=float)
    F2 = np.array([prob.evaluate(x)[0] for x in X2], dtype=float)

    return {
        "gen0": {"X": X0, "F": F0, "F_norm": F_norm0},
        "gen1_tournaments": trace_t_g1,
        "gen1_winners": winners_g1,
        "gen1": {"X": X1, "F": F1, "F_norm": F_norm1},
        "gen2_tournaments": trace_t_g2,
        "gen2_winners": winners_g2,
        "gen2": {"X": X2, "F": F2},
    }


def trace_moaco_2iters() -> dict[str, Any]:
    """
    Executes a 2-iteration hand trace of MOACO on a 4-city bi-objective TSP.
    Colony size K = 4, alpha = 1.0, beta = 1.0, rho = 0.20, Q = 100.0.
    """
    # 4 cities distance matrix D and time matrix T
    D = np.array([
        [0.0, 10.0, 20.0, 15.0],
        [10.0, 0.0, 25.0, 30.0],
        [20.0, 25.0, 0.0, 12.0],
        [15.0, 30.0, 12.0, 0.0],
    ], dtype=float)

    T = np.array([
        [0.0, 25.0, 10.0, 20.0],
        [25.0, 0.0, 15.0, 12.0],
        [10.0, 15.0, 0.0, 30.0],
        [20.0, 12.0, 30.0, 0.0],
    ], dtype=float)

    n_nodes = 4
    K = 4
    alpha = 1.0
    beta = 1.0
    rho = 0.20
    Q = 100.0

    # Visibility heuristics
    eta1 = 1.0 / np.where(D <= 1e-12, 1e-6, D)
    eta2 = 1.0 / np.where(T <= 1e-12, 1e-6, T)
    np.fill_diagonal(eta1, 0.0)
    np.fill_diagonal(eta2, 0.0)

    # Initial pheromones
    tau1_init = np.ones((n_nodes, n_nodes), dtype=float)
    tau2_init = np.ones((n_nodes, n_nodes), dtype=float)

    # Weights for each ant k in {0, 1, 2, 3}:
    # lam1 = k / 3, lam2 = 1 - lam1
    weights = [(float(k) / 3.0, 1.0 - float(k) / 3.0) for k in range(K)]

    def eval_tour(tour: Sequence[int]) -> tuple[float, float]:
        dist = sum(D[tour[i], tour[(i + 1) % n_nodes]] for i in range(n_nodes))
        time_c = sum(T[tour[i], tour[(i + 1) % n_nodes]] for i in range(n_nodes))
        return float(dist), float(time_c)

    # Iteration 1:
    # Fixed ant starting cities: ant 0 starts at 0, ant 1 starts at 1, ant 2 starts at 2, ant 3 starts at 3.
    # Constructed 4 tours for Iteration 1:
    tours_iter1 = [
        np.array([0, 1, 2, 3]),  # Ant 0: D=10+25+12+15=62, T=25+15+30+20=90
        np.array([1, 0, 3, 2]),  # Ant 1: D=10+15+12+25=62, T=25+20+30+15=90
        np.array([2, 0, 1, 3]),  # Ant 2: D=20+10+30+12=72, T=10+25+12+30=77
        np.array([3, 2, 1, 0]),  # Ant 3: D=12+25+10+15=62, T=30+15+25+20=90
    ]
    F_iter1 = np.array([eval_tour(t) for t in tours_iter1], dtype=float)
    fronts1 = fast_non_dominated_sort(F_iter1)
    non_dom_idx1 = fronts1[0]

    # Evaporation
    tau1_evap1 = (1.0 - rho) * tau1_init
    tau2_evap1 = (1.0 - rho) * tau2_init

    # Pheromone deposit by non-dominated tours
    tau1_post1 = tau1_evap1.copy()
    tau2_post1 = tau2_evap1.copy()
    deposit_details1 = []
    for idx in non_dom_idx1:
        tour = tours_iter1[idx]
        f1, f2 = F_iter1[idx]
        d_tau1 = Q / f1
        d_tau2 = Q / f2
        deposit_details1.append({"ant": idx, "d_tau1": d_tau1, "d_tau2": d_tau2})
        for i in range(n_nodes):
            u = tour[i]
            v = tour[(i + 1) % n_nodes]
            tau1_post1[u, v] += d_tau1
            tau1_post1[v, u] += d_tau1
            tau2_post1[u, v] += d_tau2
            tau2_post1[v, u] += d_tau2

    # Iteration 2:
    tours_iter2 = [
        np.array([0, 3, 2, 1]),  # Ant 0: D=15+12+25+10=62, T=20+30+15+25=90
        np.array([1, 2, 0, 3]),  # Ant 1: D=25+20+15+30=90, T=15+10+20+12=57
        np.array([2, 3, 0, 1]),  # Ant 2: D=12+15+10+25=62, T=30+20+25+15=90
        np.array([3, 0, 2, 1]),  # Ant 3: D=15+20+25+30=90, T=20+10+15+12=57
    ]
    F_iter2 = np.array([eval_tour(t) for t in tours_iter2], dtype=float)
    fronts2 = fast_non_dominated_sort(F_iter2)
    non_dom_idx2 = fronts2[0]

    tau1_evap2 = (1.0 - rho) * tau1_post1
    tau2_evap2 = (1.0 - rho) * tau2_post1
    tau1_post2 = tau1_evap2.copy()
    tau2_post2 = tau2_evap2.copy()
    deposit_details2 = []
    for idx in non_dom_idx2:
        tour = tours_iter2[idx]
        f1, f2 = F_iter2[idx]
        d_tau1 = Q / f1
        d_tau2 = Q / f2
        deposit_details2.append({"ant": idx, "d_tau1": d_tau1, "d_tau2": d_tau2})
        for i in range(n_nodes):
            u = tour[i]
            v = tour[(i + 1) % n_nodes]
            tau1_post2[u, v] += d_tau1
            tau1_post2[v, u] += d_tau1
            tau2_post2[u, v] += d_tau2
            tau2_post2[v, u] += d_tau2

    return {
        "D": D,
        "T": T,
        "weights": weights,
        "iter1": {
            "tours": tours_iter1,
            "F": F_iter1,
            "non_dom_indices": non_dom_idx1,
            "tau1_post": tau1_post1,
            "tau2_post": tau2_post1,
            "deposits": deposit_details1,
        },
        "iter2": {
            "tours": tours_iter2,
            "F": F_iter2,
            "non_dom_indices": non_dom_idx2,
            "tau1_post": tau1_post2,
            "tau2_post": tau2_post2,
            "deposits": deposit_details2,
        },
    }


def trace_gradnorm_2iters() -> dict[str, Any]:
    """
    Executes a 2-iteration hand trace of GradNorm on a 2-task shared parameter problem.
    Parameters W = [w_1, w_2]^T.
    L_1(W) = (w_1 - 1)^2 + (w_2 - 2)^2
    L_2(W) = (w_1 - 3)^2 + (w_2 - 1)^2
    alpha = 1.5, eta_model = 0.05, eta_grad = 0.025.
    Initial: W^(0) = [0.0, 0.0], weights = [1.0, 1.0].
    """
    alpha = 1.5
    eta_model = 0.05
    eta_grad = 0.025
    M = 2

    # Step 0: Initial evaluation
    W0 = np.array([0.0, 0.0], dtype=float)
    L1_0 = (W0[0] - 1.0) ** 2 + (W0[1] - 2.0) ** 2  # 1 + 4 = 5.0
    L2_0 = (W0[0] - 3.0) ** 2 + (W0[1] - 1.0) ** 2  # 9 + 1 = 10.0
    L0 = np.array([L1_0, L2_0], dtype=float)
    weights0 = np.array([1.0, 1.0], dtype=float)

    # Iteration 1:
    grad_L1_0 = np.array([2.0 * (W0[0] - 1.0), 2.0 * (W0[1] - 2.0)])  # [-2.0, -4.0]
    grad_L2_0 = np.array([2.0 * (W0[0] - 3.0), 2.0 * (W0[1] - 1.0)])  # [-6.0, -2.0]
    G0 = np.vstack([grad_L1_0, grad_L2_0])

    norm1_0 = float(np.linalg.norm(grad_L1_0))  # sqrt(20) ~ 4.472136
    norm2_0 = float(np.linalg.norm(grad_L2_0))  # sqrt(40) ~ 6.324555
    task_norms0 = np.array([norm1_0, norm2_0])

    G_weighted0 = weights0 * task_norms0  # [4.472136, 6.324555]
    mean_G0 = float(np.mean(G_weighted0))  # 5.398346

    loss_ratios0 = L0 / L0  # [1.0, 1.0]
    mean_lr0 = float(np.mean(loss_ratios0))  # 1.0
    r_task0 = loss_ratios0 / mean_lr0  # [1.0, 1.0]

    targets0 = mean_G0 * (r_task0 ** alpha)  # [5.398346, 5.398346]
    diffs0 = G_weighted0 - targets0  # [-0.926210, +0.926210]
    grad_w0 = np.sign(diffs0) * task_norms0  # [-4.472136, +6.324555]

    w_unnorm0 = np.maximum(0.01, weights0 - eta_grad * grad_w0)  # [1.111803, 0.841886]
    weights1 = (w_unnorm0 / np.sum(w_unnorm0)) * M  # [1.138158, 0.861842]

    total_grad0 = np.dot(weights1, G0)  # [-7.447368, -6.276316]
    W1 = W0 - eta_model * total_grad0  # [0.372368, 0.313816]

    # Iteration 2:
    L1_1 = (W1[0] - 1.0) ** 2 + (W1[1] - 2.0) ** 2  # (-0.627632)^2 + (-1.686184)^2 = 3.237138
    L2_1 = (W1[0] - 3.0) ** 2 + (W1[1] - 1.0) ** 2  # (-2.627632)^2 + (-0.686184)^2 = 7.375298
    L1_arr = np.array([L1_1, L2_1])

    grad_L1_1 = np.array([2.0 * (W1[0] - 1.0), 2.0 * (W1[1] - 2.0)])  # [-1.255263, -3.372368]
    grad_L2_1 = np.array([2.0 * (W1[0] - 3.0), 2.0 * (W1[1] - 1.0)])  # [-5.255263, -1.372368]
    G1 = np.vstack([grad_L1_1, grad_L2_1])

    norm1_1 = float(np.linalg.norm(grad_L1_1))  # 3.598409
    norm2_1 = float(np.linalg.norm(grad_L2_1))  # 5.431499
    task_norms1 = np.array([norm1_1, norm2_1])

    G_weighted1 = weights1 * task_norms1  # [4.095558, 4.681096]
    mean_G1 = float(np.mean(G_weighted1))  # 4.388327

    loss_ratios1 = L1_arr / L0  # [0.647428, 0.737530]
    mean_lr1 = float(np.mean(loss_ratios1))  # 0.692479
    r_task1 = loss_ratios1 / mean_lr1  # [0.934942, 1.065058]

    targets1 = mean_G1 * (r_task1 ** alpha)  # [3.966938, 4.823636]
    diffs1 = G_weighted1 - targets1  # [+0.128620, -0.142540]
    grad_w1 = np.sign(diffs1) * task_norms1  # [+3.598409, -5.431499]

    w_unnorm1 = np.maximum(0.01, weights1 - eta_grad * grad_w1)  # [1.048198, 0.997629]
    weights2 = (w_unnorm1 / np.sum(w_unnorm1)) * M  # [1.024719, 0.975281]

    total_grad1 = np.dot(weights2, G1)  # [-6.411670, -4.794174]
    W2 = W1 - eta_model * total_grad1  # [0.692952, 0.553525]

    return {
        "W0": W0, "L0": L0, "weights0": weights0,
        "iter1": {
            "G": G0, "task_norms": task_norms0, "G_weighted": G_weighted0,
            "mean_G": mean_G0, "loss_ratios": loss_ratios0, "r_task": r_task0,
            "targets": targets0, "diffs": diffs0, "grad_w": grad_w0,
            "weights_next": weights1, "total_grad": total_grad0, "W_next": W1,
        },
        "iter2": {
            "L": L1_arr, "G": G1, "task_norms": task_norms1, "G_weighted": G_weighted1,
            "mean_G": mean_G1, "loss_ratios": loss_ratios1, "r_task": r_task1,
            "targets": targets1, "diffs": diffs1, "grad_w": grad_w1,
            "weights_next": weights2, "total_grad": total_grad1, "W_next": W2,
        },
    }


# ==============================================================================
# 2. PARAMETER SENSITIVITY EXPERIMENT RUNNER
# ==============================================================================

@dataclass
class SensitivityRunRecord:
    algorithm_id: int
    algorithm_name: str
    parameter_name: str
    parameter_level: str  # "low", "default", "high"
    parameter_value: float
    seed: int
    problem_key: str
    problem_name: str
    budget: int
    actual_evals: int
    n_points: int
    hv: float
    spacing: float | None
    igd: float | None
    igd_status: str
    runtime_sec: float
    status: str


def run_parameter_sensitivity(
    results_dir: Path | str = "results",
    plots_dir: Path | str = "plots",
    seeds: Sequence[int] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    verbose: bool = True,
) -> tuple[dict[str, list[SensitivityRunRecord]], list[dict[str, Any]]]:
    """
    Executes the full 90-run parameter sensitivity study (3 algorithms x 3 settings x 10 seeds).
    
    1. NPGA (#13): t_dom in {2, 10, 25} on P3 (ZDT2), budget=1500
    2. MOACO (#25): rho in {0.05, 0.20, 0.50} on P8 (TSP 15-city), budget=800
    3. GradNorm (#37): alpha in {0.12, 1.5, 3.0} on P13(a) (Multi-Task), budget=400
    """
    res_dir = Path(results_dir)
    plt_dir = Path(plots_dir)
    csv_dir = res_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    sens_plot_dir = plt_dir / "sensitivity"
    sens_plot_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "alg_id": 13,
            "alg_name": "NPGA",
            "prob_key": "P3",
            "param_name": "t_dom",
            "budget": 1500,
            "levels": [("low", 2), ("default", 10), ("high", 25)],
            "kwarg_key": "t_dom",
        },
        {
            "alg_id": 25,
            "alg_name": "MOACO",
            "prob_key": "P8",
            "param_name": "rho",
            "budget": 800,
            "levels": [("low", 0.05), ("default", 0.20), ("high", 0.50)],
            "kwarg_key": "rho",
        },
        {
            "alg_id": 37,
            "alg_name": "GradNorm",
            "prob_key": "P13",
            "param_name": "alpha",
            "budget": 400,
            "levels": [("low", 0.12), ("default", 1.5), ("high", 3.0)],
            "kwarg_key": "alpha",
        },
    ]

    all_raw_by_alg: dict[str, list[SensitivityRunRecord]] = {}
    summary_rows: list[dict[str, Any]] = []

    for cfg in configs:
        alg_name = cfg["alg_name"]
        alg_id = cfg["alg_id"]
        prob_key = cfg["prob_key"]
        param_name = cfg["param_name"]
        budget = cfg["budget"]
        levels = cfg["levels"]
        kw_key = cfg["kwarg_key"]

        if verbose:
            print(f"\n{'='*70}\nRunning Task C Sensitivity: {alg_name} (#{alg_id}) on {prob_key} (Param: {param_name})\n{'='*70}")

        prob = get_problem(prob_key)
        alg_mod = get_algorithm(alg_id)
        true_front = prob.get_true_front(200)

        raw_records: list[SensitivityRunRecord] = []

        for level_label, val in levels:
            if verbose:
                print(f"  Setting {param_name} = {val} ({level_label}) across {len(seeds)} seeds...")

            level_hvs = []
            level_spacings = []
            level_igds = []
            level_runtimes = []
            level_evals = []

            for seed in seeds:
                t0 = time.perf_counter()
                kwargs = {kw_key: val}
                res = alg_mod.run(prob, budget=budget, seed=seed, **kwargs)
                runtime = time.perf_counter() - t0

                F_opt = res.get("F_opt", np.empty((0, prob.n_obj)))
                CV_opt = res.get("CV_opt", np.zeros(len(F_opt)))
                n_evals = res.get("n_evals", budget)

                # Filter feasible
                if len(CV_opt) == len(F_opt) and len(F_opt) > 0:
                    feas_mask = np.asarray([prob.is_feasible(cv) for cv in CV_opt], dtype=bool)
                    F_feas = F_opt[feas_mask]
                else:
                    F_feas = F_opt

                n_feas = len(F_feas)

                # Metrics
                hv_val = 0.0
                sp_val = None
                igd_val = None
                igd_stat = "unavailable"

                if n_feas > 0:
                    try:
                        F_norm = normalize_objectives(F_feas, ideal=prob.ideal_point, nadir=prob.nadir_point)
                        ref = np.ones(prob.n_obj) * 1.1
                        hv_val = float(hypervolume(F_norm, ref))
                    except Exception:
                        hv_val = 0.0

                    if n_feas > 1:
                        try:
                            F_norm = normalize_objectives(F_feas, ideal=prob.ideal_point, nadir=prob.nadir_point)
                            sp_val = float(spacing(F_norm))
                        except Exception:
                            sp_val = None

                    if true_front is not None and len(true_front) > 0:
                        try:
                            TF_norm = normalize_objectives(true_front, ideal=prob.ideal_point, nadir=prob.nadir_point)
                            F_norm = normalize_objectives(F_feas, ideal=prob.ideal_point, nadir=prob.nadir_point)
                            igd_val = float(igd(F_norm, TF_norm))
                            igd_stat = "computed"
                        except Exception:
                            igd_val = None

                rec = SensitivityRunRecord(
                    algorithm_id=alg_id,
                    algorithm_name=alg_name,
                    parameter_name=param_name,
                    parameter_level=level_label,
                    parameter_value=float(val),
                    seed=seed,
                    problem_key=prob_key,
                    problem_name=prob.name,
                    budget=budget,
                    actual_evals=n_evals,
                    n_points=n_feas,
                    hv=hv_val,
                    spacing=sp_val,
                    igd=igd_val,
                    igd_status=igd_stat,
                    runtime_sec=runtime,
                    status="works" if n_feas > 0 else "partly works",
                )
                raw_records.append(rec)

                level_hvs.append(hv_val)
                if sp_val is not None:
                    level_spacings.append(sp_val)
                if igd_val is not None:
                    level_igds.append(igd_val)
                level_runtimes.append(runtime)
                level_evals.append(n_evals)

            # Summaries for this level
            summary_rows.append({
                "algorithm": alg_name,
                "problem": prob_key,
                "parameter": param_name,
                "level": level_label,
                "value": val,
                "runs": len(seeds),
                "hv_mean": float(np.mean(level_hvs)),
                "hv_std": float(np.std(level_hvs)),
                "spacing_mean": float(np.mean(level_spacings)) if level_spacings else None,
                "spacing_std": float(np.std(level_spacings)) if level_spacings else None,
                "igd_mean": float(np.mean(level_igds)) if level_igds else None,
                "igd_std": float(np.std(level_igds)) if level_igds else None,
                "runtime_mean": float(np.mean(level_runtimes)),
                "runtime_std": float(np.std(level_runtimes)),
                "evals_mean": float(np.mean(level_evals)),
            })

        all_raw_by_alg[alg_name] = raw_records

        # Save algorithm-specific raw CSV
        slug = alg_name.lower().replace("/", "_")
        csv_file = csv_dir / f"task_c_{slug}_sensitivity_raw.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in SensitivityRunRecord.__dataclass_fields__.values()])
            writer.writeheader()
            for r in raw_records:
                writer.writerow(asdict(r))

    # Save aggregated summary CSV
    summary_csv = csv_dir / "task_c_sensitivity_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "algorithm", "problem", "parameter", "level", "value", "runs",
            "hv_mean", "hv_std", "spacing_mean", "spacing_std",
            "igd_mean", "igd_std", "runtime_mean", "runtime_std", "evals_mean"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary_rows:
            writer.writerow(s)

    # Generate comparative plots
    generate_sensitivity_plots(all_raw_by_alg, summary_rows, sens_plot_dir)

    return all_raw_by_alg, summary_rows


# ==============================================================================
# 3. SENSITIVITY PLOT GENERATION
# ==============================================================================

def generate_sensitivity_plots(
    raw_by_alg: dict[str, list[SensitivityRunRecord]],
    summary_rows: list[dict[str, Any]],
    output_dir: Path | str,
) -> None:
    """Generates informative sensitivity bar/box plots and multi-panel comparisons."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Style definitions
    colors = {"low": "#3b82f6", "default": "#10b981", "high": "#ef4444"}

    # 1. NPGA Plot (t_dom)
    npga_records = raw_by_alg.get("NPGA", [])
    if npga_records:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        levels = ["low", "default", "high"]
        x_labels = [f"Low\n(t_dom=2)", f"Default\n(t_dom=10)", f"High\n(t_dom=25)"]

        # HV boxplot
        hv_data = [[r.hv for r in npga_records if r.parameter_level == lvl] for lvl in levels]
        axes[0].boxplot(hv_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#93c5fd", color="#1d4ed8"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[0].set_title("NPGA: Hypervolume on ZDT2", fontweight="bold")
        axes[0].set_ylabel("Hypervolume (Higher is Better)")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Spacing boxplot
        sp_data = [[r.spacing for r in npga_records if r.parameter_level == lvl and r.spacing is not None] for lvl in levels]
        axes[1].boxplot(sp_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#a7f3d0", color="#047857"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[1].set_title("NPGA: Spacing on ZDT2", fontweight="bold")
        axes[1].set_ylabel("Schott's Spacing (Lower is Better)")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # IGD boxplot
        igd_data = [[r.igd for r in npga_records if r.parameter_level == lvl and r.igd is not None] for lvl in levels]
        axes[2].boxplot(igd_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#fbcfe8", color="#be185d"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[2].set_title("NPGA: IGD on ZDT2", fontweight="bold")
        axes[2].set_ylabel("IGD (Lower is Better)")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        plt.savefig(out_dir / "task_c_npga_tdom_sensitivity.png", dpi=200)
        plt.close(fig)

    # 2. MOACO Plot (rho)
    moaco_records = raw_by_alg.get("MOACO", [])
    if moaco_records:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        levels = ["low", "default", "high"]
        x_labels = [f"Low\n(rho=0.05)", f"Default\n(rho=0.20)", f"High\n(rho=0.50)"]

        hv_data = [[r.hv for r in moaco_records if r.parameter_level == lvl] for lvl in levels]
        axes[0].boxplot(hv_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#fed7aa", color="#c2410c"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[0].set_title("MOACO: Hypervolume on P8 TSP", fontweight="bold")
        axes[0].set_ylabel("Hypervolume (Higher is Better)")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        sp_data = [[r.spacing for r in moaco_records if r.parameter_level == lvl and r.spacing is not None] for lvl in levels]
        axes[1].boxplot(sp_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#d9f99d", color="#4d7c0f"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[1].set_title("MOACO: Spacing on P8 TSP", fontweight="bold")
        axes[1].set_ylabel("Schott's Spacing (Lower is Better)")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.savefig(out_dir / "task_c_moaco_rho_sensitivity.png", dpi=200)
        plt.close(fig)

    # 3. GradNorm Plot (alpha)
    grad_records = raw_by_alg.get("GradNorm", [])
    if grad_records:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        levels = ["low", "default", "high"]
        x_labels = [f"Low\n(alpha=0.12)", f"Default\n(alpha=1.5)", f"High\n(alpha=3.0)"]

        hv_data = [[r.hv for r in grad_records if r.parameter_level == lvl] for lvl in levels]
        axes[0].boxplot(hv_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#ddd6fe", color="#6d28d9"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[0].set_title("GradNorm: Hypervolume on P13 Multi-Task", fontweight="bold")
        axes[0].set_ylabel("Hypervolume (Higher is Better)")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        sp_data = [[r.spacing for r in grad_records if r.parameter_level == lvl and r.spacing is not None] for lvl in levels]
        axes[1].boxplot(sp_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#bfdbfe", color="#1d4ed8"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[1].set_title("GradNorm: Spacing on P13 Multi-Task", fontweight="bold")
        axes[1].set_ylabel("Schott's Spacing (Lower is Better)")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        igd_data = [[r.igd for r in grad_records if r.parameter_level == lvl and r.igd is not None] for lvl in levels]
        axes[2].boxplot(igd_data, tick_labels=x_labels, patch_artist=True,
                        boxprops=dict(facecolor="#fef08a", color="#a16207"),
                        medianprops=dict(color="#b91c1c", linewidth=2))
        axes[2].set_title("GradNorm: IGD on P13 Multi-Task", fontweight="bold")
        axes[2].set_ylabel("IGD (Lower is Better)")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        plt.savefig(out_dir / "task_c_gradnorm_alpha_sensitivity.png", dpi=200)
        plt.close(fig)

    # 4. Multi-panel combined summary figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    row_algs = ["NPGA", "MOACO", "GradNorm"]

    for row_idx, alg_name in enumerate(row_algs):
        alg_summary = [s for s in summary_rows if s["algorithm"] == alg_name]
        if not alg_summary:
            continue
        param = alg_summary[0]["parameter"]
        prob = alg_summary[0]["problem"]
        levels = [s["level"] for s in alg_summary]
        vals = [s["value"] for s in alg_summary]
        x_labs = [f"{lvl}\n({param}={v})" for lvl, v in zip(levels, vals)]

        # Col 0: HV
        hvs = [s["hv_mean"] for s in alg_summary]
        hv_err = [s["hv_std"] for s in alg_summary]
        axes[row_idx, 0].bar(x_labs, hvs, yerr=hv_err, capsize=4, color="#3b82f6", alpha=0.85)
        axes[row_idx, 0].set_title(f"{alg_name} ({prob}): Mean Hypervolume", fontweight="bold")
        axes[row_idx, 0].set_ylabel("Hypervolume")
        axes[row_idx, 0].grid(True, linestyle="--", alpha=0.4)

        # Col 1: Spacing
        sps = [s["spacing_mean"] if s["spacing_mean"] is not None else 0.0 for s in alg_summary]
        sp_err = [s["spacing_std"] if s["spacing_std"] is not None else 0.0 for s in alg_summary]
        axes[row_idx, 1].bar(x_labs, sps, yerr=sp_err, capsize=4, color="#10b981", alpha=0.85)
        axes[row_idx, 1].set_title(f"{alg_name} ({prob}): Mean Spacing", fontweight="bold")
        axes[row_idx, 1].set_ylabel("Schott's Spacing")
        axes[row_idx, 1].grid(True, linestyle="--", alpha=0.4)

        # Col 2: Runtime
        rts = [s["runtime_mean"] for s in alg_summary]
        rt_err = [s["runtime_std"] for s in alg_summary]
        axes[row_idx, 2].bar(x_labs, rts, yerr=rt_err, capsize=4, color="#f59e0b", alpha=0.85)
        axes[row_idx, 2].set_title(f"{alg_name} ({prob}): Mean Runtime (s)", fontweight="bold")
        axes[row_idx, 2].set_ylabel("Seconds")
        axes[row_idx, 2].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Task C Parameter Sensitivity Summary (10 Seeds per Setting)", fontsize=16, fontweight="bold")
    plt.savefig(out_dir / "task_c_all_metrics_comparison.png", dpi=200)
    plt.close(fig)
