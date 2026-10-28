"""
task_d_runner.py: Task D Head-to-Head Comparison Pipeline

Required Experiments (from Assignment PDF Section 5, Task D):
1. Continuous bi-objective:
   - P3 (unconstrained ZDT2, n=10): Compare representative algorithms across 10 seeds.
   - P5 (constrained Cantilever beam, 2 vars): Compare representative algorithms fairly with feasibility.
2. Many-objective:
   - P10 (DTLZ2, M=5): Compare NSGA-II, NSGA-III, MOEA/D, RVEA, and HypE across 10 seeds.
3. Discrete combinatorial:
   - P8 (Bi-Objective TSP, 15 cities): Compare MOEA/D-ACO, MOACO, NSGA-II, and MOMA across 10 seeds.
4. Non-front / Single-point methods & Preference Studies:
   - Algorithms 1-4: Parameter sweeps of at least 10 settings on P1 & P3.
   - Algorithms 5-7: Compromise solutions and preferences on P4, P11, P14.
   - Algorithms 8-9 (TOPSIS & VIKOR): Multi-criteria ranking of NSGA-II front, top 3, sensitivity to 0.1 weight shift.
   - Algorithms 36-37 (MGDA & GradNorm): Multi-task losses vs fixed equal weights on P13.

All stochastic comparisons use 10 independent seeds (42..51) and identical evaluation budgets.
Outputs raw per-run CSVs, summary CSVs, and comparison plots.
"""

from __future__ import annotations
import csv
import json
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
from moo.problems import get_problem, Problem
from moo.moo_utils import (
    normalize_objectives,
    hypervolume,
    spacing,
    igd,
    fast_non_dominated_sort,
)


@dataclass
class ComparisonRunRecord:
    experiment_id: str
    problem_key: str
    problem_name: str
    algorithm_id: int
    algorithm_name: str
    seed: int
    budget: int
    actual_evals: int
    n_points: int
    hv: float
    spacing: float | None
    igd: float | None
    igd_status: str
    runtime_sec: float
    status: str


# ==============================================================================
# 1. CORE HEAD-TO-HEAD EXPERIMENTAL PROTOCOL
# ==============================================================================

def execute_head_to_head_comparison(
    experiment_id: str,
    problem_key: str,
    alg_ids: Sequence[int],
    budget: int,
    seeds: Sequence[int] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    results_dir: Path | str = "results",
    plots_dir: Path | str = "plots",
    verbose: bool = True,
) -> tuple[list[ComparisonRunRecord], list[dict[str, Any]], dict[str, np.ndarray]]:
    """
    Executes a fair head-to-head comparison on a single benchmark problem across algorithms and seeds.
    """
    res_dir = Path(results_dir)
    plt_dir = Path(plots_dir)
    csv_dir = res_dir / "csv"
    comp_plot_dir = plt_dir / "comparisons"
    csv_dir.mkdir(parents=True, exist_ok=True)
    comp_plot_dir.mkdir(parents=True, exist_ok=True)

    prob = get_problem(problem_key)
    true_front = prob.get_true_front(200)

    raw_csv = csv_dir / f"task_d_{experiment_id.lower()}_raw.csv"
    summary_csv = csv_dir / f"task_d_{experiment_id.lower()}_summary.csv"
    plot_file = comp_plot_dir / f"task_d_{experiment_id.lower()}_fronts.png"

    if summary_csv.exists() and raw_csv.exists() and summary_csv.stat().st_size > 100 and plot_file.exists():
        if verbose:
            print(f"\nTask D Comparison: {experiment_id} on {prob.name} already completed ({summary_csv.name} found). Reusing existing results.")
        existing_summaries = []
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_summaries.append(row)
        return [], existing_summaries, {}

    if verbose:
        print(f"\n{'='*70}\nTask D Comparison: {experiment_id} on {prob.name} (Budget: {budget})\n{'='*70}")

    raw_records: list[ComparisonRunRecord] = []
    summary_rows: list[dict[str, Any]] = []
    combined_fronts: dict[str, np.ndarray] = {}

    for aid in alg_ids:
        alg_mod = get_algorithm(aid)
        alg_name = getattr(alg_mod, "__name__", f"Alg_{aid}").split(".")[-1]
        if hasattr(alg_mod, "initialize"):
            doc = alg_mod.__doc__ or ""
            if "alg" in alg_name:
                parts = alg_name.split("_", 1)
                alg_display = parts[1].upper() if len(parts) > 1 else alg_name
            else:
                alg_display = alg_name
        else:
            alg_display = f"Alg_{aid}"

        if verbose:
            print(f"  Testing {alg_display} (#{aid}) across {len(seeds)} seeds...")

        hvs = []
        spacings = []
        igds = []
        runtimes = []
        evals_list = []
        best_front_nat = None
        best_hv = -1.0

        for seed in seeds:
            t0 = time.perf_counter()
            res = alg_mod.run(prob, budget=budget, seed=seed)
            rt = time.perf_counter() - t0

            F_opt = res.get("F_opt", np.empty((0, prob.n_obj)))
            CV_opt = res.get("CV_opt", np.zeros(len(F_opt)))
            F_nat = res.get("F_opt_nat", None)
            if F_nat is None or len(F_nat) == 0:
                F_nat = prob.to_natural(F_opt) if len(F_opt) > 0 else np.empty((0, prob.n_obj))
            n_evals = res.get("n_evals", budget)

            # Feasibility filter
            if len(CV_opt) == len(F_opt) and len(F_opt) > 0:
                feas_mask = np.asarray([prob.is_feasible(cv) for cv in CV_opt], dtype=bool)
                F_feas = F_opt[feas_mask]
                F_feas_nat = F_nat[feas_mask]
            else:
                F_feas = F_opt
                F_feas_nat = F_nat

            n_feas = len(F_feas)

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

                if hv_val > best_hv and len(F_feas_nat) > 0:
                    best_hv = hv_val
                    best_front_nat = F_feas_nat.copy()

            rec = ComparisonRunRecord(
                experiment_id=experiment_id,
                problem_key=problem_key,
                problem_name=prob.name,
                algorithm_id=aid,
                algorithm_name=alg_display,
                seed=seed,
                budget=budget,
                actual_evals=n_evals,
                n_points=n_feas,
                hv=hv_val,
                spacing=sp_val,
                igd=igd_val,
                igd_status=igd_stat,
                runtime_sec=rt,
                status="works" if n_feas > 0 else "partly works",
            )
            raw_records.append(rec)

            hvs.append(hv_val)
            if sp_val is not None:
                spacings.append(sp_val)
            if igd_val is not None:
                igds.append(igd_val)
            runtimes.append(rt)
            evals_list.append(n_evals)

        if best_front_nat is not None:
            combined_fronts[alg_display] = best_front_nat

        summary_rows.append({
            "experiment": experiment_id,
            "problem": problem_key,
            "algorithm": alg_display,
            "algorithm_id": aid,
            "runs": len(seeds),
            "budget": budget,
            "evals_mean": float(np.mean(evals_list)),
            "hv_mean": float(np.mean(hvs)),
            "hv_std": float(np.std(hvs)),
            "spacing_mean": float(np.mean(spacings)) if spacings else None,
            "spacing_std": float(np.std(spacings)) if spacings else None,
            "igd_mean": float(np.mean(igds)) if igds else None,
            "igd_std": float(np.std(igds)) if igds else None,
            "runtime_mean": float(np.mean(runtimes)),
            "runtime_std": float(np.std(runtimes)),
        })

    # Save raw CSV
    raw_csv = csv_dir / f"task_d_{experiment_id.lower()}_raw.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in ComparisonRunRecord.__dataclass_fields__.values()])
        writer.writeheader()
        for r in raw_records:
            writer.writerow(asdict(r))

    # Save summary CSV
    summary_csv = csv_dir / f"task_d_{experiment_id.lower()}_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment", "problem", "algorithm", "algorithm_id", "runs", "budget", "evals_mean",
            "hv_mean", "hv_std", "spacing_mean", "spacing_std",
            "igd_mean", "igd_std", "runtime_mean", "runtime_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary_rows:
            writer.writerow(s)

    # Plot overlay fronts
    plot_combined_comparison(experiment_id, prob, combined_fronts, summary_rows, comp_plot_dir / f"task_d_{experiment_id.lower()}_fronts.png")

    return raw_records, summary_rows, combined_fronts


def plot_combined_comparison(
    experiment_id: str,
    problem: Problem,
    fronts: dict[str, np.ndarray],
    summary_rows: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Plots combined Pareto fronts comparing algorithms on a single problem."""
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Plot true front if available and 2D
    if problem.n_obj == 2:
        tf = problem.get_true_front(200)
        if tf is not None and len(tf) > 0:
            tf_nat = problem.to_natural(tf)
            order = np.argsort(tf_nat[:, 0])
            ax.plot(tf_nat[order, 0], tf_nat[order, 1], "k--", linewidth=1.5, label="True Front", alpha=0.7)

    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#0891b2", "#4f46e5", "#ca8a04"]
    for i, (alg_name, pts) in enumerate(fronts.items()):
        if pts is None or len(pts) == 0:
            continue
        c = palette[i % len(palette)]
        if problem.n_obj == 2:
            ax.scatter(pts[:, 0], pts[:, 1], color=c, label=alg_name, s=35, alpha=0.85, edgecolors="none")
        elif problem.n_obj >= 3:
            # For M >= 3, plot projection onto first two natural objectives
            ax.scatter(pts[:, 0], pts[:, 1], color=c, label=f"{alg_name} (f1-f2)", s=35, alpha=0.85)

    ax.set_title(f"Task D: {experiment_id} Comparison on {problem.name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Objective 1 (Natural)", fontsize=10)
    ax.set_ylabel("Objective 2 (Natural)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=9)

    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ==============================================================================
# 2. SPECIAL HANDLINGS (ALGS 1-4, 5-7, 8-9, 36-37)
# ==============================================================================

def analyze_algs_1_to_4_sweeps() -> dict[str, Any]:
    """
    Executes the required parameter sweeps of at least 10 settings for Algorithms 1-4 on P1 and P3.
    """
    from moo.algorithms import alg01_weighted_sum, alg02_epsilon_constraint, alg03_benson, alg04_weight_metric
    from moo.problems import FactorySpeedProblem, ZDT2Problem

    prob_p1 = FactorySpeedProblem()
    prob_p3 = ZDT2Problem()

    # 1. Weighted Sum sweep (10 weights)
    res_ws_p1 = alg01_weighted_sum.run(prob_p1, budget=1000, seed=42, n_weights=11)
    res_ws_p3 = alg01_weighted_sum.run(prob_p3, budget=1000, seed=42, n_weights=11)

    # 2. epsilon-Constraint sweep (10 epsilons)
    res_ec_p1 = alg02_epsilon_constraint.run(prob_p1, budget=1000, seed=42, n_eps=11)
    res_ec_p3 = alg02_epsilon_constraint.run(prob_p3, budget=1000, seed=42, n_eps=11)

    # 3. Benson's method sweep
    res_ben_p1 = alg03_benson.run(prob_p1, budget=1000, seed=42, n_weights=11)
    res_ben_p3 = alg03_benson.run(prob_p3, budget=1000, seed=42, n_weights=11)

    # 4. Weight Metric sweep (p=1, 2, inf)
    res_wm_p1 = alg04_weight_metric.run(prob_p1, budget=1000, seed=42, n_weights=11, p=2.0)
    res_wm_p3 = alg04_weight_metric.run(prob_p3, budget=1000, seed=42, n_weights=11, p=2.0)

    return {
        "p1": {
            "ws_points": len(res_ws_p1["X_opt"]),
            "ec_points": len(res_ec_p1["X_opt"]),
            "ben_points": len(res_ben_p1["X_opt"]),
            "wm_points": len(res_wm_p1["X_opt"]),
        },
        "p3_non_convex": {
            "ws_points": len(res_ws_p1["X_opt"]),
            "ws_f1_spread": [float(np.min(res_ws_p3["F_opt"][:, 0])), float(np.max(res_ws_p3["F_opt"][:, 0]))],
            "ec_points": len(res_ec_p3["X_opt"]),
            "ec_reaches_middle": bool(np.any((res_ws_p3["F_opt"][:, 0] > 0.2) & (res_ws_p3["F_opt"][:, 0] < 0.8))),
        },
    }


def analyze_algs_5_to_7_preferences() -> dict[str, Any]:
    """
    Executes and reports compromise solutions for preference/priority methods:
    - Alg 5 (Value Function) on P4 & P14
    - Alg 6 (Goal Programming) on P11 & P7
    - Alg 7 (Lexicographic) on P11 & P7
    """
    from moo.algorithms import alg05_value_function, alg06_goal_programming, alg07_lexicographic
    from moo.problems import PortfolioProblem, ProductionPlanningProblem, LaptopDesignProblem, WaterAllocationProblem

    p4 = PortfolioProblem()
    p11 = ProductionPlanningProblem()
    p14 = LaptopDesignProblem()

    # Alg 5 on P4 (risk-averse utility) & P14 (DM preferred weight/runtime)
    res_v_p4 = alg05_value_function.run(p4, budget=500, seed=42)
    res_v_p14 = alg05_value_function.run(p14, budget=500, seed=42)

    # Alg 6 on P11 (goals: profit >= 2400, units >= 600)
    res_gp_p11 = alg06_goal_programming.run(p11, budget=500, seed=42)

    # Alg 7 on P11 (profit priority 1, units priority 2)
    res_lex_p11 = alg07_lexicographic.run(p11, budget=500, seed=42)

    return {
        "alg05_p4_portfolio": {
            "weights": res_v_p4["X_opt"][0].tolist(),
            "f_natural": res_v_p4["F_opt_nat"][0].tolist(),
        },
        "alg05_p14_laptop": {
            "specs": res_v_p14["X_opt"][0].tolist(),
            "f_natural": res_v_p14["F_opt_nat"][0].tolist(),
        },
        "alg06_p11_goal_programming": {
            "solution": res_gp_p11["X_opt"][0].tolist(),
            "f_natural": res_gp_p11["F_opt_nat"][0].tolist(),
            "profit": float(res_gp_p11["F_opt_nat"][0][0]),
            "units": float(res_gp_p11["F_opt_nat"][0][1]),
            "shortfall_units": float(max(0.0, 600.0 - res_gp_p11["F_opt_nat"][0][1])),
        },
        "alg07_p11_lexicographic": {
            "solution": res_lex_p11["X_opt"][0].tolist(),
            "f_natural": res_lex_p11["F_opt_nat"][0].tolist(),
            "profit": float(res_lex_p11["F_opt_nat"][0][0]),
            "units": float(res_lex_p11["F_opt_nat"][0][1]),
        },
    }


def analyze_topsis_and_vikor_sensitivity() -> dict[str, Any]:
    """
    Ranks the NSGA-II Pareto front with TOPSIS and VIKOR.
    Reports top 3 solutions under each, shifts one weight by 0.1, and tests whether the ranking flips.
    """
    from moo.algorithms import alg08_topsis, alg09_vikor, alg14_nsga2
    from moo.problems import SupplierSelectionProblem, PortfolioProblem

    # Obtain Pareto front from P12 / P4
    prob_p12 = SupplierSelectionProblem()
    raw_table = getattr(prob_p12, "raw_data", None)
    if raw_table is None:
        from moo.problems.p12_supplier import RAW_MATRIX
        raw_table = RAW_MATRIX
    weights_base = np.array([0.3, 0.3, 0.2, 0.2])
    types = getattr(prob_p12, "criteria_min_max", ["min", "max", "min", "max"])

    top_base = alg08_topsis.topsis_rank(raw_table, weights_base, types)
    vik_base = alg09_vikor.vikor_rank(raw_table, weights_base, types, v=0.5)

    # Shift weight 0 by +0.1 and renormalize
    weights_shifted = np.array([0.4, 0.3, 0.2, 0.1])
    weights_shifted = weights_shifted / np.sum(weights_shifted)

    top_shift = alg08_topsis.topsis_rank(raw_table, weights_shifted, types)
    vik_shift = alg09_vikor.vikor_rank(raw_table, weights_shifted, types, v=0.5)

    return {
        "suppliers": [f"S{i+1}" for i in range(len(raw_table))],
        "base_weights": weights_base.tolist(),
        "shifted_weights": weights_shifted.tolist(),
        "topsis": {
            "base_scores": top_base["scores"].tolist(),
            "base_rank": top_base["rankings"].tolist(),
            "base_top3": [f"S{i+1}" for i in top_base["rankings"][:3]],
            "shift_scores": top_shift["scores"].tolist(),
            "shift_rank": top_shift["rankings"].tolist(),
            "shift_top3": [f"S{i+1}" for i in top_shift["rankings"][:3]],
            "ranking_flipped": not np.array_equal(top_base["rankings"], top_shift["rankings"]),
        },
        "vikor": {
            "base_Q": vik_base["Q"].tolist(),
            "base_rank": vik_base["rank_Q"].tolist(),
            "base_top3": [f"S{i+1}" for i in vik_base["rank_Q"][:3]],
            "shift_Q": vik_shift["Q"].tolist(),
            "shift_rank": vik_shift["rank_Q"].tolist(),
            "shift_top3": [f"S{i+1}" for i in vik_shift["rank_Q"][:3]],
            "ranking_flipped": not np.array_equal(vik_base["rank_Q"], vik_shift["rank_Q"]),
        },
    }


def analyze_mgda_and_gradnorm_on_p13() -> dict[str, Any]:
    """
    Compares task losses of MGDA (#36) and GradNorm (#37) against fixed equal weights on P13(a).
    """
    from moo.algorithms import alg36_mgda, alg37_gradnorm
    from moo.problems import MultiTaskToyProblem

    prob = MultiTaskToyProblem()

    # 1. Fixed Equal Weights gradient descent
    x = np.array([0.0, 5.0], dtype=float)
    losses_equal = []
    lr = 0.05
    for _ in range(50):
        f, _, _ = prob.evaluate(x)
        losses_equal.append(f.tolist())
        g1, g2 = prob.gradients(x)
        grad_equal = 0.5 * g1 + 0.5 * g2
        x = x - lr * grad_equal

    # 2. MGDA run
    res_mgda = alg36_mgda.run(prob, budget=250, seed=42)

    # 3. GradNorm run
    res_grad = alg37_gradnorm.run(prob, budget=250, seed=42, alpha=1.5)

    return {
        "equal_weights_final_loss": losses_equal[-1],
        "mgda_final_loss": res_mgda["F_opt"][0].tolist() if len(res_mgda["F_opt"]) > 0 else [],
        "gradnorm_final_loss": res_grad["F_opt"][0].tolist() if len(res_grad["F_opt"]) > 0 else [],
        "gradnorm_archive_size": len(res_grad["F_opt"]),
        "mgda_evals": res_mgda["n_evals"],
        "gradnorm_evals": res_grad["n_evals"],
    }


# ==============================================================================
# 3. MASTER TASK D SUITE DISPATCHER
# ==============================================================================

def run_task_d_suite(
    results_dir: Path | str = "results",
    plots_dir: Path | str = "plots",
    seeds: Sequence[int] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Runs the full Task D comparison suite required by the assignment:
    1. Continuous bi-objective on P3 (unconstrained ZDT2)
    2. Continuous bi-objective on P5 (constrained Cantilever)
    3. Many-objective on P10 (M=5 DTLZ2)
    4. Discrete combinatorial on P8 (TSP 15-city)
    5. Special studies: Algs 1-4, 5-7, 8-9 (TOPSIS/VIKOR), and 36-37 (MGDA/GradNorm)
    """
    res_dir = Path(results_dir)
    plt_dir = Path(plots_dir)

    # 1. P3 unconstrained: compare Alg 1, 2, 10, 11, 13, 14, 18, 19, 27, 33
    raw_p3, sum_p3, fronts_p3 = execute_head_to_head_comparison(
        experiment_id="P3_Unconstrained",
        problem_key="P3",
        alg_ids=[1, 2, 10, 11, 13, 14, 18, 19, 27, 33],
        budget=1500,
        seeds=seeds,
        results_dir=res_dir,
        plots_dir=plt_dir,
        verbose=verbose,
    )

    # 2. P5 constrained: compare Alg 2, 13, 14, 16, 21, 22, 23, 27
    raw_p5, sum_p5, fronts_p5 = execute_head_to_head_comparison(
        experiment_id="P5_Constrained",
        problem_key="P5",
        alg_ids=[2, 13, 14, 16, 21, 22, 23, 27],
        budget=1500,
        seeds=seeds,
        results_dir=res_dir,
        plots_dir=plt_dir,
        verbose=verbose,
    )

    # 3. P10 many-objective (M=5): compare NSGA-II (14), NSGA-III (15), MOEA/D (18), RVEA (31), HypE (34)
    raw_p10, sum_p10, fronts_p10 = execute_head_to_head_comparison(
        experiment_id="P10_ManyObj_M5",
        problem_key="P10_M5",
        alg_ids=[14, 15, 18, 31, 34],
        budget=2000,
        seeds=seeds,
        results_dir=res_dir,
        plots_dir=plt_dir,
        verbose=verbose,
    )

    # 4. P8 discrete combinatorial TSP: compare MOEA/D-ACO (20), MOACO (25), NSGA-II (14), MOMA (30)
    raw_p8, sum_p8, fronts_p8 = execute_head_to_head_comparison(
        experiment_id="P8_Discrete_TSP",
        problem_key="P8",
        alg_ids=[20, 25, 14, 30],
        budget=1000,
        seeds=seeds,
        results_dir=res_dir,
        plots_dir=plt_dir,
        verbose=verbose,
    )

    # 5. Special handling studies
    if verbose:
        print("\nRunning special handling analyses (Algs 1-4, 5-7, 8-9, 36-37)...")
    sweeps_1_4 = analyze_algs_1_to_4_sweeps()
    pref_5_7 = analyze_algs_5_to_7_preferences()
    mcdm_8_9 = analyze_topsis_and_vikor_sensitivity()
    grad_36_37 = analyze_mgda_and_gradnorm_on_p13()

    # Master summary table across all Task D comparisons
    all_summaries = sum_p3 + sum_p5 + sum_p10 + sum_p8
    master_csv = res_dir / "csv" / "task_d_all_comparisons_summary.csv"
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment", "problem", "algorithm", "algorithm_id", "runs", "budget", "evals_mean",
            "hv_mean", "hv_std", "spacing_mean", "spacing_std",
            "igd_mean", "igd_std", "runtime_mean", "runtime_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(s)

    special_json = res_dir / "csv" / "task_d_special_studies.json"
    with open(special_json, "w", encoding="utf-8") as f:
        json.dump({
            "sweeps_1_4": sweeps_1_4,
            "pref_5_7": pref_5_7,
            "mcdm_8_9": mcdm_8_9,
            "grad_36_37": grad_36_37,
        }, f, indent=2)

    return {
        "p3_summary": sum_p3,
        "p5_summary": sum_p5,
        "p10_summary": sum_p10,
        "p8_summary": sum_p8,
        "sweeps_1_4": sweeps_1_4,
        "pref_5_7": pref_5_7,
        "mcdm_8_9": mcdm_8_9,
        "grad_36_37": grad_36_37,
    }
