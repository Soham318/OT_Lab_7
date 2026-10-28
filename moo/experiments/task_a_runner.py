"""
task_a_runner.py: Task A Experiment Pipeline and Implementation Log Generator

Assignment Requirements:
- Run all 37 algorithms on at least two designated applicable problems each
  (or one if the assignment lists only one).
- Use reproducible, explicitly recorded random seeds.
- Use a configurable evaluation budget.
- Record actual evaluations, runtime, status, and metrics (HV, Spacing, IGD).
- For IGD, use valid known reference front. If unavailable, record as N/A with explanation.
- Handle constrained, continuous, many-objective, discrete, and preference-based methods.
- Generate Pareto-front plots, including true fronts where known.
- Save raw CSV results under results/csv/ and plots under plots/fronts/.
- Generate the Task A3 implementation log table.
"""

from __future__ import annotations
import csv
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from moo.algorithms import get_algorithm
from moo.problems import get_problem, Problem
from moo.moo_utils import (
    normalize_objectives,
    hypervolume,
    spacing,
    igd,
)
from moo.experiments.plot_generator import plot_pareto_front


# Exact assignment catalogue test-problem mapping from Section 3 of assignment PDF
TASK_A_CATALOG_PROBLEMS: dict[int, list[str]] = {
    1: ["P1", "P3"],           # Weighted Sum: P1, P3, P4
    2: ["P2", "P3"],           # epsilon-Constraint: P2, P3, P5
    3: ["P1", "P7"],           # Benson's Method: P1, P7
    4: ["P1", "P3"],           # Weight Metric: P1, P3
    5: ["P4", "P14"],          # Value Function: P4, P14
    6: ["P11", "P7"],          # Goal Programming: P11, P7
    7: ["P11", "P7"],          # Lexicographic: P11, P7
    8: ["P12", "P4"],          # TOPSIS: P12, plus the front from P4
    9: ["P12", "P4"],          # VIKOR: P12, plus the front from P4
    10: ["P2", "P3"],          # VEGA: P2, P3
    11: ["P3", "P5"],          # MOGA: P3, P5
    12: ["P5", "P7"],          # NSGA: P5, P7
    13: ["P3", "P5"],          # NPGA: P3, P5
    14: ["P3", "P5"],          # NSGA-II: P3, P5, P6
    15: ["P10_M3", "P10_M5"],  # NSGA-III: P10 (M = 3 and 5)
    16: ["P4", "P5"],          # SPEA2: P4, P5
    17: ["P3", "P5"],          # PESA-II: P3, P5
    18: ["P3", "P10_M3"],      # MOEA/D: P3, P10
    19: ["P3", "P5"],          # MOEA/D-DE: P3, P5
    20: ["P8"],                # MOEA/D-ACO: P8
    21: ["P5", "P6"],          # Constrained NSGA-II: P5, P6, P7
    22: ["P4", "P5"],          # MOPSO: P4, P5
    23: ["P5", "P9"],          # MOGWO: P5, P9
    24: ["P3", "P6"],          # MOABC: P3, P6
    25: ["P8"],                # MOACO: P8
    26: ["P3", "P5"],          # MOSA: P3, P5
    27: ["P3", "P5"],          # MODE: P3, P5
    28: ["P3", "P5"],          # MO-TLBO: P3, P5
    29: ["P3", "P4"],          # MOBA: P3, P4
    30: ["P5", "P9"],          # MOMA: P5, P9
    31: ["P10_M3"],            # RVEA: P10
    32: ["P3", "P14"],         # IBEA: P3, P14
    33: ["P3", "P14"],         # SMS-EMOA: P3, P14
    34: ["P10_M3"],            # HypE: P10
    35: ["P5", "P14"],         # R-NSGA-II: P5, P14
    36: ["P13"],               # MGDA: P13
    37: ["P13"],               # GradNorm: P13
}

# Full listing of problems if user requests --all-problems
TASK_A_ALL_PROBLEMS: dict[int, list[str]] = {
    1: ["P1", "P3", "P4"],
    2: ["P2", "P3", "P5"],
    3: ["P1", "P7"],
    4: ["P1", "P3"],
    5: ["P4", "P14"],
    6: ["P11", "P7"],
    7: ["P11", "P7"],
    8: ["P12", "P4"],
    9: ["P12", "P4"],
    10: ["P2", "P3"],
    11: ["P3", "P5"],
    12: ["P5", "P7"],
    13: ["P3", "P5"],
    14: ["P3", "P5", "P6"],
    15: ["P10_M3", "P10_M5"],
    16: ["P4", "P5"],
    17: ["P3", "P5"],
    18: ["P3", "P10_M3"],
    19: ["P3", "P5"],
    20: ["P8"],
    21: ["P5", "P6", "P7"],
    22: ["P4", "P5"],
    23: ["P5", "P9"],
    24: ["P3", "P6"],
    25: ["P8"],
    26: ["P3", "P5"],
    27: ["P3", "P5"],
    28: ["P3", "P5"],
    29: ["P3", "P4"],
    30: ["P5", "P9"],
    31: ["P10_M3"],
    32: ["P3", "P14"],
    33: ["P3", "P14"],
    34: ["P10_M3"],
    35: ["P5", "P14"],
    36: ["P13"],
    37: ["P13"],
}

ALGORITHM_DISPLAY_NAMES: dict[int, str] = {
    1: "Weighted Sum",
    2: "epsilon-Constraint",
    3: "Benson's Method",
    4: "Weight Metric",
    5: "Value Function",
    6: "Goal Programming",
    7: "Lexicographic",
    8: "TOPSIS",
    9: "VIKOR",
    10: "VEGA",
    11: "MOGA",
    12: "NSGA",
    13: "NPGA",
    14: "NSGA-II",
    15: "NSGA-III",
    16: "SPEA2",
    17: "PESA-II",
    18: "MOEA/D",
    19: "MOEA/D-DE",
    20: "MOEA/D-ACO",
    21: "Constrained NSGA-II",
    22: "MOPSO",
    23: "MOGWO",
    24: "MOABC",
    25: "MOACO",
    26: "MOSA",
    27: "MODE",
    28: "MO-TLBO",
    29: "MOBA",
    30: "MOMA",
    31: "RVEA",
    32: "IBEA",
    33: "SMS-EMOA",
    34: "HypE",
    35: "R-NSGA-II",
    36: "MGDA",
    37: "GradNorm",
}


def safe_print(msg: str, end: str = "\n", flush: bool = False) -> None:
    """Prints message safely handling Windows cp1252 / console encoding limitations."""
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try:
        print(msg, end=end, flush=flush)
    except UnicodeEncodeError:
        print(msg.encode(enc, errors="replace").decode(enc), end=end, flush=flush)


@dataclass
class SingleRunResult:
    alg_id: int
    alg_name: str
    problem_key: str
    problem_name: str
    budget: int
    seed: int
    actual_evals: int
    n_points: int
    hv: float
    spacing: float | None
    igd: float | None
    igd_status: str
    igd_reason: str
    runtime_sec: float
    status: str  # 'works', 'partly works', 'not done'
    plot_file: str
    front_csv_file: str
    error_msg: str
    n_infeasible: int = 0


@dataclass
class ImplementationLogEntry:
    number: int
    algorithm: str
    problems_used: str
    evaluation_budget: str
    hv: str
    igd: str
    runtime: str
    status: str


def execute_single_task_a_run(
    alg_id: int,
    prob_key: str,
    budget: int = 5000,
    seed: int = 42,
    results_dir: Path | str = "results",
    plots_dir: Path | str = "plots",
    generate_plot: bool = True,
    save_front_csv: bool = True,
) -> SingleRunResult:
    """
    Executes a single algorithm on a specific benchmark problem, computing metrics,
    saving Pareto front CSVs, and generating plots.
    """
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    csv_fronts_dir = results_dir / "fronts"
    plot_fronts_dir = plots_dir / "fronts"

    csv_fronts_dir.mkdir(parents=True, exist_ok=True)
    plot_fronts_dir.mkdir(parents=True, exist_ok=True)

    alg_name = ALGORITHM_DISPLAY_NAMES.get(alg_id, f"Algorithm {alg_id}")
    slug = (
        alg_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("ε", "eps")
        .replace("’", "")
        .replace("-", "_")
    )

    # Load problem
    try:
        problem = get_problem(prob_key)
    except Exception as e:
        return SingleRunResult(
            alg_id=alg_id,
            alg_name=alg_name,
            problem_key=prob_key,
            problem_name=prob_key,
            budget=budget,
            seed=seed,
            actual_evals=0,
            n_points=0,
            hv=0.0,
            spacing=None,
            igd=None,
            igd_status="error",
            igd_reason=f"Problem loading failed: {e}",
            runtime_sec=0.0,
            status="not done",
            plot_file="",
            front_csv_file="",
            error_msg=str(e),
            n_infeasible=0,
        )

    # Load algorithm
    try:
        alg_module = get_algorithm(alg_id)
    except Exception as e:
        return SingleRunResult(
            alg_id=alg_id,
            alg_name=alg_name,
            problem_key=prob_key,
            problem_name=problem.name,
            budget=budget,
            seed=seed,
            actual_evals=0,
            n_points=0,
            hv=0.0,
            spacing=None,
            igd=None,
            igd_status="error",
            igd_reason=f"Algorithm module loading failed: {e}",
            runtime_sec=0.0,
            status="not done",
            plot_file="",
            front_csv_file="",
            error_msg=str(e),
            n_infeasible=0,
        )

    # Execute algorithm
    t0 = time.perf_counter()
    run_err = ""
    status = "works"
    res: dict[str, Any] = {}

    try:
        res = alg_module.run(problem, budget=budget, seed=seed)
        runtime = time.perf_counter() - t0
    except Exception as e:
        runtime = time.perf_counter() - t0
        run_err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        status = "not done"

    actual_evals = int(res.get("n_evals", 0)) if "n_evals" in res else budget
    X_opt = res.get("X_opt", np.empty((0, problem.n_var)))
    F_opt = res.get("F_opt", np.empty((0, problem.n_obj)))
    N_opt = len(F_opt) if F_opt is not None else 0

    F_opt_nat = res.get("F_opt_nat", None)
    if (F_opt_nat is None or len(F_opt_nat) == 0) and N_opt > 0:
        F_opt_nat = problem.to_natural(F_opt)
    elif F_opt_nat is None:
        F_opt_nat = np.empty((0, problem.n_obj))

    CV_opt = res.get("CV_opt", None)
    if (CV_opt is None or len(CV_opt) == 0) and N_opt > 0:
        CV_opt = np.zeros(N_opt, dtype=float)
    elif CV_opt is None:
        CV_opt = np.empty(0, dtype=float)

    # Feasibility filtering: Only solutions satisfying problem feasibility tolerance are valid
    if problem.n_constr > 0 and CV_opt is not None and len(CV_opt) == N_opt and N_opt > 0:
        feas_mask = np.asarray([problem.is_feasible(cv) for cv in CV_opt], dtype=bool)
    else:
        feas_mask = np.ones(N_opt, dtype=bool)

    N_feas = int(np.sum(feas_mask))
    N_infeas = N_opt - N_feas

    # Apply the mask consistently to decision vectors, objective vectors, constraint violations, and natural coordinates
    X_feas = X_opt[feas_mask] if (X_opt is not None and len(X_opt) == N_opt) else X_opt
    F_feas = F_opt[feas_mask] if (F_opt is not None and len(F_opt) == N_opt) else F_opt
    F_feas_nat = F_opt_nat[feas_mask] if (F_opt_nat is not None and len(F_opt_nat) == N_opt) else F_opt_nat
    CV_feas = CV_opt[feas_mask] if (CV_opt is not None and len(CV_opt) == N_opt) else CV_opt

    if status != "not done":
        if N_opt == 0:
            status = "partly works"
            run_err = "0 non-dominated solutions found within budget"
        elif N_feas == 0 and problem.n_constr > 0:
            status = "partly works"
            min_cv = float(np.min(CV_opt)) if len(CV_opt) > 0 else 0.0
            run_err = f"0 feasible solutions found within budget ({N_opt} infeasible solutions discarded, min CV={min_cv:.4e})"

    # Compute metrics strictly on feasible solutions
    hv_val = 0.0
    spacing_val: float | None = None
    igd_val: float | None = None
    igd_status = "unavailable"
    igd_reason = "No analytical true front available"

    if N_feas > 0 and F_feas is not None:
        try:
            # 1. Hypervolume: normalize feasible points to [0, 1] then reference point 1.1^M
            F_norm = normalize_objectives(F_feas, ideal=problem.ideal_point, nadir=problem.nadir_point)
            ref_norm = np.ones(problem.n_obj) * 1.1
            hv_val = float(hypervolume(F_norm, ref_norm))
        except Exception:
            hv_val = 0.0

        try:
            # 2. Schott's Spacing: defined only for N >= 2 points
            if N_feas > 1:
                F_norm = normalize_objectives(F_feas, ideal=problem.ideal_point, nadir=problem.nadir_point)
                spacing_val = float(spacing(F_norm))
            else:
                spacing_val = None
        except Exception:
            spacing_val = None

        try:
            # 3. Inverted Generational Distance (IGD)
            true_front = problem.get_true_front(n_points=200)
            if true_front is not None and len(true_front) > 0:
                TF_norm = normalize_objectives(true_front, ideal=problem.ideal_point, nadir=problem.nadir_point)
                F_norm = normalize_objectives(F_feas, ideal=problem.ideal_point, nadir=problem.nadir_point)
                igd_val = float(igd(F_norm, TF_norm))
                igd_status = "computed"
                igd_reason = "Computed against analytical true Pareto front"
            else:
                igd_val = None
                igd_status = "unavailable"
                igd_reason = "Problem has no known analytical front; not fabricated"
        except Exception as e:
            igd_val = None
            igd_status = "error"
            igd_reason = f"IGD calculation error: {e}"
    elif N_opt > 0 and N_feas == 0:
        hv_val = 0.0
        spacing_val = None
        igd_val = None
        igd_status = "unavailable"
        igd_reason = "No feasible solutions found within budget"

    # Save front coordinates CSV
    front_csv_file = ""
    if save_front_csv and (N_feas > 0 or N_opt > 0):
        front_csv_path = csv_fronts_dir / f"alg{alg_id:02d}_{slug}_{prob_key}_front.csv"
        try:
            with open(front_csv_path, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                header = [f"f{m+1}_natural" for m in range(problem.n_obj)]
                header += [f"f{m+1}_internal" for m in range(problem.n_obj)]
                header.append("CV")
                header.append("is_feasible")
                writer.writerow(header)
                if N_feas > 0 and F_feas_nat is not None and len(F_feas_nat) >= N_feas and len(F_feas) >= N_feas:
                    for idx in range(N_feas):
                        row = list(F_feas_nat[idx]) + list(F_feas[idx])
                        cv_val = float(CV_feas[idx]) if (CV_feas is not None and idx < len(CV_feas)) else 0.0
                        row.append(cv_val)
                        row.append(True)
                        writer.writerow(row)
                elif N_opt > 0 and F_opt_nat is not None and len(F_opt_nat) >= N_opt and len(F_opt) >= N_opt:
                    # Infeasible run: document least-infeasible points with is_feasible=False
                    for idx in range(N_opt):
                        row = list(F_opt_nat[idx]) + list(F_opt[idx])
                        cv_val = float(CV_opt[idx]) if (CV_opt is not None and idx < len(CV_opt)) else 0.0
                        row.append(cv_val)
                        row.append(False)
                        writer.writerow(row)
            front_csv_file = str(front_csv_path)
        except Exception as e:
            front_csv_file = f"Error saving CSV: {e}"

    # Generate Pareto front plot using ONLY feasible solutions
    plot_file = ""
    if generate_plot:
        plot_path = plot_fronts_dir / f"alg{alg_id:02d}_{slug}_{prob_key}.png"
        try:
            true_front_nat = problem.get_true_front_natural(n_points=200)
            plot_pareto_front(
                F_opt_nat=F_feas_nat if (N_feas > 0 and F_feas_nat is not None) else np.empty((0, problem.n_obj)),
                problem_name=problem.name,
                alg_name=alg_name,
                save_path=plot_path,
                true_front_nat=true_front_nat,
                hv=hv_val if N_feas > 0 else None,
                spacing_val=spacing_val if (N_feas > 1 and spacing_val is not None) else None,
                igd_val=igd_val,
                budget=budget,
                evals_used=actual_evals,
                seed=seed,
            )
            plot_file = str(plot_path)
        except Exception as e:
            plot_file = f"Error generating plot: {e}"

    return SingleRunResult(
        alg_id=alg_id,
        alg_name=alg_name,
        problem_key=prob_key,
        problem_name=problem.name,
        budget=budget,
        seed=seed,
        actual_evals=actual_evals,
        n_points=N_feas,
        hv=hv_val,
        spacing=spacing_val,
        igd=igd_val,
        igd_status=igd_status,
        igd_reason=igd_reason,
        runtime_sec=runtime,
        status=status,
        plot_file=plot_file,
        front_csv_file=front_csv_file,
        error_msg=run_err,
        n_infeasible=N_infeas,
    )


# Representative smoke-test matrix covering all 7 major benchmark problem archetypes:
# Continuous, Constrained, Many-Objective (3D & 5D), Discrete TSP, Binary, Supplier MCDM, Multi-Task
SMOKE_TEST_MATRIX: list[tuple[int, str]] = [
    (1, "P1"),       # Continuous bi-objective unconstrained (Weighted Sum on Factory Speed)
    (21, "P5"),      # Constrained continuous (Constrained NSGA-II on Cantilever Beam)
    (15, "P10_M3"),  # 3D Many-objective (NSGA-III on DTLZ2 M=3)
    (18, "P10_M5"),  # 5D Many-objective (MOEA/D on DTLZ2 M=5, Monte Carlo HV & parallel coords)
    (25, "P8"),      # Discrete permutation TSP (MOACO on Delivery Routing TSP)
    (23, "P9"),      # Binary feature selection (MOGWO on WDBC Feature Selection)
    (8, "P12"),      # MCDM ranking (TOPSIS on Supplier Selection)
    (36, "P13"),     # Differentiable multi-task (MGDA on MultiTaskToy)
]


def run_task_a(
    algs: Sequence[int] | None = None,
    all_problems: bool = False,
    smoke_matrix: bool = False,
    budget: int = 5000,
    seed: int = 42,
    results_dir: Path | str = "results",
    plots_dir: Path | str = "plots",
    verbose: bool = True,
) -> tuple[list[SingleRunResult], list[ImplementationLogEntry]]:
    """
    Executes the complete Task A experiment pipeline across all requested algorithms and problems.
    Generates:
    - results/csv/task_a_runs_detail.csv (fine-grained single run records)
    - results/csv/implementation_log.csv (Table A3 master implementation log)
    - results/csv/implementation_log.md (Markdown format for easy report inclusion)
    - results/fronts/ (all obtained Pareto front points)
    - plots/fronts/ (all generated Pareto front figures)
    """
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    csv_dir = results_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    if smoke_matrix:
        # Group representative smoke test matrix by algorithm ID preserving order
        prob_map: dict[int, list[str]] = {}
        for aid, pkey in SMOKE_TEST_MATRIX:
            prob_map.setdefault(aid, []).append(pkey)
        target_algs = list(prob_map.keys())
    elif all_problems:
        prob_map = TASK_A_ALL_PROBLEMS
        target_algs = list(algs) if algs is not None else list(range(1, 38))
    else:
        prob_map = TASK_A_CATALOG_PROBLEMS
        target_algs = list(algs) if algs is not None else list(range(1, 38))

    all_run_results: list[SingleRunResult] = []
    log_entries: list[ImplementationLogEntry] = []

    total_runs = sum(len(prob_map.get(aid, [])) for aid in target_algs)
    current_run = 0

    if verbose:
        safe_print("=" * 78)
        safe_print("TASK A EXPERIMENT RUNNER -- MULTIOBJECTIVE OPTIMIZATION LAB 7")
        safe_print(f"Target Algorithms: {len(target_algs)} (IDs: {min(target_algs)} to {max(target_algs)})")
        safe_print(f"Total Scheduled Runs: {total_runs} | Budget: {budget} evals | Seed: {seed}")
        safe_print("=" * 78)

    for aid in target_algs:
        alg_name = ALGORITHM_DISPLAY_NAMES.get(aid, f"Alg {aid}")
        problems = prob_map.get(aid, [])
        if not problems:
            continue

        alg_runs: list[SingleRunResult] = []

        for p_key in problems:
            current_run += 1
            if verbose:
                safe_print(f"[{current_run:02d}/{total_runs:02d}] Alg {aid:02d} ({alg_name}) on {p_key} ...", end=" ", flush=True)

            run_res = execute_single_task_a_run(
                alg_id=aid,
                prob_key=p_key,
                budget=budget,
                seed=seed,
                results_dir=results_dir,
                plots_dir=plots_dir,
            )
            alg_runs.append(run_res)
            all_run_results.append(run_res)

            if verbose:
                igd_disp = f"IGD={run_res.igd:.4f}" if run_res.igd is not None else "IGD=N/A"
                safe_print(
                    f"Done ({run_res.runtime_sec:.2f}s) | Status: {run_res.status} | "
                    f"Points: {run_res.n_points} | HV: {run_res.hv:.4f} | {igd_disp}"
                )

        # Aggregate for Table A3 Implementation Log
        # Columns: number, algorithm, problems used, evaluation budget, HV, IGD, runtime, status
        problems_used_str = ", ".join(r.problem_key for r in alg_runs)
        budget_str = str(budget)

        hv_parts = []
        igd_parts = []
        for r in alg_runs:
            hv_parts.append(f"{r.problem_key}: {r.hv:.4f}")
            if r.igd is not None:
                igd_parts.append(f"{r.problem_key}: {r.igd:.4f}")
            else:
                igd_parts.append(f"{r.problem_key}: N/A")

        hv_str = "; ".join(hv_parts)
        igd_str = "; ".join(igd_parts)

        total_runtime = sum(r.runtime_sec for r in alg_runs)
        runtime_str = f"{total_runtime:.2f}s"

        # Overall algorithm status: 'works', 'partly works', or 'not done'
        statuses = set(r.status for r in alg_runs)
        if "not done" in statuses:
            overall_status = "not done"
        elif "partly works" in statuses:
            overall_status = "partly works"
        else:
            overall_status = "works"

        entry = ImplementationLogEntry(
            number=aid,
            algorithm=alg_name,
            problems_used=problems_used_str,
            evaluation_budget=budget_str,
            hv=hv_str,
            igd=igd_str,
            runtime=runtime_str,
            status=overall_status,
        )
        log_entries.append(entry)

    # 1. Save detailed run records to CSV
    detail_csv_path = csv_dir / "task_a_runs_detail.csv"
    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = [
            "alg_id",
            "alg_name",
            "problem_key",
            "problem_name",
            "budget",
            "seed",
            "actual_evals",
            "n_points",
            "n_infeasible",
            "hv",
            "spacing",
            "igd",
            "igd_status",
            "igd_reason",
            "runtime_sec",
            "status",
            "plot_file",
            "front_csv_file",
            "error_msg",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_run_results:
            d = asdict(r)
            if d["spacing"] is None:
                d["spacing"] = "N/A"
            if d["igd"] is None:
                d["igd"] = "N/A"
            writer.writerow(d)

    # 2. Save Table A3 Implementation Log to CSV
    log_csv_path = csv_dir / "implementation_log.csv"
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = [
            "number",
            "algorithm",
            "problems used",
            "evaluation budget",
            "HV",
            "IGD",
            "runtime",
            "status",
        ]
        writer = csv.writer(f_out)
        writer.writerow(fieldnames)
        for e in log_entries:
            writer.writerow([
                e.number,
                e.algorithm,
                e.problems_used,
                e.evaluation_budget,
                e.hv,
                e.igd,
                e.runtime,
                e.status,
            ])

    # 3. Save Table A3 Implementation Log to Markdown
    log_md_path = csv_dir / "implementation_log.md"
    with open(log_md_path, "w", encoding="utf-8") as f_out:
        f_out.write("# Task A3 — Algorithm Implementation Log\n\n")
        f_out.write("| # | Algorithm | Problems Used | Evaluation Budget | HV | IGD | Runtime | Status |\n")
        f_out.write("|---|---|---|---|---|---|---|---|\n")
        for e in log_entries:
            f_out.write(
                f"| {e.number} | **{e.algorithm}** | {e.problems_used} | {e.evaluation_budget} | "
                f"{e.hv} | {e.igd} | {e.runtime} | `{e.status}` |\n"
            )

    if verbose:
        safe_print("\n" + "=" * 78)
        safe_print("TASK A RUN COMPLETE")
        safe_print(f"Detailed Runs CSV:       {detail_csv_path}")
        safe_print(f"Implementation Log CSV:  {log_csv_path}")
        safe_print(f"Implementation Log MD:   {log_md_path}")
        safe_print(f"Front Plots Directory:   {plots_dir / 'fronts'}")
        safe_print("=" * 78)

    return all_run_results, log_entries
