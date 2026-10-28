"""
test_task_a_runner.py: Unit tests for Task A experiment runner and logging pipeline.
"""

from __future__ import annotations
import csv
import tempfile
from pathlib import Path
import numpy as np

from moo.experiments.task_a_runner import (
    TASK_A_CATALOG_PROBLEMS,
    ALGORITHM_DISPLAY_NAMES,
    execute_single_task_a_run,
    run_task_a,
)
from moo.problems import get_problem


def test_task_a_problem_catalog_mapping() -> None:
    """Verifies that all 37 algorithms map to valid benchmark problems."""
    assert len(TASK_A_CATALOG_PROBLEMS) == 37
    assert len(ALGORITHM_DISPLAY_NAMES) == 37

    for aid in range(1, 38):
        assert aid in TASK_A_CATALOG_PROBLEMS, f"Missing problem mapping for Alg {aid}"
        probs = TASK_A_CATALOG_PROBLEMS[aid]
        assert len(probs) >= 1, f"Alg {aid} must have at least 1 test problem"
        for p_key in probs:
            prob = get_problem(p_key)
            assert prob is not None, f"Failed to instantiate problem '{p_key}' for Alg {aid}"


def test_execute_single_task_a_run() -> None:
    """Verifies single run execution, metric calculations, and output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res_dir = Path(tmpdir) / "results"
        plots_dir = Path(tmpdir) / "plots"

        result = execute_single_task_a_run(
            alg_id=1,  # Weighted Sum
            prob_key="P1",  # Factory Speed
            budget=50,
            seed=42,
            results_dir=res_dir,
            plots_dir=plots_dir,
            generate_plot=True,
            save_front_csv=True,
        )

        assert result.status == "works"
        assert result.alg_id == 1
        assert result.problem_key == "P1"
        assert result.actual_evals <= 50
        assert result.n_points > 0
        assert result.hv > 0.0
        assert result.spacing >= 0.0
        assert result.igd is not None, "P1 has known true front, IGD must be computed"
        assert result.igd_status == "computed"

        # Verify generated files
        assert Path(result.plot_file).is_file()
        assert Path(result.front_csv_file).is_file()

        # Check front CSV content
        with open(result.front_csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "f1_natural" in header
            assert "CV" in header
            rows = list(reader)
            assert len(rows) == result.n_points


def test_task_a_pipeline_csv_and_md_generation() -> None:
    """Verifies complete Task A pipeline with implementation log generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res_dir = Path(tmpdir) / "results"
        plots_dir = Path(tmpdir) / "plots"

        # Run on a tiny subset: Alg 1 (Weighted Sum) on P1 and P3
        all_runs, log_entries = run_task_a(
            algs=[1],
            budget=50,
            seed=42,
            results_dir=res_dir,
            plots_dir=plots_dir,
            verbose=False,
        )

        assert len(all_runs) == 2  # P1 and P3
        assert len(log_entries) == 1

        detail_csv = res_dir / "csv" / "task_a_runs_detail.csv"
        log_csv = res_dir / "csv" / "implementation_log.csv"
        log_md = res_dir / "csv" / "implementation_log.md"

        assert detail_csv.is_file()
        assert log_csv.is_file()
        assert log_md.is_file()

        # Verify Table A3 columns
        with open(log_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected_headers = [
                "number",
                "algorithm",
                "problems used",
                "evaluation budget",
                "HV",
                "IGD",
                "runtime",
                "status",
            ]
            assert headers == expected_headers
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0][0] == "1"
            assert rows[0][1] == "Weighted Sum"
            assert "P1, P3" in rows[0][2]
            assert rows[0][7] == "works"


def test_f01_feasibility_filtering_and_zero_feasible_handling() -> None:
    """Verifies that infeasible points are excluded from metrics and zero-feasible runs are handled transparently."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res_dir = Path(tmpdir) / "results"
        plots_dir = Path(tmpdir) / "plots"

        # 1. Unconstrained problem (P1) retains 100% feasibility
        res_p1 = execute_single_task_a_run(
            alg_id=1,
            prob_key="P1",
            budget=30,
            seed=42,
            results_dir=res_dir,
            plots_dir=plots_dir,
        )
        assert res_p1.status == "works"
        assert res_p1.n_points > 0
        assert res_p1.n_infeasible == 0
        assert res_p1.hv > 0.0

        # 2. Mock a zero-feasible run on a constrained problem
        from moo.problems.base import Problem

        class AllInfeasibleProblem(Problem):
            def __init__(self):
                super().__init__(
                    name="InfeasibleDummy",
                    n_var=2,
                    n_obj=2,
                    n_constr=1,
                    xl=np.array([0.0, 0.0]),
                    xu=np.array([1.0, 1.0]),
                    feas_tol=1e-4,
                )
            def evaluate(self, x):
                return np.array([0.5, 0.5]), 2.0, np.array([2.0])
            def evaluate_natural(self, x):
                return self.evaluate(x)

        import moo.experiments.task_a_runner as runner_module
        orig_get_problem = runner_module.get_problem
        try:
            runner_module.get_problem = lambda key: AllInfeasibleProblem() if key == "DUMMY_INFEAS" else orig_get_problem(key)
            res_infeas = runner_module.execute_single_task_a_run(
                alg_id=1,
                prob_key="DUMMY_INFEAS",
                budget=20,
                seed=42,
                results_dir=res_dir,
                plots_dir=plots_dir,
            )
            assert res_infeas.status == "partly works"
            assert res_infeas.n_points == 0
            assert res_infeas.n_infeasible > 0
            assert res_infeas.hv == 0.0
            assert res_infeas.spacing is None
            assert res_infeas.igd is None
            assert "0 feasible solutions found" in res_infeas.error_msg
            assert Path(res_infeas.front_csv_file).is_file()

            # Confirm front CSV recorded is_feasible=False
            with open(res_infeas.front_csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "is_feasible" in header
                rows = list(reader)
                assert len(rows) == res_infeas.n_infeasible
                for r in rows:
                    assert r[-1] == "False"
        finally:
            runner_module.get_problem = orig_get_problem


def test_f02_topsis_and_vikor_p12_coordinates_and_finite_normalization() -> None:
    """Verifies that TOPSIS and VIKOR return proper internal coordinates, valid dimensions, and finite HV on P12."""
    from moo.problems import get_problem
    from moo.algorithms import get_algorithm
    from moo.moo_utils import normalize_objectives

    p12 = get_problem("P12")

    for aid, name in [(8, "topsis"), (9, "vikor")]:
        alg = get_algorithm(aid)
        res = alg.run(p12, budget=50, seed=42)

        # Decision vector must be the 1D integer index of the selected supplier
        assert res["X_opt"].shape == (1, 1), f"{name} X_opt shape mismatch"
        best_idx = int(res["X_opt"][0, 0])
        assert 0 <= best_idx <= 4, f"{name} invalid supplier index {best_idx}"

        # Objectives must be internal minimization: Quality (col 1) and Reliability (col 3) must be negated
        f_int = res["F_opt"][0]
        f_nat = res["F_opt_nat"][0]
        assert f_int[1] <= 0.0, f"{name} Quality must be negated in internal minimization: {f_int[1]}"
        assert f_int[3] <= 0.0, f"{name} Reliability must be negated in internal minimization: {f_int[3]}"
        assert f_nat[1] >= 0.0, f"{name} Quality must be positive in natural coordinates: {f_nat[1]}"
        assert f_nat[3] >= 0.0, f"{name} Reliability must be positive in natural coordinates: {f_nat[3]}"

        # Normalization with documented ideal/nadir must produce values in [0, 1] <= 1.1
        f_norm = normalize_objectives(res["F_opt"], ideal=p12.ideal_point, nadir=p12.nadir_point)
        assert np.all(f_norm >= -1e-6), f"{name} normalized objectives below 0: {f_norm}"
        assert np.all(f_norm <= 1.0 + 1e-6), f"{name} normalized objectives exceeded nadir: {f_norm}"

        # In task_a runner, single run must execute with HV > 0.0
        with tempfile.TemporaryDirectory() as tmpdir:
            run_res = execute_single_task_a_run(
                alg_id=aid,
                prob_key="P12",
                budget=50,
                seed=42,
                results_dir=Path(tmpdir) / "results",
                plots_dir=Path(tmpdir) / "plots",
            )
            assert run_res.status == "works"
            assert run_res.n_points == 1
            assert run_res.hv > 0.0
            assert run_res.spacing is None  # Undefined for N=1


def test_f03_moma_binary_local_search_and_p9_decoding() -> None:
    """Verifies binary-aware local search and threshold decoding on P9."""
    from moo.problems import get_problem
    from moo.algorithms.alg30_moma import local_search_binary
    from moo.algorithms import get_algorithm

    p9 = get_problem("P9")

    # 1. P9 explicit threshold >= 0.5 test
    # An active feature with float 0.8 must be counted as 1, while 0.3 is counted as 0
    x_test = np.zeros(p9.n_var)
    x_test[0] = 0.8
    x_test[1] = 0.3
    f_val, cv_val, _ = p9.evaluate(x_test)
    assert cv_val == 0.0
    assert f_val[1] == 1.0, f"Expected 1 selected feature (0.8 -> 1, 0.3 -> 0), got {f_val[1]}"

    # Test 0-to-1 flip
    x_test[1] = 0.9
    f_val2, _, _ = p9.evaluate(x_test)
    assert f_val2[1] == 2.0, "Expected 2 selected features after 0-to-1 bit flip"

    # Test 1-to-0 flip
    x_test[0] = 0.1
    f_val3, _, _ = p9.evaluate(x_test)
    assert f_val3[1] == 1.0, "Expected 1 selected feature after 1-to-0 bit flip"

    # 2. Test local_search_binary directly
    rng = np.random.default_rng(42)
    x_init = np.ones(p9.n_var)
    f_init, cv_init, _ = p9.evaluate(x_init)
    weights = np.array([0.5, 0.5])
    ideal = np.array([0.0, 1.0])

    b_x, b_f, b_cv, evals = local_search_binary(
        x=x_init,
        f=f_init,
        cv=cv_init,
        problem=p9,
        weights=weights,
        ideal_point=ideal,
        budget_rem=10,
        rng=rng,
        max_steps=5,
    )
    assert evals > 0
    assert np.all(np.isin(b_x, [0.0, 1.0])), "Local search returned non-binary vector"

    # 3. Test full MOMA on P9
    alg30 = get_algorithm(30)
    res_moma = alg30.run(p9, budget=50, seed=42)
    assert len(res_moma["X_opt"]) > 0
    for sol in res_moma["X_opt"]:
        assert np.all(np.isin(sol, [0.0, 1.0])), "MOMA returned non-binary decision vector"


def test_f04_single_point_true_front_plotting() -> None:
    """Verifies that single-point true front (P11) renders with visible scatter marker."""
    from moo.experiments.plot_generator import plot_pareto_front
    from moo.problems import get_problem

    p11 = get_problem("P11")
    true_front_nat = p11.get_true_front_natural()
    assert len(true_front_nat) == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "test_p11_front.png"
        res_path = plot_pareto_front(
            F_opt_nat=np.array([[2400.0, 500.0]]),
            problem_name=p11.name,
            alg_name="Goal Programming",
            save_path=plot_path,
            true_front_nat=true_front_nat,
            hv=0.1234,
            spacing_val=None,
            igd_val=0.0,
            budget=100,
            evals_used=100,
            seed=42,
        )
        assert Path(res_path).is_file()
        assert Path(res_path).stat().st_size > 1000


def test_smoke_matrix_execution() -> None:
    """Verifies that the expanded 8-run smoke-test matrix executes cleanly across all problem archetypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res_dir = Path(tmpdir) / "results"
        plots_dir = Path(tmpdir) / "plots"

        all_runs, log_entries = run_task_a(
            smoke_matrix=True,
            budget=20,
            seed=42,
            results_dir=res_dir,
            plots_dir=plots_dir,
            verbose=False,
        )

        assert len(all_runs) == 8, f"Expected 8 runs in smoke matrix, got {len(all_runs)}"
        assert len(log_entries) == 8

        detail_csv = res_dir / "csv" / "task_a_runs_detail.csv"
        assert detail_csv.is_file()

        with open(detail_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 8
            # Confirm n_infeasible column exists
            assert "n_infeasible" in rows[0]
            # Confirm TOPSIS on P12 has positive HV and Spacing N/A
            p12_row = [r for r in rows if r["problem_key"] == "P12"][0]
            assert float(p12_row["hv"]) > 0.0
            assert p12_row["spacing"] == "N/A"
