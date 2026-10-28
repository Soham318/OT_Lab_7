"""
main.py: CLI Entry Point for MOO Lab 7

Supported Subcommands:
- test: Runs complete unit test suite for moo_utils and problems P1-P14.
- task_a: (Phase 6) Run all 37 algorithms and generate implementation log.
- task_c_trace: (Phase 7) Run 2-iteration hand traces for NPGA, MOACO, GradNorm.
- task_c_sens: (Phase 7) Run parameter sensitivity studies for Task C.
- task_d: (Phase 8) Run head-to-head comparisons across 10 seeds.
- plots: Generate visualization figures for Pareto fronts and metrics.
- all: Execute complete experimental pipeline.
"""

from __future__ import annotations
import argparse
import inspect
import sys
import time
import importlib
from pathlib import Path


def run_tests() -> int:
    """Discovers and executes all test functions in tests/ without external dependencies."""
    print("=" * 70)
    print("MOO LAB 7: RUNNING VERIFICATION TEST SUITE")
    print("=" * 70)

    test_dir = Path(__file__).parent / "tests"
    test_files = sorted(test_dir.glob("test_*.py"))
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    start_total = time.perf_counter()

    for test_file in test_files:
        module_name = f"tests.{test_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"\n[ERROR] Failed to import {module_name}: {e}")
            return 1

        test_funcs = [
            (name, fn)
            for name, fn in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        print(f"\nRunning {len(test_funcs)} test(s) from {module_name}:")
        for name, fn in test_funcs:
            total_tests += 1
            t0 = time.perf_counter()
            try:
                fn()
                dt = time.perf_counter() - t0
                print(f"  [PASS] {name} ({dt:.4f}s)")
                passed_tests += 1
            except Exception as e:
                dt = time.perf_counter() - t0
                print(f"  [FAIL] {name} ({dt:.4f}s)")
                print(f"         Error: {e}")
                failed_tests += 1

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed_tests} PASSED, {failed_tests} FAILED in {total_time:.3f}s")
    print("=" * 70)

    return 0 if failed_tests == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiobjective Optimization Lab 7 CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: test
    subparsers.add_parser("test", help="Run automated test suite")

    # Command: run_alg
    p_run = subparsers.add_parser("run_alg", help="Run a specific algorithm on a problem")
    p_run.add_argument("--alg", type=str, required=True, help="Algorithm ID (1-37) or name (e.g. weighted_sum, nsga2, moaco, gradnorm)")
    p_run.add_argument("--prob", type=str, default="p01", help="Problem name or ID (e.g. p01, p03, p11, p14)")
    p_run.add_argument("--budget", type=int, default=1000, help="Evaluation budget")
    p_run.add_argument("--seed", type=int, default=42, help="RNG seed")

    # Command: task_a
    p_task_a = subparsers.add_parser("task_a", help="Run Task A catalog validation and implementation log")
    p_task_a.add_argument("--budget", type=int, default=5000, help="Evaluation budget per problem run (default: 5000)")
    p_task_a.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p_task_a.add_argument("--algs", type=str, default=None, help="Comma-separated algorithm IDs (e.g. '1,2,14') or ranges (e.g. '1-5')")
    p_task_a.add_argument("--all-problems", action="store_true", help="Run on all listed problems (72 runs) instead of primary 2 (68 runs)")
    p_task_a.add_argument("--results-dir", type=str, default="results", help="Directory for CSV results (default: results)")
    p_task_a.add_argument("--plots-dir", type=str, default="plots", help="Directory for front plots (default: plots)")
    p_task_a.add_argument("--smoke", action="store_true", help="Run 4-run smoke test on algorithms 1 and 2 with budget=100")
    p_task_a.add_argument("--smoke-matrix", action="store_true", help="Run representative 8-run smoke test covering all problem archetypes with budget=100")

    # Placeholder parsers for future phases
    # Task C Parsers
    p_task_c_trace = subparsers.add_parser("task_c_trace", help="Run Task C 2-iteration hand traces for NPGA, MOACO, and GradNorm")
    p_task_c_trace.add_argument("--save-dir", type=str, default="results/traces", help="Directory to save trace tables")

    p_task_c_sens = subparsers.add_parser("task_c_sens", help="Run Task C parameter sensitivity studies (90 runs)")
    p_task_c_sens.add_argument("--results-dir", type=str, default="results", help="Directory for CSV results (default: results)")
    p_task_c_sens.add_argument("--plots-dir", type=str, default="plots", help="Directory for plots (default: plots)")
    p_task_c_sens.add_argument("--seeds", type=str, default="42-51", help="Seed range (e.g. '42-51') or comma-separated list")

    # Task D Parser
    p_task_d = subparsers.add_parser("task_d", help="Run Task D head-to-head comparisons")
    p_task_d.add_argument("--results-dir", type=str, default="results", help="Directory for CSV results (default: results)")
    p_task_d.add_argument("--plots-dir", type=str, default="plots", help="Directory for plots (default: plots)")
    p_task_d.add_argument("--seeds", type=str, default="42-51", help="Seed range (e.g. '42-51') or comma-separated list")

    subparsers.add_parser("plots", help="Generate report figures")
    subparsers.add_parser("all", help="Run all pipeline steps")

    args, extra = parser.parse_known_args()

    if args.command == "test" or args.command is None:
        exit_code = run_tests()
        sys.exit(exit_code)
    elif args.command == "run_alg":
        from moo.algorithms import get_algorithm
        from moo.problems import get_problem
        try:
            alg = get_algorithm(args.alg)
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: Invalid algorithm '{args.alg}'. {e}")
            sys.exit(1)
        try:
            prob = get_problem(args.prob)
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: Invalid problem '{args.prob}'. {e}")
            sys.exit(1)
        print(f"Running {args.alg} on {prob.name} (budget={args.budget}, seed={args.seed})...")
        res = alg.run(prob, budget=args.budget, seed=args.seed)
        print(f"Completed! Evaluations used: {res['n_evals']}")
        print(f"Non-dominated points found: {len(res['X_opt'])}")
        if len(res.get("F_opt_nat", [])) > 0:
            print(f"Sample natural objective vector(s):\n{res['F_opt_nat'][:5]}")
        sys.exit(0)
    elif args.command == "task_a":
        from moo.experiments.task_a_runner import run_task_a
        is_smoke = args.smoke or args.smoke_matrix
        budget = 100 if is_smoke else args.budget
        target_algs = [1, 2] if (args.smoke and not args.smoke_matrix) else None
        if args.algs and not is_smoke:
            target_algs = []
            for item in args.algs.split(","):
                item = item.strip()
                if "-" in item:
                    start, end = map(int, item.split("-"))
                    target_algs.extend(range(start, end + 1))
                elif item.isdigit():
                    target_algs.append(int(item))
        run_task_a(
            algs=target_algs,
            all_problems=args.all_problems,
            smoke_matrix=args.smoke_matrix,
            budget=budget,
            seed=args.seed,
            results_dir=args.results_dir,
            plots_dir=args.plots_dir,
            verbose=True,
        )
        sys.exit(0)
    elif args.command == "task_c_trace":
        from moo.experiments.task_c_runner import trace_npga_2iters, trace_moaco_2iters, trace_gradnorm_2iters
        print("=" * 70)
        print("TASK C: TWO-ITERATION NUMERICAL HAND TRACES (ROLL NO. 13)")
        print("=" * 70)

        # 1. NPGA
        print("\n--- 1. NPGA (Algorithm 13) 2-Iteration Trace on P1 Factory Speed ---")
        npga_trace = trace_npga_2iters()
        print(f"Initial Population (x): {npga_trace['gen0']['X'].flatten().tolist()}")
        print(f"Initial Objectives f1, f2:\n{npga_trace['gen0']['F']}")
        print("Gen 1 Tournaments:")
        for i, t in enumerate(npga_trace["gen1_tournaments"]):
            print(f"  T{i+1}: Cand {t['cand_a']} vs Cand {t['cand_b']} against T_dom {t['t_dom']} -> A_dom={t['a_dominated']}, B_dom={t['b_dominated']} | nc_A={t['nc_a']:.4f}, nc_B={t['nc_b']:.4f} => Winner: Cand {t['winner']}")
        print(f"Gen 1 Selected Winners: {npga_trace['gen1_winners']}")
        print(f"Gen 1 Population:\n{npga_trace['gen1']['X'].flatten().tolist()}")

        # 2. MOACO
        print("\n--- 2. MOACO (Algorithm 25) 2-Iteration Trace on 4-City Bi-Objective TSP ---")
        moaco_trace = trace_moaco_2iters()
        print("Iteration 1 Tours & Costs (Distance, Time):")
        for k, (tour, f) in enumerate(zip(moaco_trace["iter1"]["tours"], moaco_trace["iter1"]["F"])):
            print(f"  Ant {k}: Tour {tour.tolist()} -> Distance={f[0]:.1f}, Time={f[1]:.1f}")
        print(f"Iteration 1 Non-Dominated Ants: {moaco_trace['iter1']['non_dom_indices']}")
        print("Iteration 2 Tours & Costs (Distance, Time):")
        for k, (tour, f) in enumerate(zip(moaco_trace["iter2"]["tours"], moaco_trace["iter2"]["F"])):
            print(f"  Ant {k}: Tour {tour.tolist()} -> Distance={f[0]:.1f}, Time={f[1]:.1f}")
        print(f"Iteration 2 Non-Dominated Ants: {moaco_trace['iter2']['non_dom_indices']}")

        # 3. GradNorm
        print("\n--- 3. GradNorm (Algorithm 37) 2-Iteration Trace on 2-Task Shared Problem ---")
        grad_trace = trace_gradnorm_2iters()
        print(f"Initial: W^(0) = {grad_trace['W0'].tolist()}, Initial Losses: L^(0) = {grad_trace['L0'].tolist()}")
        it1 = grad_trace["iter1"]
        print(f"Iteration 1: Task Norms = [{it1['task_norms'][0]:.4f}, {it1['task_norms'][1]:.4f}] | Target Norms = [{it1['targets'][0]:.4f}, {it1['targets'][1]:.4f}]")
        print(f"             Updated Weights = [{it1['weights_next'][0]:.4f}, {it1['weights_next'][1]:.4f}] (sum = {sum(it1['weights_next']):.4f})")
        print(f"             Updated Parameters W^(1) = [{it1['W_next'][0]:.4f}, {it1['W_next'][1]:.4f}]")
        it2 = grad_trace["iter2"]
        print(f"Iteration 2: Losses = [{it2['L'][0]:.4f}, {it2['L'][1]:.4f}] | Task Norms = [{it2['task_norms'][0]:.4f}, {it2['task_norms'][1]:.4f}]")
        print(f"             Target Norms = [{it2['targets'][0]:.4f}, {it2['targets'][1]:.4f}]")
        print(f"             Updated Weights = [{it2['weights_next'][0]:.4f}, {it2['weights_next'][1]:.4f}] (sum = {sum(it2['weights_next']):.4f})")
        print(f"             Updated Parameters W^(2) = [{it2['W_next'][0]:.4f}, {it2['W_next'][1]:.4f}]")

        print("\nAll 3 two-iteration hand traces successfully verified and printed!")
        sys.exit(0)
    elif args.command == "task_c_sens":
        from moo.experiments.task_c_runner import run_parameter_sensitivity
        seeds = []
        if "-" in args.seeds:
            s_start, s_end = map(int, args.seeds.split("-"))
            seeds = list(range(s_start, s_end + 1))
        else:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

        print("=" * 70)
        print("TASK C: PARAMETER SENSITIVITY PIPELINE (90 RUNS)")
        print(f"Seeds: {seeds}")
        print(f"Results Directory: {args.results_dir}")
        print(f"Plots Directory: {args.plots_dir}")
        print("=" * 70)

        run_parameter_sensitivity(
            results_dir=args.results_dir,
            plots_dir=args.plots_dir,
            seeds=seeds,
            verbose=True,
        )
        print("\nTask C parameter sensitivity experiments complete!")
        sys.exit(0)
    elif args.command == "task_d":
        from moo.experiments.task_d_runner import run_task_d_suite
        seeds = []
        if "-" in args.seeds:
            s_start, s_end = map(int, args.seeds.split("-"))
            seeds = list(range(s_start, s_end + 1))
        else:
            seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

        print("=" * 70)
        print("TASK D: HEAD-TO-HEAD COMPARISON PIPELINE")
        print(f"Seeds: {seeds}")
        print(f"Results Directory: {args.results_dir}")
        print(f"Plots Directory: {args.plots_dir}")
        print("=" * 70)

        run_task_d_suite(
            results_dir=args.results_dir,
            plots_dir=args.plots_dir,
            seeds=seeds,
            verbose=True,
        )
        print("\nTask D comparison suite complete!")
        sys.exit(0)
    else:
        print(f"Command '{args.command}' will be available in its corresponding implementation phase.")
        print("Available commands: 'python main.py test', 'python main.py task_a', 'python main.py task_c_trace', 'python main.py task_c_sens', 'python main.py task_d'.")
        sys.exit(0)


if __name__ == "__main__":
    main()
