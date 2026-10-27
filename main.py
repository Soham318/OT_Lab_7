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

    # Placeholder parser for future phases
    subparsers.add_parser("task_a", help="Run Task A catalog validation")
    subparsers.add_parser("task_c_trace", help="Run Task C hand traces")
    subparsers.add_parser("task_c_sens", help="Run Task C parameter sensitivity studies")
    subparsers.add_parser("task_d", help="Run Task D head-to-head comparisons")
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
    else:
        print(f"Command '{args.command}' will be available in its corresponding implementation phase.")
        print("Available commands: 'python main.py test', 'python main.py run_alg --alg <1-37> --prob <p01-p14>'.")
        sys.exit(0)


if __name__ == "__main__":
    main()
