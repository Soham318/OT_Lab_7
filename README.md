# Lab Assignment 7: Multiobjective Optimization
## 37 Algorithms, Trade-offs, Empirical Benchmarks, and Applications

**Optimization Techniques Laboratory · M.Tech in Artificial Intelligence & Data Science**  
**Candidate Name:** Soham Umesh Chavan  
**Roll Number Suffix:** 13 (Remainder  = 13$, Registration Ending in 13)  
**Assigned Deep-Dive Algorithms:**  
- **Algorithm 13:** NPGA (Niched Pareto Genetic Algorithm)  
- **Algorithm 25:** MOACO (Multi-Objective Ant Colony Optimization)  
- **Algorithm 37:** GradNorm (Gradient Normalization for Multi-Task Learning)  

**Date:** October 2026  
**Execution Environment:** Python 3.10+, NumPy, Matplotlib (**Zero External Optimization Toolboxes**)  

---

## 1. Project Overview & Deliverables

This repository provides a complete, self-contained, scratch implementation of all **37 Multiobjective Optimization (MOO) algorithms** and **14 benchmark problems (P1 through P14)** specified in the Lab Assignment 7 curriculum.

### Deliverables Breakdown (Tasks A through F)
- **Task A (Implementation & Pareto Visualizations):** Complete scratch implementations of Algorithms 1 through 37 in moo/algorithms/, benchmarked across 14 problems with 68 Pareto front plots in plots/fronts/ and detailed per-run metrics in 
esults/csv/task_a_runs_detail.csv.
- **Task B (Comparative Taxonomy):** Comprehensive comparative analysis of all 37 algorithms with >= 4 advantages and >= 4 disadvantages per algorithm, backed by empirical numbers, in docs/TASK_B_ALGORITHM_COMPARISON.md.
- **Task C (Deep Dives & Parameter Sensitivity):** Three 11-section study cards, six exact two-iteration numerical hand traces, and 90 sensitivity runs across 10 random seeds ({\\text{dom}}$ for NPGA, $\\rho$ for MOACO, $\\alpha$ for GradNorm) in docs/TASK_C_DEEP_DIVES.md.
- **Task D (Head-to-Head Comparative Studies):** Controlled empirical benchmarks across 10 random seeds ( \\dots 51$) under equal evaluation budgets:
  1. *P3 Unconstrained (ZDT2):* 10 algorithms x 10 seeds = **100 runs** (Winner: MODE #27, =0.4547$).
  2. *P5 Constrained (Cantilever Beam):* 8 algorithms x 10 seeds = **80 runs** (Winner: CNSGA-II #21, =0.0384, 100\\%$ feasible).
  3. *P10 Many-Objective (=5$ DTLZ2):* 5 algorithms x 10 seeds = **50 runs** (Winner: RVEA #31, =1.1395, IGD=0.2071$; HypE #34, =1.0775$; NSGA-II degraded =0.1904$).
  4. *P8 Discrete Permutation (TSP):* 4 algorithms x 10 seeds = **40 runs** (Winner: MOMA #30, =0.8310$; NSGA-II #14, =0.7205$).
  5. *Special Studies:* 17 targeted evaluations covering parameter sweeps (Algs 1-4), preference compromises (Algs 5-7), MCDM weight sensitivity (Algs 8-9), and multi-task loss balancing (Algs 36-37).
- **Task E (Algorithm Selection Guide):** Six problem-specific engineering selection case studies with rationale and negative examples in docs/TASK_E_ALGORITHM_SELECTION.md.
- **Task F (Viva Voce & Code Walkthrough):** In-depth answers to all 8 mandatory viva questions and comprehensive mutation walkthroughs in docs/TASK_F_VIVA_PREPARATION.md.
- **Master Report:** Consolidated publication-ready document in docs/FINAL_LAB_REPORT.md and docs/FINAL_LAB_REPORT.pdf.

---

## 2. Reconciled Experiment Run Counts

Across the entire project, all experimental runs are independent, logged in raw per-run CSV files, and completely reproducible:

| Experiment Suite | Description | Random Seeds | Number of Runs | Raw Evidence Artifact |
|:---|:---|:---:|:---:|:---|
| **Task A** | All 37 algorithms on assigned benchmark problems | Fixed ($) | **68 runs** | 
esults/csv/task_a_runs_detail.csv |
| **Task C** | Parameter sensitivity ({\\text{dom}}$ on P3, $\\rho$ on P8, $\\alpha$ on P13) | 10 seeds ( \\dots 51$) | **90 runs** | 
esults/csv/task_c_*_sensitivity_raw.csv |
| **Task D (Study 1)** | P3 Unconstrained ZDT2 (10 algorithms) | 10 seeds ( \\dots 51$) | **100 runs** | 
esults/csv/task_d_p3_unconstrained_raw.csv |
| **Task D (Study 2)** | P5 Constrained Cantilever (8 algorithms) | 10 seeds ( \\dots 51$) | **80 runs** | 
esults/csv/task_d_p5_constrained_raw.csv |
| **Task D (Study 3)** | P10 Many-Objective DTLZ2 =5$ (5 algorithms) | 10 seeds ( \\dots 51$) | **50 runs** | 
esults/csv/task_d_p10_manyobj_m5_raw.csv |
| **Task D (Study 4)** | P8 Discrete TSP Permutation (4 algorithms) | 10 seeds ( \\dots 51$) | **40 runs** | 
esults/csv/task_d_p8_discrete_tsp_raw.csv |
| **Grand Total (Raw CSVs)** | **Total independent stochastic runs** | — | **428 runs** | **Sum of raw CSV rows** |
| *Task D (Study 5)* | Non-CSV special studies (sweeps, preferences, MCDM, MTL) | Deterministic / 42 | *17 evaluations* | 
esults/csv/task_d_special_studies.json |

---

## 3. Quick Start & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Install minimal dependencies:
  `ash
  pip install -r requirements.txt
  `

### Running the Test Suite (Zero Regressions)
To run all 100 unit tests across the entire toolkit, problems, algorithms, and numerical hand traces:
`ash
python main.py test
`
*Expected result:* 100 passed in ~220s (exit code 0).

### Reproducing Experiments
All experiments support reproducible CLI execution with explicit seed and budget parameters:

1. **Task A Validation Run:**
   `ash
   python main.py task_a --budget 5000 --seed 42
   `

2. **Task C Numerical Hand Traces:**
   `ash
   python main.py task_c_trace
   `

3. **Task C Parameter Sensitivity Suite (90 runs):**
   `ash
   python main.py task_c_sens --seeds 42-51 --results-dir results --plots-dir plots
   `

4. **Task D Head-to-Head Comparative Studies (270 runs + 17 special study evaluations):**
   `ash
   python main.py task_d --seeds 42-51 --results-dir results --plots-dir plots
   `

---

## 4. Verification & Compliance Summary

- **Toolbox Policy:** 100% compliant. Zero external optimization frameworks (no pymoo, deap, or platypus).
- **Code Quality:** Fully typed, documented with academic citations, passing git diff --check with zero conflicts.
- **Reproducibility:** All seeds are fixed, logged in raw CSVs, and all numerical traces assert exact analytical equality to ^{-4}$ precision.
