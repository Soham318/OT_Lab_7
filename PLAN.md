# Multiobjective Optimization Lab Assignment 7 — Master Plan
**Course:** Optimization Techniques Lab · Department of Artificial Intelligence & Data Science · M.Tech  
**Topic:** Multiobjective Optimization: 37 Algorithms, Trade-offs and Applications (Chapter 5, Parts 1–4 & Bonus Module 5E)  
**Student Roll No Ending:** `13` (R = 13)  
**Assigned Task C Algorithms:**
- Algorithm 1: $((13 - 1) \bmod 37) + 1 = 13$ → **NPGA (Niched Pareto Genetic Algorithm)** (#13)
- Algorithm 2: $((13 + 11) \bmod 37) + 1 = 25$ → **MOACO (Multi-Objective Ant Colony Optimization)** (#25)
- Algorithm 3: $((13 + 23) \bmod 37) + 1 = 37$ → **GradNorm (Gradient Normalization for Multi-Task Learning)** (#37)

---

## 1. Executive Summary & Coding Restrictions

### 1.1 Strict Compliance Rules & Dependency Policy (Section 2A)
- **General-Purpose Dependencies:**
  - `numpy`: Primary engine for all multidimensional arrays, vectorized mathematical routines, objective evaluations, sorting, and random number generation across all 37 algorithms and 14 problems.
  - `matplotlib`: Dedicated visualizer for generating Pareto fronts, empirical convergence curves, and parameter sensitivity plots.
  - Standard Library (`math`, `random`, `time`, `csv`, `typing`, `dataclasses`, `pathlib`): Utilized for I/O, file management, and CLI scaffolding.
- **Limited Exception 1 — Classical Solvers (Algorithms 1–7 only):**
  - `scipy` (`scipy.optimize.minimize`, `scipy.optimize.linprog`) may **only** be used to solve the single-objective / linear programming subproblems explicitly permitted for classical methods (Algorithms 1–7: Weighted Sum, $\epsilon$-Constraint, Benson's, Weight Metric, Value Function, Goal Programming, Lexicographic).
  - The scalarizations, weight sweeps, $\epsilon$-bound sweeps, target shortfalls, deviation variable formulations, and lexicographic priority cascade logic must be 100% written by us from scratch.
  - `scipy` is **strictly prohibited** for Algorithms 8–37 and for benchmark problem evaluations.
- **Limited Exception 2 — Dataset Loading (Problem P9 only):**
  - `scikit-learn` (`sklearn.datasets.load_breast_cancer`) may **only** be used to load the raw Wisconsin Breast Cancer feature matrix and targets.
  - We must hand-code the 5-NN classifier, Euclidean distance matrix computation, and the 5-fold cross-validation loop in pure NumPy from scratch. No `sklearn.neighbors` or `sklearn.model_selection` is permitted.
- **Prohibited Optimization Libraries:**
  - Under no circumstances will optimization libraries or toolboxes be installed or imported (`pymoo`, `DEAP`, `Platypus`, `jMetalPy`, `inspyred`, `NLopt`, `dlib`, `GSL`, etc.).
- **Reproducibility & Fair Budget:**
  - Standard budget: 10,000 function evaluations across all comparative experiments (with CLI overrides).
  - Fixed seeds: Deterministic runs for single-seed validations (Task A); 10 independent fixed seeds (`[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`) reporting $\text{mean} \pm \text{std}$ for comparative benchmarks (Task D).

---

## 2. Review of Reference Repository (`Shreyansh_OTAss7/moo_lab7`)

The reference repository is implemented in C++17. A careful review reveals key architectural strengths and several gaps/simplifications to avoid in our Python codebase:

### 2.1 Organizational Strengths to Emulate
1. **Unified CLI Command Dispatcher:** Subcommands (`unit`, `log`, `trace`, `sens`, `compare`, `mcdm`, `p11`, `mtl`, `investigate`, `all`) with `--budget` and `--seeds` flags.
2. **Explicit Shared Toolkit Separation:** Metrics (exact 2D HV, Monte Carlo $\ge 3$D HV, Spacing, IGD), non-dominated sorting, crowding distance, and Deb's feasibility rules isolated into a reusable core.
3. **Structured Results Artifacts:** Tabular CSV generation for execution logs, pairwise rankings, and hypervolume progression alongside front coordinates.

### 2.2 Shortcomings & Simplifications in Reference Repo to Rectify
1. **Monolithic Algorithm Groupings:** The C++ repo grouped algorithms into 5 broad files (`classical.hpp`, `evo.hpp`, `swarm.hpp`, `aco.hpp`, `grad.hpp`). The assignment requires: *"Write each algorithm as its own module with functions to initialize, evaluate, perform one iteration and run, and a docstring listing the lecture steps it follows."* We will strictly adhere to individual algorithm modules (`alg01_weighted_sum.py` ... `alg37_gradnorm.py`).
2. **Omission of P9 & P13(b):** The reference repo deliberately skipped P9 (feature selection) and P13(b) (shared-body multi-task neural network), diverting MOGWO and MOMA to P5. In our Python project:
   - **P9** will be fully implemented with a vectorized NumPy 5-NN classifier and 5-fold cross-validation.
   - **P13(b)** will be fully implemented with a native NumPy 2-layer shared MLP with dual heads (classification + regression) executing manual forward/backward passes.
3. **Student-Specific Deep Dive:** Reference repo targeted Roll No. 45 (Algs 8, 20, 32). Our implementation specifically deep-dives into Roll No. 13: **NPGA (#13)**, **MOACO (#25)**, and **GradNorm (#37)**.

---

## 3. Proposed Modular Project Structure

```text
OT_Lab_7/
├── PLAN.md                               # This master execution plan and checklist
├── README.md                             # Setup, architecture, run instructions, reproduction commands
├── requirements.txt                      # Strict dependencies: numpy, matplotlib, scipy (Algs 1-7 only), scikit-learn (P9 data only)
├── main.py                               # CLI entry point dispatching Tasks A-F commands
├── moo/                                  # Core Python package
│   ├── __init__.py
│   ├── moo_utils.py                      # Shared toolkit (dominance, Deb's rules, sorting, HV, IGD, Spacing, Das-Dennis)
│   ├── problems/                         # Problem definitions P1 to P14
│   │   ├── __init__.py
│   │   ├── base.py                       # Abstract Base Problem class
│   │   ├── p01_factory.py                # P1: Factory speed (convex, 1-var)
│   │   ├── p02_fleet.py                  # P2: Fleet sizing (1-var, real vs int)
│   │   ├── p03_zdt2.py                   # P3: ZDT2 non-convex benchmark (10-var)
│   │   ├── p04_portfolio.py              # P4: Markowitz investment portfolio (4-var, equality constraint)
│   │   ├── p05_cantilever.py             # P5: Cantilever beam design (2-var, stress constraint)
│   │   ├── p06_binh_korn.py              # P6: Binh-Korn benchmark (2-var, 2 hard constraints)
│   │   ├── p07_water.py                  # P7: Water allocation (linear trade-off, bounds/linear constraints)
│   │   ├── p08_tsp.py                    # P8: Bi-objective TSP (15 cities, distance + time, seed 7)
│   │   ├── p09_feature_selection.py      # P9: Breast cancer feature selection (binary, hand-coded 5-NN, 5-fold CV)
│   │   ├── p10_dtlz2.py                  # P10: DTLZ2 many-objective (M=3, M=5, n=M+9)
│   │   ├── p11_production.py             # P11: Production planning (Goal Programming, Lexicographic)
│   │   ├── p12_supplier.py               # P12: Supplier selection (MCDM, 5 suppliers x 4 criteria)
│   │   ├── p13_multitask.py              # P13: Multi-task learning: (a) toy continuous, (b) NumPy MLP
│   │   └── p14_laptop.py                 # P14: Laptop design (preference reference point)
│   ├── algorithms/                       # All 37 distinct algorithm modules
│   │   ├── __init__.py                   # Algorithm registry mapping 1..37 to modules
│   │   ├── alg01_weighted_sum.py
│   │   ├── alg02_epsilon_constraint.py
│   │   ├── alg03_benson.py
│   │   ├── alg04_weight_metric.py
│   │   ├── alg05_value_function.py
│   │   ├── alg06_goal_programming.py
│   │   ├── alg07_lexicographic.py
│   │   ├── alg08_topsis.py
│   │   ├── alg09_vikor.py
│   │   ├── alg10_vega.py
│   │   ├── alg11_moga.py
│   │   ├── alg12_nsga.py
│   │   ├── alg13_npga.py                 # [Task C Deep Dive #1]
│   │   ├── alg14_nsga2.py
│   │   ├── alg15_nsga3.py
│   │   ├── alg16_spea2.py
│   │   ├── alg17_pesa2.py
│   │   ├── alg18_moead.py
│   │   ├── alg19_moead_de.py
│   │   ├── alg20_moead_aco.py
│   │   ├── alg21_constrained_nsga2.py
│   │   ├── alg22_mopso.py
│   │   ├── alg23_mogwo.py
│   │   ├── alg24_moabc.py
│   │   ├── alg25_moaco.py                # [Task C Deep Dive #2]
│   │   ├── alg26_mosa.py
│   │   ├── alg27_mode.py
│   │   ├── alg28_motlbo.py
│   │   ├── alg29_moba.py
│   │   ├── alg30_moma.py
│   │   ├── alg31_rvea.py
│   │   ├── alg32_ibea.py
│   │   ├── alg33_sms_emoa.py
│   │   ├── alg34_hype.py
│   │   ├── alg35_r_nsga2.py
│   │   ├── alg36_mgda.py
│   │   └── alg37_gradnorm.py             # [Task C Deep Dive #3]
│   └── experiments/                      # Experiment runners and evaluation drivers
│       ├── __init__.py
│       ├── task_a_runner.py              # Single-run validation, front extraction, implementation log
│       ├── task_c_runner.py              # Hand traces verification & parameter sensitivity studies
│       ├── task_d_runner.py              # Head-to-head 10-seed comparisons (P3, P5, P10-M5, P8, P12, P13)
│       └── plot_generator.py             # Matplotlib visualizer for fronts, traces, and metrics
├── tests/                                # Automated verification suite
│   ├── test_moo_utils.py                 # Unit tests for dominance, Deb's rules, sort, HV, IGD, Spacing
│   ├── test_problems.py                  # Unit tests for P1-P14 sanity points and constraint violations
│   ├── test_algorithms.py                # 37 distinct unit tests checking the key rule of every algorithm
│   └── test_task_c_traces.py             # Numerical equality verification for NPGA, MOACO, GradNorm traces
├── results/                              # Generated numerical data
│   ├── csv/
│   │   ├── implementation_log.csv        # Task A3 master log
│   │   ├── task_d_continuous_p3.csv
│   │   ├── task_d_continuous_p5.csv
│   │   ├── task_d_manyobj_p10.csv
│   │   ├── task_d_discrete_p8.csv
│   │   ├── task_c_npga_sens.csv
│   │   ├── task_c_moaco_sens.csv
│   │   └── task_c_gradnorm_sens.csv
│   └── fronts/                           # Pareto front points for all algorithms and problems
├── plots/                                # High-resolution PNG/PDF figures for report
│   ├── fronts/
│   ├── convergence/
│   └── sensitivity/
└── docs/                                 # Report deliverables and written explanations
    ├── TASK_B_MASTER_TABLE.md            # Master comparison table (all 37 algorithms, >=4 adv, >=4 disadv)
    ├── TASK_C_STUDY_CARDS.md             # Complete study cards for NPGA, MOACO, GradNorm
    ├── TASK_C_HAND_TRACES.md             # Two-iteration step-by-step arithmetic traces
    ├── TASK_E_SELECTION_REASONING.md     # 6 problem selection rationales backed by empirical data
    └── TASK_F_VIVA_PREP.md               # Line-by-line viva walkthrough and code change predictions
```

---

## 4. Standardized Interfaces & Conventions

### 4.1 Internal Objective Direction, Constraint Handling & Feasibility Convention
1. **Universal Minimization:** All objectives are internally converted to **minimization**:
   $$f_m^{\text{internal}}(x) = \begin{cases} f_m^{\text{natural}}(x) & \text{if natural problem minimizes } f_m \\ -f_m^{\text{natural}}(x) & \text{if natural problem maximizes } f_m \end{cases}$$
   Plots and reports display natural values, but all internal dominance checks, sorting, hypervolume, and distance calculations operate on minimization vectors.
2. **Standardized Constraint Violation ($CV$):**
   - All inequality constraints are formulated in standard form:
     $$g_j(x) \le 0$$
   - All equality constraints are formulated in standard form:
     $$h_k(x) = 0$$
   - The total constraint violation $CV(x)$ is computed with a fixed numerical tolerance $\text{tolerance} = 10^{-4}$ ($1\text{e-}4$):
     $$CV(x) = \sum_{j} \max(0, g_j(x)) + \sum_{k} \max(0, |h_k(x)| - \text{tolerance})$$
   - **Non-negativity Guarantee:** Because $CV(x)$ is defined as a sum of non-negative $\max(0, \cdot)$ operators, **$CV(x) \ge 0.0$ always** (it can never be negative).
3. **Consistent Definition of Feasibility:**
   - A candidate solution $x$ is defined as **feasible** if and only if:
     $$CV(x) \le \epsilon_{\text{feas}}$$
     where $\epsilon_{\text{feas}} = 10^{-4}$ is the documented numerical tolerance.
   - If $CV(x) > \epsilon_{\text{feas}}$, the solution is **infeasible**.
4. **Deb's Constrained Dominance Comparison Rule:**
   Deb's feasibility-first comparison between two candidate solutions $A$ and $B$ uses the exact same feasibility criterion:
   - Solution $A$ constrained-dominates solution $B$ ($A \prec_{\text{con}} B$) if and only if one of the following three mutually exclusive conditions holds:
     - **Case 1 (Feasible vs Infeasible):** $A$ is feasible ($CV_A \le \epsilon_{\text{feas}}$) and $B$ is infeasible ($CV_B > \epsilon_{\text{feas}}$).
     - **Case 2 (Both Infeasible):** Both $A$ and $B$ are infeasible ($CV_A > \epsilon_{\text{feas}}$ and $CV_B > \epsilon_{\text{feas}}$), and solution $A$ has a strictly smaller constraint violation than solution $B$ ($CV_A < CV_B$).
     - **Case 3 (Both Feasible):** Both $A$ and $B$ are feasible ($CV_A \le \epsilon_{\text{feas}}$ and $CV_B \le \epsilon_{\text{feas}}$), and solution $A$ Pareto-dominates solution $B$ on the objective vector:
       $$\forall m \in \{1, \dots, M\}, \, f_m^{\text{internal}}(A) \le f_m^{\text{internal}}(B) \quad \text{and} \quad \exists m \in \{1, \dots, M\}, \, f_m^{\text{internal}}(A) < f_m^{\text{internal}}(B)$$

### 4.2 Problem Class Interface (`moo/problems/base.py`)
```python
class Problem:
    name: str
    n_var: int
    n_obj: int
    n_constr: int           # Total count of inequality + equality constraints
    xl: np.ndarray          # Lower variable bounds
    xu: np.ndarray          # Upper variable bounds
    var_type: str           # 'real', 'integer', 'binary', 'permutation'
    ref_point: np.ndarray   # Reference point for Hypervolume calculation
    ideal_point: np.ndarray # Ideal objective point for normalization
    nadir_point: np.ndarray # Nadir objective point for normalization
    tolerance: float = 1e-4 # Constraint numerical tolerance
    feas_tol: float = 1e-4  # Feasibility decision tolerance
    
    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Evaluates a single decision vector or batch.
        Returns:
            F: np.ndarray of shape (n_obj,) [All minimization]
            CV: float (total constraint violation, >= 0.0)
            G: np.ndarray of shape (n_constr,) [Raw constraint violations]
        """
        ...
        
    def get_true_front(self, n_points: int = 100) -> np.ndarray | None:
        """Returns sampled points along the analytical true Pareto front (if known)."""
        ...
```

### 4.3 Algorithm Module Interface (`moo/algorithms/algXX_*.py`)
Each of the 37 algorithms will provide a standard functional API:
```python
"""
Docstring detailing:
- Lecture Deck & Algorithm Family
- Formal Algorithm Name & Original Paper Citation
- Step-by-Step Algorithmic Logic
- Default Parameter Dictionary
"""

def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """
    Initializes algorithm state, random generator, population, and evaluations counter.
    Returns: state dict containing 'X', 'F', 'CV', 'n_evals', 'iter', and algorithm-specific buffers.
    """
    ...

def evaluate(X: np.ndarray, problem: Problem, state: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates solutions, updates state['n_evals'], returns (F, CV).
    """
    ...

def step(state: dict, problem: Problem) -> dict:
    """
    Executes exactly ONE iteration / generation / cycle.
    Updates population, archive, or parameters, and increments state['iter'].
    """
    ...

def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """
    Executes initialize(), loops step() while state['n_evals'] < budget.
    Returns final dictionary:
        'X_opt': non-dominated decision vectors
        'F_opt': non-dominated objective vectors
        'history': trajectory of metrics over generations
        'n_evals': total function evaluations consumed
        'state': final internal state
    """
    ...
```

---

## 5. Shared Toolkit (`moo/moo_utils.py`) Specification

1. **`dominates(f1, f2)`:** Vectorized Pareto dominance check ($f_1 \le f_2$ and $f_1 \neq f_2$).
2. **`constrained_dominates(f1, cv1, f2, cv2, feas_tol=1e-4)`:** Deb's 3-rule feasibility-first comparison using documented tolerance $\epsilon_{\text{feas}} = 10^{-4}$.
3. **`fast_non_dominated_sort(F, CV=None, use_deb=True, feas_tol=1e-4)`:** Deb's $O(M N^2)$ non-dominated sorting algorithm returning a list of front index lists `[Front_1, Front_2, ...]`.
4. **`crowding_distance(F, front_indices)`:** Assigns $\infty$ to extrema and normalized boundary-span distances for intermediate points.
5. **`hypervolume_2d(F, ref_point)`:** Exact 2D hypervolume using coordinate sorting and rectangular decomposition.
6. **`hypervolume_monte_carlo(F, ref_point, n_samples=100000, seed=42)`:** Monte Carlo estimation of dominated hypervolume for $M \ge 3$ objectives.
7. **`hypervolume(F, ref_point)`:** Dispatcher routing to 2D exact or Monte Carlo depending on dimension.
8. **`igd(obtained_F, true_F)`:** Inverted Generational Distance:
   $$\text{IGD}(P, P^*) = \frac{1}{|P^*|} \sum_{v \in P^*} \min_{u \in P} \|u - v\|_2$$
   (Computed on normalized objective coordinates $[0, 1]$).
9. **`spacing(obtained_F)`:** Schott's Spacing metric:
   $$S = \sqrt{\frac{1}{|F|-1}\sum_{i=1}^{|F|} (d_i - \bar{d})^2}, \quad d_i = \min_{j \neq i} \sum_{m=1}^M |f_m^i - f_m^j|$$
10. **`normalize_objectives(F, ideal, nadir)`:** Linearly scales objectives into $[0, 1]$:
    $$f_m^{\text{norm}} = \frac{f_m - z_m^*}{z_m^{\text{nad}} - z_m^* + \epsilon}$$
11. **`das_dennis(p, M)`:** Generates structured uniform reference points on unit simplex $\binom{p+M-1}{M-1}$ for NSGA-III and RVEA.

---

## 6. Problem Bank Specifications (P1 to P14)

| Problem ID | Name | Variables & Domain | Objectives (Natural Form) | Constraints ($g_j(x) \le 0, h_k(x) = 0$) & Violation ($CV \ge 0$) | Sanity Check Point | Analytical True Front |
|---|---|---|---|---|---|---|
| **P1** | Factory Speed | $x \in [1, 10]$ (continuous) | $\min f_1 = (x - 3)^2$<br>$\min f_2 = (x - 7)^2$ | Bounds only: $CV = 0$ | $x=5 \to (4, 4)$<br>$x=3 \to (0, 16)$ | $x \in [3, 7]$, convex front: $f_2 = (4 - \sqrt{f_1})^2$ |
| **P2** | Fleet Sizing | $x \in [1, 10]$ (continuous vs integer) | $\min f_1 = 100/x$<br>$\min f_2 = x^2$ | Bounds only: $CV = 0$ | $x=2 \to (50, 4)$<br>$x=10 \to (10, 100)$ | All $x \in [1, 10]$ are Pareto optimal; $f_2 = 10000 / f_1^2$ |
| **P3** | ZDT2 Benchmark | $x_i \in [0, 1], i=1\dots 10$ | $\min f_1 = x_1$<br>$\min f_2 = g \cdot (1 - (f_1/g)^2)$<br>$g = 1 + \sum_{i=2}^{10} x_i$ | Bounds only: $CV = 0$ | $x_i=0 \, (\forall i \ge 2) \implies g=1$<br>$x_1=0.5 \to (0.5, 0.75)$ | Non-convex front: $f_2 = 1 - f_1^2, f_1 \in [0, 1]$ |
| **P4** | Investment Portfolio | $w_i \ge 0, i=1\dots 4$ | $\min f_1 = w^T \Sigma w$ (risk)<br>$\max f_2 = \mu^T w$ (return) | $h(w) = \sum_{i=1}^4 w_i - 1 = 0$<br>$CV = \max(0, |\sum w_i - 1| - 10^{-4})$ | $w=(0,0,1,0) \to$<br>return $0.15$, risk $0.09$ | Lower-risk boundary generated via QP / constrained sweep |
| **P5** | Cantilever Beam | $b \in [0.01, 0.10]$ m<br>$h \in [0.02, 0.20]$ m | $\min f_1 = 7850 L b h$ (mass)<br>$\min f_2 = \frac{4 P L^3}{E b h^3}$ (deflection) | $g(b, h) = 2.4\times 10^{-5} - b h^2 \le 0$<br>$CV = \max(0, 2.4\times 10^{-5} - b h^2)$ | $(b, h) = (0.01, 0.05) \to$<br>mass $3.925$ kg, defl $0.016$ m, stress $240$ MPa | Active constraint boundary $b h^2 = 2.4\times 10^{-5}$ |
| **P6** | Binh-Korn Benchmark | $x_1 \in [0, 5], x_2 \in [0, 3]$ | $\min f_1 = 4 x_1^2 + 4 x_2^2$<br>$\min f_2 = (x_1 - 5)^2 + (x_2 - 5)^2$ | $g_1: (x_1-5)^2 + x_2^2 - 25 \le 0$<br>$g_2: 7.7 - ((x_1-8)^2 + (x_2+3)^2) \le 0$<br>$CV = \max(0, g_1) + \max(0, g_2)$ | $(0, 0) \to (0, 50)$ (feasible)<br>$(5, 3) \to (136, 4)$ (feasible) | Curve connecting $(0, 0)$ to $(5, 3)$ bounded by $g_1$ |
| **P7** | Water Allocation | $x_1 \in [10, 70]$ (agri)<br>$x_2 \in [20, 80]$ (industry) | $\max f_1 = 0.8 x_1$<br>$\max f_2 = 1.2 x_2$ | $g_1: x_1 + x_2 - 90 \le 0$<br>$g_2: 10 - x_1 \le 0, \, g_3: 20 - x_2 \le 0$<br>$CV = \sum_{j=1}^3 \max(0, g_j)$ | $(10, 80) \to (8, 96)$<br>$(70, 20) \to (56, 24)$ | Line $x_1 + x_2 = 90$; linear Pareto front |
| **P8** | Bi-Objective TSP | Tour permutation of 15 cities | $\min f_1 = \sum d_{ij}$ (distance)<br>$\min f_2 = \sum t_{ij}$ (time) | Valid permutation of 15 cities;<br>$CV = \text{missing/duplicate count}$ | Seed 7 coordinates in $[0, 100]^2$, $t_{ij} = d_{ij} u_{ij}$ | Evaluated via greedy NN heuristics and non-dominated archive |
| **P9** | Feature Selection | Binary mask $\{0, 1\}^{30}$ | $\min f_1 = 1 - \text{Acc}_{\text{5-CV}}$<br>$\min f_2 = \sum x_i$ (features) | $g(x) = 1 - \sum_{i=1}^{30} x_i \le 0$<br>$CV = \max(0, 1 - \sum x_i)$ | All-ones mask $[1, 1, \dots, 1]$ as accuracy baseline | Pareto trade-off between subset size and 5-NN accuracy |
| **P10** | DTLZ2 Benchmark | $n = M + 9$ variables in $[0, 1]$ ($M=3, M=5$) | $\min f_1 \dots f_M$ (spherical manifold) | Bounds only: $CV = 0$ | $x_i = 0.5 \, (\forall i \ge M) \implies g=0$<br>$\sum f_m^2 = 1$ | Unit hypersphere octant $\sum_{m=1}^M f_m^2 = 1$ |
| **P11** | Production Planning | $x_A, x_B \ge 0$ | $\max f_1 = 4 x_A + 6 x_B$<br>$\max f_2 = x_A + x_B$ | $g_1: 2 x_A + 3 x_B - 1200 \le 0$<br>$g_2: 4 x_A + 2 x_B - 1600 \le 0$<br>$CV = \max(0, g_1) + \max(0, g_2)$ | Corners: $(0,0), (400,0), (300,200), (0,400)$. Max profit 2400. | Line segment between $(300, 200)$ and $(0, 400)$ |
| **P12** | Supplier Selection | 5 discrete alternatives $\times$ 4 criteria | 4 criteria: Cost (min), Quality (max), Delivery (min), Reliability (max) | Weights: $(0.3, 0.3, 0.2, 0.2)$; discrete alternatives ($CV = 0$) | S1 to S5 raw table from PDF Page 12 | MCDM ranking via TOPSIS and VIKOR ($v=0.25, 0.5, 0.75$) |
| **P13** | Multi-Task Learning | (a) $x \in \mathbb{R}^2$, start $(0, 5)$<br>(b) NumPy Shared MLP | (a) $f_1 = \|x - a\|^2, f_2 = \|x - b\|^2$<br>(b) Classification + Regression loss | Continuous parameters ($CV = 0$) | (a) $a=(3,3), b=(-3,-3) \implies$ segment from $a$ to $b$ | Segment $x_1 = x_2 \in [-3, 3]$ for (a) |
| **P14** | Laptop Design | $x_1 \in [13, 17]$ (screen)<br>$x_2 \in [40, 90]$ (battery) | $\min f_1 = 0.4 + 0.05 x_1 + 0.012 x_2$<br>$\max f_2 = \frac{0.2 x_2}{0.5 + 0.03 x_1}$ | Bounds only: $CV = 0$ | $x=(13, 40) \to (1.53\text{kg}, 9.0\text{h})$<br>$x=(13, 90) \to (2.13\text{kg}, 20.2\text{h})$ | All front points have $x_1 = 13$; curve across $x_2 \in [40, 90]$ |

---

## 7. Complete Algorithm Catalogue Breakdown (Algorithms 1 to 37)

The 37 algorithms are grouped into their lecture families:

### 7.1 Classical Methods (Lecture Part 1 & 5E) — Solved via Scipy subproblem or explicit search
1. **Alg 01: Weighted Sum Method:** $\min \sum w_m f_m(x)$, sweeping weight vector $w$.
2. **Alg 02: $\epsilon$-Constraint Method:** $\min f_k(x)$ subject to $f_m(x) \le \epsilon_m$ for $m \neq k$.
3. **Alg 03: Benson's Method:** Maximize weighted shortfall distance past reference point: $\max \sum w_m (z_m - f_m(x))$.
4. **Alg 04: Weight Metric ($p$-norm, Tchebycheff):** Minimize $\left(\sum w_m |f_m - z_m^*|^p\right)^{1/p}$ for $p=1, 2$ and $\max_m w_m |f_m - z_m^*|$ for $p=\infty$.
5. **Alg 05: Value Function Method:** Maximize explicit decision-maker utility $U(f_1, \dots, f_M)$.
6. **Alg 06: Goal Programming:** Minimize unwanted deviational variables $d_m^+, d_m^-$ from aspirational targets.
7. **Alg 07: Lexicographic Method:** Prioritized sequential optimization where higher priority objectives are constrained by tolerance.

### 7.2 Ranking / MCDM Methods (Lecture Part 1 & 5E)
8. **Alg 08: TOPSIS:** Technique for Order Preference by Similarity to Ideal Solution (relative closeness to ideal vs anti-ideal).
9. **Alg 09: VIKOR:** Compromise ranking based on $S$ (group utility), $R$ (individual regret), and $Q$ compromise index with weight $v$.

### 7.3 Early Non-Elitist Evolutionary Algorithms (Lecture Part 2)
10. **Alg 10: VEGA (Vector Evaluated Genetic Algorithm):** Population divided into sub-populations, each selected on a single objective.
11. **Alg 11: MOGA (Fonseca & Fleming):** Rank assigned by dominance count ($1 +$ number of dominating individuals), fitness sharing in objective space.
12. **Alg 12: NSGA (Srinivas & Deb):** Non-dominated sorting into successive fronts; dummy fitness assignment with sharing within each front.
13. **Alg 13: NPGA (Niched Pareto GA):** **[Task C Deep Dive #1]** Pareto tournament against comparison set $t_{\text{dom}}$, niche count tie-breaker.

### 7.4 Modern Elitist Multiobjective Evolutionary Algorithms (Lecture Part 3 & 5E)
14. **Alg 14: NSGA-II (Deb et al.):** Fast non-dominated sort, crowding distance, $(N+N)$ elitist survival.
15. **Alg 15: NSGA-III (Deb & Jain):** Reference points on unit hyperplane replace crowding distance for many-objective selection ($M \ge 3$).
16. **Alg 16: SPEA2 (Zitzler et al.):** Fine-grained strength value, raw fitness, $k$-th nearest neighbor density, fixed-size external archive.
17. **Alg 17: PESA-II (Corne et al.):** Region-based selection where hyperboxes in grid archive are selected instead of individual solutions.
18. **Alg 18: MOEA/D (Zhang & Li):** Decomposes MOP into $N$ scalar subproblems (Tchebycheff/PBI) solved collaboratively within weight neighborhoods.
19. **Alg 19: MOEA/D-DE:** MOEA/D incorporating Differential Evolution crossover/mutation operators for continuous domains.
20. **Alg 20: MOEA/D-ACO:** Decomposition-based ant colony optimization for combinatorial domains (P8 TSP).

### 7.5 Constrained, Swarm & Nature-Inspired Algorithms (Lecture Part 4)
21. **Alg 21: Constrained NSGA-II:** Incorporating Deb's feasibility-first comparison directly into NSGA-II tournament selection.
22. **Alg 22: MOPSO (Multi-Objective Particle Swarm Optimization):** External Pareto repository with adaptive grid selection for global best guides.
23. **Alg 23: MOGWO (Multi-Objective Grey Wolf Optimizer):** Leaders $\alpha, \beta, \delta$ chosen from archive using roulette wheel over grid hyperboxes.
24. **Alg 24: MOABC (Multi-Objective Artificial Bee Colony):** Employed, onlooker, and scout bees with Pareto dominance greedy updates and external archive.
25. **Alg 25: MOACO (Multi-Objective Ant Colony Optimization):** **[Task C Deep Dive #2]** Multiple pheromone trails and heuristic matrices for bi-objective TSP.
26. **Alg 26: MOSA (Multi-Objective Simulated Annealing):** Neighbor generation with multiobjective Boltzmann acceptance probability and cooling schedule.
27. **Alg 27: MODE (Multi-Objective Differential Evolution):** DE mutation/crossover with one-to-one Pareto dominance replacement and archive.
28. **Alg 28: MO-TLBO (Teaching-Learning-Based Optimization):** Teacher phase and learner phase generalized to Pareto dominance with shrinking teaching factor.
29. **Alg 29: MOBA (Multi-Objective Bat Algorithm):** Frequency tuning, loudness decay, and pulse emission rate schedules driving Pareto velocity updates.
30. **Alg 30: MOMA (Multi-Objective Memetic Algorithm):** NSGA-II global evolutionary search interleaved with local search (e.g. pattern search or 2-opt).

### 7.6 Many-Objective, Indicator, Preference & Gradient Methods (Lecture Bonus Module 5E)
31. **Alg 31: RVEA (Reference Vector Guided Evolutionary Algorithm):** Angle-Penalized Distance (APD) balancing convergence and diversity dynamically.
32. **Alg 32: IBEA (Indicator-Based Evolutionary Algorithm):** Binary $\epsilon$-indicator guiding environmental selection without explicit diversity preservation.
33. **Alg 33: SMS-EMOA:** Steady-state $(\mu + 1)$ evolutionary algorithm discarding the individual with the minimum exclusive hypervolume contribution.
34. **Alg 34: HypE:** Hypervolume estimation algorithm utilizing Monte Carlo integration to assess individual hypervolume contributions in high dimensions.
35. **Alg 35: R-NSGA-II:** Preference-based NSGA-II steering solutions toward user-specified reference points via Euclidean distance and $\epsilon$-clearing.
36. **Alg 36: MGDA (Multiple Gradient Descent Algorithm):** Computes minimum-norm convex combination of objective gradients to establish a common Pareto descent vector.
37. **Alg 37: GradNorm:** **[Task C Deep Dive #3]** Dynamic task balancing via gradient normalization balancing gradient magnitudes and relative training rates.

---

## 8. Extra Task C Plans: NPGA (#13), MOACO (#25), GradNorm (#37)

### 8.1 Algorithm 13: NPGA (Niched Pareto Genetic Algorithm)
- **Study Card:**
  - *Origin:* Horn, Nafpliotis, & Goldberg (ICGA 1994).
  - *Problem it was invented to fix:* High computational overhead of Goldberg’s original $O(M N^3)$ non-dominated sorting and severe parameter sensitivity in early fitness sharing.
  - *Core Concept:* Instead of sorting the whole population, tournament selection picks 2 candidates $c_1, c_2$ and compares both against a small random comparison set of size $t_{\text{dom}}$ drawn from the population.
  - *Tournament Logic:*
    1. If $c_1$ dominates the comparison set and $c_2$ does not $\implies c_1$ wins.
    2. If $c_2$ dominates the comparison set and $c_1$ does not $\implies c_2$ wins.
    3. If neither or both dominate the comparison set (a tie) $\implies$ calculate niche count $m(c_1)$ and $m(c_2)$ using triangular sharing function $Sh(d) = \max(0, 1 - d/\sigma_{\text{share}})$. The candidate with the smaller niche count (least crowded) wins.
  - *Time Complexity:* $O(N \cdot t_{\text{dom}} \cdot M)$ per generation—massively faster than NSGA ($O(M N^3)$).
- **Two-Iteration Hand Trace Plan:**
  - *Tiny Instance:* Population $N = 4$ on P1 ($x \in [1, 10], f_1 = (x-3)^2, f_2 = (x-7)^2$).
  - Parameters: $t_{\text{dom}} = 2, \sigma_{\text{share}} = 2.0, p_c = 1.0, p_m = 0.0$.
  - Explicit seed fixes initial population: e.g., $x^{(0)} = [2.0, 4.0, 6.0, 8.0]$.
  - Step-by-step arithmetic for tournament comparisons, niche sharing calculations, crossover pairs, and offspring creation across Iteration 1 and Iteration 2.
  - Unit test `tests/test_task_c_traces.py` ensures the Python code produces identical floats to 6 decimal places.
- **Sensitivity Study Plan:**
  - Key parameter: Tournament comparison sample size $t_{\text{dom}} \in \{2, 10, 25\}$ (or $\sigma_{\text{share}} \in \{0.05, 0.20, 0.50\}$).
  - Run on P3 (ZDT2) and P5 (Cantilever) across 10 random seeds (seeds 42–51).
  - Metrics tracked: Hypervolume (HV), Spacing ($S$), and IGD.
  - Deliverable: Boxplots of HV and Spacing and Pareto front overlay plots demonstrating how small $t_{\text{dom}}$ causes drift due to weak selection pressure while large $t_{\text{dom}}$ causes premature clustering.

### 8.2 Algorithm 25: MOACO (Multi-Objective Ant Colony Optimization)
- **Study Card:**
  - *Origin:* Barán & Schaerer (2003) / Alaya, Solnon, & Ghedira (2007).
  - *Problem it was invented to fix:* Single-pheromone ACO cannot explore non-convex or conflicting combinatorial Pareto sets in multi-objective routing.
  - *Core Concept:* Maintains two distinct pheromone matrices $\tau^1_{ij}$ (distance pheromone) and $\tau^2_{ij}$ (time pheromone), and two heuristic visibility matrices $\eta^1_{ij} = 1/d_{ij}$ and $\eta^2_{ij} = 1/t_{ij}$.
  - *Ant Transition Rule:* Ant $k$ moving from city $i$ to unvisited city $j$ chooses with probability:
    $$P_{ij}^k = \frac{[\tau^1_{ij}]^{\alpha_1} [\tau^2_{ij}]^{\alpha_2} [\eta^1_{ij}]^{\beta_1} [\eta^2_{ij}]^{\beta_2}}{\sum_{l \in \text{allowed}} [\tau^1_{il}]^{\alpha_1} [\tau^2_{il}]^{\alpha_2} [\eta^1_{il}]^{\beta_1} [\eta^2_{il}]^{\beta_2}}$$
  - *Pheromone Update:* Evaporation $\tau_{ij}^m \leftarrow (1 - \rho) \tau_{ij}^m$. Deposit $\Delta \tau_{ij}^m$ only performed by ants belonging to the non-dominated Pareto front of the iteration.
- **Two-Iteration Hand Trace Plan:**
  - *Tiny Instance:* 4 cities with fixed symmetric distance matrix $D$ and travel time matrix $T$. Colony size $K = 4$ ants.
  - Fixed initial pheromones $\tau_{ij}^1 = \tau_{ij}^2 = 1.0$.
  - Hand trace each ant's probabilistic transitions using deterministic pseudo-random draws, compute tour lengths and tour times, perform non-dominated sort on the 4 tours, evaporate pheromone, and deposit pheromone for Iteration 1 and Iteration 2.
  - Code verification in `tests/test_task_c_traces.py` confirming tour choices and pheromone matrices match to $10^{-6}$.
- **Sensitivity Study Plan:**
  - Key parameter: Pheromone evaporation rate $\rho \in \{0.05, 0.20, 0.50\}$ (or heuristic exponent $\beta \in \{1.0, 2.0, 5.0\}$).
  - Run on P8 (15-city bi-objective TSP with seed 7 coordinates) across 10 random seeds.
  - Track final Pareto front spread, distance-time trade-off, and Hypervolume.
  - Deliverable: Comparative front plots and convergence curves demonstrating premature stagnation at high $\rho$ vs excessive exploration at low $\rho$.

### 8.3 Algorithm 37: GradNorm (Gradient Normalization for Multi-Task Learning)
- **Study Card:**
  - *Origin:* Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, & Andrew Rabinovich (ICML 2018).
  - *Problem it was invented to fix:* Gradients from different task losses frequently have vastly differing magnitudes, causing one task to dominate model parameter updates while others starve.
  - *Core Concept:* Dynamically adapts task loss weights $w_i(t)$ at each training step such that:
    1. Gradient norms of all tasks with respect to shared parameters are balanced.
    2. Tasks that train slower receive higher relative gradient weight (balancing training pace).
  - *Mathematical Formulation:*
    - Total loss: $L_{\text{total}}(t) = \sum_{i=1}^M w_i(t) L_i(t)$.
    - Gradient norm: $G_W^{(i)}(t) = \|\nabla_W (w_i(t) L_i(t))\|_2$.
    - Average norm: $\bar{G}_W(t) = \frac{1}{M} \sum_{i=1}^M G_W^{(i)}(t)$.
    - Relative inverse training rate: $\tilde{L}_i(t) = L_i(t) / L_i(0)$, and $r_i(t) = \tilde{L}_i(t) / \left(\frac{1}{M}\sum_j \tilde{L}_j(t)\right)$.
    - Target gradient norm: $\text{Target}_i(t) = \bar{G}_W(t) \cdot [r_i(t)]^\alpha$.
    - Weight loss: $L_{\text{grad}}(w_1, \dots, w_M) = \sum_{i=1}^M |G_W^{(i)}(t) - \text{Target}_i(t)|$.
    - Update $w_i \leftarrow w_i - \eta_{\text{grad}} \nabla_{w_i} L_{\text{grad}}$, and renormalize $\sum_{i=1}^M w_i(t) = M$.
- **Two-Iteration Hand Trace Plan:**
  - *Tiny Instance:* 2 tasks on P13(a) or a 2-parameter shared layer $W = [w_{11}, w_{22}]$ with initial values, task losses $L_1, L_2$, learning rate $\eta = 0.01$, $\eta_{\text{grad}} = 0.025$, and $\alpha = 1.5$.
  - Arithmetic trace computing $L_1(0), L_2(0)$, initial gradients $\nabla_W L_1, \nabla_W L_2$, gradient norms $G^{(1)}, G^{(2)}$, target norms, $L_{\text{grad}}$ gradient, weight adjustments, normalization to sum to 2, and parameter updates for Iteration 1 and Iteration 2.
  - Code verification in `tests/test_task_c_traces.py` ensuring exact numerical convergence.
- **Sensitivity Study Plan:**
  - Key parameter: Asymmetry restoring force parameter $\alpha \in \{0.12, 1.5, 3.0\}$.
  - Run on P13 multi-task setup across 10 independent initialization seeds.
  - Metrics tracked: Ratio of task losses $L_1/L_2$, gradient norm ratio $G^{(1)}/G^{(2)}$, and convergence rate.
  - Deliverable: Multi-panel loss and weight trajectory plots comparing GradNorm against fixed equal weighting ($w_1 = w_2 = 1.0$) and MGDA.

---

## 9. Unit Testing & Verification Strategy

Every algorithm and component must pass rigorous verification prior to experimental evaluation:
1. **Toolkit Verification (`tests/test_moo_utils.py`):**
   - Pareto dominance: test all 4 combinations (dominates, dominated, equal, non-dominating).
   - Deb's feasibility rules: 4 candidate test cases from lecture:
     - Feasible $A$ ($CV_A \le 10^{-4}$) vs Infeasible $B$ ($CV_B = 2.0$) $\implies A$ preferred.
     - Infeasible $B$ ($CV_B = 2.0$) vs Infeasible $C$ ($CV_C = 5.0$) $\implies B$ preferred.
     - Feasible $A$ vs Feasible $D$ $\implies$ evaluated by Pareto dominance.
   - Fast non-dominated sort: verified against known hand-sorted 6-point benchmark.
   - Exact 2D Hypervolume: verified against analytical rectangle area with reference point $(1.1, 1.1)$.
   - Spacing: verified against uniformly spaced points ($S = 0$) vs clustered points ($S > 0$).
   - IGD: verified against sampled front points (exact matches yield $\text{IGD} = 0$).
2. **Problem Sanity Tests (`tests/test_problems.py`):**
   - P1 to P14 sanity points evaluated against PDF specified values:
     - P1: $x=5 \to (4, 4)$
     - P2: $x=2 \to (50, 4)$
     - P5: $(0.01, 0.05) \to 3.925$ kg, $0.016$ m, $240$ MPa ($CV \le 10^{-4}$)
     - P6: $(0,0) \to (0, 50)$ ($CV \le 10^{-4}$); $(5, 3) \to (136, 4)$ ($CV \le 10^{-4}$)
   - Bounds checking and constraint violation flags validated.
3. **Algorithm Key-Rule Unit Tests (`tests/test_algorithms.py`):**
   - 37 dedicated test functions checking each algorithm's signature rule (e.g., Alg 21 prioritizes feasibility; Alg 15 correctly projects onto Das-Dennis reference lines; Alg 13 niche-count tie-breaks identical tournament winners; Alg 36 finds minimum-norm vector in convex hull).
4. **Task C Hand-Trace Regression Tests (`tests/test_task_c_traces.py`):**
   - Exact numerical equality verification for NPGA, MOACO, and GradNorm against hand-calculated trace steps.

---

## 10. CLI Architecture & Experiment Runner (`main.py`)

The project CLI will support clean, reproducible commands:
```bash
# 1. Run all unit tests
python main.py test

# 2. Run Task A validation & generate results/csv/implementation_log.csv
python main.py task_a --budget 10000

# 3. Run Task C traces (NPGA, MOACO, GradNorm) and verify against hand calculations
python main.py task_c_trace

# 4. Run Task C sensitivity studies (10 seeds, 3 parameter values)
python main.py task_c_sens --seeds 10 --budget 10000

# 5. Run Task D head-to-head comparisons (P3, P5, P10-M5, P8, P12, P13)
python main.py task_d --seeds 10 --budget 10000

# 6. Generate all figures and plots for report
python main.py plots

# 7. Run entire pipeline end-to-end
python main.py all --seeds 10 --budget 10000
```

---

## 11. Assignment Ambiguities, Inconsistencies & Open Decisions

During complete analysis of the assignment PDF, the following items require explicit notation and clarification:

1. **P3 Formula Notation:**
   - *Issue:* PDF Page 9 states: $g = 1 + \sum_{i=2}^{10} x_i$ and in parentheses notes "(this is $1 + 9 \cdot \sum x_i / (n-1)$ with $n=10$)".
   - *Resolution:* Because $n=10$, $9/(n-1) = 9/9 = 1$. The two expressions are mathematically identical. We will use the exact summation $1 + \sum_{i=2}^{10} x_i$.
2. **P8 Coordinate & Travel Time Generation:**
   - *Issue:* Coordinates use `np.random.seed(7)`. Does the symmetric speed multiplier $u_{ij} \sim U(0.5, 2.0)$ continue from the same seed stream or reseed?
   - *Resolution:* In `moo/problems/p08_tsp.py`, we fix `np.random.seed(7)` and sequentially generate:
     1. $15 \times 2$ coordinate array.
     2. Symmetric $15 \times 15$ matrix $u_{ij}$ with $u_{ii} = 0$.
     3. Export exact coordinates, distance matrix $D$, and travel time matrix $T$ to `results/p08_matrices.csv` for 100% deterministic reproducibility.
3. **Feature Selection (P9) vs Missing CSV in Reference Repo:**
   - *Issue:* The C++ reference repo skipped P9 because it did not have an offline CSV.
   - *Resolution:* In Python, the assignment explicitly permits `from sklearn.datasets import load_breast_cancer`. We will download/load it once, save an offline fallback `moo/problems/data/wdbc.csv`, and implement our own pure NumPy 5-NN classifier and 5-fold cross-validation loop.
4. **P13 Multi-Task Learning Scope:**
   - *Issue:* P13 includes Part (a) toy continuous problem and Part (b) shared-body neural network.
   - *Resolution:* We will implement both:
     - P13(a) for transparent 2D descent trajectory plotting and Task C GradNorm hand traces.
     - P13(b) for testing GradNorm vs MGDA vs fixed weights on a small 2-layer shared MLP with manual NumPy forward/backward backpropagation.
5. **Algorithm Test Coverage in Task A (1 vs 2 Problems):**
   - *Issue:* Rule A2.3 requires running each algorithm on at least 2 problems from its "Test problems" column, but notes "(one if only one is listed)".
   - *Resolution:* Algorithms 20 (MOEA/D-ACO), 25 (MOACO), 31 (RVEA), 34 (HypE), 36 (MGDA), and 37 (GradNorm) have only 1 problem listed in the catalogue table. For these, 1 problem is evaluated. For all other algorithms, at least 2 problems are evaluated.
6. **Reference Point for Hypervolume:**
   - *Issue:* Different problems have different objective scales.
   - *Resolution:* Objectives are normalized to $[0, 1]$ using each problem's ideal and nadir bounds, and hypervolume is computed with reference point $r = [1.1, 1.1, \dots, 1.1]$ across all algorithms.

---

## 12. Master Implementation Checklist Grouped by Tasks A-F

### Phase 1: Foundation & Shared Toolkit (Session 1)
- [x] Create Python project skeleton (`moo/`, `problems/`, `algorithms/`, `tests/`, `results/`, `plots/`, `docs/`)
- [x] Implement `moo/moo_utils.py` (dominance, Deb's rules with $\epsilon_{\text{feas}} = 10^{-4}$, non-dominated sort, crowding distance, 2D exact HV, Monte Carlo HV, IGD, Spacing, Das-Dennis)
- [x] Implement `moo/problems/base.py` with standard $CV \ge 0.0$ and tolerance $10^{-4}$
- [x] Implement problems P1 to P7 (`p01_factory.py` to `p07_water.py`)
- [x] Implement problems P8 to P14 (`p08_tsp.py` to `p14_laptop.py`)
- [x] Write unit tests `tests/test_moo_utils.py` and `tests/test_problems.py` and verify all tests pass (24/24 passed in 2.0s)

### Phase 2: Classical, Ranking & Early Evolutionary Algorithms (Algs 1–13)
- [ ] Implement Classical solvers: Algs 01–07 (`alg01_weighted_sum.py` to `alg07_lexicographic.py`) using `scipy.optimize` strictly for single-objective subproblems
- [ ] Implement Ranking / MCDM: Algs 08–09 (`alg08_topsis.py`, `alg09_vikor.py`)
- [ ] Implement Non-Elitist EAs: Algs 10–12 (`alg10_vega.py`, `alg11_moga.py`, `alg12_nsga.py`)
- [ ] Implement **Alg 13: NPGA** (`alg13_npga.py`) [Task C Deep Dive #1]
- [ ] Add unit tests for Algs 01–13 in `tests/test_algorithms.py`

### Phase 3: Elitist & Decomposition EAs (Algs 14–20, 21)
- [ ] Implement Alg 14: NSGA-II (`alg14_nsga2.py`)
- [ ] Implement Alg 15: NSGA-III (`alg15_nsga3.py`)
- [ ] Implement Alg 16: SPEA2 (`alg16_spea2.py`)
- [ ] Implement Alg 17: PESA-II (`alg17_pesa2.py`)
- [ ] Implement Algs 18–20: MOEA/D, MOEA/D-DE, MOEA/D-ACO
- [ ] Implement Alg 21: Constrained NSGA-II (`alg21_constrained_nsga2.py`)
- [ ] Add unit tests for Algs 14–21 in `tests/test_algorithms.py`

### Phase 4: Swarm, Physics & Hybrid Metaheuristics (Algs 22–30)
- [ ] Implement Swarm methods: Algs 22–24 (`alg22_mopso.py`, `alg23_mogwo.py`, `alg24_moabc.py`)
- [ ] Implement **Alg 25: MOACO** (`alg25_moaco.py`) [Task C Deep Dive #2]
- [ ] Implement Physics & Human-inspired: Algs 26–29 (`alg26_mosa.py`, `alg27_mode.py`, `alg28_motlbo.py`, `alg29_moba.py`)
- [ ] Implement Alg 30: MOMA (`alg30_moma.py`)
- [ ] Add unit tests for Algs 22–30 in `tests/test_algorithms.py`

### Phase 5: Many-Objective, Indicator, Preference & Gradient Methods (Algs 31–37)
- [ ] Implement Many-Objective & Indicator: Algs 31–34 (`alg31_rvea.py`, `alg32_ibea.py`, `alg33_sms_emoa.py`, `alg34_hype.py`)
- [ ] Implement Preference-based: Alg 35 (`alg35_r_nsga2.py`)
- [ ] Implement Gradient-based: Alg 36: MGDA (`alg36_mgda.py`)
- [ ] Implement **Alg 37: GradNorm** (`alg37_gradnorm.py`) [Task C Deep Dive #3]
- [ ] Add unit tests for Algs 31–37 in `tests/test_algorithms.py` and ensure 100% of unit tests pass

### Phase 6: Task A Execution & Implementation Log (Task A3)
- [ ] Execute Task A runner across all 37 algorithms on their test problems
- [ ] Save Pareto front data in `results/fronts/`
- [ ] Generate `results/csv/implementation_log.csv` (number, algorithm, problems, budget, HV, IGD, runtime, status)
- [ ] Plot individual Pareto fronts with true fronts overlaid

### Phase 7: Task C Deep Dive Execution (NPGA, MOACO, GradNorm)
- [ ] Write detailed study cards in `docs/TASK_C_STUDY_CARDS.md`
- [ ] Perform and document two-iteration hand traces in `docs/TASK_C_HAND_TRACES.md`
- [ ] Implement `tests/test_task_c_traces.py` and verify code reproduces hand trace numbers
- [ ] Run 10-seed sensitivity studies for NPGA, MOACO, and GradNorm
- [ ] Generate sensitivity comparison plots and CSV logs

### Phase 8: Task D Head-to-Head Comparative Experiments
- [ ] Continuous Bi-Objective: Run all applicable algorithms on P3 (unconstrained) and P5 (constrained) across 10 seeds
- [ ] Many-Objective: Run NSGA-II, NSGA-III, MOEA/D, RVEA, HypE on P10 ($M=5$) across 10 seeds
- [ ] Discrete: Run MOEA/D-ACO, MOACO, NSGA-II, MOMA on P8 (TSP) across 10 seeds
- [ ] MCDM weight sensitivity: Run TOPSIS & VIKOR on P4 NSGA-II front and P12
- [ ] Multi-Task: Compare MGDA, GradNorm, and fixed equal weights on P13
- [ ] Generate ranked summary tables and overlay Pareto front plots

### Phase 9: Task B, Task E & Task F Documentation & Viva Preparation
- [ ] Compile Task B Master Comparison Table in `docs/TASK_B_MASTER_TABLE.md` (all 37 algorithms, $\ge 4$ advantages, $\ge 4$ disadvantages, citing empirical Task A/D data)
- [ ] Write Task B family comparison summary (diversity, convergence, constraints, non-convex fronts)
- [ ] Write Task E problem selection reasoning in `docs/TASK_E_SELECTION_REASONING.md` for 6 problems backed by experimental data
- [ ] Prepare Task F Viva and Code Walkthrough guide in `docs/TASK_F_VIVA_PREP.md` answering the 8 core viva questions
- [ ] Assemble comprehensive PDF Report and build instructions
