# OT Lab 7 — Task B: Comprehensive Multi-Objective Optimization Algorithm Comparison

**Course:** Optimization Techniques Lab · Department of Artificial Intelligence & Data Science
**Student:** Soham Umesh Chavan (Roll No. Ending in 13, $R=13$)
**Repository:** [https://github.com/Soham318/OT_Lab_7](https://github.com/Soham318/OT_Lab_7)
**Document Purpose:** Rigorous, publication-grade comparative analysis of all 37 Multiobjective Optimization (MOO) algorithms implemented from scratch in the repository.

---

## 1. Executive Summary & Mathematical Foundations

Multi-Objective Optimization (MOO) addresses decision problems with two or more conflicting objective functions that must be optimized simultaneously over a defined feasible decision space $S \subseteq \mathbb{R}^n$. In general mathematical form, an unconstrained or constrained multiobjective optimization problem is formulated as:

$$\min_{x \in S} F(x) = \big(f_1(x), f_2(x), \dots, f_M(x)\big)^T$$
$$\text{subject to } g_j(x) \le 0 \quad (j = 1, \dots, J), \quad h_k(x) = 0 \quad (k = 1, \dots, K)$$

Because objectives are inherently conflicting, a single global solution that minimizes all objectives simultaneously rarely exists. Instead, the goal is to discover the set of **Pareto-optimal** solutions representing optimal trade-offs.

### Key Definitions & Terminology
- **Pareto Dominance ($\prec$):** A candidate decision vector $x_1$ Pareto-dominates $x_2$ ($x_1 \prec x_2$) if and only if $x_1$ is no worse than $x_2$ in all objectives ($\forall m, f_m(x_1) \le f_m(x_2)$) and strictly better in at least one objective ($\exists m, f_m(x_1) < f_m(x_2)$).
- **Pareto Optimality:** A solution $x^* \in S$ is Pareto-optimal if there is no feasible $x \in S$ such that $x \prec x^*$. The set of all Pareto-optimal decision vectors is the **Pareto Set ($PS$)**, and its image in objective space is the **Pareto Front ($PF$)**.
- **Deb's Constrained Dominance ($\prec_{\text{con}}$):** In constrained MOO, solution $A$ constrained-dominates solution $B$ ($A \prec_{\text{con}} B$) if: (1) $A$ is feasible and $B$ is infeasible; (2) both are infeasible and $A$ has a strictly smaller total constraint violation ($CV_A < CV_B$); or (3) both are feasible and $A$ Pareto-dominates $B$ on objectives. Total constraint violation is defined as $CV(x) = \sum_{j=1}^J \max(0, g_j(x)) + \sum_{k=1}^K \max(0, |h_k(x)| - \text{tol})$ with numerical tolerance $\epsilon_{\text{feas}} = 10^{-4}$.
- **Technical Abbreviations Used:** Multi-Objective Evolutionary Algorithm (MOEA), Multi-Criteria Decision Making (MCDM), Simulated Binary Crossover (SBX), Differential Evolution (DE), Hypervolume (HV), Inverted Generational Distance (IGD), Angle-Penalized Distance (APD), Sequential Least Squares Programming (SLSQP), Karush-Kuhn-Tucker (KKT), Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Artificial Bee Colony (ABC), Ant Colony Optimization (ACO), Simulated Annealing (SA), Teaching-Learning-Based Optimization (TLBO), Bat Algorithm (BA), Memetic Algorithm (MA), Multiple Gradient Descent Algorithm (MGDA).

---

## 2. Individual Profiles for All 37 Algorithms

Every algorithm is profiled below in the exact canonical assignment order (1 through 37). Each profile provides the core operational idea, distinguishing mechanism, at least four specific advantages, at least four specific disadvantages, suitable problem characteristics, limitations, project implementation details, and asymptotic time/space complexity.

### Algorithm 1: Weighted Sum
- **Module File:** `moo/algorithms/alg01_weighted_sum.py`
- **Algorithm Family:** Classical Scalarization
- **General Approach:** Deterministic / Scalarization-based (Single-objective solver sweeps)

#### 1. Core Idea
Transforms a multiobjective optimization problem into a series of single-objective scalar subproblems by assigning a non-negative weight $w_m \ge 0$ to each objective function $f_m(x)$ such that $\sum_{m=1}^M w_m = 1$. By systematically varying the weight vector $w$ over the unit simplex and solving the scalar optimization problem $\min_{x \in S} \sum_{m=1}^M w_m f_m(x)$ using an established single-objective numerical optimizer, a discrete set of Pareto-optimal solutions is traced.

#### 2. Defining Mechanism
Linear scalarization using a static convex combination of objective functions, where every Pareto point corresponds to the tangency point between a hyperplane with normal vector $w$ and the feasible objective space.

#### 3. Advantages
- Computationally lightweight per run, requiring only standard single-objective optimization algorithms (e.g., gradient descent, Quasi-Newton, SLSQP, or simplex methods).
- Guaranteed to return a weakly Pareto-optimal solution for any non-negative weight vector $w \ge 0$, and strictly Pareto-optimal if all weights are strictly positive ($w_m > 0$).
- Conceptually intuitive for decision-makers who can naturally express relative importance as trade-off percentages or monetary weights.
- Seamlessly integrates existing industrial legacy solvers and mathematical programming environments without requiring evolutionary population scaffolding.

#### 4. Disadvantages
- Inability to locate non-convex (concave) Pareto-optimal solutions, as any supporting hyperplane tangency misses points lying within non-convex indentations.
- Uniformly spaced weight vectors do not produce uniformly distributed Pareto points; points cluster tightly in steep regions and leave large gaps in flat regions.
- Highly sensitive to objective scaling; differences in magnitude or physical units between objectives distort the search unless rigorous a priori normalization is enforced.
- Requires executing an entirely separate optimization trajectory for every desired Pareto point, scaling poorly when a dense front representation is needed.

#### 5. Suitable Problems
Suited for low-dimensional problems ($M=2$ or $M=3$) with strictly convex Pareto fronts, smooth continuous decision spaces, and situations where single-objective optimization routines are already mature.

#### 6. Limitations
Requires known objective bounds for proper normalization; cannot capture non-convex fronts; requires solving multiple independent runs from scratch.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg01_weighted_sum.py`. Subproblems are solved using `scipy.optimize.minimize` with SLSQP (incorporating analytical or numerical gradients) with a robust Nelder-Mead fallback. Evaluated across continuous problems P1, P3, P4, P5, P6, P7, P10, P11, P14.

#### 8. Complexity Analysis
Time: $O(K \cdot C_{\text{single}})$ where $K$ is the number of weight vectors in the sweep and $C_{\text{single}}$ is the convergence cost of the single-objective solver. Space: $O(n + M)$ per subproblem.

---

### Algorithm 2: ε-Constraint
- **Module File:** `moo/algorithms/alg02_epsilon_constraint.py`
- **Algorithm Family:** Classical Scalarization
- **General Approach:** Deterministic / Constrained Scalarization

#### 1. Core Idea
Retains one primary objective $f_k(x)$ for minimization while converting the remaining $M-1$ objectives into explicit inequality constraints: $f_m(x) \le \epsilon_m$ for all $m \ne k$. By systematically sweeping the constraint bound vector $\epsilon = (\epsilon_1, \dots, \epsilon_{k-1}, \epsilon_{k+1}, \dots, \epsilon_M)$ across the known range of each objective, the optimizer discovers boundary points of the feasible objective space.

#### 2. Defining Mechanism
Constrained single-objective reformulation that isolates a single objective function and slices the objective space using orthogonal bounding hyperplanes $\epsilon_m$.

#### 3. Advantages
- Overcomes the fundamental limitation of Weighted Sum: capable of finding every Pareto-optimal point even along non-convex and disconnected Pareto fronts.
- Directly handles problems where one objective is of primary concern (e.g., cost or safety) while others merely need to satisfy acceptable thresholds.
- Scaling and unit differences between distinct objectives do not distort the primary objective optimization because objectives are not summed together.
- The Lagrange multipliers associated with the active $\epsilon$-constraints provide direct sensitivity information and local trade-off rates (shadow prices).

#### 4. Disadvantages
- Choosing appropriate ranges and grid spacings for $\epsilon_m$ requires prior knowledge of the individual minima and maxima (payoff table or nadir estimation).
- Improperly chosen $\epsilon_m$ thresholds frequently yield infeasible optimization subproblems, wasting computational evaluations.
- The number of grid combinations scales exponentially as $O(G^{M-1})$, where $G$ is the grid resolution per objective, making it intractable for $M > 4$.
- Solving non-linearly constrained subproblems is significantly harder for numerical optimizers than unconstrained or linearly weighted formulations.

#### 5. Suitable Problems
Ideal for bi-objective and tri-objective problems with non-convex Pareto fronts, engineering designs with hard environmental or regulatory caps, and mixed-integer formulations.

#### 6. Limitations
Requires computing the payoff table beforehand; high sensitivity to constraint tolerances; risk of weakly Pareto-optimal solutions unless augmented forms are used.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg02_epsilon_constraint.py`. Primary objective is selected, secondary bounds are varied linearly between estimated ideal and nadir bounds, and subproblems are solved via SLSQP with constraint handling. Evaluated on P1, P3, P5, P6.

#### 8. Complexity Analysis
Time: $O(G^{M-1} \cdot C_{\text{constr}})$ where $G$ is grid points per secondary objective and $C_{\text{constr}}$ is the cost of constrained nonlinear programming. Space: $O(n + M)$.

---

### Algorithm 3: Benson’s Method
- **Module File:** `moo/algorithms/alg03_benson.py`
- **Algorithm Family:** Classical Scalarization
- **General Approach:** Deterministic / Reference Point Shortfall Maximization

#### 1. Core Idea
Begins with an exterior, weakly feasible, or dominated reference point $z^0 \in \mathbb{R}^M$ such that $z_m^0 \ge f_m(x)$ for all $x \in S$. It introduces non-negative shortfall variables $u_m = z_m^0 - f_m(x) \ge 0$ measuring how much each objective improves upon the reference point. Benson's formulation then maximizes the weighted sum of these shortfalls $\max \sum_{m=1}^M w_m (z_m^0 - f_m(x))$ subject to $f_m(x) \le z_m^0$ and $x \in S$, pulling the solution directly onto the non-dominated Pareto boundary.

#### 2. Defining Mechanism
Direct maximization of coordinate shortfalls between a dominated reference point and the feasible objective space, projecting dominated points onto the Pareto front.

#### 3. Advantages
- Guarantees that the resulting solution is strictly Pareto-optimal (not merely weakly Pareto-optimal) if all weights $w_m > 0$.
- Can serve as a rigorous Pareto-efficiency test: if the optimal objective shortfall value is zero when initialized with $f(x^0)$, $x^0$ is already Pareto-optimal.
- Avoids division by zero or singularity issues that can plague metric-based distance formulations when reference points coincide with the ideal point.
- Provides a natural linear programming formulation when the underlying problem constraints and objective functions are linear.

#### 4. Disadvantages
- Requires generating an explicit reference point $z^0$ that strictly bounds the objective space from the dominated side.
- Does not naturally generate an evenly distributed Pareto front; sweeping weights $w$ on shortfalls still suffers from non-uniform point distributions on non-convex boundaries.
- If $z^0$ is chosen too far from the true front, numerical scaling can bias the projection toward extreme vertices of the feasible polytope.
- Inefficient for generating complete Pareto fronts compared to population-based evolutionary methods, requiring sequential mathematical programming solves.

#### 5. Suitable Problems
Particularly effective for multiobjective linear programs (MOLP), convex multi-criteria problems, and post-processing candidate solutions to ensure Pareto efficiency.

#### 6. Limitations
Restricted to convex/linear formulations for global optimality guarantees; non-trivial reference point initialization; poor front diversity under uniform weights.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg03_benson.py`. Uses SciPy minimize to maximize total shortfall from estimated upper objective bounds, tested on P1, P3, P7, P11.

#### 8. Complexity Analysis
Time: $O(K \cdot C_{\text{NLP}})$ for $K$ reference/weight sweeps. Space: $O(n + M)$.

---

### Algorithm 4: Weight Metric
- **Module File:** `moo/algorithms/alg04_weight_metric.py`
- **Algorithm Family:** Classical Scalarization
- **General Approach:** Deterministic / Compromise Programming via $L_p$ Metric

#### 1. Core Idea
Measures the distance between an achievable objective vector $f(x)$ and an unattainable ideal target $z^* = (z_1^*, \dots, z_M^*)$ using a weighted $L_p$ norm: $\min_{x \in S} \left( \sum_{m=1}^M w_m |f_m(x) - z_m^*|^p \right)^{1/p}$. When $p=1$, it reduces to weighted sum; when $p=2$, it minimizes Euclidean distance; and as $p \to \infty$, it yields the weighted Tchebycheff metric $\min_{x \in S} \max_{m} w_m |f_m(x) - z_m^*|$, which minimizes the worst-case weighted regret.

#### 2. Defining Mechanism
Distance minimization to the ideal point in objective space using $L_p$ metrics, especially the augmented Tchebycheff metric with a small regularization term $\rho \sum w_m |f_m - z_m^*|$.

#### 3. Advantages
- The weighted Tchebycheff metric ($p=\infty$) can reach every Pareto-optimal point of any non-convex Pareto set with appropriate weights.
- Allows decision-makers to incorporate geometric preferences by adjusting the ideal point $z^*$ and weighting factors $w$.
- Augmented Tchebycheff formulations eliminate weakly Pareto-optimal solutions by adding a small positive linear term.
- Provides a continuous bridge between aggregate group utility ($p=1$), Euclidean compromise ($p=2$), and minimax fairness ($p=\infty$).

#### 4. Disadvantages
- Requires prior computation or estimation of the ideal objective point $z^* = (\min f_1, \dots, \min f_M)$.
- The pure Tchebycheff metric ($p=\infty$) can produce weakly Pareto-optimal solutions on horizontal or vertical front segments unless augmented.
- The non-smooth $\max$ operator requires introducing an auxiliary scalar $\alpha$ with $M$ linear inequalities, complicating optimization for derivative-based solvers.
- High sensitivity to objective normalization; unequal scaling severely distorts the metric hyperspheres/hypercubes.

#### 5. Suitable Problems
Well-suited for compromise programming, non-convex bi-objective and tri-objective problems, and operational research where balancing worst-case performance is required.

#### 6. Limitations
Non-differentiability of $p=\infty$; requires reliable ideal point estimation; requires normalization to avoid metric distortion.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg04_weight_metric.py`. Implements $p=1, 2$ and augmented Tchebycheff ($p=\infty$ with $\rho=10^{-4}$) using SciPy SLSQP. Evaluated on P1, P3, P4, P5.

#### 8. Complexity Analysis
Time: $O(K \cdot C_{\text{NLP}})$ where $K$ is the number of weight vectors. Space: $O(n + M)$.

---

### Algorithm 5: Value Function
- **Module File:** `moo/algorithms/alg05_value_function.py`
- **Algorithm Family:** Classical Utility / Preference
- **General Approach:** Deterministic / Direct Utility Optimization

#### 1. Core Idea
Assumes that the decision-maker possesses an underlying scalar utility or value function $U: \mathbb{R}^M \to \mathbb{R}$ that completely captures their preference structure, risk tolerance, and trade-off rates across all objectives. The multiobjective optimization problem is replaced by the single-objective problem: $\max_{x \in S} U(f_1(x), \dots, f_M(x))$ (or minimizing an aggregate cost/loss function), transforming the search into a targeted quest for the single optimal compromise point.

#### 2. Defining Mechanism
Direct optimization of an explicit multi-attribute utility function (e.g., additive, Cobb-Douglas multiplicative, or exponential risk utility) that maps the objective vector to a single scalar preference value.

#### 3. Advantages
- Delivers exactly one actionable, optimal decision tailored to the stakeholder's explicit value system, eliminating decision paralysis.
- Naturally handles any number of objectives ($M \ge 3$) because the problem is immediately collapsed into a single scalar objective.
- Can incorporate non-linear trade-offs, diminishing marginal returns, and risk aversion parameters directly into the utility formulation.
- Completely sidesteps the heavy computational burden of computing, storing, and visualizing high-dimensional Pareto frontiers.

#### 4. Disadvantages
- Severe practical difficulty in eliciting and validating an explicit mathematical utility function from human decision-makers.
- Assumes preference consistency across the entire objective space, ignoring cognitive biases and local trade-off shifts.
- Provides no visibility into alternative Pareto-optimal trade-offs that might offer massive improvements in one objective for a tiny sacrifice in utility.
- If the utility function is non-linear or non-convex, it can introduce multiple local optima that trap gradient-based optimization solvers.

#### 5. Suitable Problems
Best applied when strong econometric, economic, or policy models exist with well-calibrated utility functions (e.g., portfolio utility with known risk aversion).

#### 6. Limitations
Complete reliance on a priori preference articulation; failure to provide trade-off context; risk of model misspecification.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg05_value_function.py`. Evaluates weighted Cobb-Douglas and polynomial utility forms, optimizing via SciPy SLSQP. Evaluated on P1, P3, P4, P7.

#### 8. Complexity Analysis
Time: $O(C_{\text{single}})$ (only a single optimization run is required). Space: $O(n + M)$.

---

### Algorithm 6: Goal Programming
- **Module File:** `moo/algorithms/alg06_goal_programming.py`
- **Algorithm Family:** Classical Scalarization
- **General Approach:** Deterministic / Deviational Goal Satisfaction

#### 1. Core Idea
Instead of minimizing objectives directly, the decision-maker specifies aspirational targets $T_m$ for each objective. For each goal, positive and negative deviational variables are introduced: $f_m(x) - d_m^+ + d_m^- = T_m$, with $d_m^+, d_m^- \ge 0$ and $d_m^+ \cdot d_m^- = 0$. The optimizer minimizes an achievement function penalizing unwanted deviations. In Weighted Goal Programming, a weighted sum $\sum (w_m^+ d_m^+ + w_m^- d_m^-)$ is minimized; in Preemptive (Lexicographic) Goal Programming, priority levels are satisfied hierarchically.

#### 2. Defining Mechanism
Splitting objective shortfalls into directional deviational variables ($d_m^+, d_m^-$) and minimizing only the unwanted deviation components relative to explicit aspiration targets.

#### 3. Advantages
- Intuitively aligns with real-world managerial decision-making, where executives think in terms of quarterly budgets, deadlines, and production quotas rather than abstract Pareto fronts.
- Can distinguish between overachievement and underachievement, penalizing only undesirable deviations (e.g., costs exceeding budget vs profit exceeding target).
- Supports both weighted (Archimedean) and preemptive (lexicographic priority) goal structures to reflect non-negotiable versus negotiable goals.
- Can be cast as a linear program for linear constraints and objectives, allowing industrial solvers to handle thousands of variables efficiently.

#### 4. Disadvantages
- If target goals $T_m$ are set too pessimistically (inside the feasible region), the algorithm stops at the interior target, yielding an inefficient, Pareto-dominated point.
- Requires substantial upfront cognitive effort to set realistic, mutually consistent target levels for all objectives.
- Weighted goal programming is highly sensitive to the normalization of deviational variables across disparate physical units.
- Preemptive goal programming can completely starve lower-priority goals if higher-priority goals consume all degrees of freedom.

#### 5. Suitable Problems
Widely utilized in manufacturing scheduling, production planning (P11), financial budgeting, supply chain management, and resource allocation with known targets.

#### 6. Limitations
Risk of Pareto inefficiency under pessimistic targets; requires expert elicitation of targets and priorities; normalization sensitive.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg06_goal_programming.py`. Supports both weighted and preemptive modes with explicit $d_m^+, d_m^-$ deviation tracking. Tested thoroughly on production planning P11 and P7.

#### 8. Complexity Analysis
Time: $O(P \cdot C_{\text{NLP}})$ where $P$ is the number of priority levels (or 1 for weighted mode). Space: $O(n + 2M)$.

---

### Algorithm 7: Lexicographic
- **Module File:** `moo/algorithms/alg07_lexicographic.py`
- **Algorithm Family:** Classical Hierarchical
- **General Approach:** Deterministic / Sequential Priority Cascade

#### 1. Core Idea
Ranks objectives in a strict hierarchical order of priority: $f_1 \gg f_2 \gg \dots \gg f_M$. The algorithm first solves $\min_{x \in S} f_1(x)$ to obtain optimal value $f_1^*$. In the second step, it solves $\min_{x \in S} f_2(x)$ subject to the additional constraint $f_1(x) \le f_1^* + \tau_1$, where $\tau_1 \ge 0$ is a tolerance window. This sequential cascade continues through all $M$ objectives, progressively narrowing the feasible space while preserving earlier achievements.

#### 2. Defining Mechanism
Strict non-compensatory priority ordering where lower-priority objectives are only evaluated across the alternate optimal solutions (or within a tolerance window) of higher-priority objectives.

#### 3. Advantages
- Reflects non-compensatory decision policies where certain criteria (e.g., human safety or legal compliance) must strictly dominate economic criteria without trade-offs.
- Eliminates the subjective and error-prone task of assigning numerical weights or trade-off rates between incomparable dimensions.
- Guarantees finding a Pareto-optimal solution if all objectives are strictly minimized across the cascade.
- Computationally efficient because it solves a sequence of standard single-objective constrained optimization problems without evolutionary overhead.

#### 4. Disadvantages
- Severe risk of lower-priority starvation: if the highest-priority objective has a unique global minimum, subsequent objectives have no influence whatsoever.
- Requires introducing arbitrary tolerance parameters (slacks) $\tau_k$ if the user wants lower-priority objectives to have meaningful impact.
- The final solution is extremely sensitive to the chosen priority permutation; swapping the rank of two objectives produces completely divergent outcomes.
- Does not provide a complete Pareto front, requiring $M!$ priority permutations to explore full trade-off possibilities.

#### 5. Suitable Problems
Safety-critical engineering, medical triage, mission-critical space/defense systems, and legal/regulatory compliance workflows where primary requirements cannot be traded off.

#### 6. Limitations
Strict priority hierarchy required; lower-level objectives often redundant without slacks; permutation explosion ($M!$) for full exploration.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg07_lexicographic.py`. Implements strict priority cascading with configurable relative/absolute tolerance slacks $\tau_k$, tested on P7 and P11.

#### 8. Complexity Analysis
Time: $O(M \cdot C_{\text{NLP}})$ (exactly $M$ sequential constrained solves). Space: $O(n + M)$.

---

### Algorithm 8: TOPSIS
- **Module File:** `moo/algorithms/alg08_topsis.py`
- **Algorithm Family:** Multi-Criteria Decision Making (MCDM)
- **General Approach:** Deterministic / Geometric Relative Closeness to Ideal

#### 1. Core Idea
Technique for Order Preference by Similarity to Ideal Solution (Hwang & Yoon, 1981). Given a decision matrix of $N$ alternatives evaluated on $M$ criteria: 1) Vector-normalizes the matrix to eliminate unit differences; 2) Multiplies by criteria weights $w_m$; 3) Identifies the Positive Ideal Solution (PIS, best score per criterion) and Negative Ideal Solution (NIS, worst score per criterion); 4) Calculates Euclidean distances $S_i^+$ to PIS and $S_i^-$ to NIS; 5) Ranks alternatives by relative closeness $C_i = \frac{S_i^-}{S_i^+ + S_i^-} \in [0, 1]$.

#### 2. Defining Mechanism
Ranking discrete alternatives by maximizing the distance to the Negative Ideal Solution while simultaneously minimizing the distance to the Positive Ideal Solution.

#### 3. Advantages
- Extremely fast analytical computation ($O(N \cdot M)$), requiring zero iterative optimization or function evaluations.
- Simultaneously considers proximity to ideal performance and avoidance of worst-case failure modes.
- Easily handles dozens of heterogeneous criteria with disparate physical units via vector normalization.
- Provides a clear, continuous ranking index $C_i \in [0, 1]$ allowing complete preference ordering of all evaluated alternatives.

#### 4. Disadvantages
- Does not generate new solutions or explore decision space; can only rank a finite pre-existing set of candidates.
- Susceptible to rank reversal phenomenon: adding or removing an irrelevant alternative can alter the relative ranking of remaining alternatives.
- Euclidean distance assumption treats criteria as orthogonal, ignoring inter-attribute correlations and covariance.
- Sensitive to subjective weight assignments; small perturbations in weights can alter top-ranked candidates.

#### 5. Suitable Problems
A posteriori selection from Pareto front archives, supplier selection (P12), vendor evaluation, material selection, and capital project ranking.

#### 6. Limitations
Requires pre-computed alternatives; cannot perform continuous search; rank reversal vulnerability; assumes Euclidean metric.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg08_topsis.py`. Takes decision matrix, weights, and criteria directions; performs vector normalization and computes relative closeness $C_i$. Evaluated on P12 and P4 front. Implementation Fallback: When invoked via `run(problem, budget)` on a continuous problem without an explicit `alternatives_matrix` or `get_decision_matrix()`, the module samples $N = \max(1, \min(20, \text{budget}))$ uniform random candidates across the decision bounds $[x_l, x_u]$. It evaluates their natural objective values ($n_{\text{evals}} \le 20$) without transforming to minimization, constructs an ad-hoc decision matrix, and ranks this finite sample table. It returns the single top-ranked candidate as $X_{\text{opt}}$ with an evaluated front of size $\le 20$, functioning strictly as a candidate-ranking filter rather than an exploratory Pareto front generator.

#### 8. Complexity Analysis
Time: $O(N \cdot M)$ where $N$ is candidate count and $M$ is criteria count. Space: $O(N \cdot M)$.

---

### Algorithm 9: VIKOR
- **Module File:** `moo/algorithms/alg09_vikor.py`
- **Algorithm Family:** Multi-Criteria Decision Making (MCDM)
- **General Approach:** Deterministic / Compromise Ranking Index

#### 1. Core Idea
VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje, Opricovic, 1998) determines a compromise solution for a discrete set of alternatives under conflicting criteria. It computes: 1) Group utility $S_i = \sum_{m=1}^M w_m \frac{f_m^* - f_{mi}}{f_m^* - f_m^-}$ (Manhattan distance); 2) Individual regret $R_i = \max_{m} w_m \frac{f_m^* - f_{mi}}{f_m^* - f_m^-}$ (Chebyshev distance); 3) Compromise index $Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1-v) \frac{R_i - R^*}{R^- - R^*}$ where $v$ weights consensus vs regret; 4) Validates Acceptable Advantage ($Q(a'') - Q(a') \ge DQ$) and Acceptable Stability in decision making.

#### 2. Defining Mechanism
Dual-metric compromise index $Q$ integrating $L_1$ group utility and $L_\infty$ individual regret, accompanied by two rigorous decision-validation acceptance criteria.

#### 3. Advantages
- Explicitly models compromise by balancing collective group benefit (majority rule) against personal grievance (individual regret).
- Includes formal acceptance criteria (Acceptable Advantage and Acceptable Stability) that identify when a single alternative is clearly superior versus when a compromise set must be reported.
- Linear normalization preserves relative proportional distances between alternative values better than vector normalization.
- Enables strategic sensitivity analysis by varying the decision mechanism weight $v \in [0, 1]$ (consensus vs veto power).

#### 4. Disadvantages
- Cannot explore continuous or combinatorial search spaces; operates strictly on static alternative tables.
- Normalization requires reliable determination of best ($f^*$) and worst ($f^-$) values, which are easily skewed by unrealistic outlier options.
- When the acceptable advantage threshold $DQ = 1/(N-1)$ is not met, the method outputs a subset rather than a definitive ranking, requiring secondary adjudication.
- Subject to rank reversal when alternatives are added or deleted from the comparison pool.

#### 5. Suitable Problems
Public policy decision-making, municipal planning, collaborative engineering design, supplier selection (P12), and ranking Pareto front subsets.

#### 6. Limitations
Discrete evaluation only; potential for multi-alternative tie results under tight advantage thresholds; weight sensitivity.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg09_vikor.py`. Computes group utility $S$, individual regret $R$, and compromise index $Q$ across configurable $v$ values (e.g. $v=0.25, 0.5, 0.75$) and validates compromise conditions C1 (Acceptable Advantage) and C2 (Acceptable Stability). Tested on P12. Implementation Fallback: When invoked via `run(problem, budget)` on a continuous problem lacking an alternatives matrix, the module samples $N = \max(1, \min(20, \text{budget}))$ uniform candidates across decision bounds $[x_l, x_u]$, evaluates their natural objectives ($n_{\text{evals}} \le 20$), and ranks this finite sample table. It returns the single top compromise candidate as $X_{\text{opt}}$ along with the sampled front, operating strictly as a multi-criteria ranking post-processor rather than a generative Pareto front optimizer.

#### 8. Complexity Analysis
Time: $O(N \cdot M + N \log N)$ where $N$ is candidate alternative count and $M$ is criteria count (matrix normalization, utility distance aggregation, and final sorting). Space: $O(N \cdot M)$.

---

### Algorithm 10: VEGA
- **Module File:** `moo/algorithms/alg10_vega.py`
- **Algorithm Family:** First-Generation MOEA
- **General Approach:** Population-based / Heuristic Objective Decomposition

#### 1. Core Idea
The Vector Evaluated Genetic Algorithm (Schaffer, 1985) was the first multi-objective evolutionary algorithm. In each generation, the population of size $N$ is conceptually partitioned into $M$ equal sub-populations of size $N/M$. Each sub-population undergoes selection (e.g., proportional or tournament) based entirely on a single objective $f_m$. The resulting selected sub-populations are then merged together, shuffled uniformly, and subjected to standard crossover and mutation to produce the next generation.

#### 2. Defining Mechanism
Fractional population partitioning where each cohort is selected based on a single objective dimension before global recombination.

#### 3. Advantages
- Historically foundational and exceptionally easy to implement by wrapping single-objective selection operators without Pareto sorting.
- Computationally rapid per generation, requiring only $O(N)$ selection operations without expensive non-dominated sorting or distance calculations.
- Naturally excels at finding and refining the extreme boundary points (individual objective minima) of the Pareto front.
- Requires no niche parameters, sharing radii, or reference lines to maintain basic population dispersion.

#### 4. Disadvantages
- Severe selection bias toward extreme solutions (speciation), often completely missing intermediate compromise solutions on the Pareto front.
- Mating an individual excellent in $f_1$ with an individual excellent in $f_2$ frequently produces 'middling' offspring that are mediocre in both objectives (genetic drift).
- Cannot navigate non-convex or concave Pareto geometries effectively, collapsing population diversity into disconnected clusters.
- Lacks elitism, leading to significant generational fitness oscillations and loss of discovered Pareto-optimal solutions.

#### 5. Suitable Problems
Historical benchmarking, simple convex bi-objective problems where finding individual objective extremes is the primary priority.

#### 6. Limitations
Ineffective for $M \ge 3$; fails on non-convex fronts; severe intermediate point neglect; non-elitist.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg10_vega.py`. Divides population into $M$ cohorts, performs tournament selection on each objective, shuffles, applies SBX crossover and polynomial mutation. Evaluated on P1, P2, P3.

#### 8. Complexity Analysis
Time: $O(N \cdot n + N \cdot M)$ per generation (linear in population size, dimension, and objectives). Space: $O(N \cdot n)$.

---

### Algorithm 11: MOGA
- **Module File:** `moo/algorithms/alg11_moga.py`
- **Algorithm Family:** First-Generation MOEA
- **General Approach:** Population-based / Pareto Dominance Ranking with Objective Niche Sharing

#### 1. Core Idea
Multi-Objective Genetic Algorithm (Fonseca & Fleming, 1993). An individual $x_i$ is evaluated by counting how many individuals in the current population dominate it: $r_i = 1 + |\{j \in P \mid x_j \prec x_i\}|$. All non-dominated individuals receive rank 1. Fitness is assigned by interpolating a monotonically decreasing function over sorted ranks, and then averaged among individuals of identical rank. To maintain diversity, shared fitness is calculated using a triangular sharing function $Sh(d) = \max(0, 1 - d/\sigma_{\text{share}})$ based on Euclidean distance in objective space.

#### 2. Defining Mechanism
Dominance-count ranking ($1 +$ count of dominating solutions) paired with objective-space niche sharing among individuals with identical rank.

#### 3. Advantages
- Ensures all non-dominated individuals receive identical baseline reproductive opportunity, removing the extreme-point bias of VEGA.
- Can discover and preserve solutions across non-convex and disconnected Pareto frontiers.
- Introduced structured objective-space fitness sharing, encouraging the population to distribute evenly across discovered trade-offs.
- Flexible ranking framework that allows straightforward integration of goal or priority preferences.

#### 4. Disadvantages
- Extreme sensitivity to the niche sharing radius parameter $\sigma_{\text{share}}$; inappropriate values cause either overcrowding or severe genetic drift.
- High computational cost of $O(N^2)$ for pairwise objective distance calculations in every generation.
- Lacks an elitist survival mechanism, meaning high-quality Pareto solutions can be lost due to destructive crossover or mutation.
- Selection pressure degrades rapidly as the fraction of non-dominated individuals increases in higher-dimensional objective spaces.

#### 5. Suitable Problems
Continuous and discrete bi-objective engineering problems where a well-spread front is desired and parameters can be calibrated.

#### 6. Limitations
Requires manual tuning of $\sigma_{\text{share}}$; $O(N^2)$ sharing overhead; lacks elitism; poor scaling for $M \ge 4$.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg11_moga.py`. Implements dominance count ranking, rank-averaged fitness assignment, and normalized objective-space sharing. Evaluated on P1, P2, P3, P6.

#### 8. Complexity Analysis
Time: $O(M N^2)$ for dominance ranking + $O(M N^2)$ for niche sharing per generation. Space: $O(N \cdot n + N^2)$.

---

### Algorithm 12: NSGA
- **Module File:** `moo/algorithms/alg12_nsga.py`
- **Algorithm Family:** First-Generation MOEA
- **General Approach:** Population-based / Non-Dominated Layered Sorting with Front Dummy Fitness

#### 1. Core Idea
Non-Dominated Sorting Genetic Algorithm (Srinivas & Deb, 1994). The population is classified into hierarchical non-dominated layers (fronts). All non-dominated solutions are identified as Front 1, assigned a dummy fitness $F_1$, and shared within Front 1 using a phenotypic sharing function $Sh(d)$. To preserve hierarchy, Front 2 is identified from remaining solutions, assigned a dummy fitness $F_2 = \min(F_{1,\text{shared}}) - \epsilon$, and shared within Front 2. This continues until all solutions are classified, followed by standard stochastic selection.

#### 2. Defining Mechanism
Layer-by-layer non-dominated front classification paired with dummy fitness assignment that explicitly guarantees any point in front $k$ has higher fitness than any point in front $k+1$.

#### 3. Advantages
- Establishes a principled Pareto hierarchy where superior front members always have higher selection probability than inferior front members.
- Maintains excellent diversity across fronts by performing fitness sharing strictly within each individual front layer.
- Capable of tracing complex, disconnected, and non-convex Pareto fronts without shape distortions.
- Provides a natural multi-front population distribution that prevents premature convergence to local sub-fronts.

#### 4. Disadvantages
- High computational complexity of the original sorting algorithm: $O(M N^3)$ in the worst case, making it very slow for large populations.
- Relies heavily on the user-defined sharing parameter $\sigma_{\text{share}}$, which requires prior knowledge of the front spread.
- Original NSGA lacks elitism, allowing elite solutions from early generations to be discarded during stochastic reproduction.
- Dummy fitness step sizes $\epsilon$ can lead to numerical underflow or precision loss when many fronts are present.

#### 5. Suitable Problems
Bi-objective engineering optimization problems with continuous or combinatorial variables where population sizes are modest ($N \le 100$).

#### 6. Limitations
$O(M N^3)$ sorting overhead; sensitive to sharing radius; non-elitist; replaced in modern applications by NSGA-II.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg12_nsga.py`. Uses classic layered sorting, front dummy fitness assignment with margin $\epsilon=0.05$, and objective-space niche sharing. Evaluated on P1, P2, P3, P5.

#### 8. Complexity Analysis
Time: $O(M N^3)$ sorting + $O(N^2)$ sharing per generation. Space: $O(N \cdot n + N^2)$.

---

### Algorithm 13: NPGA
- **Module File:** `moo/algorithms/alg13_npga.py`
- **Algorithm Family:** First-Generation MOEA (Task C Deep Dive #1)
- **General Approach:** Population-based / Pareto Tournament Selection against Sampled Comparison Set

#### 1. Core Idea
Niched Pareto Genetic Algorithm (Horn, Nafpliotis, & Goldberg, 1994). Specifically designed to eliminate the heavy $O(M N^3)$ sorting overhead of NSGA and MOGA. In each selection step: 1) Two candidates $c_1, c_2$ are randomly picked from the population; 2) A comparison set of size $t_{\text{dom}}$ is randomly sampled from the population; 3) If $c_1$ dominates the comparison set and $c_2$ does not, $c_1$ wins; if $c_2$ dominates and $c_1$ does not, $c_2$ wins; 4) If both or neither dominate the comparison set (a tie), the candidate with the smaller niche count (calculated using sharing radius $\sigma_{\text{share}}$) wins.

#### 2. Defining Mechanism
Pareto tournament selection against a small random comparison set of size $t_{\text{dom}}$ combined with niche-count tie-breaking.

#### 3. Advantages
- Exceptional computational efficiency: replaces $O(M N^3)$ sorting with $O(N \cdot t_{\text{dom}} \cdot M)$ tournament checks, scaling linearly with population size for fixed $t_{\text{dom}}$.
- Niche sharing calculations are only evaluated on demand when a tournament tie occurs, dramatically reducing distance evaluations.
- Enables fine-grained tuning of selection pressure by simply adjusting the comparison set parameter $t_{\text{dom}}$ (typically 5% to 15% of $N$).
- Demonstrated strong performance on binary and combinatorial optimization problems (e.g., knapsack, feature selection).

#### 4. Disadvantages
- Introduces two sensitive interdependent hyperparameters: comparison set size $t_{\text{dom}}$ and sharing radius $\sigma_{\text{share}}$.
- When $t_{\text{dom}}$ is too small ($t_{\text{dom}} < 5$), selection pressure is insufficient to drive the population toward the true Pareto front.
- When $t_{\text{dom}}$ is too large ($t_{\text{dom}} > 20$), premature convergence occurs and diversity collapses despite niche tie-breaking.
- Original formulation lacks elitism, requiring an external archive or modern $(\mu+\lambda)$ survival to guarantee monotonic hypervolume growth.

#### 5. Suitable Problems
Continuous and discrete multi-objective problems requiring rapid generations, large populations ($N \ge 500$), and binary feature selection (P9).

#### 6. Limitations
Requires careful tuning of $t_{\text{dom}}$ and $\sigma_{\text{share}}$; stochastic selection error from comparison set sampling; non-elitist.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg13_npga.py` as our Roll 13 Task C Deep Dive #1. Verified with exact 2-iteration hand traces on P1 and 10-seed parameter sensitivity studies across $t_{\text{dom}} \in \{2, 10, 25\}$ on P3 and P5.

#### 8. Complexity Analysis
Time: $O(N \cdot t_{\text{dom}} \cdot M)$ per generation (massively faster than NSGA). Space: $O(N \cdot n)$.

---

### Algorithm 14: NSGA-II
- **Module File:** `moo/algorithms/alg14_nsga2.py`
- **Algorithm Family:** Second-Generation Elitist MOEA
- **General Approach:** Population-based / Elitist Non-Dominated Sorting with Crowding Distance

#### 1. Core Idea
Elitist Non-Dominated Sorting Genetic Algorithm II (Deb, Pratap, Agarwal, & Meyarivan, IEEE TEVC 2002). Replaces the $O(M N^3)$ sorting of NSGA with a fast $O(M N^2)$ non-dominated sorting algorithm that maintains domination counts $n_p$ and lists of dominated solutions $S_p$. Replaces manual fitness sharing ($\sigma_{\text{share}}$) with a parameter-free crowding distance metric measuring the cuboid perimeter formed by neighboring solutions along each objective. Parent population $P_t$ and offspring population $Q_t$ of size $N$ are pooled into $R_t$ ($2N$), sorted into fronts, and the best $N$ solutions are selected elitistically with the critical front truncated by crowding distance.

#### 2. Defining Mechanism
Fast $O(M N^2)$ non-dominated sorting combined with parameter-free crowding distance truncation inside a $(N+N)$ elitist selection loop.

#### 3. Advantages
- Fast $O(M N^2)$ non-dominated sorting reduces computational overhead dramatically compared to original NSGA ($O(M N^3)$).
- Parameter-free diversity preservation: crowding distance completely eliminates the need to specify or calibrate a sharing radius $\sigma_{\text{share}}$.
- Strict $(N+N)$ elitist survival guarantees that non-dominated solutions discovered in earlier generations cannot be lost, ensuring monotonic front improvement.
- Exceptional robustness and versatility across continuous, binary, and combinatorial domains; widely considered the standard benchmark algorithm in the MOO literature.

#### 4. Disadvantages
- Severe performance degradation in many-objective problems ($M \ge 4$): nearly all population members become mutually non-dominated, rendering Pareto sorting ineffective.
- Crowding distance has an inherent boundary bias, over-favoring extreme points and producing sparse distributions in high-dimensional manifolds.
- Crowding distance calculations only measure distances along individual objective axes, failing to capture lateral crowding or cluster geometry.
- Parent-offspring pool of size $2N$ doubles the memory and sorting overhead relative to steady-state algorithms.

#### 5. Suitable Problems
Continuous bi-objective and tri-objective benchmarks (P1, P2, P3, P4, P5, P6, P7), mixed-integer engineering design, and general-purpose 2-3 objective optimization.

#### 6. Limitations
Ineffective for $M \ge 4$; boundary point bias in high dimensions; requires specialized constraint handling extensions for heavily constrained spaces.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg14_nsga2.py`. Features fast non-dominated sorting, normalized crowding distance calculation, SBX crossover ($\eta_c=20$), polynomial mutation ($\eta_m=20$), and binary tournament selection. Evaluated on P1, P2, P3, P4, P5, P6, P7.

#### 8. Complexity Analysis
Time: $O(M N^2)$ per generation for sorting + $O(M N \log N)$ for crowding distance. Space: $O(M N + n N)$.

---

### Algorithm 15: NSGA-III
- **Module File:** `moo/algorithms/alg15_nsga3.py`
- **Algorithm Family:** Many-Objective Reference-Based MOEA
- **General Approach:** Population-based / Reference-Point-Based Selection on Hyperplane

#### 1. Core Idea
Reference-Point-Based Many-Objective Evolutionary Algorithm (Deb & Jain, IEEE TEVC 2014). Designed specifically to conquer the many-objective breakdown of NSGA-II ($M \ge 4$). It retains fast non-dominated sorting to sort the $2N$ pool into fronts. In the critical front $F_l$, instead of crowding distance, it: 1) Normalizes objectives using an adaptive hyperplane formed by extreme points; 2) Generates structured uniform reference points on the unit simplex (Das-Dennis method); 3) Associates each population member with its closest reference line using perpendicular distance; 4) Executes a niche-preservation selection mechanism that prioritizes reference points with the lowest current association counts (niche count $\rho_j$).

#### 2. Defining Mechanism
Reference-point-guided environmental selection using Das-Dennis structured simplex points and perpendicular Euclidean distance association instead of crowding distance.

#### 3. Advantages
- Successfully scales to many-objective problems ($M = 3, 5, 10, 15$) where NSGA-II completely loses selection pressure.
- Maintains a highly uniform, structured distribution of solutions across the entire high-dimensional Pareto manifold.
- Eliminates the boundary bias of crowding distance, providing balanced coverage of both center and boundary regions.
- Adaptive normalization accurately maps disparate objective ranges onto a standardized unit simplex without manual user weighting.

#### 4. Disadvantages
- Combinatorial explosion of reference points: the number of reference points $\binom{M+p-1}{p}$ grows rapidly with $M$, requiring huge population sizes unless layered sampling is used.
- Assumes the true Pareto front aligns reasonably well with the unit hyperplane; performance degrades on irregular, disconnected, or inverted simplex fronts.
- Computing extreme points, matrix inversion for hyperplane intercepts, and perpendicular distances adds significant computational overhead.
- Offers no distinct advantage over NSGA-II on simple 2-objective problems where crowding distance is already highly effective.

#### 5. Suitable Problems
Many-objective continuous benchmarks (P10 DTLZ2 with $M=3, 5, 10$), complex industrial multi-criteria design with 4 to 15 conflicting criteria.

#### 6. Limitations
Reference point combinatorial scaling; sensitive to intercept computation when extreme points are collinear; overkill for $M=2$.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg15_nsga3.py`. Implements Das-Dennis reference point generator, extreme point intercept estimation, perpendicular distance association, and niche-preservation truncation. Evaluated on P10.

#### 8. Complexity Analysis
Time: $O(M N^2)$ for sorting + $O(N K M)$ for reference point association, where $K$ is reference point count. Space: $O(K M + N n)$.

---

### Algorithm 16: SPEA2
- **Module File:** `moo/algorithms/alg16_spea2.py`
- **Algorithm Family:** Second-Generation Elitist MOEA
- **General Approach:** Population-based / Strength-Based Fitness with $k$-NN Density Estimation

#### 1. Core Idea
Strength Pareto Evolutionary Algorithm 2 (Zitzler, Laumanns, & Thiele, 2001). Maintains a regular population of size $N$ and a fixed-size external archive of size $\bar{N}$. In each generation: 1) Calculates strength $S(i)$ for every individual in both pools, equal to the number of solutions it dominates; 2) Calculates raw fitness $R(i) = \sum_{j \succ i} S(j)$, summing the strength of all solutions that dominate it (non-dominated solutions have $R(i)=0$); 3) Adds density $D(i) = 1 / (\sigma_i^k + 2)$ where $\sigma_i^k$ is Euclidean distance to the $k$-th nearest neighbor ($k = \sqrt{N+\bar{N}}$); 4) Total fitness $F(i) = R(i) + D(i)$; 5) Fills the archive with all $F(i) < 1$ solutions; if the archive is overfilled, an iterative truncation procedure successively removes the solution with the minimum distance to its nearest neighbors.

#### 2. Defining Mechanism
Combined strength-dominance fitness with $k$-nearest-neighbor ($k$-NN) density estimation and nearest-neighbor archive truncation.

#### 3. Advantages
- Fine-grained fitness assignment: solutions dominating many individuals get rewarded, while solutions dominated by many individuals are severely penalized.
- Density estimation using the $k$-th nearest neighbor ($k = \sqrt{N+\bar{N}}$) provides a more statistically robust estimate of local crowding than crowding distance.
- The archive truncation operator preserves boundary solutions and systematically eliminates clustering without distorting front shape.
- External archive guarantees strong elitism and preserves high-quality non-dominated solutions independently of mating population dynamics.

#### 4. Disadvantages
- High computational cost: computing pairwise distances for $N+\bar{N}$ solutions and sorting distances takes $O((N+\bar{N})^2 \log(N+\bar{N}))$.
- Archive truncation is particularly slow when many non-dominated solutions are discovered, requiring iterative distance updates.
- Requires specifying two population size parameters: regular population $N$ and external archive size $\bar{N}$.
- Performance still degrades on many-objective problems ($M \ge 5$) as the raw fitness $R(i)$ becomes 0 for almost the entire population.

#### 5. Suitable Problems
Continuous and combinatorial multi-objective problems with 2 to 4 objectives where high-quality front diversity and rigorous elitism are required.

#### 6. Limitations
Computational overhead of $k$-NN sorting; dual population parameter tuning; degrades for $M \ge 5$.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg16_spea2.py`. Features strength calculation, raw fitness summation, $k = \lfloor \sqrt{N+\bar{N}} \rfloor$ density estimation, and iterative nearest-neighbor archive truncation. Evaluated on P1, P3, P5, P6.

#### 8. Complexity Analysis
Time: $O(M (N+\bar{N})^2 + (N+\bar{N})^2 \log(N+\bar{N}))$ for strength calculation, raw fitness summation, and all-pairs $k$-NN density sorting per generation, where $N$ is population size, $\bar{N}$ is archive size, and $M$ is objective count. If the non-dominated set overfills the archive ($L = |F_1| > \bar{N}$), iterative nearest-neighbor truncation executes $L - \bar{N}$ successive removal steps, where each step sorts the active distance sub-matrix in $O(L^2 \log L)$, contributing up to $O((L - \bar{N}) L^2 \log L)$ worst-case truncation time. Space: $O((N+\bar{N}) (n + M + (N+\bar{N})))$.

---

### Algorithm 17: PESA-II
- **Module File:** `moo/algorithms/alg17_pesa2.py`
- **Algorithm Family:** Second-Generation Elitist MOEA
- **General Approach:** Population-based / Region-Based Hyperbox Selection on Adaptive Grid Archive

#### 1. Core Idea
Pareto Envelope-based Selection Algorithm II (Corne, Jerram, Knowles, & Oates, PPSN 2001). Introduces region-based selection over an external non-dominated archive. The objective space containing the current archive is dynamically partitioned into a hypergrid of equal-sized hyperboxes. Instead of selecting individual solutions, PESA-II's selection operator selects a hyperbox using binary tournament based on squeeze factor (the number of archive solutions residing in the box; lower squeeze factor wins). Once a hyperbox is selected, an individual solution within that box is chosen uniformly at random for genetic recombination.

#### 2. Defining Mechanism
Region-based selection where the primary selection unit is the grid hyperbox rather than the candidate solution, inherently penalizing crowded regions.

#### 3. Advantages
- Significantly reduces selection bias toward densely clustered regions of the Pareto front compared to individual-based tournament selection.
- Fast selection execution: tournament selection over hyperboxes avoids pairwise distance calculations and non-dominated sorting of the offspring.
- Adaptive grid dynamically recalculates objective ranges, accommodating expanding or contracting Pareto fronts without manual bounds.
- Strong elitism achieved through maintaining an external bounded archive of non-dominated solutions.

#### 4. Disadvantages
- Sensitive to the grid division parameter $K_g$ (subdivisions per dimension); too coarse a grid loses resolution, while too fine a grid puts every solution in its own box.
- Grid coordinate assignment has $O(K_g^M)$ potential boxes, becoming memory-prohibitive or sparse in many-objective spaces ($M \ge 5$).
- Rigid orthogonal grid boundaries can arbitrarily split natural clusters, penalizing solutions that happen to fall close to a boundary.
- Internal population is recreated purely from the archive in each generation, which can discard useful genetic diversity from dominated intermediate points.

#### 5. Suitable Problems
Continuous and discrete multi-objective problems with 2 to 4 objectives, routing problems (P8), and resource allocation where rapid diversity maintenance is needed.

#### 6. Limitations
Grid resolution sensitivity; exponential grid space scaling in $M$; artificial boundary artifacts.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg17_pesa2.py`. Features an `AdaptiveHypergrid` archive, squeeze factor tracking, binary tournament over occupied boxes, and random intra-box selection. Evaluated on P1, P3, P8.

#### 8. Complexity Analysis
Time: $O(N \cdot \bar{N} \cdot M + N \log N)$ per generation, where $N$ is offspring population size, $\bar{N}$ is external archive capacity, and $M$ is objective count. Region-based selection over occupied hyperboxes takes $O(N \log N_{\text{box}})$; evaluating and incorporating each of the $N$ offspring requires pairwise Pareto dominance checks against all $\bar{N}$ archive members, taking $O(N \cdot \bar{N} \cdot M)$ dominant time. Grid recalculation and squeeze factor updates take $O(M \cdot \bar{N})$. Space: $O(\bar{N} (n+M) + N (n+M) + N_g)$ where $N_g$ is occupied hyperbox count.

---

### Algorithm 18: MOEA/D
- **Module File:** `moo/algorithms/alg18_moead.py`
- **Algorithm Family:** Decomposition-Based MOEA
- **General Approach:** Population-based / Collaborative Decomposition into Scalar Subproblems

#### 1. Core Idea
Multiobjective Evolutionary Algorithm based on Decomposition (Zhang & Li, IEEE TEVC 2007). Decomposes a multi-objective problem into $N$ scalar optimization subproblems using a set of evenly distributed weight vectors $\lambda^1, \dots, \lambda^N$. Common scalarization methods include Tchebycheff, Weighted Sum, or Penalty-based Boundary Intersection (PBI). For each subproblem $i$, a neighborhood $B(i)$ of the $T$ closest weight vectors is established. In each iteration, subproblem $i$ selects parents exclusively from its neighborhood $B(i)$, generates a new offspring, and updates the solutions of neighboring subproblems if the offspring improves their scalar fitness.

#### 2. Defining Mechanism
Explicit decomposition of the Pareto front into $N$ localized scalar subproblems solved simultaneously through neighborhood-restricted mating and replacement.

#### 3. Advantages
- Completely eliminates the computational cost of non-dominated sorting ($O(M N^2)$) and crowding distance calculations.
- Neighborhood mating and replacement maintain exceptionally strong population diversity naturally across the entire objective space.
- Seamlessly integrates existing single-objective heuristics, local searches, and domain-specific operators into each subproblem.
- Provides an ideal framework for mathematical decomposition, parallel computing, and distributed optimization architectures.

#### 4. Disadvantages
- Uniform weight vectors do not necessarily produce uniformly distributed solutions on complex, non-linear, or steep Pareto fronts.
- Performance depends heavily on the chosen decomposition approach (Tchebycheff vs PBI) and the penalty parameter $\theta$ in PBI.
- Requires careful tuning of the neighborhood size $T$ and maximum replacement count $n_r$ to prevent premature neighborhood takeover.
- Struggles with irregular, disconnected, or non-simplex Pareto front shapes where many predefined weight vectors point to non-existent regions.

#### 5. Suitable Problems
Continuous multi-objective optimization (P1, P3, P4, P5), problems with complex variable linkages, and applications where fast evaluations are critical.

#### 6. Limitations
Decomposition method dependent; weight mismatch on irregular Pareto geometries; neighborhood parameter tuning required.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg18_moead.py`. Generates uniform weights on unit simplex, establishes Euclidean weight neighborhoods $B(i)$ of size $T=10$, uses Tchebycheff scalarization with dynamic ideal point tracking. Evaluated on P1, P3, P4.

#### 8. Complexity Analysis
Time: $O(N \cdot T \cdot M)$ per generation (linear in population size $N$ for fixed neighborhood $T$). Space: $O(N (n + M + T))$.

---

### Algorithm 19: MOEA/D-DE
- **Module File:** `moo/algorithms/alg19_moead_de.py`
- **Algorithm Family:** Decomposition-Based MOEA
- **General Approach:** Population-based / Decomposition with Differential Evolution Operators

#### 1. Core Idea
MOEA/D with Differential Evolution (Li & Zhang, IEEE TEVC 2009). Enhances the original MOEA/D algorithm by replacing standard simulated binary crossover (SBX) with Differential Evolution (DE) mutation and binomial crossover operators: $v_i = x_{r1} + F (x_{r2} - x_{r3})$, followed by crossover with target vector $x_i$ and polynomial mutation. Parents $r_1, r_2, r_3$ are chosen from the subproblem's neighborhood $B(i)$ with high probability $\delta$, or from the global population with probability $1-\delta$. The resulting offspring updates up to $n_r$ neighborhood solutions.

#### 2. Defining Mechanism
Integration of Differential Evolution difference-vector operators ($DE/rand/1/bin$) into the localized subproblem neighborhood framework of MOEA/D.

#### 3. Advantages
- Significantly outperforms SBX-based MOEA/D on continuous problems with rotated, non-separable, or correlated decision variable landscapes.
- Self-adapting step sizes: difference vectors automatically scale down as the population converges, facilitating fine local exploitation.
- Neighborhood-restricted DE prevents disruptive large-scale mutations across distant regions of the Pareto front.
- Includes a replacement limit parameter $n_r$ that prevents a single elite offspring from dominating too many subproblems, preserving diversity.

#### 4. Disadvantages
- Inapplicable to combinatorial, permutation (e.g. TSP), or discrete categorical problems without specialized continuous relaxations.
- Introduces additional hyperparameters: DE scale factor $F$, crossover rate $CR$, neighborhood selection probability $\delta$, and replacement limit $n_r$.
- Still inherits the fundamental vulnerability of MOEA/D to irregular, non-convex, or disconnected Pareto front shapes.
- Can suffer from stagnation if the neighborhood population loses diversity in the difference vectors.

#### 5. Suitable Problems
Continuous benchmarks with complex landscape linkages (ZDT, DTLZ, WFG series), engineering parameter optimization (P3, P4, P5).

#### 6. Limitations
Continuous domains only; sensitive to $F$ and $CR$; requires well-distributed weight vectors.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg19_moead_de.py`. Implements $DE/rand/1/bin$ with $F=0.5, CR=1.0$, polynomial mutation, neighborhood mating probability $\delta=0.9$, and replacement limit $n_r=2$. Evaluated on P1, P3, P5.

#### 8. Complexity Analysis
Time: $O(N \cdot T \cdot M)$ per generation. Space: $O(N (n + M + T))$.

---

### Algorithm 20: MOEA/D-ACO
- **Module File:** `moo/algorithms/alg20_moead_aco.py`
- **Algorithm Family:** Decomposition-Based Metaheuristic
- **General Approach:** Population-based / Decomposition with Ant Colony Combinatorial Construction

#### 1. Core Idea
Combines decomposition with Ant Colony Optimization for multiobjective combinatorial optimization (e.g., bi-objective TSP, P8). Each subproblem $i$ has an associated weight vector $\lambda^i = (\lambda_1^i, \lambda_2^i)$. An ant assigned to subproblem $i$ constructs a discrete tour using a composite heuristic visibility: $\eta_{jk}^i = [1/d_{jk}]^{\lambda_1^i} \cdot [1/t_{jk}]^{\lambda_2^i}$ and shared or subproblem-specific pheromone trails. Solutions are evaluated against the scalarized objective, and the best solutions in each neighborhood deposit pheromones along their visited edges.

#### 2. Defining Mechanism
Decomposition of combinatorial routing objectives using subproblem-weighted heuristic visibilities and neighborhood-focused ant colony pheromone deposition.

#### 3. Advantages
- Extends decomposition principles to combinatorial permutation domains where vector crossover operators are awkward or invalid.
- Allows ants assigned to different subproblems to explore different trade-offs (e.g., cost-heavy vs time-heavy tours) concurrently.
- Collaborative pheromone updates between neighboring subproblems allow shared edge knowledge transfer across similar trade-off regions.
- Constructs strictly valid permutations naturally, completely eliminating constraint violations and repair overhead in TSP routing.

#### 4. Disadvantages
- Substantial memory requirements to maintain pheromone matrices and heuristic tables across multiple subproblem colonies.
- Difficult to balance global pheromone exploration versus neighborhood exploitation without premature stagnation on popular edges.
- Evaluation of tour distances and times for full ant colonies in each iteration can consume the function evaluation budget rapidly.
- Combinatorial nature prevents using standard derivative or gradient acceleration techniques.

#### 5. Suitable Problems
Multi-objective Traveling Salesperson Problem (P8), vehicle routing, network routing, and discrete permutation scheduling.

#### 6. Limitations
Combinatorial / graph routing only; memory intensive; parameter tuning of ant colony coefficients ($\alpha, \beta, \rho$).

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg20_moead_aco.py`. Intended domain: Discrete combinatorial permutation routing (e.g. P8 bi-objective TSP), where each ant assigned to subproblem $i$ constructs a valid tour permutation using composite heuristic visibility $\eta_{jk}^i = [1/d_{jk}]^{\lambda_1^i} \cdot [1/t_{jk}]^{\lambda_2^i}$ and shared pheromone trails. Continuous Problem Fallback: When executed on non-permutation continuous decision spaces (`problem.var_type != "permutation"`), the candidate construction routine bypasses all ant colony graph traversal, pheromones, and heuristic matrices entirely, falling back to uniform random sampling `rng.uniform(problem.xl, problem.xu)` across the decision bounds. Evaluated on P8 TSP.

#### 8. Complexity Analysis
Time: $O(N_{\text{ants}} \cdot C^2 + N_{\text{sub}} \cdot T)$ per generation where $C$ is city count, $N_{\text{ants}}$ is ant count, and $T$ is neighborhood size. Space: $O(C^2 + N_{\text{sub}} (C + M))$.

---

### Algorithm 21: Constrained NSGA-II
- **Module File:** `moo/algorithms/alg21_cnsga2.py`
- **Algorithm Family:** Constrained Evolutionary Algorithm
- **General Approach:** Population-based / Elitist Non-Dominated Sorting with Deb’s Feasibility Rules

#### 1. Core Idea
Constrained NSGA-II (Deb, Pratap, Agarwal, & Meyarivan, 2002). Eliminates subjective penalty functions by embedding Deb's constrained-dominance principle directly into tournament selection and sorting: 1) Any feasible solution dominates any infeasible solution; 2) Between two infeasible solutions, the one with smaller total constraint violation $CV$ dominates; 3) Between two feasible solutions, standard Pareto dominance applies. In our standardized framework, $CV = \sum \max(0, g_j(x)) + \sum \max(0, |h_k(x)| - 10^{-4})$ with $\epsilon_{\text{feas}} = 10^{-4}$.

#### 2. Defining Mechanism
Feasibility-first constrained-dominance binary tournament and sorting, ensuring infeasible solutions are ranked purely by violation severity without penalty multipliers.

#### 3. Advantages
- Completely eliminates penalty parameters, penalty weights, and penalty schedules, which are notoriously difficult to tune in multi-objective problems.
- Guarantees that feasible solutions are always protected and favored over any infeasible solution during environmental selection.
- Directs the search toward the boundary of the feasible space, which is where constrained Pareto-optimal solutions typically reside.
- Seamlessly integrates with the existing fast non-dominated sorting and crowding distance mechanisms of NSGA-II.

#### 4. Disadvantages
- In problems with disjoint, disconnected feasible regions separated by wide infeasible barriers, the algorithm can easily become trapped in a sub-optimal feasible island.
- When the feasible region is extremely small (low volume ratio), the population may spend many generations wandering before finding a single feasible point.
- Heavily favors solutions with low $CV$ over solutions with high $CV$ that might have outstanding objective values and lie just across the boundary.
- Inherits the many-objective crowding distance breakdown of standard NSGA-II for $M \ge 4$.

#### 5. Suitable Problems
Constrained engineering design problems (P5 Cantilever Beam, P6 Binh-Korn), portfolio optimization with budget equalities (P4), and constrained water allocation (P7).

#### 6. Limitations
Vulnerable to disconnected feasible regions; slow on needle-in-a-haystack feasible spaces; limited to $M \le 3$.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg21_cnsga2.py`. Replaces Pareto dominance in NSGA-II with `constrained_dominates` using standard tolerance $10^{-4}$. Evaluated thoroughly on P4, P5, P6, P7.

#### 8. Complexity Analysis
Time: $O(M N^2 + N \cdot K_{\text{constr}})$ per generation. Space: $O(N (n + M + K_{\text{constr}}))$.

---

### Algorithm 22: MOPSO
- **Module File:** `moo/algorithms/alg22_mopso.py`
- **Algorithm Family:** Swarm Intelligence
- **General Approach:** Population-based / Particle Swarm with External Pareto Repository and Adaptive Grid

#### 1. Core Idea
Multi-Objective Particle Swarm Optimization (Coello Coello, Pulido, & Lechuga, IEEE TEVC 2004). Extends single-objective PSO by maintaining an external repository (archive) of non-dominated solutions. In each generation: 1) Personal best $pbest_i$ is updated via Pareto dominance (if new position dominates $pbest_i$, replace; if mutually non-dominating, pick randomly); 2) Objective space of the repository is partitioned into an adaptive grid; 3) A global best guide $gbest_i$ is selected from the repository using roulette wheel selection favoring less crowded hypercubes; 4) Velocity is updated: $v_i = w v_i + c_1 r_1 (pbest_i - x_i) + c_2 r_2 (gbest_i - x_i)$; 5) Position is updated with boundary clipping and mutation.

#### 2. Defining Mechanism
Adaptive grid repository selection for $gbest$ particle guidance combined with Pareto-dominance personal best tracking.

#### 3. Advantages
- Extremely rapid convergence toward the true Pareto front on continuous unimodal and moderately multimodal landscapes.
- Conceptually elegant kinematic equations with low operational overhead per particle update.
- External repository with adaptive grid maintains a well-distributed archive of non-dominated solutions independently of swarm size.
- Turbulence / mutation operators can be easily incorporated to perturb particles and prevent premature swarm stagnation.

#### 4. Disadvantages
- Prone to premature convergence: swarm momentum can cause particles to collapse onto a localized segment of the Pareto front.
- Requires careful tuning and damping of the inertia weight $w$, cognitive parameter $c_1$, social parameter $c_2$, and maximum velocity $v_{\max}$.
- Difficult to apply directly to discrete, integer, or permutation-based decision spaces without velocity discretization.
- Particles frequently overshoot boundaries in constrained problems, requiring aggressive repair or velocity resetting.

#### 5. Suitable Problems
Continuous unconstrained or box-constrained bi-objective problems (P1, P2, P3, P14), parameter calibration, and rapid prototyping.

#### 6. Limitations
Continuous domains primarily; parameter sensitivity ($w, c_1, c_2, v_{\max}$); premature convergence on complex multimodal fronts.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg22_mopso.py`. Features inertia weight decay ($w: 0.9 \to 0.4$), $c_1=1.5, c_2=1.5$, velocity clamping, adaptive grid archive of size 100, and mutation operator. Evaluated on P1, P3, P14.

#### 8. Complexity Analysis
Time: $O(N \cdot n + N_{\text{arch}}^2 \cdot M)$ per generation. Space: $O((N + N_{\text{arch}}) (n + M))$.

---

### Algorithm 23: MOGWO
- **Module File:** `moo/algorithms/alg23_mogwo.py`
- **Algorithm Family:** Swarm Intelligence
- **General Approach:** Population-based / Grey Wolf Hunting Dynamics with Archive Leader Selection

#### 1. Core Idea
Multi-Objective Grey Wolf Optimizer (Mirjalili, Saremi, Mirjalili, & Coelho, 2016). Mimics the social hierarchy and hunting behavior of grey wolves. Maintains an external archive of non-dominated solutions partitioned by an adaptive hypercube grid. In each iteration: 1) Three distinct leaders—alpha ($\alpha$), beta ($\beta$), and delta ($\delta$)—are selected from the least crowded archive hypercubes via roulette wheel selection; 2) For each wolf, hunting positions relative to $\alpha, \beta, \delta$ are computed using coefficient vectors $A = 2a \cdot r_1 - a$ and $C = 2r_2$, where $a$ linearly decreases from 2 to 0; 3) The wolf's position is updated to the average of the three hunting vectors: $x(t+1) = (X_1 + X_2 + X_3)/3$.

#### 2. Defining Mechanism
Hierarchical hunting position update averaging three elite leader solutions ($\alpha, \beta, \delta$) selected from an adaptive grid archive.

#### 3. Advantages
- Three-leader guidance mechanism avoids the single-guide trap of standard PSO, significantly improving exploration and diversity.
- Linear decrease of exploration parameter $a$ from 2 to 0 provides smooth, natural transition from global exploration to fine local exploitation.
- Adaptive grid archive ensures non-dominated solutions are preserved and evenly distributed across the objective space.
- No velocity vectors or inertia parameters to maintain, simplifying particle state representation.

#### 4. Disadvantages
- If $\alpha, \beta, \delta$ are selected from opposite extreme corners of the Pareto front, their average position $(X_1+X_2+X_3)/3$ can pull the wolf into an unpromising interior region.
- Struggles with highly non-convex or disconnected Pareto front topologies where linear combinations of leaders fall into infeasible or dominated space.
- Does not naturally handle discrete or combinatorial decision variables without special mapping functions.
- Archive maintenance and hypercube grid updates add computational overhead as archive size grows.

#### 5. Suitable Problems
Continuous bi-objective and tri-objective engineering problems (P1, P3, P5, P6, P14), continuous mechanical design optimization.

#### 6. Limitations
Continuous domains; linear averaging assumption can fail on concave fronts; archive grid overhead.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg23_mogwo.py`. Implements adaptive grid archive (size 100), roulette wheel selection for $\alpha, \beta, \delta$ leaders, and parameter $a$ linear annealing. Evaluated on P1, P3, P5, P6.

#### 8. Complexity Analysis
Time: $O(N \cdot n + M (N + N_{\text{arch}})^2 + N_{\text{arch}} \cdot M)$ per generation, where $N$ is pack size, $N_{\text{arch}}$ is external archive size, $n$ is decision variable dimension, and $M$ is objective count. Wolf hunting updates cost $O(N \cdot n)$; the archive update stacks pack and archive into a combined pool of size $N + N_{\text{arch}}$, requiring pairwise non-dominated sorting in $O(M (N + N_{\text{arch}})^2)$ dominant time, followed by $O(N_{\text{arch}} \cdot M)$ adaptive grid binning and truncation. Space: $O((N + N_{\text{arch}}) (n + M))$.

---

### Algorithm 24: MOABC
- **Module File:** `moo/algorithms/alg24_moabc.py`
- **Algorithm Family:** Swarm Intelligence
- **General Approach:** Population-based / Foraging Colony Decomposition into Employed, Onlooker, and Scout Bees

#### 1. Core Idea
Multi-Objective Artificial Bee Colony (Hedayatzadeh et al., 2010; Akbari et al., 2012). Simulates the intelligent foraging behavior of honey bee swarms. The colony consists of three groups: 1) Employed bees: associated with specific food sources, generate a mutant neighbor along a random dimension: $v_{ij} = x_{ij} + \phi_{ij} (x_{ij} - x_{kj})$, evaluate Pareto dominance, and increment a failure counter if no improvement occurs; 2) Onlooker bees: observe waggle dances and select food sources with probability proportional to Pareto rank or crowding distance, applying further local perturbations; 3) Scout bees: if a food source's failure counter exceeds a limit parameter, it is abandoned and the scout generates a completely new random solution.

#### 2. Defining Mechanism
Three-phase foraging division (Employed, Onlooker, Scout) with Pareto dominance greedy acceptance and an explicit abandonment threshold for stagnation escape.

#### 3. Advantages
- Scout bee phase provides automatic, principled global re-initialization for exhausted food sources, preventing permanent stagnation in local Pareto traps.
- Fewer algorithm-specific control parameters to tune compared to GA (no crossover/mutation probabilities) or PSO (no velocity limits).
- Balanced division of labor: employed bees exploit existing patches, onlooker bees intensify search on top fronts, and scouts explore globally.
- Can handle multimodal objective landscapes effectively due to the distributed foraging behavior.

#### 4. Disadvantages
- Perturbing only one decision variable at a time ($v_{ij} = x_{ij} + \phi (x_{ij} - x_{kj})$) leads to sluggish convergence on non-separable functions with strong variable interactions.
- Greedy selection in employed/onlooker phases requires deciding how to handle mutually non-dominating candidate solutions.
- Scout bee re-initialization can waste evaluation budget if the limit parameter is set too low, discarding promising semi-converged regions.
- Scales poorly to many-objective spaces ($M \ge 4$) where dominance-based onlooker selection loses discriminatory power.

#### 5. Suitable Problems
Continuous multimodal multi-objective optimization (P1, P3, P5, P6), parameter tuning, and problems with deceptive local Pareto fronts.

#### 6. Limitations
Sluggish on non-separable variables; requires tuning the abandonment `limit` parameter; non-dominated replacement ambiguity.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg24_moabc.py`. Implements employed phase, onlooker phase with dominance rank probabilities, scout abandonment (`limit = 20`), and an external Pareto archive. Evaluated on P1, P3, P5.

#### 8. Complexity Analysis
Time: $O(N \cdot n + M (N + N_{\text{arch}})^2 + N_{\text{arch}} \cdot M)$ per generation, where $N$ is colony size, $N_{\text{arch}}$ is archive size, $n$ is decision variable dimension, and $M$ is objective count. Food source perturbations take $O(N \cdot n)$; the archive update combines population and archive into $N + N_{\text{arch}}$ candidates, requiring pairwise non-dominated sorting in $O(M (N + N_{\text{arch}})^2)$ dominant time, followed by $O(N_{\text{arch}} \cdot M)$ grid truncation. Space: $O((N + N_{\text{arch}}) (n + M))$.

---

### Algorithm 25: MOACO
- **Module File:** `moo/algorithms/alg25_moaco.py`
- **Algorithm Family:** Nature-Inspired / Ant Colony (Task C Deep Dive #2)
- **General Approach:** Population-based / Multiple Pheromones and Heuristics for Combinatorial Routing

#### 1. Core Idea
Multi-Objective Ant Colony Optimization (Barán & Schaerer, 2003; Alaya, Solnon, & Ghedira, 2007). Designed for multiobjective combinatorial problems such as the bi-objective Traveling Salesperson Problem (P8). It maintains two distinct pheromone matrices: $\tau_{ij}^1$ (distance-minimizing trail) and $\tau_{ij}^2$ (time-minimizing trail), along with two heuristic visibility matrices $\eta_{ij}^1 = 1/d_{ij}$ and $\eta_{ij}^2 = 1/t_{ij}$. An ant $k$ situated at city $i$ chooses the next unvisited city $j$ according to the transition probability: $P_{ij}^k \propto [\tau_{ij}^1]^{\alpha_1} [\tau_{ij}^2]^{\alpha_2} [\eta_{ij}^1]^{\beta_1} [\eta_{ij}^2]^{\beta_2}$. In the pheromone update, evaporation $\tau_{ij}^m \leftarrow (1-\rho) \tau_{ij}^m$ occurs, and only ants achieving non-dominated tours in the iteration deposit new pheromone.

#### 2. Defining Mechanism
Dual pheromone and heuristic matrices integrated into a multiplicative ant transition rule, with exclusive pheromone reinforcement from iteration non-dominated ants.

#### 3. Advantages
- Native combinatorial formulation: naturally generates valid tour permutations, completely bypassing repair operators or penalty functions.
- Multi-pheromone representation captures conflicting graph path preferences directly (e.g., short roads vs high-speed highways).
- Pheromone evaporation prevents premature convergence and allows the colony to explore alternative Pareto paths dynamically.
- Pheromone reinforcement restricted strictly to the iteration non-dominated front focuses computational search exclusively on Pareto-efficient edges.

#### 4. Disadvantages
- High spatial memory complexity of $O(M \cdot C^2)$ to store all pheromone and visibility matrices for $C$ cities and $M$ objectives.
- High sensitivity to the evaporation rate $\rho$: small $\rho$ leads to sluggish adaptation, while large $\rho$ erases learned paths prematurely.
- Inapplicable to continuous decision spaces without artificial spatial discretization schemes.
- Computational cost scales quadratically with graph size ($O(C^2)$), making it slow for very large routing instances ($C > 200$).

#### 5. Suitable Problems
Bi-objective Traveling Salesperson Problem (P8), multi-objective vehicle routing (VRP), network packet routing, and combinatorial scheduling.

#### 6. Limitations
Restricted to graph/permutation problems; memory footprint scales with $M \cdot C^2$; parameter tuning of $(\alpha, \beta, \rho)$.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg25_moaco.py` as our Roll 13 Task C Deep Dive #2. Intended domain: Multi-objective combinatorial permutation routing (e.g. P8 bi-objective TSP), where ants construct valid tour permutations using dual pheromone matrices ($\tau^1, \tau^2$) and problem distance/time matrices ($D, T$). Non-TSP / Continuous Fallback: For general problems lacking distance and time matrices ($D, T$), the module initializes synthetic heuristic matrices $\eta_1$ and $\eta_2$ as constant matrices of ones ($\eta_{ij}^1 = \eta_{ij}^2 = 1.0$) with zero diagonal. Furthermore, when executed on continuous variable problems, ant colony graph traversal is bypassed; ants instead sample Gaussian perturbations around randomly selected archive solutions (`base + noise` clipped to $[x_l, x_u]$). Verified with exact 2-iteration hand traces on a 4-city instance and 10-seed sensitivity studies across $\rho \in \{0.05, 0.20, 0.50\}$ on P8.

#### 8. Complexity Analysis
Time: $O(K \cdot C^2 + K \cdot \bar{N} \cdot M)$ per generation where $K$ is ant count, $C$ is city count, and $\bar{N}$ is archive size. Space: $O(M \cdot C^2 + K \cdot C + \bar{N} (C + M))$.

---

### Algorithm 26: MOSA
- **Module File:** `moo/algorithms/alg26_mosa.py`
- **Algorithm Family:** Physics-Inspired / Simulated Annealing
- **General Approach:** Trajectory / Population / Multiobjective Metropolis-Hastings Annealing

#### 1. Core Idea
Multi-Objective Simulated Annealing (Suppapitnarm et al., 2000; Bandyopadhyay et al., 2008). Generalizes physical annealing to multiobjective landscapes. In each step: 1) A candidate solution generates a neighbor $x'$ via Gaussian or discrete perturbation; 2) If $x'$ Pareto-dominates the current solution $x$, it is always accepted ($P=1$); 3) If $x'$ is dominated by $x$, it is accepted with a temperature-dependent Boltzmann probability $P = \exp(-\Delta E / T)$, where $\Delta E$ represents normalized objective deterioration; 4) If $x'$ and $x$ are mutually non-dominating, it is accepted or evaluated via archive dominance; 5) Temperature cools via geometric schedule $T_{k+1} = \alpha T_k$ (with $\alpha \approx 0.95$); 6) An external archive stores all non-dominated points discovered throughout the thermal trajectory.

#### 2. Defining Mechanism
Multiobjective Boltzmann thermal acceptance criterion allowing probabilistic acceptance of dominated solutions to escape local Pareto optima.

#### 3. Advantages
- Outstanding global search and escape capabilities: high initial temperature allows broad exploratory traversal over deep deceptive valleys.
- Theoretical asymptotic convergence to global Pareto-optimal sets under logarithmic cooling schedules.
- Versatile applicability across continuous, integer, and discrete permutation problems with minimal architectural changes.
- External archive guarantees that high-quality non-dominated solutions discovered during exploratory hot phases are permanently preserved.

#### 4. Disadvantages
- Very slow convergence rate compared to population-based swarm or differential evolution algorithms, requiring many temperature stages.
- Sensitivity to the cooling schedule parameters: initial temperature $T_0$, cooling rate $\alpha$, and epoch length.
- Ambiguity in defining the multiobjective energy difference $\Delta E$ (e.g., maximum component deterioration, average shortfall, or dominance rank change).
- Trajectory-based search provides less natural parallel population diversity than generational genetic algorithms.

#### 5. Suitable Problems
Complex rugged landscapes with severe multimodal local Pareto traps (P1, P3), combinatorial layout, and scheduling problems.

#### 6. Limitations
Slow convergence; requires careful cooling schedule design; defining multi-objective energy difference $\Delta E$.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg26_mosa.py`. Features normalized average shortfall energy difference $\Delta E$, geometric cooling schedule ($T_0=1.0, \alpha=0.95$), and an external Pareto archive. Evaluated on P1, P3, P8.

#### 8. Complexity Analysis
Time: $O(K_{\text{steps}} \cdot M + N_{\text{arch}}^2)$ across the temperature cooling schedule. Space: $O(N_{\text{arch}} (n + M) + n)$.

---

### Algorithm 27: MODE
- **Module File:** `moo/algorithms/alg27_mode.py`
- **Algorithm Family:** Differential Evolution
- **General Approach:** Population-based / Multi-Objective Differential Evolution with One-to-One Selection

#### 1. Core Idea
Multi-Objective Differential Evolution (Babu & Jehan, 2003; Storn & Price extension). Adapts single-objective Differential Evolution to multiobjective problems. For each target vector $x_i$ in population $P$: 1) Generates a mutant vector $v_i = x_{r1} + F (x_{r2} - x_{r3})$ using randomly chosen distinct parents; 2) Performs binomial crossover with target $x_i$ using crossover rate $CR$ to yield trial vector $u_i$; 3) Applies one-to-one selection: if $u_i$ dominates $x_i$, $u_i$ replaces $x_i$; if $x_i$ dominates $u_i$, $u_i$ is discarded; if $u_i$ and $x_i$ are mutually non-dominating, $u_i$ is admitted to the population or external archive; 4) Population is pruned back to size $N$ using non-dominated sorting and crowding distance.

#### 2. Defining Mechanism
Direct one-to-one parent-offspring Pareto dominance comparison coupled with self-scaling difference-vector mutation.

#### 3. Advantages
- Remarkable convergence speed on continuous multi-objective problems due to directional difference-vector arithmetic.
- Scale-invariant search: difference vectors automatically adjust step size to the current population spread without explicit step size scheduling.
- One-to-one selection ensures that no child replaces its parent unless it is at least non-dominated by it, maintaining generational stability.
- Significantly fewer hyperparameters than standard genetic algorithms (principally just scale factor $F$ and crossover probability $CR$).

#### 4. Disadvantages
- Cannot be directly applied to discrete, combinatorial, or permutation variables without specialized discretization mappings.
- When trial and target vectors are mutually non-dominating, greedy one-to-one selection faces an ambiguity that can discard valuable diversity if not handled via archiving.
- Prone to premature stagnation if the population difference vectors become parallel or collapse onto a lower-dimensional manifold.
- Does not naturally scale to many-objective problems ($M \ge 4$) without replacing Pareto dominance sorting with reference vectors or indicators.

#### 5. Suitable Problems
Continuous unconstrained and constrained benchmarks (P1, P3, P5, P6), continuous parameter identification, and aerodynamic design.

#### 6. Limitations
Continuous domains only; parameter tuning of $(F, CR)$; diversity loss under mutual non-dominance without archive.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg27_mode.py`. Implements $DE/rand/1/bin$ ($F=0.5, CR=0.8$), boundary reflection, one-to-one dominance selection, and crowding-distance truncation. Evaluated on P1, P3, P5.

#### 8. Complexity Analysis
Time: $O(N \cdot n + M N^2)$ per generation. Space: $O(N (n + M))$.

---

### Algorithm 28: MO-TLBO
- **Module File:** `moo/algorithms/alg28_motlbo.py`
- **Algorithm Family:** Human-Inspired Metaheuristic
- **General Approach:** Population-based / Classroom Metaphor (Teacher Phase and Learner Phase)

#### 1. Core Idea
Multi-Objective Teaching-Learning-Based Optimization (Rao et al., 2011; Zou et al., 2015). Metaphor of classroom education without algorithm-specific control parameters. Each generation consists of two phases: 1) Teacher Phase: An elite teacher $X_{\text{teacher}}$ is selected from an external non-dominated archive; the mean position $M_{\text{mean}}$ of the learners is calculated; each learner $X_i$ updates toward the teacher: $X_{\text{new}} = X_i + r (X_{\text{teacher}} - T_F M_{\text{mean}})$ where $T_F \in \{1, 2\}$ is a teaching factor; 2) Learner Phase: For each learner $X_i$, another learner $X_j$ is randomly picked; if $X_i$ dominates $X_j$, $X_i$ moves away from $X_j$; if $X_j$ dominates $X_i$, $X_i$ moves toward $X_j$; 3) All updates are accepted based on Deb's constrained dominance.

#### 2. Defining Mechanism
Two-phase classroom teaching and pairwise peer-learning dynamics with a dynamic teaching factor $T_F \in \{1, 2\}$ and zero algorithm-specific control parameters.

#### 3. Advantages
- Parameter-less concept: eliminates the frustrating burden of tuning mutation rates, crossover probabilities, or velocity limits.
- Intuitive, structured balance: Teacher phase provides strong directed exploitation while Learner phase maintains decentralized peer exploration.
- Direct incorporation of Deb's constrained-dominance rules ensures seamless, penalty-free constraint handling.
- Fast convergence on continuous engineering problems due to collective knowledge pooling across the population.

#### 4. Disadvantages
- Calculating the classroom mean vector $M_{\text{mean}}$ assumes a convex unimodal distribution; on concave or multimodal fronts, the mean often lies in an infeasible or dominated void.
- Selecting a single teacher from the external archive can cause the entire population to surge toward a single region of the Pareto front, harming diversity.
- Pairwise learner comparison is inefficient when most learners become mutually non-dominating, leading to random directional drift.
- Restricted to continuous spaces; adaptation to permutation routing (TSP) requires awkward artificial heuristic encodings.

#### 5. Suitable Problems
Continuous constrained engineering design (P4, P5, P6, P7), mechanical component design, and benchmark continuous problems.

#### 6. Limitations
Continuous domains; mean calculation deceptive on non-convex fronts; diversity preservation requires archive niching.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg28_motlbo.py`. Selects teacher from least crowded archive grid cell, calculates teaching factor $T_F = \text{round}(1 + \text{rand}())$, executes peer learner phase with Deb's rules. Evaluated on P1, P3, P5.

#### 8. Complexity Analysis
Time: $O(N \cdot n + M (N + N_{\text{arch}})^2 + M N_{\text{nd}} \log N_{\text{nd}})$ per generation, where $N$ is learner population size, $N_{\text{arch}}$ is external archive size, $n$ is decision variable dimension, and $M$ is objective count. Teacher and Learner phases execute $O(N \cdot n)$ continuous vector operations; environmental archive maintenance pools learners and archive into $N + N_{\text{arch}}$ candidates, requiring pairwise non-dominated sorting in $O(M (N + N_{\text{arch}})^2)$ dominant time. Truncating excess non-dominated solutions via crowding distance adds $O(M N_{\text{nd}} \log N_{\text{nd}})$ time. Space: $O((N + N_{\text{arch}}) (n + M))$.

---

### Algorithm 29: MOBA
- **Module File:** `moo/algorithms/alg29_moba.py`
- **Algorithm Family:** Swarm Intelligence
- **General Approach:** Population-based / Microbat Echolocation with Dynamic Frequency, Loudness, and Pulse Rate

#### 1. Core Idea
Multi-Objective Bat Algorithm (Yang, 2011). Simulates the echolocation capability of microbats. Each virtual bat $i$ possesses a position $x_i$, velocity $v_i$, acoustic frequency $f_i \in [f_{\min}, f_{\max}]$, loudness $A_i$, and pulse emission rate $r_i$. In each generation: 1) Frequency is assigned randomly: $f_i = f_{\min} + (f_{\max} - f_{\min}) \beta$; 2) Velocity and position are updated guided by an elite solution selected from an external Pareto archive: $v_i = v_i + (x_i - x_{\text{best}}) f_i, x_i = x_i + v_i$; 3) If a random number exceeds pulse rate $r_i$, a localized random walk around a Pareto solution is performed with step size proportional to average loudness $A_t$; 4) If the new solution improves Pareto dominance and a random check passes, loudness decreases ($A_i \leftarrow \alpha A_i$) and pulse emission rate increases ($r_i \leftarrow r_i [1 - e^{-\gamma t}]$).

#### 2. Defining Mechanism
Acoustic frequency modulation driving velocity updates combined with loudness-controlled local exploitation and pulse-rate-controlled global exploration.

#### 3. Advantages
- Dynamic automatic transition from broad global search (high loudness, low pulse rate) to intense local exploitation (low loudness, high pulse rate).
- Frequency tuning provides adaptive step sizes that help the swarm navigate varying landscape scales.
- Local random walk operator helps bats escape shallow local Pareto traps.
- External archive ensures non-dominated solutions are permanently tracked and used as echolocation guides.

#### 4. Disadvantages
- Complex parameter interaction between frequency bounds $[f_{\min}, f_{\max}]$, loudness cooling $\alpha$, and pulse acceleration $\gamma$.
- Bat positions can easily blow up or violate boundary constraints if frequency and velocity are not tightly clamped.
- Does not naturally handle discrete or combinatorial decision spaces.
- Can suffer from premature convergence toward an arbitrary single section of the Pareto front if archive guide selection is biased.

#### 5. Suitable Problems
Continuous engineering parameter optimization (P1, P3, P5, P6), signal processing, and low-dimensional nonlinear trade-off problems.

#### 6. Limitations
Continuous domains; delicate hyperparameter tuning $(\alpha, \gamma, f_{\min}, f_{\max})$; boundary clipping sensitive.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg29_moba.py`. Features frequency modulation $[0, 2]$, velocity clamping, loudness decay $\alpha=0.9$, pulse acceleration $\gamma=0.05$, and an external Pareto archive. Evaluated on P1, P3, P5.

#### 8. Complexity Analysis
Time: $O(N \cdot n + M (N + N_{\text{arch}})^2)$ per generation, where $N$ is bat population size, $N_{\text{arch}}$ is archive size, $n$ is decision variable dimension, and $M$ is objective count. Echolocation kinematics and loudness-controlled random walks take $O(N \cdot n)$; the archive update combines population and archive into a candidate pool of size $N + N_{\text{arch}}$, requiring pairwise non-dominated sorting in $O(M (N + N_{\text{arch}})^2)$ dominant time. Space: $O((N + N_{\text{arch}}) (n + M))$.

---

### Algorithm 30: MOMA
- **Module File:** `moo/algorithms/alg30_moma.py`
- **Algorithm Family:** Hybrid / Memetic Algorithm
- **General Approach:** Population-based / Global Evolutionary Search Interleaved with Local Search

#### 1. Core Idea
Multi-Objective Memetic Algorithm (Knowles & Corne, 2000; Ishibuchi et al.). Hybridizes population-based global search with specialized local search memes. It executes a global evolutionary cycle (such as NSGA-II non-dominated sorting and tournament selection). Periodically, a subset of high-potential individuals (e.g., front 1 members or solutions with high crowding distance) is selected to undergo dedicated local search refinement (such as Hooke-Jeeves pattern search, localized gradient descent, or 2-opt permutation swaps for TSP). The refined solutions replace their parents, injecting high-precision improvements back into the evolutionary gene pool.

#### 2. Defining Mechanism
Explicit interleaving of global multiobjective evolutionary search with targeted, budget-limited single- or multi-objective local search operators.

#### 3. Advantages
- Achieves significantly higher convergence accuracy and front proximity (lower IGD) than pure evolutionary algorithms.
- Highly effective for combinatorial problems like TSP (P8) when hybridized with domain-specific operators like 2-opt or 3-opt.
- Accelerates convergence by rapidly polishing rough evolutionary solutions onto the local Pareto-efficient boundary.
- Flexible modular architecture that allows swapping in different local search heuristics depending on problem domain.

#### 4. Disadvantages
- Local search easily consumes a disproportionate share of the evaluation budget, leaving insufficient evaluations for global population exploration.
- Risk of premature diversity loss if local search pulls too many population members toward the same local Pareto attractor.
- Requires deciding which solutions to improve, when to apply local search, and how many local steps to allocate.
- Significantly higher implementation and runtime complexity than homogeneous evolutionary algorithms.

#### 5. Suitable Problems
Continuous precision engineering (P3, P5), combinatorial routing like bi-objective TSP (P8), and complex industrial problems where local search heuristics exist.

#### 6. Limitations
Severe evaluation budget consumption; delicate local/global budget allocation; danger of local basin trapping.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg30_moma.py`. Uses NSGA-II as global framework, applying pattern search local moves to a fraction ($p_{\text{local}}=0.2$) of Front 1 solutions with budget caps. Evaluated on P3, P5, P8.

#### 8. Complexity Analysis
Time: $O(M N^2 + N_{\text{local}} \cdot K_{\text{local}} \cdot C_{\text{eval}})$ per generation. Space: $O(N (n + M))$.

---

### Algorithm 31: RVEA
- **Module File:** `moo/algorithms/alg31_rvea.py`
- **Algorithm Family:** Many-Objective Reference-Vector MOEA
- **General Approach:** Population-based / Reference-Vector-Guided Angle-Penalized Distance Selection

#### 1. Core Idea
Reference Vector Guided Evolutionary Algorithm (Cheng, Jin, Olhofer, & Sendhoff, IEEE TEVC 2016). Tailored for many-objective optimization ($M \ge 3$). Uses a set of predefined reference vectors to partition the objective space into conical sub-regions. Individuals are assigned to their nearest reference vector based on the angle between them. Within each reference vector's subpopulation, environmental selection retains the single best solution minimizing the Angle-Penalized Distance: $d_{\text{APD}} = (1 + P(t) \cdot \frac{\theta}{\gamma}) \cdot \|f\|$, where $\|f\|$ is convergence distance to ideal, $\theta$ is angle to reference vector, $\gamma$ is minimal angle to neighboring vectors, and $P(t) = M \cdot (t/t_{\max})^\alpha$ is an adaptive penalty that increases over generations.

#### 2. Defining Mechanism
Angle-Penalized Distance (APD) metric that adaptively penalizes angular divergence from reference vectors as search progresses from exploration to convergence.

#### 3. Advantages
- Seamlessly scales to many-objective problems ($M = 3, 5, 10, 15$) where Pareto dominance selection completely breaks down.
- Dynamic penalty parameter $P(t)$ automatically emphasizes diversity in early generations and convergence in late generations.
- Objective normalization by current extreme points allows it to handle disparate objective scales without manual weight tweaking.
- More flexible than decomposition methods: solutions can migrate between neighboring reference cones dynamically.

#### 4. Disadvantages
- Predefined uniform reference vectors perform sub-optimally when the true Pareto front is disconnected, degenerated, or highly irregular.
- Computing angles between all individuals and reference vectors adds computational overhead in high dimensions.
- Sensitivity to the angle penalty exponent $\alpha$ (typically set to 2.0).
- Offers no compelling benefit over NSGA-II on standard bi-objective problems ($M=2$).

#### 5. Suitable Problems
Many-objective continuous benchmarks (P10 DTLZ2 with $M=3, 5$), complex many-criteria engineering design, and high-dimensional trade-off analysis.

#### 6. Limitations
Requires structured reference vectors; irregular front shape sensitivity; unnecessary overhead for $M \le 2$.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg31_rvea.py`. Generates uniform reference vectors via Das-Dennis, normalizes objectives, and calculates APD with $P(t) = M (t/t_{\max})^2$. Evaluated on P10.

#### 8. Complexity Analysis
Time: $O(N \cdot K_{\text{ref}} \cdot M)$ per generation where $K_{\text{ref}}$ is reference vector count. Space: $O((N + K_{\text{ref}}) (n + M))$.

---

### Algorithm 32: IBEA
- **Module File:** `moo/algorithms/alg32_ibea.py`
- **Algorithm Family:** Indicator-Based MOEA
- **General Approach:** Population-based / Direct Binary Performance Indicator Selection

#### 1. Core Idea
Indicator-Based Evolutionary Algorithm (Zitzler & Künzli, PPSN 2004). Instead of relying on heuristic diversity schemes (like crowding distance or fitness sharing), IBEA guides selection directly using a formal continuous quality indicator, most commonly the additive binary $\epsilon$-indicator: $I_{\epsilon+}(x_1, x_2) = \max_{m} (f_m(x_1) - f_m(x_2))$. The fitness of an individual $x_1$ measures its total indicator loss across the population: $F(x_1) = \sum_{x_2 \ne x_1} -\exp(-I_{\epsilon+}(x_2, x_1) / (\kappa \cdot c))$, where $\kappa > 0$ is a scaling factor. An iterative environmental selection successively removes the individual whose deletion minimizes fitness disruption, updating remaining fitness values.

#### 2. Defining Mechanism
Fitness assignment based on aggregated pairwise binary indicator comparisons ($I_{\epsilon+}$ or binary hypervolume) with iterative worst-loss truncation.

#### 3. Advantages
- Theoretically grounded: directly optimizes the mathematical indicator that will ultimately be used to evaluate the final front quality.
- Maintains strong selection pressure in many-objective problems ($M \ge 4$) without requiring non-dominated sorting or reference lines.
- Can incorporate decision-maker preferences seamlessly by modifying the indicator formulation or weighting criteria.
- Produces exceptionally smooth, well-converged Pareto sets without boundary-clumping artifacts.

#### 4. Disadvantages
- Computing the $N \times N$ pairwise indicator matrix in every generation requires $O(N^2 \cdot M)$ calculations.
- Iterative removal of the worst individual requires updating all remaining fitness values, adding $O(N^2)$ sequential steps.
- Sensitive to the scaling parameter $\kappa$; if $\kappa$ is too small or too large, fitness values saturate or become uniform.
- Additive $\epsilon$-indicator requires accurate objective normalization to prevent large-magnitude objectives from dominating the max operator.

#### 5. Suitable Problems
Many-objective problems (P10), continuous multi-objective design, and applications where optimizing $\epsilon$-dominance or Hypervolume is the explicit goal.

#### 6. Limitations
Pairwise $O(N^2)$ indicator computation; scaling parameter $\kappa$ calibration; normalization dependent.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg32_ibea.py`. Implements additive binary $\epsilon$-indicator ($I_{\epsilon+}$), pairwise indicator matrix computation, exponential fitness aggregation with $\kappa=0.05$, and iterative removal. Evaluated on P1, P3, P10.

#### 8. Complexity Analysis
Time: $O(M N^2)$ per generation for pairwise matrix + $O(N^2)$ for environmental truncation. Space: $O(N^2 + N(n+M))$.

---

### Algorithm 33: SMS-EMOA
- **Module File:** `moo/algorithms/alg33_sms_emoa.py`
- **Algorithm Family:** Indicator-Based MOEA
- **General Approach:** Population-based / Steady-State $(\mu+1)$ with Exclusive Hypervolume Contribution Selection

#### 1. Core Idea
S-Metric Selection Evolutionary Multi-Objective Algorithm (Emmerich, Beume, & Naujoks, 2005; Beume, Naujoks, & Emmerich, 2007). Operates in a steady-state $(\mu+1)$ framework. In each iteration: 1) Two parents generate a single offspring via crossover and mutation; 2) The offspring joins the population, forming a pool of size $\mu+1$; 3) Non-dominated sorting stratifies the pool into fronts; 4) If the worst front contains multiple solutions, the algorithm computes the exclusive hypervolume contribution of each individual: $\Delta \text{HV}(i) = \text{HV}(F_{\text{worst}}) - \text{HV}(F_{\text{worst}} \setminus \{i\})$; 5) The solution with the smallest $\Delta \text{HV}$ is permanently discarded.

#### 2. Defining Mechanism
Steady-state $(\mu+1)$ replacement discarding the individual that contributes the minimal exclusive hypervolume to the worst non-dominated front.

#### 3. Advantages
- Theoretically optimal diversity and convergence: hypervolume is strictly Pareto-compliant (if $A \prec B$, $\text{HV}(A) > \text{HV}(B)$).
- Monotonic hypervolume growth: the hypervolume of the population non-dominated front can never decrease across generations.
- Steady-state $(\mu+1)$ replacement immediately integrates high-quality offspring into subsequent parent selection without generational delay.
- Produces exquisitely distributed Pareto fronts with natural, balanced spacing between adjacent solutions.

#### 4. Disadvantages
- Computing exact hypervolume contributions is computationally \#P-hard, making it intractable for $M \ge 4$ without Monte Carlo estimation.
- Generating only one offspring per iteration requires many iterations to update a population of size $\mu$, increasing sorting overhead per evaluation.
- Requires specifying a reference point for hypervolume calculation; inappropriate reference points skew boundary point survival.
- Slow runtimes on large populations ($N > 200$) due to repeated hypervolume recalculations.

#### 5. Suitable Problems
Bi-objective and tri-objective problems requiring the highest possible front quality, uniform spread, and maximum hypervolume (P1, P3, P4, P5, P6).

#### 6. Limitations
Exponential complexity in $M$ (intractable for exact $M \ge 4$); reference point sensitive; high per-evaluation sorting overhead.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg33_sms_emoa.py`. Operates in a steady-state $(\mu+1)$ scheme. Uses fast non-dominated sorting to stratify the pooled population, followed by exclusive hypervolume contribution calculation on the worst front ($F_{\text{worst}}$). For $M=2$, exact rectangular hypervolume slicing is used ($O(K \log K)$). For $M \ge 3$, our implementation computes hypervolume contributions via Monte Carlo estimation ($N_s = 10,000$ samples), avoiding the exponential bottleneck of exact slicing. Evaluated on P1, P3, P5.

#### 8. Complexity Analysis
Time: Per steady-state $(\mu+1)$ step (which performs exactly 1 objective evaluation): fast non-dominated sorting of the pooled population of size $\mu+1$ costs $\Theta(M \mu^2)$, and exclusive hypervolume contribution on the worst front ($K = |F_{\text{worst}}| \le \mu+1$) costs $\Theta(K \log K) \le O(\mu \log \mu)$ for $M=2$ (exact rectangular slicing). For $M \ge 3$, our implementation uses Monte Carlo estimation with $N_s = 10,000$ samples taking $\Theta(K^2 \cdot N_s \cdot M)$ time per step (distinguished from canonical theoretical exact hypervolume slicing which is \#P-hard for arbitrary $M$, scaling as $O(\mu^{M/2} \log \mu)$ or $O(\mu^{M-1})$ for fixed $M \ge 3$). Cumulative Batch Complexity: Over an arbitrary batch of $B$ objective evaluations (requiring $B$ sequential steady-state steps), cumulative non-dominated sorting costs $\Theta(B \cdot M \mu^2)$, and cumulative 2D hypervolume contributions cost $O(B \cdot \mu \log \mu)$. When adopting the common generation-equivalent comparison convention of $B = \mu$ objective evaluations (matching the evaluation count of one generational MOEA cycle), cumulative sorting becomes $\Theta(M \mu^3)$ and 2D hypervolume contributions become $O(\mu^2 \log \mu)$. Space: $O(\mu (n + M) + N_{\text{arch}} (n + M))$ where $\mu$ is population size and $N_{\text{arch}}$ is external archive capacity.

---

### Algorithm 34: HypE
- **Module File:** `moo/algorithms/alg34_hype.py`
- **Algorithm Family:** Indicator-Based / Many-Objective MOEA
- **General Approach:** Population-based / Hypervolume Estimation via Monte Carlo Integration

#### 1. Core Idea
Hypervolume Estimation Algorithm for Many-Objective Optimization (Bader & Zitzler, IEEE TEVC 2011). Resolves the exponential bottleneck of SMS-EMOA by estimating hypervolume contributions via Monte Carlo sampling. A reference region bounded by the current population ideal and a reference point is populated with $N_s$ uniformly distributed random sampling points. For each sample point, the algorithm determines how many population members dominate it. Points dominated by exactly $k$ solutions contribute weight to those $k$ solutions. This provides an unbiased statistical estimate of individual and shared hypervolume contributions, scaling linearly with sample size $N_s$ rather than exponentially with $M$.

#### 2. Defining Mechanism
Monte Carlo integration of partitioned hypervolume contributions combined with an exact multi-point weighting scheme.

#### 3. Advantages
- Successfully scales hypervolume-based evolutionary search to many-objective spaces ($M = 5, 10, 15$) where exact HV calculation is impossible.
- User can directly tune the computational complexity vs estimation accuracy trade-off by adjusting the sample count $N_s$.
- Maintains all theoretical virtues of hypervolume optimization (strict Pareto compliance, natural diversity, boundary preservation).
- Significantly faster than exact hypervolume methods on 4-objective and 5-objective benchmarks.

#### 4. Disadvantages
- Monte Carlo estimation introduces stochastic noise into ranking; two identical evaluations can produce slightly different fitness values.
- High sample counts ($N_s \ge 100,000$) are required to maintain ranking accuracy in very high dimensions ($M \ge 10$), increasing CPU time.
- Highly dependent on setting a proper reference point; poor reference points distort sample density across the front.
- More complex to implement correctly than reference-point methods like NSGA-III or RVEA.

#### 5. Suitable Problems
Many-objective continuous benchmarks (P10 DTLZ2 with $M=5, 10$), complex engineering trade-offs where hypervolume maximization is required.

#### 6. Limitations
Monte Carlo sampling variance; requires high sample counts for high $M$; sensitive to reference point bounds.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg34_hype.py`. Features Monte Carlo integration ($N_s=10,000$), weight vector calculation, and hypervolume loss truncation. Evaluated on P10.

#### 8. Complexity Analysis
Time: $O(N_s \cdot N \cdot M)$ per generation where $N_s$ is sample count. Space: $O(N (n + M) + N_s \cdot M)$.

---

### Algorithm 35: R-NSGA-II
- **Module File:** `moo/algorithms/alg35_r_nsga2.py`
- **Algorithm Family:** Preference-Based MOEA
- **General Approach:** Population-based / Reference-Point-Guided Search with $\epsilon$-Clearing

#### 1. Core Idea
Reference-Point-Based NSGA-II (Deb, Sundar, Udaya Bhaskar Rao, & Meyarivan, 2006). Modifies NSGA-II to incorporate user-defined reference points $z^{\text{pref}}$ representing preferred regions of the Pareto front. In the environmental selection: 1) Non-dominated sorting is retained; 2) In the critical front, solutions are evaluated by their normalized Euclidean distance to the nearest reference point; 3) To prevent all solutions from collapsing onto a single point, an $\epsilon$-clearing distance threshold is enforced: if two solutions fall within $\epsilon_{\text{clear}}$ of each other, the one farther from the reference point is penalized or eliminated, producing a controlled spread inside the preferred region.

#### 2. Defining Mechanism
Reference-point Euclidean distance ranking combined with an explicit $\epsilon$-clearing neighborhood mechanism to steer search toward preferred Regions of Interest (ROI).

#### 3. Advantages
- Dramatically saves computational evaluation budget by ignoring irrelevant portions of the Pareto front that the decision-maker has no interest in.
- Can handle multiple distinct reference points simultaneously, discovering separate regions of interest concurrently.
- $\epsilon$-clearing parameter prevents population overcrowding, ensuring a well-spread, diverse cluster of solutions around each preference target.
- Scales better to higher objective counts than standard NSGA-II because search is localized to narrow conical rays in objective space.

#### 4. Disadvantages
- Requires decision-makers to articulate specific reference points $z^{\text{pref}}$ before or during optimization.
- If reference points are set in physically impossible or unpromising directions, the algorithm expends evaluations searching unreachable regions.
- Provides zero visibility into unexpected, high-performing trade-offs that lie outside the specified regions of interest.
- Requires tuning the $\epsilon$-clearing threshold parameter to balance local clustering vs local spread.

#### 5. Suitable Problems
Laptop design with customer preference points (P14), preference-guided engineering design, and interactive optimization workflows.

#### 6. Limitations
Requires a priori preference targets; no global front visibility; $\epsilon$-clearing parameter sensitive.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg35_r_nsga2.py`. Replaces crowding distance with normalized distance to reference points and enforces $\epsilon$-clearing ($0.05$). Evaluated on P14 and P3.

#### 8. Complexity Analysis
Time: $O(M N^2 + N \log N)$ per generation. Space: $O(N (n + M) + K_{\text{pref}} M)$.

---

### Algorithm 36: MGDA
- **Module File:** `moo/algorithms/alg36_mgda.py`
- **Algorithm Family:** Gradient-Based Multiobjective
- **General Approach:** Deterministic / Trajectory Descent via Minimum-Norm Convex Gradient Combination

#### 1. Core Idea
Multiple Gradient Descent Algorithm (Désidéri, 2012; Sener & Koltun, NeurIPS 2018). Extends classical gradient descent to multiobjective optimization. At each iteration for parameters $\theta$: 1) Computes the gradient $\nabla_\theta f_m(\theta)$ for all $M$ objectives; 2) Solves the convex subproblem to find the minimum-norm vector in the convex hull of gradients: $\min_\alpha \| \sum_{m=1}^M \alpha_m \nabla f_m(\theta) \|^2$ subject to $\sum \alpha_m = 1, \alpha_m \ge 0$; 3) If the minimum norm is zero, the current solution satisfies the Pareto-critical Karush-Kuhn-Tucker (KKT) conditions and search terminates; 4) Otherwise, the vector $d = -\sum_{m=1}^M \alpha_m \nabla f_m(\theta)$ is a proven descent direction that simultaneously decreases every objective: $\nabla f_m(\theta)^T d < 0$ for all $m$.

#### 2. Defining Mechanism
Computing the minimum-norm vector in the convex hull of objective gradients to establish a mathematically guaranteed common Pareto descent direction.

#### 3. Advantages
- Guarantees simultaneous descent: every optimization step improves (or preserves) all objective values concurrently without trading one off against another.
- Principled Pareto-critical stopping condition: automatically terminates when the minimum norm $\|d\| < \epsilon$, proving local Pareto optimality.
- Extremely fast local convergence rate on smooth continuous landscapes due to exact first-order gradient information.
- Can be scaled to deep neural networks (Sener & Koltun) using Frank-Wolfe optimization over shared feature layer gradients.

#### 4. Disadvantages
- Requires smooth, continuously differentiable objective functions; completely inapplicable to black-box, noisy, discrete, or non-smooth problems.
- Finds only a single Pareto-critical point per optimization trajectory; exploring the full Pareto front requires multiple random restarts or sweeps.
- Susceptible to getting trapped at local Pareto-critical points on non-convex multi-objective landscapes.
- Can cause task gradient conflict in multi-task networks if gradients point in directly opposing directions, driving the descent step to zero.

#### 5. Suitable Problems
Multi-Task Learning (P13), deep neural network multi-loss optimization, smooth continuous trajectory optimization.

#### 6. Limitations
Requires exact gradients; single-point trajectory; traps at local KKT points; gradient conflict stagnation.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg36_mgda.py`. All actual objective evaluations are recorded strictly in `state["n_evals"]` and counted against the configured evaluation budget, with explicit checks (`if evals_used + 2 > budget_rem: break`) ensuring that evaluations never overshoot the limit. The implementation clearly distinguishes between numerical and analytical evaluation accounting:
- **Numerical central finite-difference gradient estimation:** On black-box benchmarks where analytical derivatives are unavailable, each decision variable $j \in \{1, \dots, n\}$ is perturbed by $\pm \epsilon$ with boundary clipping. If a coordinate is fixed ($x_{l, j} == x_{u, j}$ or $|dx| < 10^{-12}$), the evaluation is skipped; otherwise, 2 function evaluations are executed, consuming up to $2n$ objective evaluations per step for gradient estimation.
- **Candidate update evaluation:** If remaining budget permits after gradient calculation, evaluating the updated candidate solution $x_{\text{new}}$ consumes up to one additional objective evaluation per step.
- **Maximum full-step count on the numerical-gradient path:** Combining gradient estimation and candidate update evaluation yields a maximum full-step count of $2n + 1$ objective evaluations, subject to the remaining budget and any skipped fixed variables.
- **Analytical-gradient path:** When exact gradients are supplied via `problem.compute_gradients` (e.g., P13), no objective evaluations are consumed for gradient calculation ($0$ evaluations); the candidate update can require one evaluation if evaluated.
Evaluated on P13.

#### 8. Complexity Analysis
Time: On the analytical-gradient path, time complexity is $O(M \cdot n + M^3)$ per descent step where $n$ is decision parameter dimension, $M$ is objective count, and $M^3$ bounds the Frank-Wolfe simplex projection, incurring at most 1 candidate evaluation ($C_f$). On the numerical-gradient path, finite-difference estimation performs up to $2n$ evaluations and candidate update requires up to 1 evaluation (maximum $2n+1$ evaluations per full step), incurring $O((2n+1) \cdot C_f + M \cdot n + M^3)$ computational time, where $C_f$ is single objective evaluation cost. Space: $O(M \cdot n)$ to store the gradient matrix $G$ plus $O(N_{\text{arch}} (n + M))$ for the bounded non-dominated archive.

---

### Algorithm 37: GradNorm
- **Module File:** `moo/algorithms/alg37_gradnorm.py`
- **Algorithm Family:** Multi-Task Gradient Balancing (Task C Deep Dive #3)
- **General Approach:** Deterministic / Dynamic Gradient Normalization for Shared-Body Neural Networks

#### 1. Core Idea
Gradient Normalization for Multi-Task Learning (Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, & Andrew Rabinovich, ICML 2018). Addresses task imbalance in multi-task neural networks where tasks with larger gradient scales dominate parameter updates. At each step $t$: 1) Weighted loss is $L_{\text{total}} = \sum w_i(t) L_i(t)$; 2) Calculates gradient norms with respect to shared layer $W$: $G_W^{(i)}(t) = \| \nabla_W (w_i(t) L_i(t)) \|_2$; 3) Computes average norm $\bar{G}_W(t) = \frac{1}{M} \sum G_W^{(i)}(t)$; 4) Calculates relative inverse training rate $r_i(t) = (L_i(t)/L_i(0)) / \text{mean}_j(L_j(t)/L_j(0))$; 5) Defines target gradient norm $\text{Target}_i(t) = \bar{G}_W(t) [r_i(t)]^\alpha$; 6) Updates weights $w_i(t)$ via gradient descent on $L_{\text{grad}} = \sum |G_W^{(i)}(t) - \text{Target}_i(t)|$ and renormalizes $\sum w_i(t) = M$.

#### 2. Defining Mechanism
Dynamic task weight adaptation that simultaneously balances shared gradient magnitudes and inversely proportional task training rates.

#### 3. Advantages
- Eliminates manual task weight grid search in multi-task deep learning architectures, automating loss balancing dynamically.
- Pace-balancing mechanism: tasks that learn slower receive higher target gradient norms, ensuring all tasks train at synchronized rates.
- Scale-balancing mechanism: prevents high-magnitude loss functions from drowning out low-magnitude loss functions in shared layers.
- Proven empirically to improve generalization across multiple benchmarks (e.g., joint depth estimation and semantic segmentation).

#### 4. Disadvantages
- Requires computing gradients with respect to shared layer $W$ for every task separately, adding backward pass computational overhead.
- Requires selecting which layer to designate as shared layer $W$; typically chosen as the final shared feature layer.
- Introduces asymmetry restoring force hyperparameter $\alpha$ and weight learning rate $\eta_{\text{grad}}$ that require calibration.
- Assumes initial loss values $L_i(0)$ accurately reflect task difficulty; noisy or fluctuating initial loss estimates degrade target pacing.

#### 5. Suitable Problems
Multi-Task Learning with shared representation neural networks (P13), multi-objective deep learning, and multi-head classification/regression networks.

#### 6. Limitations
Restricted to gradient-based neural network architectures; backward pass overhead; requires initial loss $L_i(0)$ baseline.

#### 7. Project Implementation Notes
Implemented in `moo/algorithms/alg37_gradnorm.py` as our Roll 13 Task C Deep Dive #3. This module represents the project's NumPy-based adaptation of GradNorm for multiobjective optimization, not an exact reproduction of the original neural-network method (which designates a specific internal shared parameter layer $W$ within a deep neural network and computes task gradients via backpropagation). In this multiobjective adaptation, dynamic task weight balancing is applied directly to benchmark decision parameters. All actual objective evaluations are recorded strictly in `state["n_evals"]` and count against the configured evaluation budget, with explicit checks preventing overshooting. The evaluation accounting clearly distinguishes:
- **Baseline task-loss evaluation:** Exactly one objective evaluation is performed at the current parameter vector $x_{\text{cur}}$ at the start of each step to obtain task losses $L_m(t)$ and evaluate constraint violations.
- **Numerical central finite differences:** Up to $2n$ additional evaluations are executed to compute the $(M \times n)$ Jacobian matrix via perturbations of $\pm \epsilon$ across the $n$ decision variables (skipping any fixed variables where $|dx| < 10^{-12}$).
- **Maximum full-step count on the numerical-gradient path:** Combining the baseline task-loss evaluation and numerical finite differences yields a maximum full-step count of $2n + 1$ objective evaluations, subject to the remaining budget and skipped fixed variables.
- **Analytical-gradient path:** When analytical gradients are available via `problem.compute_gradients`, no perturbation evaluations are performed for gradient calculation; however, the baseline task-loss evaluation (1 evaluation) still occurs in the existing implementation to compute current losses $L_m(t)$ and relative inverse training rates $r_m(t)$.
Evaluated on P13 multi-task problem and 2-layer shared MLP with manual backprop across $\alpha \in \{0.12, 1.5, 3.0\}$.

#### 8. Complexity Analysis
Time: In our NumPy-based multiobjective adaptation on the analytical-gradient path, time complexity is $O(C_f + M \cdot n)$ per step (1 baseline loss evaluation costing $C_f$ plus gradient aggregation across $M$ objectives and $n$ decision variables). On the numerical-gradient path, finite differences perform up to $2n$ evaluations alongside the 1 baseline evaluation (maximum $2n+1$ evaluations per full step), yielding time complexity $O((2n+1) \cdot C_f + M \cdot n)$. In canonical deep neural network implementations, runtime scales instead as $O(M \cdot C_{\text{backward}} + M \cdot |W|)$ per training step, where $C_{\text{backward}}$ is backpropagation cost through designated shared layer $W$. Space: $O(M \cdot n)$ to store the gradient matrix $G$ plus $O(N_{\text{arch}} (n + M))$ for the bounded non-dominated archive.

---

## 3. Master Comparison Tables

### Table A: Master Algorithm Comparison (All 37 Algorithms)

| # | Algorithm Name | Family | Approach | Main Selection / Search Mechanism | Typical Objectives | Variable Suitability | Main Strength | Main Weakness |
|---|---|---|---|---|---|---|---|---|
| 1 | **Weighted Sum** | Classical Scalarization | Deterministic | Convex combination of objectives minimized via gradient or pattern search across a... | 2 to 3 objectives | Continuous (standard), can be ad... | Extremely simple to formulate and computationally rapid for co... | Mathematically incapable of discovering solutions on non-conve... |
| 2 | **ε-Constraint** | Classical Scalarization | Deterministic | Single primary objective minimized while remaining $M-1$ objectives are constraine... | 2 to 4 objectives | Continuous, integer, or mixed-in... | Capable of discovering solutions on non-convex Pareto fronts b... | Selection of feasible and non-redundant $\epsilon$ bound grids... |
| 3 | **Benson’s Method** | Classical Scalarization | Deterministic | Maximization of sum of shortfalls from an initial dominated reference point subjec... | 2 to 5 objectives | Continuous and linear / convex p... | Guarantees strong Pareto optimality and generates an efficient... | Requires constructing an exterior dominated reference point an... |
| 4 | **Weight Metric** | Classical Scalarization | Deterministic | Minimization of weighted distance ($L_1, L_2$, or $L_\infty$ Tchebycheff) from can... | 2 to 5 objectives | Continuous and smooth nonlinear ... | The Augmented Tchebycheff ($L_\infty$) metric is theoretically... | The $L_\infty$ norm is non-differentiable at equal-coordinate ... |
| 5 | **Value Function** | Classical Utility / Preference | Deterministic | Maximization of an explicit decision-maker utility function $U(f_1(x), \dots, f_M(... | 2 to 10+ objectives | Continuous, discrete, or mixed d... | Directly yields the single most preferred compromise solution ... | Eliciting an accurate, mathematically consistent utility funct... |
| 6 | **Goal Programming** | Classical Scalarization | Deterministic | Minimization of positive and negative deviational variables from prescribed aspira... | 2 to 10+ objectives | Continuous, integer, and linear ... | Directly accommodates aspirational business targets and distin... | Can yield dominated solutions if targets are set within the in... |
| 7 | **Lexicographic** | Classical Hierarchical | Deterministic | Sequential optimization of objectives in strict order of priority, constraining hi... | 2 to 5 objectives | Continuous, integer, or combinat... | Completely eliminates the need to specify arbitrary weighting ... | Higher-priority objectives frequently have unique optima, leav... |
| 8 | **TOPSIS** | Multi-Criteria Decision Making (MCDM) | Deterministic | Evaluation and ranking of a pre-existing discrete candidate matrix based on Euclid... | 2 to 20+ criteria | Discrete alternatives matrix (fi... | Conceptually elegant, computationally instantaneous, and accou... | Is a ranking/filtering tool for given alternatives, NOT a gene... |
| 9 | **VIKOR** | Multi-Criteria Decision Making (MCDM) | Deterministic | Multi-criteria ranking based on group utility ($S$), individual regret ($R$), and ... | 2 to 20+ criteria | Discrete alternatives matrix | Explicitly balances majority consensus ($S$) against maximum i... | Does not perform search in decision space; linear normalizatio... |
| 10 | **VEGA** | First-Generation MOEA | Population-based | Sub-population partitioning where each fractional cohort of size $N/M$ is selected... | 2 to 3 objectives | Continuous, binary, or discrete ... | Pioneering historic baseline (Schaffer, 1985); extremely simpl... | Severe speciation bias: strongly favors extreme points on indi... |
| 11 | **MOGA** | First-Generation MOEA | Population-based | Rank assigned as $1 +$ number of dominating individuals in population; fitness int... | 2 to 3 objectives | Continuous, binary, or integer v... | First algorithm to introduce Pareto dominance ranking to guide... | Fitness sharing requires $O(N^2)$ pairwise distance calculatio... |
| 12 | **NSGA** | First-Generation MOEA | Population-based | Successive front decomposition where each non-dominated layer receives a dummy fit... | 2 to 3 objectives | Continuous, binary, or combinato... | Systematically stratifies the population into hierarchical Par... | High computational complexity of original non-dominated sortin... |
| 13 | **NPGA** | First-Generation MOEA (Task C Deep Dive #1) | Population-based | Binary tournament where 2 candidates are compared against a random comparison set ... | 2 to 3 objectives | Continuous, binary, or combinato... | Dramatically reduces computational overhead to $O(N \cdot t_{\... | High sensitivity to tournament comparison size $t_{\text{dom}}... |
| 14 | **NSGA-II** | Second-Generation Elitist MOEA | Population-based | Fast $O(M N^2)$ non-dominated sorting, parameter-free crowding distance assignment... | 2 to 3 objectives | Continuous, discrete, combinator... | Gold-standard benchmark of modern multiobjective optimization;... | Crowding distance mechanism collapses in many-objective spaces... |
| 15 | **NSGA-III** | Many-Objective Reference-Based MOEA | Population-based | Replaces crowding distance in NSGA-II with structured reference points on a normal... | 3 to 15+ objectives (Many-Objective Optimization) | Continuous, discrete, or mixed v... | Maintains excellent diversity and selection pressure across 4 ... | The Das-Dennis reference point grid scales combinatorially wit... |
| 16 | **SPEA2** | Second-Generation Elitist MOEA | Population-based | Fine-grained strength value ($S$), raw dominance fitness ($R$), $k$-th nearest nei... | 2 to 4 objectives | Continuous, binary, or combinato... | Fine-grained fitness assignment that considers both dominating... | High computational cost of $O((N+\bar{N})^2 \log(N+\bar{N}))$ ... |
| 17 | **PESA-II** | Second-Generation Elitist MOEA | Population-based | Objective space partitioned into adaptive hyperboxes; selection unit is the hyperb... | 2 to 4 objectives | Continuous, binary, or combinato... | Region-based selection ensures uniform sampling across objecti... | Performance depends heavily on the grid resolution parameter (... |
| 18 | **MOEA/D** | Decomposition-Based MOEA | Population-based | Decomposes MOP into $N$ scalar subproblems with uniform weight vectors; each subpr... | 2 to 5 objectives | Continuous, discrete, or combina... | Transforms multiobjective search into collaborative scalar opt... | Predefined uniform weight vectors perform poorly on disconnect... |
| 19 | **MOEA/D-DE** | Decomposition-Based MOEA | Population-based | Decomposition framework utilizing DE/rand/1/bin difference vectors and polynomial ... | 2 to 5 objectives | Continuous variables with strong... | Differential Evolution reproduction operators provide exceptio... | Restricted strictly to continuous variable domains; DE scaling... |
| 20 | **MOEA/D-ACO** | Decomposition-Based Metaheuristic | Population-based | Decomposes combinatorial multiobjective problems into scalar subproblems where ind... | 2 to 3 objectives | Combinatorial, permutation, and ... | Brings the power of decomposition to discrete combinatorial op... | High memory footprint of maintaining multiple pheromone/heuris... |
| 21 | **Constrained NSGA-II** | Constrained Evolutionary Algorithm | Population-based | Direct integration of Deb's 3-rule constrained-dominance comparison into binary to... | 2 to 3 objectives | Continuous, discrete, or mixed c... | Parameter-free constraint handling: strictly drives the popula... | Can get trapped in disconnected feasible islands if large infe... |
| 22 | **MOPSO** | Swarm Intelligence | Population-based | Particles update velocity using personal best ($pbest$) and a global best ($gbest$... | 2 to 3 objectives | Continuous decision variables | Very rapid convergence velocity on continuous landscapes due t... | High risk of premature convergence to local Pareto fronts; vel... |
| 23 | **MOGWO** | Swarm Intelligence | Population-based | Pack positions updated by encircling and hunting guided by alpha ($\alpha$), beta ... | 2 to 3 objectives | Continuous decision variables | Hierarchical multi-leader guidance ($\alpha, \beta, \delta$) p... | Selecting three leaders from an external archive can cause the... |
| 24 | **MOABC** | Swarm Intelligence | Population-based | Employed bees perturb food sources; onlooker bees choose sources based on Pareto r... | 2 to 3 objectives | Continuous and discrete numerica... | Scout bee abandonment mechanism provides an outstanding escape... | One-dimensional perturbation operator in employed phase can re... |
| 25 | **MOACO** | Nature-Inspired / Ant Colony (Task C Deep Dive #2) | Population-based | Colony of ants construct discrete tours using dual pheromone matrices ($\tau^1, \t... | 2 to 3 objectives | Combinatorial, permutation, and ... | Purpose-built for multiobjective combinatorial routing; constr... | High memory footprint ($O(M \cdot C^2)$ for pheromone/heuristi... |
| 26 | **MOSA** | Physics-Inspired / Simulated Annealing | Trajectory | Neighbor generation with multiobjective Boltzmann acceptance probability, geometri... | 2 to 4 objectives | Continuous, discrete, or combina... | Theoretical guarantee of asymptotic convergence to the global ... | Inherently slow convergence speed requiring a large number of ... |
| 27 | **MODE** | Differential Evolution | Population-based | DE mutation and crossover generating trial vectors; one-to-one Pareto dominance re... | 2 to 3 objectives | Continuous decision variables | Self-adapting difference vectors provide unmatched convergence... | Restricted strictly to continuous domains; one-to-one selectio... |
| 28 | **MO-TLBO** | Human-Inspired Metaheuristic | Population-based | Teacher Phase pulls learners toward class mean and an elite teacher; Learner Phase... | 2 to 3 objectives | Continuous decision variables | Algorithm-specific parameter-less philosophy: requires no cros... | Class mean vector calculation can be deceptive on non-convex o... |
| 29 | **MOBA** | Swarm Intelligence | Population-based | Bat velocity tuned via echolocation frequency; local search triggered conditionall... | 2 to 3 objectives | Continuous decision variables | Dynamic switching between global echolocation flight and fine ... | High parameter complexity: simultaneous tuning of frequency ra... |
| 30 | **MOMA** | Hybrid / Memetic Algorithm | Population-based | Combines global population exploration (NSGA-II framework) with intensive local re... | 2 to 3 objectives | Continuous, discrete, or combina... | Synergistic hybridization: evolutionary operators locate globa... | Local search can consume an enormous fraction of the total eva... |
| 31 | **RVEA** | Many-Objective Reference-Vector MOEA | Population-based | Decomposes objective space using reference vectors; selects one solution per cone ... | 3 to 15+ objectives (Many-Objective Optimization) | Continuous, discrete, or mixed d... | Dynamically shifts balance from diversity early on to sharp co... | Requires predefined reference vectors that can struggle on irr... |
| 32 | **IBEA** | Indicator-Based MOEA | Population-based | Environmental selection guided directly by binary indicators (additive $\epsilon$-... | 2 to 10+ objectives | Continuous, binary, or combinato... | Directly couples evolutionary selection pressure to formal per... | Computing the full pairwise indicator matrix in every generati... |
| 33 | **SMS-EMOA** | Indicator-Based MOEA | Population-based | Steady-state $(\mu+1)$ loop; non-dominated sorting classifies pool; the individual... | 2 to 3 objectives (exact), scales to many objectives with approximations | Continuous, discrete, or combina... | Guarantees that every population replacement strictly preserve... | Exact hypervolume contribution computation is \#P-hard, scalin... |
| 34 | **HypE** | Indicator-Based / Many-Objective MOEA | Population-based | Replaces exact hypervolume calculation with Monte Carlo sampling boxes to assign i... | 3 to 15+ objectives (Many-Objective Optimization) | Continuous, discrete, or combina... | Enables hypervolume-driven search to scale to 5, 10, or 20 obj... | Stochastic variance in Monte Carlo estimation can introduce no... |
| 35 | **R-NSGA-II** | Preference-Based MOEA | Population-based | Replaces crowding distance in NSGA-II with normalized Euclidean distance to user r... | 2 to 10+ objectives | Continuous, discrete, or mixed d... | Focuses computational effort exclusively on user-specified Reg... | Does not explore the full Pareto front; performance depends en... |
| 36 | **MGDA** | Gradient-Based Multiobjective | Deterministic | Finds the minimum-norm vector in the convex hull of objective gradients via Quadra... | 2 to 10+ differentiable objectives | Continuous, smooth, and differen... | Theoretically guarantees descent on ALL objectives simultaneou... | Requires smooth, differentiable objective functions; finds onl... |
| 37 | **GradNorm** | Multi-Task Gradient Balancing (Task C Deep Dive #3) | Deterministic | Dynamically balances gradient magnitudes and relative training rates of disparate ... | 2 to 10+ differentiable task losses | Continuous deep neural network p... | Prevents gradient dominance in multi-task learning where high-... | Requires selecting a shared representation layer $W$ for gradi... |

---

### Table B: Critical Algorithmic Differences & Comparative Analysis

This section examines the key mathematical, geometric, and operational differences between fundamental algorithm pairs and evolutionary generations.

#### Weighted Sum vs. ε-Constraint (Scalarization Geometry & Non-Convexity)
- **Mathematical Formulation:** Weighted Sum linearly aggregates all objectives into a single scalar $\min \sum w_m f_m(x)$ using a normal vector $w$. In contrast, $\epsilon$-Constraint selects one primary objective $f_k(x)$ for minimization while converting the remaining $M-1$ objectives into explicit inequality constraints: $f_m(x) \le \epsilon_m$ for $m \ne k$.
- **Pareto Front Geometry:** Weighted Sum can **only** locate solutions on the convex hull of the Pareto front; any solution residing on a non-convex (concave) or indented boundary can never be a tangency point of a supporting hyperplane and is completely unfindable regardless of the weight vector chosen. $\epsilon$-Constraint slices the objective space with orthogonal bounding hyperplanes $\epsilon_m$, allowing it to locate every Pareto-optimal point regardless of whether the front is convex, non-convex, or disconnected.
- **Objective Scaling:** Weighted Sum is severely distorted by differences in physical units and magnitudes between objectives, requiring precise a priori normalization. $\epsilon$-Constraint does not combine objectives additively, making the primary objective minimization completely invariant to the numerical scaling of the constrained secondary objectives.
- **Subproblem Feasibility:** Every weight vector in Weighted Sum yields a feasible subproblem (assuming the original decision space is non-empty). In $\epsilon$-Constraint, poorly chosen $\epsilon_m$ thresholds frequently lead to empty feasible regions, wasting optimization cycles on infeasible subproblems unless an accurate payoff table or nadir estimation is computed beforehand.

#### Benson’s Method vs. Ordinary Scalarization (Efficiency Guarantees & Reference Point Projection)
- **Direction of Search:** Ordinary scalarization (e.g., Weighted Sum) pulls candidate solutions toward the origin or an ideal point $z^*$. Benson's Method operates in the opposite direction: it starts from an exterior, dominated, or weakly feasible reference point $z^0$ ($z_m^0 \ge f_m(x)$) and maximizes the total shortfall $\sum w_m (z_m^0 - f_m(x))$ subject to $f_m(x) \le z_m^0$.
- **Efficiency Guarantees:** Ordinary weighted sum with non-negative weights $w_m \ge 0$ only guarantees *weak* Pareto optimality (solutions on vertical/horizontal boundaries may be dominated). Benson's method strictly guarantees *strong* Pareto optimality whenever all weights are strictly positive ($w_m > 0$).
- **Pareto Efficiency Testing:** Benson's method provides a constructive mathematical test for Pareto efficiency: if a candidate point $x^0$ is plugged in as $z^0$ and the maximum shortfall is zero, $x^0$ is rigorously proven to be Pareto-optimal. If the shortfall is positive, the optimizer immediately returns an efficient point that strictly dominates $x^0$.
- **Linear Programming Adaptability:** For multiobjective linear programs (MOLP), Benson's method formulates a single primal-dual linear programming structure that avoids non-linear distance norms and hyperbolic boundary shapes, whereas metric scalarizations ($L_p$) introduce non-linear or non-differentiable contours.

#### MOGA, NSGA, NPGA, and NSGA-II (Evolution of First- vs. Second-Generation MOEAs)
- **Ranking Mechanism:**
  - **MOGA (Fonseca & Fleming, 1993):** Ranks each individual by $1 +$ the number of population members that dominate it ($r_i = 1 + |\{j \mid x_j \prec x_i\}|$). Does not partition into discrete fronts.
  - **NSGA (Srinivas & Deb, 1994):** Classifies the population into discrete hierarchical fronts ($F_1, F_2, \dots$) using an iterative $O(M N^3)$ sorting algorithm, assigning dummy fitness values that decrease strictly front by front.
  - **NPGA (Horn et al., 1994):** Bypasses global ranking entirely. Conducts binary tournament selection against a small random comparison set of size $t_{\text{dom}}$: a candidate wins if it dominates the sample while its opponent does not.
  - **NSGA-II (Deb et al., 2002):** Introduces fast non-dominated sorting ($O(M N^2)$) using domination counters $n_p$ and lists of dominated solutions $S_p$, stratifying the combined parent-offspring pool ($2N$).
- **Diversity Maintenance:**
  - **MOGA:** Phenotypic/objective fitness sharing using a sharing radius $\sigma_{\text{share}}$ applied across individuals of identical rank.
  - **NSGA:** Niche fitness sharing using $\sigma_{\text{share}}$ applied strictly within each individual front layer before assigning dummy fitness to the next layer.
  - **NPGA:** Niche count tie-breaker using $\sigma_{\text{share}}$ evaluated only when both tournament candidates tie in dominating the comparison set.
  - **NSGA-II:** Completely parameter-free crowding distance measuring the cuboid perimeter formed by neighboring solutions, totally eliminating $\sigma_{\text{share}}$.
- **Elitism & Complexity:**
  - MOGA, NSGA, and original NPGA are non-elitist with generational replacement, allowing high-quality trade-offs to be lost ($O(M N^2)$ to $O(M N^3)$ complexity).
  - NSGA-II enforces strict $(\mu+\lambda)$ elitism by pooling parents and offspring ($2N$), guaranteeing monotonic front progression in $O(M N^2)$ time.

#### NSGA-II vs. NSGA-III (Objective Scalability & Crowding vs. Reference Points)
- **Diversity Operator in Environmental Selection:** NSGA-II relies on crowding distance, which computes the sum of normalized distances between adjacent solutions along each objective axis. NSGA-III completely replaces crowding distance with structured reference points generated on a normalized unit hyperplane (Das-Dennis method).
- **Behavior in Many-Objective Spaces ($M \ge 4$):**
  - In NSGA-II, as $M$ increases, the proportion of mutually non-dominated solutions in a random population approaches 100% (e.g., $>90\%$ for $M=5$, $>99\%$ for $M=10$). Consequently, all solutions end up in Front 1, and selection pressure is determined entirely by crowding distance. Because crowding distance favors extreme points along each axis, the population collapses to the coordinate axes and outer boundaries, leaving the center of the Pareto front hollow and unpopulated.
  - In NSGA-III, non-dominated sorting is retained, but solutions in the critical front are associated with structured reference lines via perpendicular distance. A niche-preservation count $\rho_j$ tracks how many solutions are assigned to each reference point, explicitly prioritizing under-represented reference directions. This maintains uniform coverage across the entire high-dimensional manifold.
- **Normalization Strategy:** NSGA-II performs 1D min-max normalization independently per objective. NSGA-III computes an adaptive hyperplane by identifying extreme points and solving a system of linear equations to calculate intercept points, effectively mapping complex irregular Pareto manifolds to a standardized simplex.
- **Target Use Case:** NSGA-II remains the gold standard for bi-objective and tri-objective optimization ($M=2, 3$). NSGA-III is explicitly engineered for many-objective optimization ($M = 4$ to $15+$).

#### SPEA2 vs. PESA-II (Continuous $k$-NN Density vs. Region-Based Grid Selection)
- **Selection Unit:** SPEA2 performs binary tournament selection over individual candidate solutions based on a fine-grained scalar fitness $F(i) = R(i) + D(i)$. PESA-II introduces region-based selection: the primary selection unit is a hyperbox within an adaptive grid, and tournament selection chooses the hyperbox with the lowest squeeze factor before randomly picking an individual within that box.
- **Density Estimation Mechanism:**
  - SPEA2 calculates exact Euclidean distance to the $k$-th nearest neighbor in objective space ($k = \lfloor \sqrt{N+\bar{N}} \rfloor$), providing a smooth, continuous density metric that accurately detects local clustering without artificial boundaries.
  - PESA-II overlays a discrete orthogonal hypergrid across the objective space. Density is measured simply as the integer count of solutions (squeeze factor) falling inside each hypercube.
- **Computational Complexity:**
  - SPEA2 requires all-pairs distance matrix computation and row-wise sorting, incurring $O((N+\bar{N})^2 \log(N+\bar{N}))$ time, which becomes very heavy for large archives.
  - PESA-II computes grid coordinates in $O(M \cdot \bar{N})$ time, completely bypassing all-pairs distance sorting during selection.
- **Boundary Preservation:** SPEA2's iterative nearest-neighbor archive truncation explicitly protects boundary and extreme points. PESA-II's grid can split natural boundary clusters if an arbitrary grid division line happens to bisect them, creating discretization artifacts.

#### MOEA/D vs. MOEA/D-DE (Crossover Mechanics & Landscape Adaptability)
- **Reproduction Operators:** Standard MOEA/D (Zhang & Li, 2007) relies on Simulated Binary Crossover (SBX) and polynomial mutation. MOEA/D-DE (Li & Zhang, 2009) replaces SBX with Differential Evolution (DE) mutation ($v_i = x_{r1} + F (x_{r2} - x_{r3})$) and binomial crossover, followed by polynomial mutation.
- **Variable Linkage & Rotation:** SBX searches along coordinate-aligned axes, making it struggle on continuous problems with rotated, non-separable, or strongly correlated decision variables. DE generates difference vectors directly from parent differences ($x_{r2} - x_{r3}$), naturally aligning search directions with the contour lines of the landscape.
- **Self-Adapting Step Sizes:** In MOEA/D-DE, as the neighborhood population converges toward the true Pareto front, the magnitude of the difference vectors automatically contracts without requiring scheduled step size decay, providing seamless transition from global exploration to fine local exploitation.
- **Parent Sampling Scope:** Standard MOEA/D selects parents strictly from the neighborhood $B(i)$. MOEA/D-DE introduces a neighborhood mating probability $\delta$ (typically 0.9): with probability $\delta$, parents are drawn from $B(i)$; with probability $1-\delta$, parents are drawn globally from the entire population, preventing stagnation in isolated neighborhoods.

#### MOEA/D-ACO vs. MOACO (Decomposition Guidance vs. Multi-Pheromone Dominance)
- **Fundamental Framework:** MOEA/D-ACO applies decomposition to combinatorial routing: it breaks the bi-objective problem into $N$ scalar subproblems with fixed weight vectors $\lambda^i$, directing individual ants to optimize specific scalar trade-offs. MOACO uses Pareto dominance: ants search the space using a combined multi-pheromone transition rule, and only iteration-wide Pareto-optimal ants deposit pheromone.
- **Pheromone Architecture:**
  - MOACO maintains two global pheromone matrices ($\tau^1$ for distance, $\tau^2$ for time) that are updated collectively by non-dominated tours.
  - MOEA/D-ACO maintains either subproblem-specific pheromone matrices or a shared pheromone system that is reinforced based on neighborhood subproblem improvements.
- **Heuristic Information Integration:** In MOEA/D-ACO, heuristic visibility is scalarized directly by subproblem weights ($\eta_{jk}^i = [1/d_{jk}]^{\lambda_1^i} [1/t_{jk}]^{\lambda_2^i}$), systematically steering ant $i$ along trade-off ray $\lambda^i$. In MOACO, all ants use the same multiplicative heuristic composite with fixed exponents $\beta_1, \beta_2$.
- **Diversity Maintenance:** MOEA/D-ACO enforces diversity by distributing ants across predefined weight vectors $\lambda^i$ along the unit simplex. MOACO relies on pheromone evaporation $\rho$ and Pareto front archive filtering to maintain spread across alternative paths.

#### MOPSO vs. MOGWO (Swarm Kinematics vs. Hierarchical Leader Guidance)
- **Motion Equations & Leadership:**
  - MOPSO updates particle position using a physical velocity vector driven by two attractors: personal best ($pbest_i$) and a single global best ($gbest_i$) drawn from an external grid archive.
  - MOGWO completely eliminates velocity vectors and inertia weights. Instead, each wolf updates its position by computing three hunting displacement vectors directed toward three elite pack leaders—alpha ($\alpha$), beta ($\beta$), and delta ($\delta$)—selected from the least crowded archive cells, averaging their positions: $x(t+1) = (X_1 + X_2 + X_3)/3$.
- **Exploration vs. Exploitation Balance:** MOPSO balances search dynamically via inertia weight damping ($w: 0.9 \to 0.4$) and acceleration coefficients $c_1, c_2$. MOGWO achieves this transition by linearly decaying an encircling coefficient $a$ from 2 to 0 across iterations.
- **Susceptibility to Trapping & Boundary Oscillation:** MOPSO is prone to velocity explosion and boundary overshooting if velocities are not tightly clamped, and can collapse if particles chase a dominated $gbest$. MOGWO's three-leader guidance provides broader triangulation that resists single-point collapse, but averaging leaders from opposite ends of a non-convex front can pull wolves into interior dominated regions.
- **State Representation:** MOPSO requires storing both position $x_i$ and velocity $v_i$ for each particle, plus historical $pbest_i$. MOGWO stores only the current position $x_i$, reducing particle memory overhead by half.

#### MOABC vs. MOSA (Population Foraging vs. Trajectory Thermal Acceptance)
- **Search Paradigm:** MOABC is a population-based swarm intelligence algorithm inspired by honey bee foraging, maintaining distinct functional cohorts (Employed, Onlooker, Scout bees). MOSA is a physics-inspired trajectory/population metaheuristic based on thermodynamics and statistical mechanics, transitioning from high-temperature random exploration to low-temperature annealing.
- **Stagnation & Escape Mechanisms:**
  - MOABC escapes local Pareto traps using a deterministic abandonment threshold: if a food source fails to improve after `limit` trials, a scout bee abandons it and instantly re-initializes a new solution uniformly at random across the decision space.
  - MOSA escapes local Pareto traps via probabilistic uphill moves: dominated or non-improving solutions are accepted with Boltzmann probability $P = \exp(-\Delta E / T)$, which begins near 1.0 at high temperatures and gradually freezes to 0.0.
- **Perturbation Strategy:** MOABC perturbs only one decision variable at a time ($v_{ij} = x_{ij} + \phi (x_{ij} - x_{kj})$), creating coordinate-aligned search steps. MOSA typically perturbs all dimensions simultaneously using Gaussian or Cauchy random walks, exploring spherical neighborhoods.
- **Convergence Rate:** MOABC converges substantially faster on continuous problems due to swarm information sharing in the onlooker phase. MOSA converges slowly because it requires hundreds of temperature cooling epochs to guarantee asymptotic optimality.

#### MODE vs. MO-TLBO (Vector Difference Arithmetic vs. Classroom Metaphor)
- **Algorithmic Philosophy & Hyperparameters:**
  - MODE relies heavily on tuned numerical hyperparameters: the Differential Evolution scale factor $F$ and crossover probability $CR$.
  - MO-TLBO is founded on a 'parameter-less' philosophy: it requires zero algorithm-specific control parameters (no mutation rate, no crossover rate), relying purely on population size and generation count.
- **Perturbation & Update Dynamics:**
  - MODE updates solutions using directional difference vectors between random population members ($x_{r1} - x_{r2}$), which self-scale as the population converges.
  - MO-TLBO updates solutions through two structured social phases: the Teacher Phase (where learners are pulled toward the class mean and an elite archive teacher) and the Learner Phase (where pairs of learners interact and move toward/away from each other based on Pareto dominance).
- **Selection Operator:** MODE performs greedy one-to-one selection between the mutant/trial vector and its parent target vector, updating an external archive upon non-dominance. MO-TLBO accepts updates in both phases using Deb's feasibility-first constrained dominance rules.
- **Landscape Biases:** MODE excels on rotated, non-separable continuous landscapes due to difference-vector invariance. MO-TLBO can struggle on non-convex or disconnected Pareto fronts because computing the arithmetic mean $M_{\text{mean}}$ of the population assumes a convex unimodal distribution.

#### RVEA, IBEA, SMS-EMOA, and HypE (Taxonomy of Advanced Many-Objective & Indicator Algorithms)
- **Underlying Guidance Mechanism:**
  - **RVEA (Cheng et al., 2016):** Reference-vector-guided decomposition using Angle-Penalized Distance (APD), balancing convergence distance and angular divergence dynamically.
  - **IBEA (Zitzler & Künzli, 2004):** Direct binary indicator optimization (typically additive binary $\epsilon$-indicator $I_{\epsilon+}$), evaluating pairwise population dominance losses.
  - **SMS-EMOA (Emmerich et al., 2005):** Steady-state $(\mu+1)$ selection discarding the single solution in the worst front that contributes the minimum exclusive hypervolume $\Delta \text{HV}$.
  - **HypE (Bader & Zitzler, 2011):** Hypervolume estimation via Monte Carlo sampling integration, approximating individual hypervolume contributions in high dimensions.
- **Objective Scalability ($M$):**
  - **SMS-EMOA:** Computationally limited to $M \le 3$ for exact hypervolume calculation due to the \#P-hard complexity of exact hypervolume slicing ($O(N^{M/2})$).
  - **IBEA:** Scales cleanly to $M = 5$ to $10+$ because computing the pairwise additive $\epsilon$-indicator matrix requires only $O(M N^2)$ time.
  - **RVEA:** Scales efficiently to $M = 3$ to $15+$ objectives by partitioning objective space with reference vectors without any hypervolume calculations.
  - **HypE:** Scales to $M = 5$ to $20+$ objectives by replacing exact hypervolume slicing with $N_s$ Monte Carlo sampling points, decoupling runtime from exponential $M$ growth.
- **Diversity vs. Indicator Philosophy:**
  - RVEA enforces diversity geometrically through angular cone partitioning around reference vectors.
  - IBEA, SMS-EMOA, and HypE require no geometric reference lines, grids, or sharing parameters; diversity is an inherent mathematical byproduct of maximizing the indicator.
- **Generational Model:** SMS-EMOA is strictly steady-state ($(\mu+1)$ replacement). IBEA, RVEA, and HypE use generational or $(\mu+\lambda)$ environmental selection.

#### NSGA-II vs. R-NSGA-II (A Posteriori Global Front vs. A Priori Region of Interest (ROI))
- **Optimization Scope:** NSGA-II searches globally to discover a complete, evenly distributed Pareto front across the entire objective space. R-NSGA-II incorporates a priori user preferences to focus search exclusively on localized Regions of Interest (ROI) centered around one or more user-supplied reference points $z^{\text{pref}}$.
- **Crowding Distance vs. Preference Distance:**
  - NSGA-II uses crowding distance in the critical front to reward solutions that are far from their neighbors, actively expanding the spread of the front.
  - R-NSGA-II replaces crowding distance with normalized Euclidean distance to the closest reference point: solutions closer to $z^{\text{pref}}$ receive higher survival priority, steering the population toward preferred trade-offs.
- **Niche Control:** In R-NSGA-II, if multiple solutions crowd too tightly around a reference point, an explicit $\epsilon$-clearing parameter $\epsilon_{\text{clear}}$ eliminates or penalizes redundant neighbors, ensuring a controlled, diverse cluster within the ROI rather than a collapsed single point.
- **Evaluation Efficiency:** When a stakeholder only cares about a specific trade-off range (e.g., laptop weight under 1.8 kg and battery life over 12 hours, P14), NSGA-II wastes over 80% of its evaluation budget mapping irrelevant boundary points. R-NSGA-II concentrates 100% of the budget on the desired compromise sector, finding higher-density, higher-precision solutions in the target region.

#### MGDA vs. GradNorm (Multi-Objective Common Descent vs. Multi-Task Gradient Balancing)
- **Primary Problem Domain:** MGDA (Désidéri, 2012; Sener & Koltun, 2018) is a general mathematical optimization algorithm for finding Pareto-optimal stationary points on differentiable multi-objective continuous functions. GradNorm (Chen et al., ICML 2018) is a specialized deep learning algorithm for multi-task neural networks with shared representations and multiple task-specific heads.
- **Mathematical Objective:**
  - MGDA finds a single common descent vector $d = -\sum \alpha_m \nabla f_m(\theta)$ by solving a quadratic program to find the minimum-norm vector in the convex hull of task gradients ($\min_\alpha \| \sum \alpha_m \nabla f_m \|^2$). It guarantees simultaneous descent: $\nabla f_m^T d < 0$ for all $m$.
  - GradNorm dynamically adapts the scalar task loss weights $w_i(t)$ in the composite loss $L_{\text{total}} = \sum w_i(t) L_i(t)$ to balance the gradient $L_2$-norms across tasks with respect to a designated shared layer $W$, penalizing tasks whose gradient norms deviate from a dynamic target pace.
- **Handling Conflicting Gradients:**
  - If two task gradients point in opposite directions (angle $> 90^\circ$), MGDA's minimum norm falls toward zero; if gradients are diametrically opposed, the descent step collapses to zero (Pareto-critical KKT point), causing training to halt completely.
  - GradNorm never halts training due to gradient conflict. It scales the loss weights $w_i(t)$ to ensure that the slower-learning task exerts equal or greater gradient magnitude on the shared parameters, forcing the shared representation to learn balanced features.
- **Computational Overhead:** MGDA requires computing full network parameter gradients for each task separately and solving a QP/Frank-Wolfe problem at every iteration. GradNorm computes gradients only with respect to the final shared layer $W$, adding minimal backpropagation overhead during training.

---

### Table C: Practical Algorithm Selection Guide by Use Case

To guide engineering practitioners and researchers in selecting the most suitable optimization method, algorithms are grouped below by practical operational scenario.

#### Use Case: Two-Objective Optimization (Bi-Objective Problems)
- **Recommended Algorithms:** NSGA-II (Alg 14), SMS-EMOA (Alg 33), MODE (Alg 27), MOPSO (Alg 22)
- **Technical Justification:** For $M=2$, Pareto dominance provides immense discriminatory power: solutions easily dominate or are dominated by others. NSGA-II is the premier general-purpose choice, offering robust $O(N^2)$ fast sorting, parameter-free crowding distance, and strong $(\mu+\lambda)$ elitism. For problems where maximum front spread and hypervolume are critical, SMS-EMOA provides theoretically optimal spacing because exact 2D hypervolume slicing is fast ($O(N \log N)$). For smooth continuous landscapes with correlated variables, MODE achieves faster convergence via difference-vector scaling, while MOPSO provides rapid initial front discovery on unimodal continuous benchmarks.
- **Implementation Caveats & Warnings:** Avoid NSGA-III or HypE for $M=2$; reference lines and Monte Carlo estimation introduce unnecessary computational overhead without improving 2D front quality. Avoid classical Weighted Sum if the front is suspected to be non-convex.

#### Use Case: Many-Objective Optimization ($M \ge 4$ to $15+$ Objectives)
- **Recommended Algorithms:** NSGA-III (Alg 15), RVEA (Alg 31), IBEA (Alg 32), HypE (Alg 34)
- **Technical Justification:** In many-objective spaces, the curse of dimensionality causes almost the entire population to become mutually non-dominated in early generations, completely neutralizing standard Pareto dominance sorting. NSGA-III overcomes this by replacing crowding distance with structured Das-Dennis reference points on a normalized hyperplane, maintaining uniform spread across high dimensions. RVEA partitions the objective space using reference vectors and balances convergence and diversity adaptively via Angle-Penalized Distance (APD). IBEA bypasses geometric reference points entirely by guiding selection via pairwise additive $\epsilon$-indicators ($I_{\epsilon+}$). HypE enables hypervolume-driven search to scale to 15 objectives by substituting exact \#P-hard hypervolume calculation with Monte Carlo integration.
- **Implementation Caveats & Warnings:** NSGA-II and SPEA2 fail in this regime due to crowding distance boundary collapse and uniform zero raw fitness. Das-Dennis reference points in NSGA-III scale combinatorially $\binom{M+p-1}{p}$, requiring two-layer grids for $M > 8$.

#### Use Case: Constrained Multiobjective Optimization
- **Recommended Algorithms:** Constrained NSGA-II (Alg 21), MO-TLBO with Deb’s Rules (Alg 28), MODE with Deb’s Rules (Alg 27)
- **Technical Justification:** Handling non-linear equality and inequality constraints without subjective penalty tuning is essential in engineering. Constrained NSGA-II directly embeds Deb's three feasibility-first comparison rules into tournament selection and sorting: feasible solutions always beat infeasible ones, infeasible solutions are ranked purely by total constraint violation $CV$, and Pareto dominance applies only when both solutions are feasible. This naturally drives the population toward the boundary of the feasible region (where optimal trade-offs reside) without requiring penalty parameter calibration. MO-TLBO and MODE integrated with Deb's rules provide effective continuous alternatives with parameter-free or self-adaptive step sizes.
- **Implementation Caveats & Warnings:** In problems with disjoint feasible islands separated by wide infeasible barriers, Deb's rules can trap the population in a sub-optimal feasible island. In our framework, total constraint violation $CV$ is guaranteed non-negative with standard tolerance $\epsilon_{\text{feas}} = 10^{-4}$.

#### Use Case: Discrete and Combinatorial Optimization (e.g. TSP, Routing, Scheduling)
- **Recommended Algorithms:** MOACO (Alg 25), MOEA/D-ACO (Alg 20), MOMA with 2-Opt (Alg 30), NPGA (Alg 13)
- **Technical Justification:** Combinatorial spaces like permutation routing (P8 bi-objective TSP) break standard continuous vector operators (e.g., polynomial mutation and SBX require awkward repair heuristics). MOACO is purpose-built for combinatorial routing: it uses dual pheromone matrices (distance and time) and dual heuristic visibility matrices to construct strictly valid permutations naturally. MOEA/D-ACO applies decomposition to combinatorial routing, assigning ants to specific trade-off directions along the unit simplex. MOMA hybridizes evolutionary search with discrete local search (2-opt edge swaps), rapidly polishing rough tours. NPGA provides fast, lightweight binary tournament selection for binary feature selection (P9).
- **Implementation Caveats & Warnings:** Continuous swarm algorithms (MOPSO, MOGWO, MOBA) and gradient methods (MGDA) are completely inapplicable without artificial discretization mappings. MOACO requires $O(M \cdot C^2)$ memory for pheromone matrices.

#### Use Case: Preference-Based Optimization & Known Regions of Interest (ROI)
- **Recommended Algorithms:** R-NSGA-II (Alg 35), Goal Programming (Alg 06), Value Function (Alg 05), Lexicographic (Alg 07)
- **Technical Justification:** When decision-makers possess clear prior preferences or target goals, searching the entire global Pareto front is wasteful and computationally inefficient. R-NSGA-II replaces crowding distance with normalized Euclidean distance to user-specified reference points $z^{\text{pref}}$, enforcing an $\epsilon$-clearing distance to generate a dense, well-distributed cluster of solutions focused strictly in the Region of Interest (P14 laptop design). Goal Programming minimizes unwanted positive and negative deviations from explicit operational targets (P11 production planning). The Value Function method directly maximizes an articulated multi-attribute utility function. Lexicographic optimization strictly satisfies non-negotiable primary requirements before optimizing secondary criteria.
- **Implementation Caveats & Warnings:** Goal Programming can stop at an interior, Pareto-dominated point if targets are set too pessimistically. R-NSGA-II provides zero visibility into alternative trade-offs lying outside the specified preference cone.

#### Use Case: Differentiable Multi-Task Learning (Deep Neural Networks)
- **Recommended Algorithms:** GradNorm (Alg 37), MGDA (Alg 36)
- **Technical Justification:** Multi-task learning (MTL) involves optimizing deep neural networks with a shared representation and multiple task-specific loss functions (P13). Standard evolutionary algorithms cannot scale to millions of network weights. MGDA computes the minimum-norm vector in the convex hull of task gradients via quadratic programming, finding a common descent direction that guarantees simultaneous descent on all task losses without trading off. GradNorm dynamically balances task loss weights $w_i(t)$ at each training step by equalizing gradient norms with respect to a shared layer and pace-balancing tasks according to relative training rates.
- **Implementation Caveats & Warnings:** MGDA can halt prematurely if task gradients are in direct opposition (norm goes to zero), causing gradient conflict. GradNorm avoids conflict stalling by adjusting loss weights while allowing standard SGD/Adam backpropagation, but requires designating a shared representation layer $W$.

#### Use Case: Expensive Objective Evaluations (High CPU/Simulation Cost)
- **Recommended Algorithms:** Classical $\epsilon$-Constraint (Alg 02), Benson’s Method (Alg 03), TOPSIS / VIKOR on Archives (Algs 08, 09), IBEA (Alg 32)
- **Technical Justification:** When each objective function evaluation requires complex CFD, FEA, or physical laboratory testing (e.g. budget $< 300$ evaluations), population-based genetic algorithms cannot converge effectively. The $\epsilon$-Constraint method can trace high-precision Pareto points in dozens of function calls using gradient-based SQP solvers. Benson's method quickly projects any initial candidate onto the efficient boundary. In such scenarios, running a lightweight surrogate or generating a modest set of candidate alternatives followed by TOPSIS or VIKOR ranking provides optimal decision support without exhausting evaluation budgets.
- **Implementation Caveats & Warnings:** Population-based MOEAs typically require 5,000 to 20,000 evaluations to form a dense front; running them with budgets $< 100$ produces coarse, highly under-converged approximations.

---

## 5. Theoretical Synthesis & Fundamental Trade-Offs

The systematic study of all 37 multiobjective optimization algorithms reveals fundamental theoretical and practical trade-offs that govern algorithm selection, design, and performance. No single algorithm dominates all problem domains; rather, algorithm performance is dictated by the interaction between the optimization mechanism, problem landscape geometry, objective dimensionality, and decision-maker preference structure.

### 5.1 Convergence vs. Diversity Trade-Off
The dual primary goals of multiobjective optimization are:
1. **Convergence (Proximity):** Driving candidate solutions as close as possible to the true theoretical Pareto-optimal set $P^*$ (minimizing generational distance).
2. **Diversity (Spread & Uniformity):** Distributing solutions uniformly across the entire extent of the Pareto front $PF^*$ without clustering, gaps, or boundary omission (minimizing spacing $S$ and maximizing spread).

In evolutionary search, these two goals constantly compete for reproductive selection pressure:
- **Dominance-Driven Selection (MOGA, NSGA):** Provides strong convergence toward non-dominated boundaries, but when the number of non-dominated solutions exceeds the population budget, convergence stalls unless diversity preservation operators intervene.
- **Crowding Distance (NSGA-II):** Achieves diversity without parameters, but suffers from boundary-point bias, often preserving extreme outliers at the expense of well-distributed intermediate trade-offs.
- **Decomposition (MOEA/D):** Replaces density-based diversity with a geometric framework of predefined weight vectors. While this enforces uniform diversity, convergence relies heavily on whether the problem's Pareto geometry matches the decomposition metric (e.g., Tchebycheff vs PBI).
- **Indicator Selection (SMS-EMOA, IBEA):** Harmonizes convergence and diversity into a single mathematical scalar (e.g., Hypervolume contribution or additive $\epsilon$-indicator). While hypervolume maximization theoretically guarantees both convergence to the true front and optimal boundary spread, computing it exactly is computationally intractable for $M \ge 4$.

### 5.2 Objective Scalability: The Many-Objective Breakdown ($M \ge 4$)
A profound structural transition occurs when moving from bi-objective/tri-objective problems ($M \in \{2, 3\}$) to many-objective optimization ($M \ge 4$ to $15+$):
- **Pareto Dominance Resistance:** In a random population of size $N=100$ in an $M$-dimensional objective space, the probability that two randomly generated solutions are mutually non-dominated approaches $1 - (1/2)^{M-1}$. For $M=10$, over $99\%$ of the population is mutually non-dominated in the very first generation.
- **Consequence for Pareto-Based Algorithms:** Algorithms relying primarily on Pareto dominance (NSGA-II, SPEA2, MOGA) suffer complete loss of selection pressure. In NSGA-II, all solutions fall into Front 1; selection is governed entirely by crowding distance, which collapses to coordinate axes and causes extreme clustering.
- **Methodological Solutions for Many-Objective Spaces:**
  1. *Reference-Point / Directional Niching (NSGA-III, RVEA):* Discard distance-based crowding in favor of structured simplex reference lines (Das-Dennis) or Angle-Penalized Distance cones.
  2. *Decomposition (MOEA/D, MOEA/D-DE):* Converts the problem into $N$ scalar subproblems, completely bypassing dominance sorting.
  3. *Indicator-Based Selection (IBEA, HypE):* Measures continuous degrees of preference or Monte Carlo estimated hypervolume contributions, preserving fine-grained selection gradients regardless of $M$.

### 5.3 A Priori, A Posteriori, and Interactive Preference Articulation
Algorithms differ fundamentally in when and how decision-maker (DM) preferences are incorporated:
1. **A Priori Methods (Algs 1, 4, 5, 6, 7, 35):** Preferences (weights, utility functions, targets, reference points) are articulated *before* optimization. The search is steered directly toward a specific Region of Interest (ROI).
   - *Advantage:* Highly efficient; avoids wasting evaluations exploring unwanted trade-offs.
   - *Risk:* If preferences are misspecified or based on flawed assumptions about what is achievable, the algorithm converges to an unsatisfactory or unfeasible region.
2. **A Posteriori Methods (Algs 10–20, 21–34):** The algorithm maps the complete Pareto front without prior preferences. The DM inspects the generated trade-off surface and selects a compromise solution *after* optimization.
   - *Advantage:* Provides complete visibility into trade-offs and unexpected breakthrough solutions.
   - *Risk:* Heavy computational burden to map high-dimensional manifolds; causes decision paralysis when presenting hundreds of complex trade-offs to human stakeholders.
3. **MCDM Selection on Archives (Algs 8, 9):** Bridges a posteriori generation and final decision: TOPSIS and VIKOR evaluate the discrete archive generated by an a posteriori MOEA, systematically ranking candidate solutions using structured geometric or compromise indices.

### 5.4 Constraint Handling Paradigms
Constrained multiobjective optimization introduces conflicting feasibility and optimality pressures:
- **Penalty Function Approach:** Penalizes objective values proportionally to constraint violation $CV$. Highly sensitive to penalty coefficients; small penalties yield infeasible solutions, while large penalties trap the population in local feasible basins.
- **Deb's Feasibility-First Rules (Alg 21):** Strictly prioritizes feasibility over objective performance. Parameter-free, exceptionally robust for finding boundary optima, but can struggle when the feasible region consists of tiny disconnected islands.
- **Decomposition-Based Feasibility:** MOEA/D incorporates constraint violation directly into scalar subproblems (e.g., constrained Tchebycheff), balancing constraint satisfaction along each weight vector independently.

---

---

## 6. Foundational Literature References

The theoretical principles, mathematical formulations, and algorithmic mechanics documented across all 37 algorithms are established upon the following foundational publications:

1. **Weighted Sum & Scalarization Foundations:**
   - Kuhn, H. W., & Tucker, A. W. (1951). Nonlinear programming. *Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability*, 481–492.
   - Zadeh, L. (1963). Optimality and non-scalar-valued performance criteria. *IEEE Transactions on Automatic Control*, 8(1), 59–60.
2. **ε-Constraint Method:**
   - Haimes, Y. Y., Lasdon, L. S., & Wismer, D. A. (1971). On a bicriterion formulation of the problems of integrated system identification and system optimization. *IEEE Transactions on Systems, Man, and Cybernetics*, 1(3), 296–297.
3. **Benson’s Method:**
   - Benson, H. P. (1978). Existence of efficient solutions for vector maximization problems. *Journal of Optimization Theory and Applications*, 26(4), 569–580.
4. **Weight Metric & Tchebycheff Approaches:**
   - Bowman, V. J. (1976). On the relationship using Chebyshev norms and the efficient frontier of multiple-criteria objectives. *Lecture Notes in Economics and Mathematical Systems*, 130, 76–85.
   - Steuer, R. E., & Choo, E. U. (1983). An interactive Chebyshev procedure for multiple objective programming. *Mathematical Programming*, 26(3), 326–344.
5. **Goal Programming & Multi-Attribute Utility:**
   - Charnes, A., & Cooper, W. W. (1961). *Management Models and Industrial Applications of Linear Programming*. John Wiley & Sons, New York.
   - Keeney, R. L., & Raiffa, H. (1976). *Decisions with Multiple Objectives: Preferences and Value Trade-Offs*. John Wiley & Sons, New York.
6. **MCDM: TOPSIS and VIKOR:**
   - Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag, Berlin/Heidelberg.
   - Opricovic, S. (1998). *Multicriteria Optimization of Civil Engineering Systems*. Faculty of Civil Engineering, Belgrade.
   - Opricovic, S., & Tzeng, G. H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445–455.
7. **First-Generation MOEAs:**
   - Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic algorithms. *Proceedings of the 1st International Conference on Genetic Algorithms*, 93–100.
   - Fonseca, C. M., & Fleming, P. J. (1993). Genetic Algorithms for Multiobjective Optimization: Formulation, Discussion and Generalization. *Proceedings of the 5th International Conference on Genetic Algorithms*, 416–423.
   - Srinivas, N., & Deb, K. (1994). Muiltiobjective optimization using nondominated sorting in genetic algorithms. *Evolutionary Computation*, 2(3), 221–248.
   - Horn, J., Nafpliotis, N., & Goldberg, D. E. (1994). A niched Pareto genetic algorithm for multiobjective optimization. *Proceedings of the 1st IEEE Conference on Evolutionary Computation*, 82–87.
8. **Second-Generation Elitist & Decomposition Algorithms:**
   - Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
   - Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the strength Pareto evolutionary algorithm. *TIK-Report*, 103, ETH Zurich.
   - Corne, D. W., Jerram, N. R., Knowles, J. D., & Oates, M. J. (2001). PESA-II: Region-based selection in evolutionary multiobjective optimization. *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)*, 283–290.
   - Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.
   - Li, H., & Zhang, Q. (2009). Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II. *IEEE Transactions on Evolutionary Computation*, 13(2), 284–302.
9. **Many-Objective & Indicator-Based Algorithms:**
   - Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.
   - Cheng, R., Jin, Y., Olhofer, M., & Sendhoff, B. (2016). A reference vector guided evolutionary algorithm for many-objective optimization. *IEEE Transactions on Evolutionary Computation*, 20(5), 773–791.
   - Zitzler, E., & Künzli, S. (2004). Indicator-based selection in multiobjective search. *Parallel Problem Solving from Nature (PPSN VIII)*, 832–842.
   - Beume, N., Naujoks, B., & Emmerich, M. (2007). SMS-EMOA: Multiobjective selection based on dominated hypervolume. *European Journal of Operational Research*, 181(3), 1653–1669.
   - Bader, J., & Zitzler, E. (2011). HypE: An algorithm for fast hypervolume-based many-objective optimization. *Evolutionary Computation*, 19(1), 45–76.
   - Deb, K., Sundar, J., Udaya Bhaskar Rao, N., & Meyarivan, T. (2006). Reference point based multi-objective optimization using evolutionary algorithms. *International Journal of Computational Intelligence Research*, 2(3), 273–286.
10. **Swarm, Physics & Hybrid Metaheuristics:**
    - Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives with particle swarm optimization. *IEEE Transactions on Evolutionary Computation*, 8(3), 256–279.
    - Mirjalili, S., Saremi, S., Mirjalili, S. M., & Coelho, L. D. S. (2016). Multi-objective grey wolf optimizer: A novel algorithm for multi-criterion optimization. *Expert Systems with Applications*, 55, 304–318.
    - Hedayatzadeh, R., Hasanizadeh, B., Akbari, R., & Ziarati, K. (2010). A multi-objective artificial bee colony for optimizing multi-objective problems. *Proceedings of the 3rd International Conference on Advanced Computer Theory and Engineering (ICACTE 2010)*, V5-277–V5-281. (Expanded in: Akbari, R., Hedayatzadeh, R., Ziarati, K., & Hassanizadeh, B. (2012). A multi-objective artificial bee colony algorithm. *Swarm and Evolutionary Computation*, 2, 39–52).
    - Barán, B., & Schaerer, M. (2003). A multiobjective ant colony system for vehicle routing problem with time windows. *Proceedings of the 21st IASTED International Conference*, 97–102.
    - Alaya, I., Solnon, C., & Ghedira, K. (2007). Ant colony optimization for multi-objective optimization problems. *19th IEEE International Conference on Tools with Artificial Intelligence (ICTAI)*, 450–457.
    - Suppapitnarm, A., Seffen, K. A., Parks, G. T., & Clarkson, P. J. (2000). A simulated annealing algorithm for multiobjective optimization. *Engineering Optimization*, 33(1), 59–85.
    - Babu, B. V., & Jehan, M. M. L. (2003). Differential evolution for multi-objective optimization. *Proceedings of the 2003 Congress on Evolutionary Computation (CEC)*, 4, 2696–2703.
    - Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011). Teaching-learning-based optimization: a novel method for constrained mechanical design optimization problems. *Computer-Aided Design*, 43(3), 303–315.
    - Yang, X. S. (2011). Bat algorithm for multi-objective optimisation. *International Journal of Bio-Inspired Computation*, 3(5), 267–274.
    - Knowles, J. D., & Corne, D. W. (2000). M-PAES: A memetic algorithm for multiobjective optimization. *Proceedings of the 2000 Congress on Evolutionary Computation*, 1, 325–332.
11. **Gradient & Multi-Task Learning:**
    - Désidéri, J. A. (2012). Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. *Comptes Rendus Mathematique*, 350(5-6), 313–318.
    - Sener, O., & Koltun, V. (2018). Multi-task learning as multi-objective optimization. *Advances in Neural Information Processing Systems (NeurIPS 2018)*, 31, 527–538.
    - Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. *International Conference on Machine Learning (ICML 2018)*, 794–803.
