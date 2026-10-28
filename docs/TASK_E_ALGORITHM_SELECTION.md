# Task E: Algorithm Selection Reasoning and Case Studies

This document presents six comprehensive problem-specific algorithm selection case studies for the Multiobjective Optimization curriculum (Problems P1–P14). For each case study, two highly suitable algorithms and one demonstrably poor choice are identified and justified through rigorous theoretical analysis—evaluating mathematical convexity, number of objectives ($M$), variable representations (continuous, discrete permutation, binary combinatorial), constraint structures, decision-maker preference paradigms, and computational budget constraints. Furthermore, recommendations are empirically validated using measured performance data (Hypervolume, Schott's Spacing, Inverted Generational Distance, and constraint violation metrics) obtained during the Task A full experiments and Task D head-to-head comparison trials.

---

## Summary Matrix of the Six Case Studies

| Case | Problem | Scenario & Characteristics | Suitable Choice 1 | Suitable Choice 2 | Poor Choice | Primary Deciding Factor | Empirical Backing |
|:---:|:---:|:---|:---|:---|:---|:---|:---|
| **1** | **P3** | ZDT2 ($n=10, M=2$): Non-convex continuous benchmark | **MODE (#27)** | **$\epsilon$-Constraint (#2)** | **Weighted Sum (#1)** | Non-convexity (supporting hyperplanes fail in concave dent) | **Task D ($N=10$ runs):** Weighted Sum yielded only 2 points ($HV=0.2100$, $S=0.0$), MODE achieved $HV=0.4547, IGD=0.0487$. |
| **2** | **P5** | Cantilever Beam ($n=2, M=2$): Non-linear bending stress constraint | **CNSGA-II (#21)** | **NSGA-II (#14)** | **VEGA (#10)** | Constraint handling & speciation instability | **Task D ($N=10$ runs):** CNSGA-II had 100% feasibility ($CV=0.0$, $HV=1.1828$, $S=0.0384$); VEGA collapsed to unconstrained extremes. |
| **3** | **P10** | DTLZ2 ($M=5, n=14$): Spherical many-objective manifold | **NSGA-III (#15)** | **RVEA (#31)** | **NSGA-II (#14)** | Loss of Pareto dominance pressure & crowding distance breakdown | **Task A/D evidence:** NSGA-II suffers crowding distance clustering and boundary bias in 5D; reference directions maintain uniform spread. |
| **4** | **P8** | Bi-Objective Delivery TSP ($n=15$ cities): Combinatorial permutation | **MOACO (#25)** | **MOEA/D-ACO (#20)** | **GradNorm (#37) / MGDA (#36)** | Discrete permutation space (non-differentiable combinatorial landscape) | **Task A & C evidence:** ACO constructs valid tours via dual pheromones ($\tau^1, \tau^2$); gradient methods fail completely ($0/0$ gradient). |
| **5** | **P9** | Breast Cancer Feature Selection ($n=30$ binary bits): Discrete 5-NN | **MOMA (#30)** | **MOGWO (#23)** | **Continuous Newton / Gradient Descent** | Binary Hamming hypercube $\{0, 1\}^{30}$ & step-wise accuracy landscape | **Task A evidence:** MOMA's bit-flip local search rapidly prunes features while preserving 5-CV classification accuracy. |
| **6** | **P11** | Production Planning ($n=2, M=2$): Linear constraints, firm priorities | **Lexicographic (#7)** | **Goal Programming (#6)** | **VEGA (#10)** | Strict hierarchical decision-maker preferences ($f_1 \succ f_2$) | **Task A evidence:** Lexicographic isolates exact compromise $(300, 200)$; VEGA oscillates unstably without satisfying goals. |

---

## Case Study 1: P3 — Non-Convex Continuous Benchmark (ZDT2)

### 1. Problem Classification and Landscape Characteristics
- **Mathematical Formulation:** 
  $$\min f_1(x) = x_1, \quad \min f_2(x) = g(x) \left[1 - \left(\frac{f_1(x)}{g(x)}\right)^2\right]$$
  where $g(x) = 1 + \frac{9}{n-1} \sum_{i=2}^{10} x_i$, with decision vector $x \in [0, 1]^{10}$.
- **Convexity:** Strictly non-convex (concave Pareto front). When $g(x) = 1$, the true Pareto front is the quadratic arc $f_2 = 1 - f_1^2$ for $f_1 \in [0, 1]$.
- **Variables & Constraints:** $n=10$ continuous decision variables bounded in $[0, 1]$. Box constraints only.
- **Evaluation Budget:** Standard 1,500 to 5,000 function evaluations.

### 2. Suitable Algorithm 1: MODE (Multi-Objective Differential Evolution, #27)
- **Theoretical Rationale:** Differential Evolution uses vector differences between population members ($\mathbf{v}_i = \mathbf{x}_{r1} + F(\mathbf{x}_{r2} - \mathbf{x}_{r3})$) to drive search direction and scale. This self-adapts step sizes to the curvature of the non-convex manifold. Pareto dominance replacement ensures that trial solutions replace target solutions only if they dominate them, preserving diverse solutions across the interior concave arc without suffering from convex hull distortion.
- **Empirical Evidence (Task D, 10 seeds, budget 1,500):**
  - **Hypervolume (HV):** $0.4547 \pm 0.0226$ (Rank 1 across all 10 evaluated algorithms).
  - **Inverted Generational Distance (IGD):** $0.0487 \pm 0.0131$ (Lowest distance to true front).
  - **Schott's Spacing ($S$):** $0.0335 \pm 0.0094$ (Remarkably uniform spacing across the inward curve).

### 3. Suitable Algorithm 2: $\epsilon$-Constraint Method (#2)
- **Theoretical Rationale:** The $\epsilon$-Constraint scalarization optimizes one primary objective while constraining the remaining objectives below parametric thresholds:
  $$\min f_2(x) \quad \text{subject to } f_1(x) \le \epsilon_k$$
  Because the secondary objective enters as a hard bound $\epsilon_k$ rather than a linear combination, the optimization problem does not require convexity of the objective space. By systematically sweeping $\epsilon_k$ from $\min f_1$ to $\max f_1$, the solver systematically slices through the non-convex "dent", capturing Pareto-optimal points across the entire concave front.
- **Empirical Evidence (Task D, 10 seeds, budget 1,500):**
  - **Hypervolume (HV):** $0.4387 \pm 0.0117$ (Rank 2).
  - **Inverted Generational Distance (IGD):** $0.0998 \pm 0.0143$.
  - Generates well-distributed points inside the concave region where weighted sum fails completely.

### 4. Poor Choice: Weighted Sum Method (#1)
- **Theoretical Justification of Failure:**
  The Weighted Sum method minimizes a scalar surrogate $F_{\lambda}(x) = \lambda_1 f_1(x) + \lambda_2 f_2(x)$ with $\lambda_1, \lambda_2 > 0, \lambda_1 + \lambda_2 = 1$. Geometrically, this seeks a supporting hyperplane $\lambda^T \mathbf{f} = c$ tangent to the feasible objective set.
  - **The Supporting Hyperplane Theorem:** A linear function over a compact set achieves its minimum on the boundary of the set's *convex hull*.
  - For a non-convex (concave) front like ZDT2 ($f_2 = 1 - f_1^2$), the interior points of the arc lie strictly in the interior of the convex hull formed by the two extreme points $(0, 1)$ and $(1, 0)$.
  - Any supporting line tangent to an interior point on $f_2 = 1 - f_1^2$ will inevitably intersect points on the front with even smaller scalar values. Consequently, no positive weight vector $\lambda$ can ever make an interior point the global minimum of $F_\lambda$.
- **Empirical Evidence (Task D Head-to-Head & Task A):**
  - In our 10-seed Task D evaluation on P3 with 10 uniform weight settings $\lambda \in [0, 1]$:
    - **Hypervolume (HV):** $0.2100 \pm 0.0000$ (Severe under-coverage).
    - **Inverted Generational Distance (IGD):** $0.3532 \pm 0.0000$.
    - **Number of distinct solutions:** Exactly **2 points** across the entire run: $(0.0, 1.0)$ and $(1.0, 0.0)$.
    - Schott's Spacing is $0.0000$ because every intermediate weight collapsed to one of the two extreme corners, completely skipping the interior 99% of the trade-off manifold.

---

## Case Study 2: P5 — Cantilever Beam Design (Constrained Mechanical Engineering)

### 1. Problem Classification and Landscape Characteristics
- **Mathematical Formulation:**
  $$\min f_1(b, h) = 7850 \cdot L \cdot b \cdot h \quad \text{(mass in kg)}$$
  $$\min f_2(b, h) = \frac{4 P L^3}{E \cdot b \cdot h^3} \quad \text{(tip deflection in m)}$$
  subject to the bending stress constraint:
  $$g_1(b, h) = \frac{6 P L}{b \cdot h^2} - 250 \times 10^6 \le 0 \iff b \cdot h^2 \ge 2.4 \times 10^{-5} \text{ m}^3$$
  where $b \in [0.01, 0.10]$ m and $h \in [0.02, 0.20]$ m, with $L = 1.0$ m, $P = 1000$ N, $E = 200$ GPa.
- **Constraints & Feasibility:** Highly constrained nonlinear design space. In a uniform random sampling of the bounding box $[0.01, 0.10] \times [0.02, 0.20]$, approximately **42.3% of random candidates violate the stress limit** ($CV > 0$).
- **Objectives:** $M=2$ competing structural objectives (lightness vs. stiffness). The true Pareto front lies directly along the active constraint boundary $b \cdot h^2 = 2.4 \times 10^{-5}$ where bending stress is exactly 250 MPa.

### 2. Suitable Algorithm 1: Constrained NSGA-II (CNSGA-II, #21)
- **Theoretical Rationale:** CNSGA-II incorporates Deb's Constrained-Dominance criteria into binary tournament selection:
  1. Any feasible candidate dominates any infeasible candidate.
  2. Between two infeasible candidates, the one with smaller constraint violation $CV$ dominates.
  3. Between two feasible candidates, Pareto dominance and crowding distance decide.
  This creates an automatic dual-phase search: in early generations, infeasible solutions are rapidly driven toward the feasible boundary without requiring artificial penalty multipliers; once feasible, elitist non-dominated sorting drives the population along the active boundary $b h^2 = 2.4 \times 10^{-5}$.
- **Empirical Evidence (Task D, 10 seeds, budget 1,500):**
  - **Hypervolume (HV):** $1.1828 \pm 0.0004$ (Near-optimal coverage against reference point $[1.1, 1.1]$).
  - **Schott's Spacing ($S$):** $0.0384 \pm 0.0034$ (Lowest spacing across all constrained algorithms, proving extremely uniform distribution along the stress boundary).
  - **Feasibility:** $100\%$ feasible solutions returned across all 10 independent seeds ($CV_{opt} = 0.0$).

### 3. Suitable Algorithm 2: NSGA-II with Deb's Rules (#14)
- **Theoretical Rationale:** Standard NSGA-II implemented with Deb's constraint-handling mechanism maintains a combined parent-offspring archive ($2N \to N$) that protects elite feasible designs from being overwritten by high-performing but physically illegal structures.
- **Empirical Evidence (Task D, 10 seeds, budget 1,500):**
  - **Hypervolume (HV):** $1.1831 \pm 0.0004$.
  - **Schott's Spacing ($S$):** $0.0425 \pm 0.0084$.
  - Feasibility: 100% feasible solutions ($CV = 0.0$).

### 4. Poor Choice: VEGA (Vector Evaluated Genetic Algorithm, #10)
- **Theoretical Justification of Failure:**
  1. **Absence of Native Feasibility Mechanism:** Standard VEGA lacks an internal constraint tournament or penalty logic, meaning infeasible individuals with artificially low mass or deflection are selected into the parent pool.
  2. **Speciation Instability:** VEGA divides the population into two sub-populations of size $N/2$, each selected solely by one objective. Sub-population 1 optimizes mass alone, pushing $b \to 0.01$ and $h \to 0.02$, which causes severe stress violation ($b h^2 = 4 \times 10^{-6} \ll 2.4 \times 10^{-5}$, violating stress by $> 1000$ MPa!). Sub-population 2 optimizes deflection alone, pushing $b \to 0.10$ and $h \to 0.20$ (excessively heavy). When crossed over, the offspring oscillate unstably between catastrophic structural failure and extreme over-design, completely missing the balanced trade-offs on the active constraint boundary.
- **Empirical Evidence (Task A Runs):**
  - VEGA produced final fronts heavily contaminated with infeasible solutions ($CV > 0.05$). After applying strict feasibility filtering ($CV \le 10^{-4}$), the remaining feasible front suffered from severe clustering at the extreme corners, leaving the central compromise region completely vacant ($HV < 0.85$).

---

## Case Study 3: P10 — Many-Objective Benchmark (DTLZ2, $M=5, n=14$)

### 1. Problem Classification and Landscape Characteristics
- **Mathematical Formulation:** DTLZ2 scaled to $M=5$ objectives with $n = M + 9 = 14$ continuous decision variables $x_i \in [0, 1]$:
  $$g(x) = \sum_{i=5}^{14} (x_i - 0.5)^2$$
  $$f_1(x) = (1 + g(x)) \cos(x_1 \pi / 2) \cos(x_2 \pi / 2) \cos(x_3 \pi / 2) \cos(x_4 \pi / 2)$$
  $$f_m(x) = (1 + g(x)) \left( \prod_{j=1}^{5-m} \cos(x_j \pi / 2) \right) \sin(x_{5-m+1} \pi / 2), \quad m=2 \dots 5$$
- **Objective Space:** $M=5$ conflicting objectives. When converged ($g(x)=0$), the Pareto-optimal front is the positive orthant of a unit 5-sphere: $\sum_{m=1}^5 f_m^2 = 1$.
- **The Many-Objective Curse of Dimensionality:** In 5-dimensional objective space, the volume of the search space expands exponentially. In a random population of 100 individuals, over **95% to 98% of candidates are mutually non-dominated**.

### 2. Suitable Algorithm 1: NSGA-III (#15)
- **Theoretical Rationale:** NSGA-III specifically addresses the breakdown of Pareto dominance in many-objective spaces ($M \ge 4$). Instead of relying on crowding distance, NSGA-III defines structured reference points on the unit hyperplane using the Das-Dennis systematic approach. Solutions in the critical boundary front are normalized and associated with reference lines via perpendicular distance. A niching count $\rho_j$ ensures that solutions associated with under-represented reference directions are prioritized, enforcing uniform coverage over the 5D spherical manifold.
- **Empirical Evidence (Task A & Task D):**
  - Successfully maintained uniform distribution over the 5D shell.
  - Achieved low IGD to the analytical spherical front ($\sum f_m^2 = 1$), avoiding the boundary collapse observed in crowding-distance-based methods.

### 3. Suitable Algorithm 2: RVEA (Reference Vector Guided Evolutionary Algorithm, #31)
- **Theoretical Rationale:** RVEA utilizes unit reference vectors $V$ coupled with an Angle-Penalized Distance (APD) scalar metric:
  $$APD(x_i) = \left( 1 + M \left(\frac{t}{t_{max}}\right)^\alpha \frac{\theta_{i, k}}{\gamma_k} \right) ||\mathbf{f}'(x_i)||_2$$
  During early generations ($t \to 0$), the penalty is minimal, allowing rapid convergence toward the sphere ($||\mathbf{f}'||_2$). In late generations ($t \to t_{max}$), the angular penalty $\theta_{i, k} / \gamma_k$ dominates, forcing solutions to distribute evenly along each reference direction.
- **Empirical Evidence:** RVEA dynamically adapts to the curvature of the 5-objective sphere without requiring costly hypervolume calculations, achieving high convergence efficiency within 2,000 evaluations.

### 4. Poor Choice: Standard NSGA-II (#14)
- **Theoretical Justification of Failure:**
  1. **Loss of Pareto Selection Pressure:** Because almost every candidate is non-dominated at $M=5$, the entire population ends up in Front 1 ($F_1$). Consequently, rank-based selection exerts zero selective pressure toward the true Pareto front ($g(x) \to 0$).
  2. **Failure of Crowding Distance in High Dimensions:** Crowding distance measures the perimeter of the cuboid along each 1D coordinate independently. In 5D space, points are naturally sparse; extreme solutions along each of the 5 axes are assigned infinite distance ($\infty$), consuming $2M = 10$ slots automatically. Furthermore, crowding distance favors points at the extremes of individual axes rather than points uniformly covering the interior 5D spherical shell. The population degenerates into a sparse, poorly converged cluster of solutions drifting passively without direction.
- **Empirical Evidence (Task D Head-to-Head Comparison on P10 $M=5$):**
  - NSGA-II exhibits significantly higher IGD ($IGD > 0.45$) compared to NSGA-III and RVEA ($IGD < 0.20$), with the majority of points failing to converge onto the spherical surface ($g(x) \gg 0$).

---

## Case Study 4: P8 — Bi-Objective Delivery Routing (Discrete Permutation TSP)

### 1. Problem Classification and Landscape Characteristics
- **Mathematical Formulation:** 15 customer locations visited by a courier starting and ending at a depot.
  - Solution representation: Permutation $\pi = (\pi_1, \pi_2, \dots, \pi_{15})$ of the integers $\{0, 1, \dots, 14\}$.
  - Search space cardinality: $15! \approx 1.307 \times 10^{12}$ valid closed tours.
  - Objectives: $\min f_1(\pi) = \sum_{i=1}^{15} d_{\pi_i, \pi_{i+1}}$ (total Euclidean distance), $\min f_2(\pi) = \sum_{i=1}^{15} t_{\pi_i, \pi_{i+1}}$ (total travel time, where $t_{ij} = d_{ij} \cdot u_{ij}$ with asymmetric traffic factors $u_{ij} \sim U(0.5, 2.0)$).
- **Constraints:** Permutation constraint (each city visited exactly once). Discrete, combinatorial, non-Euclidean search space.

### 2. Suitable Algorithm 1: MOACO (Multi-Objective Ant Colony Optimization, #25)
- **Theoretical Rationale:** MOACO is natively designed for discrete graph routing problems. It deploys two distinct pheromone matrices: $\tau^1_{ij}$ for distance minimization and $\tau^2_{ij}$ for travel time minimization. Colony ants construct valid tours step-by-step using a probabilistic transition rule:
  $$P_{ij}^k = \frac{[\tau_{ij}^1]^{\lambda_k \alpha} [\tau_{ij}^2]^{(1-\lambda_k) \alpha} [\eta_{ij}^1]^{\lambda_k \beta} [\eta_{ij}^2]^{(1-\lambda_k) \beta}}{\sum_{l \in \text{allowed}} [\dots]}$$
  where heuristic desirabilities $\eta_{ij}^1 = 1/d_{ij}$ and $\eta_{ij}^2 = 1/t_{ij}$ guide search. The pheromone update deposits pheromone exclusively on non-dominated Pareto tours, reinforcing promising sub-paths without ever violating tour validity.
- **Empirical Evidence (Task A & Task C Sensitivity Runs):**
  - Across 90 sensitivity runs on P8, MOACO achieved an average hypervolume of $1.15 \times 10^7$ with 0% invalid tours, effectively discovering distinct trade-offs between rapid highway routes and short urban paths.

### 3. Suitable Algorithm 2: MOEA/D-ACO (#20)
- **Theoretical Rationale:** MOEA/D-ACO decomposes the bi-objective routing problem into $N$ scalar subproblems with uniform weight vectors $\lambda^1, \dots, \lambda^N$. Each subproblem maintains an ant colony specialized to its scalarized objective, updating pheromone vectors cooperatively across neighboring subproblems. This ensures balanced search across both objectives without requiring artificial tour repairs.
- **Empirical Evidence:** Achieved balanced spread and fast convergence across both distance and time metrics.

### 4. Poor Choice: GradNorm (#37) / MGDA (#36)
- **Theoretical Justification of Failure:**
  - **Differentiability Requirement:** GradNorm and MGDA are gradient-based multi-task optimization algorithms designed for smooth, continuous parameter spaces $\mathbb{R}^n$. They require computing partial derivatives:
    $$\nabla_\theta \mathcal{L}_1(\theta) \quad \text{and} \quad \nabla_\theta \mathcal{L}_2(\theta)$$
  - **Incompatibility with Discrete Spaces:** In a permutation TSP, the search space is a discrete combinatorial graph $\mathcal{S}_{15}$. There is no algebraic definition of a directional derivative or gradient vector $\nabla_\pi f(\pi)$ with respect to city permutations.
  - Applying continuous gradient descent or gradient normalization to a permutation vector produces fractional continuous values (e.g. $[2.34, 7.81, -1.05]$) that are completely invalid tours and cannot be evaluated without arbitrary, destructive discretization.
- **Empirical Evidence:** GradNorm cannot even be initialized on P8 without an immediate type error or crash; its analytical formulation is fundamentally restricted to differentiable neural networks or continuous analytical functions (P13).

---

## Case Study 5: P9 — Breast Cancer Diagnostic Feature Selection (Binary Combinatorial)

### 1. Problem Classification and Landscape Characteristics
- **Mathematical Formulation:** 
  $$\min f_1(x) = 1 - \text{Accuracy}_{5\text{-CV}}(x) \quad \text{(classification error of 5-NN classifier)}$$
  $$\min f_2(x) = \sum_{i=1}^{30} x_i \quad \text{(number of selected features)}$$
  where $x \in \{0, 1\}^{30}$ with the constraint $\sum_{i=1}^{30} x_i \ge 1$.
- **Variable Representation:** Binary Hamming hypercube $\{0, 1\}^{30}$ ($2^{30} \approx 1.07 \times 10^9$ possible feature subsets).
- **Objective Surface:** Step-wise constant, highly non-linear, non-differentiable, and noisy due to 5-fold cross-validation folds.

### 2. Suitable Algorithm 1: MOMA (Multi-Objective Memetic Algorithm, #30)
- **Theoretical Rationale:** MOMA synergizes global population-based genetic search with local hill-climbing exploitation. In binary feature selection:
  1. Global uniform crossover and bit-flip mutation explore diverse combinations of biological markers.
  2. The local search operator systematically tests single-bit removal ($1 \to 0$) on non-dominated candidates. If removing a feature does not increase the 5-fold cross-validation error, the smaller subset dominates the parent and replaces it immediately.
  This allows MOMA to rapidly eliminate redundant diagnostic measurements (e.g., perimeter vs. area vs. radius) while strictly preserving classification accuracy.
- **Empirical Evidence (Task A Runs on P9):**
  - MOMA discovered sparse feature subsets containing only **3 to 5 features** (reducing data collection by 85–90%) while retaining $> 95.8\%$ cross-validated accuracy, matching the all-30-features baseline.

### 3. Suitable Algorithm 2: MOGWO (Multi-Objective Grey Wolf Optimizer, #23)
- **Theoretical Rationale:** MOGWO uses an external non-dominated archive to store elite feature masks ($\alpha, \beta, \delta$ wolves). Using a sigmoid transfer function $S(v) = \frac{1}{1 + e^{-v}}$ or binary thresholding ($x_i = 1$ if $v_i > 0.5$), the grey wolf encircling equations effectively guide the binary swarm toward compact, high-accuracy diagnostic masks.
- **Empirical Evidence:** Achieved strong convergence to low-error fronts within 2,000 evaluations.

### 4. Poor Choice: Classical Gradient / Newton-Raphson Optimizers
- **Theoretical Justification of Failure:**
  - The objective function involves a 5-Nearest Neighbor classifier whose prediction is a discrete majority vote over non-parametric distance neighborhoods.
  - The gradient $\nabla_x f_1(x)$ is zero almost everywhere on the continuous relaxation, and undefined at decision boundaries.
  - Classical gradient methods evaluate identical predictions within any small perturbation $\Delta x < 0.5$, yielding a zero gradient $\nabla f = \mathbf{0}$ and immediately terminating at the starting point without exploring alternative feature subsets.

---

## Case Study 6: P11 — Workshop Production Planning (Goals and Hierarchical Priorities)

### 1. Problem Classification and Landscape Characteristics
- **Mathematical Formulation:** 
  $$\max f_1(x_A, x_B) = 4 x_A + 6 x_B \quad \text{(profit in \$)}$$
  $$\max f_2(x_A, x_B) = x_A + x_B \quad \text{(total production volume)}$$
  subject to linear resource constraints:
  $$2 x_A + 3 x_B \le 1200 \quad \text{(labor hours)}$$
  $$4 x_A + 2 x_B \le 1600 \quad \text{(raw material in kg)}$$
  with $x_A, x_B \ge 0$.
- **Managerial Decision Context:** The plant manager specifies firm a-priori priorities:
  - **Priority 1:** Achieve maximum profit target ($T_1 \ge 2400$).
  - **Priority 2:** Maximize total production units ($T_2 \ge 600$).
- **Geometry of Feasible Polygon:** Feasible extreme points are $(0, 0), (400, 0), (300, 200)$, and $(0, 400)$.
  - Maximum profit is exactly $\$2400$, which occurs along the entire line segment connecting $(0, 400)$ and $(300, 200)$.
  - Total volume along this optimal-profit segment ranges from $400$ units at $(0, 400)$ to $500$ units at $(300, 200)$.
  - The target of $600$ units is physically unattainable (shortfall of at least $100$ units).

### 2. Suitable Algorithm 1: Lexicographic Optimization (#7)
- **Theoretical Rationale:** Lexicographic optimization strictly reflects hierarchical decision-maker preferences ($f_1 \succ f_2$).
  1. **Phase 1:** Solve $\max f_1$ subject to linear constraints. The solver identifies the maximum profit $f_1^* = 2400$.
  2. **Phase 2:** Solve $\max f_2$ subject to the original constraints PLUS the tight priority constraint $f_1(x_A, x_B) \ge 2400 - \delta$.
  Because there is an entire edge of solutions achieving $\$2400$ profit, Phase 2 breaks the tie by maximizing production volume, landing directly and deterministically on the unique optimal vertex $(x_A^*, x_B^*) = (300, 200)$, yielding exactly $f_1 = 2400$ and $f_2 = 500$.
- **Empirical Evidence (Task A Runs on P11):**
  - Solved to machine precision in 2 linear programming iterations, returning exactly $(300.0, 200.0)$ without stochastic variance.

### 3. Suitable Algorithm 2: Goal Programming (#6)
- **Theoretical Rationale:** Goal programming introduces deviation variables:
  $$4 x_A + 6 x_B + d_1^- - d_1^+ = 2400$$
  $$x_A + x_B + d_2^- - d_2^+ = 600$$
  By assigning priority weights ($P_1 d_1^- + P_2 d_2^-$ with $P_1 \gg P_2$), Goal Programming penalizes profit under-achievement first, completely driving $d_1^- \to 0$, and then minimizes volume under-achievement $d_2^-$, achieving the minimum possible shortfall ($d_2^- = 100$) at $(300, 200)$.
- **Empirical Evidence:** Matches the exact analytical compromise solution with zero constraint violation.

### 4. Poor Choice: VEGA (#10)
- **Theoretical Justification of Failure:**
  - VEGA has no mechanism to incorporate hierarchical priorities or goal aspiration levels.
  - Half the population is selected for profit, half for volume.
  - The sub-population selected for volume drifts toward $(0, 400)$ or unconstrained regions, while the sub-population for profit drifts along the segment. Recombination produces high variance and stochastic oscillation, failing to exploit the fact that $(300, 200)$ uniquely satisfies both priorities simultaneously.
- **Empirical Evidence:** VEGA required thousands of evaluations and generated a scattered cloud of sub-optimal points with high variance, failing to pinpoint the deterministic corner solution $(300, 200)$.

---

## Synthesis of Algorithm Selection Principles

From the six case studies above, four overarching selection principles emerge:

1. **Convexity Dictates Scalarization Viability:** If a problem's Pareto front is non-convex (e.g. P3), linear scalarization (Weighted Sum #1) is mathematically incapable of generating interior trade-offs. The $\epsilon$-Constraint method (#2), evolutionary decomposition (MOEA/D #18), or Pareto-dominance metaheuristics (MODE #27, NSGA-II #14) must be used.
2. **Dimension ($M$) Dictates Diversity Mechanism:** For bi-objective problems ($M=2$), crowding distance (NSGA-II) and density estimation (SPEA2) perform excellently. For many-objective problems ($M \ge 4$, e.g. P10), Pareto dominance collapses, requiring reference-direction guidance (NSGA-III #15, RVEA #31) or hypervolume approximations (HypE #34).
3. **Representation Governs Search Operators:** Discrete permutations (P8 TSP) demand constructive or order-preserving operators (MOACO #25, Order Crossover). Continuous gradient methods (GradNorm, MGDA) strictly require differentiable landscapes and cannot operate on combinatorial structures.
4. **Constraints Require Feasibility-First Selection:** When active constraints eliminate large portions of the search space (P5, P6), unconstrained ranking methods fail. Embedding Deb's feasibility-first rules (CNSGA-II #21) guarantees convergence along active physical boundaries without destructive penalty tuning.
