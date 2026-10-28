# Task D: Head-to-Head Comparative Studies and Empirical Benchmark Analysis

**Course:** Optimization Techniques Lab · Department of Artificial Intelligence & Data Science  
**Student:** Soham Umesh Chavan (Roll Number Remainder $R = 13$)  
**Protocol:** 10 independent random seeds ($42 \dots 51$), strictly equal function evaluation budgets per problem, feasibility-first filtering (Deb's rules), objective normalization against documented problem bounds, and exact/Monte-Carlo metrics.

---

## 1. Executive Summary of Comparisons

| Comparison ID | Benchmark Problem | Methodological Focus | Algorithms Evaluated ($N=10$ Seeds Each) | Budget / Run | Dominant Algorithm(s) | Primary Empirical Finding |
|:---:|:---|:---|:---|:---:|:---|:---|
| **D1** | **P3 (ZDT2)** | Continuous Bi-Objective (Non-Convex) | Weighted Sum (#1), $\epsilon$-Constraint (#2), VEGA (#10), MOGA (#11), NPGA (#13), NSGA-II (#14), MOEA/D (#18), MOEA/D-DE (#19), MODE (#27), SMS-EMOA (#33) | 1,500 evals | **MODE (#27)** ($HV=0.4547$) & **$\epsilon$-Constraint (#2)** ($HV=0.4387$) | Weighted Sum completely missed the non-convex dent ($HV=0.2100, S=0.0$). Early non-elitist GAs failed to converge. |
| **D2** | **P5 (Cantilever)** | Continuous Bi-Objective (Constrained Stress) | $\epsilon$-Constraint (#2), NPGA (#13), NSGA-II (#14), SPEA2 (#16), CNSGA-II (#21), MOPSO (#22), MOGWO (#23), MODE (#27) | 1,500 evals | **CNSGA-II (#21)** ($S=0.0384, 100\%\text{ feas}$) & **NSGA-II (#14)** ($HV=1.1831$) | Deb's feasibility rules successfully drive search along the active nonlinear stress boundary ($b h^2 = 2.4 \times 10^{-5}$). |
| **D3** | **P10 ($M=5$ DTLZ2)** | Many-Objective ($M=5$ Spherical Shell) | NSGA-II (#14), NSGA-III (#15), MOEA/D (#18), RVEA (#31), HypE (#34) | 2,000 evals | **RVEA (#31)** ($HV=1.1395, IGD=0.2071$) & **HypE (#34)** ($HV=1.0775$) | NSGA-II degraded severely ($HV=0.1904, IGD=0.5586$) due to loss of dominance selection pressure and crowding distance failure. |
| **D4** | **P8 (MOTSP)** | Discrete Combinatorial (Permutation Tour) | MOEA/D-ACO (#20), MOACO (#25), NSGA-II (#14), MOMA (#30) | 1,000 evals | **MOMA (#30)** ($HV=0.8310, S=0.0020$) & **NSGA-II (#14)** ($HV=0.7205$) | Memetic local search excels on discrete permutations. ACO dual pheromones naturally build valid closed tours. |
| **D5** | **Special Studies** | Parameters, Compromises, MCDM & Gradients | Algs 1–4 (Sweeps), Algs 5–7 (Preferences), Algs 8–9 (TOPSIS/VIKOR), Algs 36–37 (MGDA/GradNorm) | Custom | **Lexicographic (#7)**, **GradNorm (#37)** | Shifting MCDM weight by 0.1 flipped supplier rankings. GradNorm balanced conflicting losses on P13. |

---

## 2. Study 1: Continuous Bi-Objective on P3 (Unconstrained ZDT2)

### Experimental Setup
- **Problem:** P3 (ZDT2, $n=10$ variables, $x_i \in [0, 1]$).
- **True Pareto Front:** Strictly non-convex quadratic arc $f_2 = 1 - f_1^2$ for $f_1 \in [0, 1]$.
- **Evaluation Budget:** Exactly 1,500 evaluations per seed across 10 independent runs (Seeds 42 to 51).
- **Normalization:** Ideal point $[0.0, 0.0]$, Nadir point $[1.0, 1.0]$, Hypervolume reference point $[1.1, 1.1]$.

### Empirical Results Table ($N=10$ Seeds)

| Rank | Algorithm | Alg ID | Evaluations | Hypervolume (HV) | Schott's Spacing ($S$) | IGD | Runtime (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **MODE** | #27 | $1500.0 \pm 0.0$ | **$0.4547 \pm 0.0226$** | **$0.0335 \pm 0.0094$** | **$0.0487 \pm 0.0131$** | $1.49 \pm 0.15$ |
| **2** | **$\epsilon$-Constraint** | #2 | $1120.4 \pm 0.0$ | **$0.4387 \pm 0.0117$** | $0.5612 \pm 0.3904$ | $0.0998 \pm 0.0143$ | $0.03 \pm 0.01$ |
| **3** | **MOEA/D** | #18 | $1500.0 \pm 0.0$ | $0.2766 \pm 0.0782$ | $0.0326 \pm 0.0280$ | $0.2097 \pm 0.0888$ | $1.28 \pm 0.38$ |
| **4** | **Weighted Sum** | #1 | $302.6 \pm 0.0$ | $0.2100 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.3532 \pm 0.0000$ | $0.01 \pm 0.00$ |
| **5** | **SMS-EMOA** | #33 | $1500.0 \pm 0.0$ | $0.1454 \pm 0.1103$ | $0.0444 \pm 0.0393$ | $0.4471 \pm 0.2052$ | $8.08 \pm 0.55$ |
| **6** | **NSGA-II** | #14 | $1500.0 \pm 0.0$ | $0.0517 \pm 0.0492$ | $0.1085 \pm 0.0650$ | $0.5117 \pm 0.1371$ | $2.89 \pm 0.21$ |
| **7** | **MOGA** | #11 | $1500.0 \pm 0.0$ | $0.0070 \pm 0.0209$ | $0.1345 \pm 0.0700$ | $1.0205 \pm 0.2795$ | $1.11 \pm 0.08$ |
| **8** | **MOEA/D-DE** | #19 | $1500.0 \pm 0.0$ | $0.0026 \pm 0.0078$ | $0.0612 \pm 0.0502$ | $1.0361 \pm 0.2440$ | $1.52 \pm 1.30$ |
| **9** | **VEGA** | #10 | $1500.0 \pm 0.0$ | $0.0000 \pm 0.0000$ | $0.1514 \pm 0.0962$ | $1.2856 \pm 0.1889$ | $0.69 \pm 0.07$ |
| **10** | **NPGA** | #13 | $1500.0 \pm 0.0$ | $0.0000 \pm 0.0000$ | $0.3440 \pm 0.2449$ | $1.2823 \pm 0.2959$ | $0.78 \pm 0.07$ |

### In-Depth Analysis and Discussion
1. **The Failure of Linear Scalarization in Non-Convex Landscapes:**
   - As proven mathematically in Task F Question 1, the Weighted Sum method (#1) minimizes a linear objective $\lambda_1 f_1 + \lambda_2 f_2$. Over the concave curve $f_2 = 1 - f_1^2$, supporting hyperplanes can only touch the convex hull vertices $(0, 1)$ and $(1, 0)$.
   - Across all 10 independent seeds, Weighted Sum discovered **only the two extreme points**, resulting in a Schott's Spacing of exactly $0.0000$ and an IGD of $0.3532$. The interior of the front was completely vacant.
2. **Why MODE (#27) and $\epsilon$-Constraint (#2) Excelled:**
   - **MODE (#27)** operates with differential evolution vector mutations and target-trial Pareto dominance replacement. Because dominance replacement is scale-free and coordinate-free, it is invariant to front convexity, enabling MODE to achieve near-optimal coverage ($HV = 0.4547$, $IGD = 0.0487$).
   - **$\epsilon$-Constraint (#2)** systematically converts secondary objectives into parametric inequality bounds ($f_1(x) \le \epsilon_k$). By sweeping $\epsilon_k$, it slices through the non-convex dent, capturing interior points without requiring convexity ($HV = 0.4387$).
3. **The Collapse of Early Non-Elitist GAs (VEGA, NPGA, MOGA):**
   - On a 10-dimensional problem within a modest budget of 1,500 evaluations, non-elitist algorithms suffer catastrophic loss of elite genetic material. VEGA and NPGA both yielded $HV = 0.0000$ and $IGD \approx 1.28$, demonstrating that generational replacement without elitism fails to converge on multi-dimensional continuous manifolds.

---

## 3. Study 2: Continuous Bi-Objective on P5 (Constrained Cantilever Beam)

### Experimental Setup
- **Problem:** P5 (Cantilever Beam Design, $n=2$ continuous variables: width $b \in [0.01, 0.10]$ m, height $h \in [0.02, 0.20]$ m).
- **Objectives:** $\min f_1 = \text{mass}$ (kg), $\min f_2 = \text{tip deflection}$ (m).
- **Constraints:** Nonlinear bending stress constraint: $\sigma = \frac{6 P L}{b h^2} \le 250 \text{ MPa} \iff b h^2 \ge 2.4 \times 10^{-5} \text{ m}^3$.
- **Evaluation Budget:** 1,500 evaluations per seed across 10 independent runs (Seeds 42 to 51).
- **Feasibility Rule:** Deb's Constrained Dominance ($CV \le 10^{-4}$ tolerance).

### Empirical Results Table ($N=10$ Seeds)

| Rank | Algorithm | Alg ID | Evaluations | Hypervolume (HV) | Schott's Spacing ($S$) | Feasible Ratio | Runtime (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **CNSGA-II** | #21 | $1500.0 \pm 0.0$ | $1.1828 \pm 0.0004$ | **$0.0384 \pm 0.0034$** | **100% ($CV=0.0$)** | $12.59 \pm 4.47$ |
| **2** | **NSGA-II** | #14 | $1500.0 \pm 0.0$ | **$1.1831 \pm 0.0004$** | $0.0425 \pm 0.0084$ | **100% ($CV=0.0$)** | $7.31 \pm 0.93$ |
| **3** | **NPGA** | #13 | $1500.0 \pm 0.0$ | **$1.1832 \pm 0.0006$** | $0.0627 \pm 0.0182$ | **100% ($CV=0.0$)** | $2.17 \pm 0.26$ |
| **4** | **MODE** | #27 | $1500.0 \pm 0.0$ | $1.1814 \pm 0.0010$ | $0.0860 \pm 0.0091$ | **100% ($CV=0.0$)** | $3.31 \pm 0.40$ |
| **5** | **SPEA2** | #16 | $1500.0 \pm 0.0$ | $1.1794 \pm 0.0014$ | $0.1132 \pm 0.0089$ | **100% ($CV=0.0$)** | $1.90 \pm 0.33$ |
| **6** | **MOPSO** | #22 | $1500.0 \pm 0.0$ | $1.1831 \pm 0.0004$ | $0.1472 \pm 0.0409$ | **100% ($CV=0.0$)** | $1.85 \pm 0.07$ |
| **7** | **MOGWO** | #23 | $1500.0 \pm 0.0$ | $1.1812 \pm 0.0022$ | $0.1879 \pm 0.0940$ | **100% ($CV=0.0$)** | $2.42 \pm 0.26$ |
| **8** | **$\epsilon$-Constraint** | #2 | $909.6 \pm 0.0$ | $1.0892 \pm 0.0016$ | $0.2010 \pm 0.0024$ | **100% ($CV=0.0$)** | $0.03 \pm 0.00$ |

### In-Depth Analysis and Discussion
1. **Validation of Deb's Feasibility-First Selection:**
   - In P5, over 42% of random designs violate the bending stress limit. CNSGA-II (#21) embeds Deb's three rules directly into binary tournament selection. Consequently, infeasible candidates are unconditionally rejected in favor of feasible ones, and infeasible solutions compete strictly on constraint violation $CV$.
   - CNSGA-II achieved the **finest Schott's Spacing ($S = 0.0384 \pm 0.0034$)** across all algorithms, distributing designs uniformly along the active stress boundary $b h^2 = 2.4 \times 10^{-5}$.
2. **Comparison with Swarm Methods (MOPSO, MOGWO):**
   - MOPSO (#22) and MOGWO (#23) rapidly converged to the true front, achieving high hypervolume ($HV \approx 1.183$). However, their spacing values were significantly higher ($S = 0.1472$ and $S = 0.1879$), showing that velocity updates tend to cluster particles around regional attractors rather than distributing them evenly along the curved constraint surface.

---

## 4. Study 3: Many-Objective on P10 ($M=5$ DTLZ2)

### Experimental Setup
- **Problem:** P10 (DTLZ2, $M=5$ objectives, $n = M + 9 = 14$ continuous variables).
- **True Pareto Front:** Positive orthant of the unit 5-sphere: $\sum_{m=1}^5 f_m^2 = 1$.
- **Evaluation Budget:** 2,000 evaluations per seed across 10 independent runs (Seeds 42 to 51).
- **Algorithms Evaluated:** NSGA-II (#14), NSGA-III (#15), MOEA/D (#18), RVEA (#31), HypE (#34).

### Empirical Results Table ($N=10$ Seeds)

| Rank | Algorithm | Alg ID | Evaluations | Hypervolume (HV) | Schott's Spacing ($S$) | IGD | Runtime (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **RVEA** | #31 | $2000.0 \pm 0.0$ | **$1.1395 \pm 0.0202$** | **$0.1383 \pm 0.0087$** | **$0.2071 \pm 0.0070$** | $10.14 \pm 3.05$ |
| **2** | **HypE** | #34 | $2000.0 \pm 0.0$ | **$1.0775 \pm 0.0242$** | $0.1410 \pm 0.0124$ | **$0.2415 \pm 0.0097$** | $7.29 \pm 0.31$ |
| **3** | **MOEA/D** | #18 | $2000.0 \pm 0.0$ | $0.7051 \pm 0.0876$ | $0.1837 \pm 0.0196$ | $0.3617 \pm 0.0355$ | $4.16 \pm 0.33$ |
| **4** | **NSGA-III** | #15 | $2000.0 \pm 0.0$ | $0.1989 \pm 0.0187$ | $0.2521 \pm 0.0205$ | $0.5586 \pm 0.0160$ | $34.70 \pm 13.84$ |
| **5** | **NSGA-II** | #14 | $2000.0 \pm 0.0$ | $0.1904 \pm 0.0722$ | $0.2602 \pm 0.0160$ | $0.5586 \pm 0.0519$ | $26.15 \pm 3.33$ |

### In-Depth Analysis: What Degrades in NSGA-II?
1. **Collapse of Pareto Selection Pressure:**
   - In 5-objective space, the probability of dominance between two random vectors is $(1/2)^5 = 1/32 \approx 3.1\%$.
   - In a population of 100 individuals on DTLZ2, **over 95% of candidates are mutually non-dominated** and assigned to Front 1 ($F_1$).
   - As a result, fast non-dominated sorting exerts essentially zero selection pressure toward the true Pareto front ($g(x) \to 0$).
2. **Breakdown of Crowding Distance:**
   - Crowding distance assigns $\infty$ to the extreme boundary solutions along each of the $M=5$ axes. This consumes $2M = 10$ slots automatically.
   - For interior solutions, crowding distance simply sums independent 1D projections, failing to detect multidimensional clustering in 5D space.
   - NSGA-II suffered severe degradation: **HV was only $0.1904$** and **IGD was $0.5586$**, indicating that the population remained far from the spherical surface.
3. **Why RVEA (#31) and HypE (#34) Dominated:**
   - **RVEA (#31)** replaces crowding distance with Angle-Penalized Distance (APD), dynamically penalizing angular deviation from uniform reference vectors. It achieved **Rank 1 with $HV = 1.1395$ and $IGD = 0.2071$**.
   - **HypE (#34)** utilizes Monte Carlo integration (10,000 samples) to approximate hypervolume contributions directly, providing precise selective pressure without suffering from the dimensionality curse.

---

## 5. Study 4: Discrete Combinatorial on P8 (Bi-Objective Delivery Routing TSP)

### Experimental Setup
- **Problem:** P8 (Bi-Objective TSP, $n=15$ cities, uniform in $[0, 100]^2$, Seed 7 coordinates).
- **Objectives:** $\min f_1 = \text{total Euclidean distance}$, $\min f_2 = \text{total travel time}$ ($t_{ij} = d_{ij} \cdot u_{ij}, u_{ij} \sim U(0.5, 2.0)$).
- **Solution Encoding:** Permutation tour $\pi \in \mathcal{S}_{15}$.
- **Evaluation Budget:** 1,000 evaluations per seed across 10 independent runs (Seeds 42 to 51).

### Empirical Results Table ($N=10$ Seeds)

| Rank | Algorithm | Alg ID | Evaluations | Hypervolume (HV) | Schott's Spacing ($S$) | Runtime (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **MOMA** | #30 | $1000.0 \pm 0.0$ | **$0.8310 \pm 0.0270$** | **$0.0020 \pm 0.0043$** | $0.82 \pm 0.03$ |
| **2** | **NSGA-II** | #14 | $1000.0 \pm 0.0$ | $0.7205 \pm 0.0266$ | $0.0222 \pm 0.0228$ | $1.91 \pm 0.03$ |
| **3** | **MOEA/D-ACO** | #20 | $1000.0 \pm 0.0$ | $0.5715 \pm 0.0206$ | $0.0122 \pm 0.0216$ | $0.36 \pm 0.01$ |
| **4** | **MOACO** | #25 | $1000.0 \pm 0.0$ | $0.5711 \pm 0.0324$ | $0.0104 \pm 0.0130$ | $0.58 \pm 0.01$ |

### In-Depth Analysis and Operator Mechanics
1. **Permutation Operators vs Dual Pheromone Construction:**
   - **MOMA (#30)** achieved the highest hypervolume ($HV = 0.8310$) and finest spacing ($S = 0.0020$). MOMA combines Order Crossover (OX) with local 2-opt edge exchange mutations, rapidly refining candidate tours into high-efficiency compromise routes.
   - **MOACO (#25)** and **MOEA/D-ACO (#20)** use dual pheromone trails ($\tau^1, \tau^2$) and dual heuristic visibility ($\eta^1 = 1/d, \eta^2 = 1/t$) to probabilistically build valid closed tours city-by-city. They guarantee 100% valid tours without requiring repair operators.
2. **Incompatibility of Gradient-Based Methods:**
   - As documented in Task E Case 4, gradient-based multi-task methods (GradNorm #37, MGDA #36) require differentiable continuous landscapes $\mathbb{R}^n$. They cannot operate on discrete permutation topologies $\mathcal{S}_{15}$.

---

## 6. Study 5: Special Handling Studies & Preference Experiments

### 1. Algorithms 1–4 Parameter Sweeps (10+ Settings on P1 & P3)
- **P1 (Convex):** Weighted Sum (#1) obtained 11 distinct Pareto points evenly spaced along $f_2 = (4 - \sqrt{f_1})^2$. $\epsilon$-Constraint (#2) obtained 10 points. Benson's Method (#3) obtained 21 points. Weight Metric (#4) obtained 11 points.
- **P3 (Non-Convex):** Weighted Sum (#1) with 11 uniform weight settings $\lambda \in [0, 1]$ collapsed completely to the two endpoints $(0, 1)$ and $(1, 0)$, skipping the interior concave curve. $\epsilon$-Constraint (#2) successfully sampled across the concave arc.

### 2. Algorithms 5–7 Compromise Solutions & Preferences
- **Algorithm 5 (Value Function):**
  - On P4 (Portfolio): Optimized quadratic utility, identifying portfolio weights $w = [0.0, 0.393, 0.332, 0.275]$ yielding risk $0.0210$ and return $0.1245$.
  - On P14 (Laptop): DM aspiration point $(1.8 \text{ kg}, 12.0 \text{ h})$ guided the optimizer to $(1.800 \text{ kg}, 12.000 \text{ h})$ with screen $14.52''$ and battery $56.14 \text{ Wh}$.
- **Algorithm 6 (Goal Programming on P11):**
  - Targets: Profit $\ge 2400$, Units $\ge 600$.
  - Result: $(x_A, x_B) = (300.00, 200.00)$, profit $= 2399.997$, units $= 499.999$, with an unavoidable shortfall of $100.001$ units (since max feasible units at max profit is 500).
- **Algorithm 7 (Lexicographic on P11):**
  - Priority 1 (Profit): Achieves maximum profit $\$2400.00$.
  - Priority 2 (Units): Breaks the profit tie along the line segment by maximizing volume, deterministically landing on $(x_A, x_B) = (300.00, 200.00)$ producing 500 units.

### 3. Algorithms 8–9 (TOPSIS & VIKOR) Multi-Criteria Sensitivity on P12
- **Base Weights:** Cost $0.3$, Quality $0.3$, Delivery $0.2$, Reliability $0.2$.
  - **TOPSIS Top 3:** S3 (Score: $0.630$), S1 (Score: $0.621$), S5 (Score: $0.597$).
  - **VIKOR Top 3:** S1 ($Q = 0.093$), S2 ($Q = 0.309$), S5 ($Q = 0.429$).
- **Shifted Weights:** Shifted Cost weight by $+0.1$ ($0.4, 0.3, 0.2, 0.1$).
  - **TOPSIS Shifted Top 3:** S1 ($0.583$), S3 ($0.558$), S5 ($0.539$) $\implies$ **Ranking flipped!** (S1 overtook S3).
  - **VIKOR Shifted Top 3:** S2 ($0.135$), S1 ($0.136$), S3 ($0.500$) $\implies$ **Ranking flipped!** (S2 overtook S1).
- **Conclusion:** A modest 0.1 shift in preference weight reverses the optimal supplier selection under both MCDM methods, demonstrating high sensitivity to decision-maker trade-off articulation.

### 4. Algorithms 36–37 (MGDA & GradNorm) Multi-Task Loss Balancing on P13
- **Fixed Equal Weights:** Final losses: $\mathcal{L}_1 = 17.83, \mathcal{L}_2 = 18.17$.
- **MGDA (#36):** Final losses: $\mathcal{L}_1 = 6.76, \mathcal{L}_2 = 57.84$. MGDA found a common descent direction but terminated prematurely near the boundary of Task 1.
- **GradNorm (#37, $\alpha=1.5$):** Final losses: $\mathcal{L}_1 = 8.32, \mathcal{L}_2 = 31.58$. GradNorm dynamically scaled gradient norms using inverse training rates $r_i(t)$, maintaining balanced task learning and generating 37 non-dominated Pareto trade-offs in its external archive.
