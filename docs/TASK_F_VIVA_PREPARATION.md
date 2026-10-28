# Task F: Comprehensive Viva Examination and Code Walkthrough Preparation

**Candidate Roll Number:** Remainder 13 ($R = 13$, Registration Number suffix 13)  
**Assigned Deep-Dive Algorithms:**
- **Algorithm 13:** NPGA (Niched Pareto Genetic Algorithm) — Early Non-Elitist Evolutionary
- **Algorithm 25:** MOACO (Multi-Objective Ant Colony Optimization) — Swarm Metaheuristic
- **Algorithm 37:** GradNorm (Gradient Normalization for Multi-Task Learning) — Multi-Task Gradient-Based

---

## Part 1: Detailed Answers to the Eight Mandatory Viva Questions

### Question 1: Why can Weighted Sum never reach a dented part of a non-convex front?

#### 1. Mathematical Derivation and Geometric Principle
The Weighted Sum scalarization transforms an $M$-objective problem into a single parametric surrogate:
$$\min_{x \in \mathcal{X}} F_\lambda(x) = \sum_{m=1}^M \lambda_m f_m(x) \quad \text{subject to } \lambda_m \ge 0, \quad \sum_{m=1}^M \lambda_m = 1$$
Geometrically, in the objective space $\mathcal{Z} = \{f(x) : x \in \mathcal{X}\}$, the equation $\sum_{m=1}^M \lambda_m f_m = c$ defines a hyperplane with outward normal vector $\lambda = (\lambda_1, \dots, \lambda_M)^T$. Minimizing $F_\lambda(x)$ is geometrically equivalent to shifting this hyperplane parallel to itself from $-\infty$ until it first contacts the feasible objective set $\mathcal{Z}$. This makes the hyperplane a **supporting hyperplane** to the set $\mathcal{Z}$ at the contact point $f(x^*)$.

#### 2. The Convex Hull Limitation
By the **Supporting Hyperplane Theorem**, a supporting hyperplane to any arbitrary set $\mathcal{Z}$ is simultaneously a supporting hyperplane to the **convex hull** of $\mathcal{Z}$, denoted $\text{conv}(\mathcal{Z})$. Consequently, the global minimum of a linear function over a compact set is strictly attained at the extreme boundary points of its convex hull:
$$\min_{z \in \mathcal{Z}} \lambda^T z = \min_{z \in \text{conv}(\mathcal{Z})} \lambda^T z$$

#### 3. Why Interior Points in a "Dented" Front Are Missed
Consider a non-convex (inwardly curved / concave) Pareto front, such as the ZDT2 benchmark (Problem P3) where the true front is defined by $f_2 = 1 - f_1^2$ for $f_1 \in [0, 1]$:
- The interior points of this inward curve (e.g., $f_1 = 0.5, f_2 = 0.75$) lie strictly inside the interior of $\text{conv}(\mathcal{Z})$.
- The boundary of $\text{conv}(\mathcal{Z})$ is simply the straight line chord connecting the two extreme vertices $(0, 1)$ and $(1, 0)$.
- If we construct a line $\lambda_1 f_1 + \lambda_2 f_2 = c$ tangent to any interior point on $f_2 = 1 - f_1^2$, the concave curvature of the boundary curves below the tangent line. Therefore, feasible points exist further in the direction of the normal with smaller scalar values.
- Depending on the slope $-\lambda_1 / \lambda_2$:
  - If $\lambda_1 > \lambda_2$, the extreme point $(0, 1)$ achieves a strictly lower value of $F_\lambda$.
  - If $\lambda_1 < \lambda_2$, the extreme point $(1, 0)$ achieves a strictly lower value of $F_\lambda$.
  - At the critical threshold $\lambda_1 = \lambda_2$, both extreme points are tied, and intermediate points still have higher scalar values.
- Thus, **no positive weight vector $\lambda$ exists for which an interior point of the dent is the global minimizer**.

#### 4. Empirical Evidence from Our Runs
In our Task D head-to-head comparison on P3 (ZDT2) evaluated across 10 independent seeds with 10 uniform weight settings:
- **Number of distinct solutions found:** Exactly **2 points** across all runs: $(0.0, 1.0)$ and $(1.0, 0.0)$.
- **Schott's Spacing ($S$):** $0.0000 \pm 0.0000$.
- **Hypervolume (HV):** $0.2100 \pm 0.0000$, with high IGD ($0.3532$).
- In contrast, methods that do not rely on linear scalarization (e.g. $\epsilon$-Constraint #2, which bounds one objective, or MODE #27, which uses Pareto dominance replacement) successfully traversed the dent, achieving $HV = 0.4387$ and $HV = 0.4547$ respectively.

---

### Question 2: Why did VEGA collapse towards the extremes in your run?

#### 1. The Core Mechanism of VEGA (Algorithm 10)
In David Schaffer's Vector Evaluated Genetic Algorithm (1985), the total population $N$ is divided equally into $M$ disjoint sub-populations of size $N/M$:
$$\mathcal{P} = \mathcal{P}_1 \cup \mathcal{P}_2 \cup \dots \cup \mathcal{P}_M, \quad |\mathcal{P}_m| = N/M$$
- Sub-population $\mathcal{P}_1$ is selected from the previous generation strictly on the basis of objective $f_1$, completely ignoring $f_2, \dots, f_M$.
- Sub-population $\mathcal{P}_2$ is selected strictly on the basis of objective $f_2$, completely ignoring $f_1, f_3, \dots, f_M$.
- After selection, the sub-populations are merged into a single pool of size $N$, shuffled, and subjected to standard crossover and mutation to create the offspring generation.

#### 2. The Speciation / Extreme Selection Pressure Defect
The fundamental flaw in VEGA is that **selection pressure in each sub-population is 100% single-objective**:
- Consider a candidate $A$ that is exceptional in $f_1$ ($f_1 = 0.01$) but terrible in $f_2$ ($f_2 = 100.0$). Candidate $A$ has maximal fitness in sub-population $\mathcal{P}_1$ and easily survives.
- Consider candidate $B$ that is exceptional in $f_2$ ($f_2 = 0.02$) but terrible in $f_1$ ($f_1 = 95.0$). Candidate $B$ has maximal fitness in sub-population $\mathcal{P}_2$ and easily survives.
- Now consider candidate $C$, a **high-quality compromise solution** that represents the true center of the Pareto front ($f_1 = 0.50, f_2 = 0.50$):
  - In sub-population $\mathcal{P}_1$, candidate $C$'s $f_1 = 0.50$ is substantially worse than candidate $A$'s $0.01$. Candidate $C$ is outcompeted and eliminated.
  - In sub-population $\mathcal{P}_2$, candidate $C$'s $f_2 = 0.50$ is substantially worse than candidate $B$'s $0.02$. Candidate $C$ is outcompeted and eliminated.
- Consequently, balanced compromise individuals are systematically culled from both sub-populations in every single generation.

#### 3. Genetic Recombination Fails to Restore Compromises
When the two extreme sub-populations are recombined, crossover between extreme-$f_1$ specialists and extreme-$f_2$ specialists frequently produces intermediate offspring. However, in the very next selection phase, these intermediate offspring are eliminated once again in favor of extreme specialists. This causes the population to rapidly bifurcate into two disjoint, non-communicating clusters residing at the extreme boundary endpoints of the Pareto front.

#### 4. Empirical Evidence from Our Runs
In our Task A and Task D runs on P2 (Fleet Sizing) and P3 (ZDT2):
- On P3, VEGA's final non-dominated front collapsed into two tight clusters located at $(0.002, 0.999)$ and $(0.985, 0.021)$.
- Zero individuals survived in the interior interval $f_1 \in [0.15, 0.85]$.
- Because the interior front was unpopulated, the dominated hypervolume against the reference point was near zero ($HV = 0.0000 \pm 0.0000$), and the Inverted Generational Distance remained severe ($IGD = 1.2856 \pm 0.1889$).

---

### Question 3: What does elitism guarantee, and which of your algorithms have it?

#### 1. Theoretical Definition and Formal Guarantee of Elitism
In evolutionary and metaheuristic multiobjective optimization, **elitism** is the algorithmic mechanism ensuring that non-dominated solutions discovered in generation $t$ are preserved and prevented from being replaced by dominated (inferior) solutions in generation $t+1$.
- **Mathematical Guarantee (Monotonic Non-Deterioration):** Let $\mathcal{A}_t$ denote the set of non-dominated solutions known at generation $t$. Elitism guarantees that for all $t$:
  $$\forall y \in \mathcal{A}_{t+1}, \quad \nexists x \in \mathcal{A}_t \text{ such that } x \succ y$$
  That is, the hypervolume of the known Pareto front is monotonically non-decreasing over time: $HV(\mathcal{A}_{t+1}) \ge HV(\mathcal{A}_t)$.
- **Rudolph's Convergence Theorem (1998):** Rudolph mathematically proved that a canonical genetic algorithm without elitism *cannot* converge to the global optimum of an arbitrary optimization problem with probability 1. Due to stochastic selection and destructive variation (crossover and mutation), superior alleles can be lost via genetic drift. Elitism is the strictly necessary condition to guarantee asymptotic convergence to the true Pareto set.

#### 2. Categorization of Our 37 Implemented Algorithms

##### A. Non-Elitist Algorithms (No Guarantee of Preservation)
These early algorithms select parents solely from the current generation $\mathcal{P}_t$ to produce offspring $\mathcal{Q}_t$, and then completely discard $\mathcal{P}_t$ in favor of $\mathcal{Q}_t$:
1. **VEGA (#10):** Sub-populations selected without archival memory.
2. **MOGA (#11):** Goldbergs's dominance-count ranking with fitness sharing; parents are discarded.
3. **NSGA (#12):** Srinivas & Deb's dummy fitness assignment; generational replacement without an elite archive.
4. **NPGA (#13):** Horn, Nafpliotis & Goldberg's Pareto tournament selection; generational replacement without an archive.

##### B. Elitist Algorithms via Combined $(N + N \to N)$ Selection
These algorithms combine parent population $\mathcal{P}_t$ and offspring population $\mathcal{Q}_t$ into a joint pool $\mathcal{R}_t = \mathcal{P}_t \cup \mathcal{Q}_t$ of size $2N$, apply non-dominated sorting and secondary diversity truncation, guaranteeing that elite solutions cannot be lost:
1. **NSGA-II (#14):** Fast non-dominated sort + crowding distance.
2. **NSGA-III (#15):** Non-dominated sort + Das-Dennis reference line niching.
3. **Constrained NSGA-II (#21):** Deb's constrained-dominance sort + crowding distance.
4. **MODE (#27):** Target-trial one-to-one dominance replacement ($x_i$ replaced only if $u_i \succ x_i$).
5. **MOMA (#30):** Memetic hybrid using $(N+N)$ selection with local hill-climbing.
6. **HypE (#34):** Non-dominated sort + Monte Carlo hypervolume contribution truncation.

##### C. Elitist Algorithms via Explicit External Archives
These algorithms maintain an external non-dominated archive $\mathcal{E}$ updated at every generation/iteration:
1. **SPEA2 (#16):** Environmental selection with fixed-size archive using $k$-th nearest neighbor density.
2. **PESA-II (#17):** Hyperbox grid archive with region-based selection.
3. **MOEA/D (#18) & MOEA/D-DE (#19):** External Pareto archive + localized neighborhood replacement.
4. **MOEA/D-ACO (#20):** Ant colony routing with external non-dominated tour repository.
5. **MOPSO (#22):** Personal best ($pbest$) and external archive global best ($gbest$).
6. **MOGWO (#23):** External archive supplying $\alpha, \beta, \delta$ wolf leaders.
7. **MOABC (#24):** External archive with dominance-based food source updates.
8. **MOSA (#26):** Simulated Annealing with external non-dominated list.
9. **MO-TLBO (#28) & MOBA (#29):** External Pareto archives.
10. **RVEA (#31):** External archive + angle-penalized distance reference vector selection.
11. **IBEA (#32):** Environmental selection based on binary $\epsilon$-indicator fitness.
12. **SMS-EMOA (#33):** Steady-state $(N+1)$ selection discarding the solution with minimal hypervolume contribution.

---

### Question 4: Why does crowding distance stop working well with many objectives, and what replaces it?

#### 1. The Phenomenon of High-Dimensional Objective Space ($M \ge 4, 5$)
When the number of objectives grows, two interrelated mathematical breakdowns occur:
1. **Loss of Selection Pressure from Pareto Dominance:**
   The probability that a random vector $u$ dominates another random vector $v$ in $M$ independent objectives scales as $(1/2)^M$.
   - For $M = 2$: $P(\text{dominance}) = 1/4 = 25\%$.
   - For $M = 3$: $P(\text{dominance}) = 1/8 = 12.5\%$.
   - For $M = 5$: $P(\text{dominance}) = 1/32 = 3.1\%$.
   - In a population of $N = 100$ individuals on DTLZ2 with $M = 5$, **over 95% of the population is mutually non-dominated** and assigned to Front 1 ($F_1$). Consequently, fast non-dominated sorting provides zero discriminative selection pressure.
2. **The Entire Selection Burden Falls on Crowding Distance:**
   Since nearly all individuals are in $F_1$, environmental selection must select $N$ individuals from $2N$ candidates using *only* crowding distance.

#### 2. Why Crowding Distance Degrades Severely
Crowding distance (Deb et al., 2002) is computed as:
$$CD(i) = \sum_{m=1}^M \frac{f_m(i+1) - f_m(i-1)}{f_m^{max} - f_m^{min}}$$
This metric fails in high dimensions due to three structural weaknesses:
1. **Independent 1D Projection Flaw:** Crowding distance computes distances along each objective coordinate independently and sums them. In high dimensions, an individual can be isolated along one single axis while being heavily crowded in the $(M-1)$-dimensional subspace. Independent 1D projections cannot detect multidimensional clustering.
2. **Boundary Point Saturation:** In NSGA-II, the boundary points with minimum and maximum values along each objective axis are assigned an infinite crowding distance ($CD = \infty$). In an $M$-objective problem, there are $2M$ such boundary points:
   - For $M=2$, there are only 4 boundary points.
   - For $M=5$, there are 10 boundary points.
   - For $M=10$, there are 20 boundary points.
   These extreme boundary solutions automatically survive every generation regardless of their convergence quality, drawing the population away from the interior manifold and creating unnatural "star-burst" clustering at the axes.
3. **Curse of Dimensionality / Point Sparsity:** As dimension increases, points in the unit hypercube become equidistant from one another. The variance of crowding distances collapses, causing selection to degenerate into stochastic random choice.

#### 3. What Replaces Crowding Distance?
Modern Many-Objective Evolutionary Algorithms replace crowding distance with structured geometric mechanisms:
1. **Structured Reference Lines (Das-Dennis Simplex in NSGA-III #15):**
   NSGA-III replaces crowding distance with pre-defined reference points on the unit simplex:
   $$H = \binom{M + p - 1}{p}$$
   Candidates are normalized and associated with their geometrically nearest reference line via perpendicular distance $d^\perp$. An explicit niching counter $\rho_j$ ensures that solutions are selected to cover each reference direction uniformly, guaranteeing full coverage of the manifold.
2. **Angle-Penalized Distance (APD in RVEA #31):**
   RVEA dynamically weights the acute angle between candidate objective vectors and uniform reference vectors:
   $$APD(x) = \left( 1 + M \left(\frac{t}{t_{max}}\right)^\alpha \frac{\theta}{\gamma} \right) ||f'(x)||_2$$
   This balances convergence ($||f'||_2$) and angular uniformity ($\theta$) without requiring non-dominated sorting.
3. **Decomposition into Scalar Subproblems (MOEA/D #18):**
   Converts the many-objective search into $N$ scalar subproblems using Tchebycheff or penalty-based boundary intersection (PBI), entirely avoiding dominance comparison.
4. **Hypervolume Approximations (HypE #34):**
   Uses Monte Carlo integration to estimate the exact hypervolume contribution of each candidate in high dimensions.

---

### Question 5: Which of your algorithms needs a preference from the decision-maker, and what happens if it is wrong?

#### 1. Inventory of Preference-Dependent Algorithms in Our Catalogue

| # | Algorithm | Preference Input Required from Decision-Maker | Nature of Parameter |
|:---:|:---|:---|:---|
| **1** | **Weighted Sum** | Relative trade-off weights $\lambda = (\lambda_1, \dots, \lambda_M)$ | A-priori weights ($\sum \lambda_m = 1$) |
| **2** | **$\epsilon$-Constraint** | Target objective index $k$ and bound thresholds $\epsilon_j$ for $j \neq k$ | Hard upper bounds on secondary objectives |
| **4** | **Weight Metric** | Ideal/reference point $z^*$ and norm parameter $p \in \{1, 2, \infty\}$ | Aspiration vector and metric curvature |
| **5** | **Value Function** | Explicit mathematical utility function $U(f_1, \dots, f_M)$ | Non-linear marginal utility / trade-off elasticity |
| **6** | **Goal Programming** | Aspiration targets $T_m$ and priority weights $P_k$ for deviations $d_m^+, d_m^-$ | Rigid goal levels and penalty priorities |
| **7** | **Lexicographic** | Strict ordinal ranking of objectives ($f_1 \succ f_2 \succ \dots$) and tolerances $\delta_m$ | Hierarchical priority order and relaxation bands |
| **8** | **TOPSIS** | Attribute weights $w_m$ ($\sum w_m = 1$) | Criterion importance for ideal distance |
| **9** | **VIKOR** | Attribute weights $w_m$ and strategy weight $v \in [0, 1]$ | Balance between group utility and individual regret |
| **35** | **R-NSGA-II** | Desired reference point $z^{pref}$ and clearing radius $\epsilon$ | Objective-space aspiration vector |

#### 2. What Happens If the Decision-Maker's Preference Is Wrong?

1. **In Goal Programming (#6):**
   - **Case A: Unattainable Goals ($T$ lies beyond the Pareto front):** The algorithm spends its entire evaluation budget minimizing under-achievement deviations ($d^-$). The returned compromise point is highly sensitive to the scaling of deviations; without normalization, the objective with larger numeric scale dominates, yielding a heavily biased solution.
   - **Case B: Sub-optimal / Interior Goals ($T$ is dominated):** If the DM sets conservative targets that lie *inside* the feasible region, Goal Programming achieves $d^- = 0$ at an interior, Pareto-dominated point! The algorithm terminates with zero deviation, delivering a solution where every objective could have been significantly improved without penalty.
2. **In Lexicographic Optimization (#7):**
   - If the DM erroneously ranks an objective with a single, sharp global optimum as Priority 1 ($f_1$), Phase 1 terminates at that unique point.
   - Because no alternative solutions achieve the optimal value of $f_1$, **the remaining $M-1$ objectives are completely ignored**! A catastrophic sacrifice in quality, safety, or cost can occur to gain an imperceptible fraction of a percent in $f_1$.
3. **In R-NSGA-II (#35):**
   - If $z^{pref}$ is chosen far from the true Pareto front or in a physically unreasonable direction, the modified Euclidean distance metric forces the evolutionary population to cluster tightly around the projection of $z^{pref}$.
   - The algorithm experiences premature diversity collapse: all solutions converge onto a single tiny patch of the front, completely blinding the DM to vastly superior trade-offs located elsewhere on the manifold.
4. **In MCDM Ranking (TOPSIS #8 and VIKOR #9):**
   - In our Task D P12 sensitivity study, shifting a single weight by $0.1$ altered the top-ranked supplier. If weights do not reflect true stakeholder trade-offs, MCDM algorithms select inferior alternatives that may fail critical performance standards.

---

### Question 6: Explain Deb's three constraint rules with one numerical example.

#### 1. Formal Definition of Deb's Constrained-Dominance Rules (Deb 2000)
Given two candidate solutions $A$ and $B$, evaluate their objective vectors $F(A), F(B)$ and their total constraint violations $CV(A), CV(B)$, defined for $J$ inequality constraints $g_j(x) \le 0$ as:
$$CV(x) = \sum_{j=1}^J \max(0, g_j(x))$$
Solution $A$ is said to **constrained-dominate** solution $B$ ($A \succ_c B$) if and only if any of the following three conditions holds:
1. **Rule 1 (Feasible vs. Infeasible):** Solution $A$ is feasible ($CV(A) = 0$) and Solution $B$ is infeasible ($CV(B) > 0$).  
   *Principle:* Any feasible solution is unconditionally preferred to any infeasible solution.
2. **Rule 2 (Infeasible vs. Infeasible):** Both solutions $A$ and $B$ are infeasible ($CV(A) > 0$ and $CV(B) > 0$), and $CV(A) < CV(B)$.  
   *Principle:* Between two infeasible designs, the one closer to the feasible boundary is preferred. Objective values are completely ignored.
3. **Rule 3 (Feasible vs. Feasible):** Both solutions $A$ and $B$ are feasible ($CV(A) = 0$ and $CV(B) = 0$), and solution $A$ Pareto-dominates solution $B$ under standard dominance:
   $$\forall m \in \{1, \dots, M\}, \quad f_m(A) \le f_m(B) \quad \text{and} \quad \exists m \text{ such that } f_m(A) < f_m(B)$$

#### 2. Detailed Numerical Example (Problem P5: Cantilever Beam)
Recall Problem P5:
- Objectives: $\min f_1 = \text{mass}$ (kg), $\min f_2 = \text{tip deflection}$ (m).
- Constraint: Bending stress $\sigma = \frac{6 P L}{b h^2} \le 250$ MPa, giving $CV = \max(0, \sigma - 250 \times 10^6) \times 10^{-6}$ (in MPa).

Consider four candidate beam designs:
- **Design $A$:** $f_1 = 4.0$ kg, $f_2 = 0.020$ m, Stress $\sigma = 240$ MPa $\implies CV(A) = 0.0$ MPa (Feasible)
- **Design $B$:** $f_1 = 2.0$ kg, $f_2 = 0.010$ m, Stress $\sigma = 265$ MPa $\implies CV(B) = 15.0$ MPa (Infeasible by 15 MPa)
- **Design $C$:** $f_1 = 1.5$ kg, $f_2 = 0.005$ m, Stress $\sigma = 310$ MPa $\implies CV(C) = 60.0$ MPa (Infeasible by 60 MPa)
- **Design $D$:** $f_1 = 5.5$ kg, $f_2 = 0.035$ m, Stress $\sigma = 210$ MPa $\implies CV(D) = 0.0$ MPa (Feasible)

##### Application of the Three Rules:
- **Comparison 1 ($A$ vs. $B$ — Demonstrating Rule 1):**
  - Notice that Design $B$ has vastly superior mass ($2.0 < 4.0$) and superior stiffness ($0.010 < 0.020$). In unconstrained optimization, $B$ would strongly dominate $A$.
  - However, $CV(A) = 0$ (feasible) while $CV(B) = 15.0 > 0$ (infeasible).
  - By **Rule 1**, **$A$ constrained-dominates $B$ ($A \succ_c B$)**. The superior objective values of $B$ are completely rejected because the beam will physically fracture.
- **Comparison 2 ($B$ vs. $C$ — Demonstrating Rule 2):**
  - Both $B$ and $C$ are infeasible ($CV(B) = 15.0 > 0, CV(C) = 60.0 > 0$).
  - Design $C$ has lower mass and lower deflection than $B$.
  - By **Rule 2**, objective values are entirely discarded. Since $CV(B) = 15.0 < 60.0 = CV(C)$, **$B$ constrained-dominates $C$ ($B \succ_c C$)**. This provides a smooth, monotonic gradient driving the population toward the feasible region without requiring arbitrary penalty multipliers.
- **Comparison 3 ($A$ vs. $D$ — Demonstrating Rule 3):**
  - Both $A$ and $D$ are feasible ($CV(A) = 0, CV(D) = 0$).
  - Now, standard Pareto dominance applies: $f_1(A) = 4.0 < 5.5 = f_1(D)$ and $f_2(A) = 0.020 < 0.035 = f_2(D)$.
  - By **Rule 3**, **$A$ constrained-dominates $D$ ($A \succ_c D$)**.

---

### Question 7: How would you detect that your algorithm has prematurely converged?

Premature convergence occurs when an optimizer becomes trapped in a sub-optimal local Pareto front or collapses onto a narrow region of the search space before discovering the full, true trade-off front. In our project, premature convergence is diagnosed using four rigorous quantitative indicators:

#### 1. Metric Decoupling: Stagnant Hypervolume with High Inverted Generational Distance (IGD)
- **Symptom:** The Hypervolume (HV) reaches an asymptote over multiple generations ($\Delta HV < 10^{-6}$), yet the Inverted Generational Distance (IGD) against the known reference front remains significantly above zero ($IGD \gg 0.05$).
- **Diagnosis:** The population has converged tightly onto a localized attractor, but this attractor is either a local Pareto front or a tiny segment of the global front. Because IGD samples points uniformly across the entire true front, missing regions result in high nearest-neighbor distances, exposing the failure that HV alone cannot reveal.

#### 2. Collapse of Decision-Space Variance (Genotypic Diversity Loss)
- **Symptom:** The standard deviation of the decision variables across the population drops towards machine precision:
  $$\sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_{i, j} - \bar{x}_j)^2} < 10^{-5} \quad \forall j \in \{1, \dots, n\}$$
- **Diagnosis:** The population has suffered genetic drift or loss of diversity. Almost all chromosomes have become identical. In genetic algorithms, SBX crossover between near-identical parents produces offspring identical to parents, effectively halting search progress.

#### 3. Deterioration of Schott's Spacing ($S$) and Objective Range Collapse
- **Symptom:** Schott's Spacing $S$ approaches zero or undefined, while the objective range $[f_m^{min}, f_m^{max}]$ collapses into a narrow band representing $< 5\%$ of the known Pareto span.
- **Diagnosis:** The algorithm has suffered crowding collapse, clustering all individuals onto a single compromise point or boundary endpoint.

#### 4. Offspring Acceptance / Replacement Rate Decay
- **Symptom:** In external-archive or differential evolution methods (e.g. MODE #27, MOEA/D #18), the proportion of newly generated offspring that successfully enter the non-dominated archive or replace a neighborhood solution drops below $1\%$.
- **Diagnosis:** The algorithm's step size (mutation rate $\eta_m$ or differential scale $F$) is mismatched with the local landscape curvature, causing all trials to be dominated by the incumbent archive.

---

### Question 8: Why do TOPSIS and VIKOR not count as optimizers?

#### 1. Fundamental Definition of an Optimizer
An **optimization algorithm** is a computational procedure designed to navigate and search through a continuous or discrete solution space $\mathcal{X}$. An optimizer:
1. Generates new trial candidate solutions $x \in \mathcal{X}$ via mathematical operators (gradient steps, mutation, crossover, ant tour construction, velocity updates).
2. Evaluates the candidate solutions against the underlying mathematical problem $F(x) = (f_1(x), \dots, f_M(x))^T$.
3. Dynamically refines the search trajectory to discover previously unknown Pareto-optimal designs.

#### 2. The Operational Reality of TOPSIS and VIKOR
TOPSIS (Hwang & Yoon, 1981) and VIKOR (Opricovic, 1998) are **Multi-Criteria Decision Making (MCDM) ranking techniques**, not optimization algorithms. They possess zero generative capability:
- **Input Requirement:** They strictly accept as input a **pre-existing, finite, fixed matrix of discrete alternatives**:
  $$\mathbf{X} \in \mathbb{R}^{K \times M}$$
  where $K$ is the number of already-known candidate solutions (e.g., the 5 suppliers in Problem P12, or the 100 solutions on an already-computed NSGA-II Pareto front) and $M$ is the number of evaluated criteria.
- **Algorithmic Role:**
  - **TOPSIS** normalizes the performance matrix, identifies the ideal solution $A^*$ and anti-ideal solution $A^-$, computes Euclidean distances $D_i^*$ and $D_i^-$, and ranks the $K$ choices by relative closeness $C_i = D_i^- / (D_i^* + D_i^-)$.
  - **VIKOR** computes the Manhattan group utility $S_i$, the Chebyshev individual regret $R_i$, and the aggregated compromise index $Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1-v) \frac{R_i - R^*}{R^- - R^*}$, returning a compromise ranking and checking stability conditions.
- **Why They Cannot Optimize:**
  - Neither TOPSIS nor VIKOR can perturb decision variables, generate new alternatives, or explore the design space.
  - If given 5 terrible candidate designs, TOPSIS and VIKOR will simply rank which of the 5 terrible designs is least bad. They cannot discover a 6th design that Pareto-dominates all five.
  - They operate strictly in **post-processing** (a-posteriori decision support) to help human decision-makers select a single preferred compromise solution from an already-generated Pareto front.

---

## Part 2: Code Walkthroughs for Assigned Deep-Dive Algorithms ($R=13$)

### Walkthrough 1: Algorithm 13 — NPGA (Niched Pareto Genetic Algorithm)

#### 1. Code Architecture and Implementation Details
- **Source File:** `moo/algorithms/alg13_npga.py`
- **Lecture Family:** Early Non-Elitist Evolutionary Algorithms (Part 2, Chapter 5)
- **Original Citation:** Horn, J., Nafpliotis, N., & Goldberg, D. E. (1994). *"A niched Pareto genetic algorithm for multiobjective optimization."* Proceedings of the First IEEE Conference on Evolutionary Computation.
- **Entry Points:**
  - `initialize(problem, seed=42, pop_size=100, t_dom=10, sigma_share=0.1, **kwargs) -> dict`
  - `step(state, problem, prob_c=0.9, eta_c=15.0, eta_m=20.0) -> dict`
  - `run(problem, budget=10000, seed=42, **kwargs) -> dict`

#### 2. Key Internal Functions and Execution Flow
1. **Pareto Tournament Selection (`pareto_tournament`):**
   - Two candidate individuals $p_1, p_2$ are randomly selected from the population.
   - A comparison set $C_{dom}$ of size $t_{dom}$ is sampled uniformly at random from the population.
   - Candidate $p_1$ is tested for dominance against $C_{dom}$; candidate $p_2$ is tested for dominance against $C_{dom}$.
   - **Case 1:** $p_1$ is non-dominated by $C_{dom}$ while $p_2$ is dominated $\implies p_1$ is selected.
   - **Case 2:** $p_2$ is non-dominated by $C_{dom}$ while $p_1$ is dominated $\implies p_2$ is selected.
   - **Case 3 (Tie-Break):** If both are non-dominated or both are dominated, the tie is broken by **niche count sharing**:
     - Compute normalized niche counts $nc(p_1)$ and $nc(p_2)$ against the current population using sharing radius $\sigma_{share}$:
       $$Sh(d) = 1 - (d / \sigma_{share}) \quad \text{if } d < \sigma_{share}, \text{ else } 0$$
     - The candidate in the less crowded niche ($\min nc$) wins the tournament.
2. **Generational Variation and Replacement:**
   - Selected parents undergo Simulated Binary Crossover (SBX, `eta_c=15.0`) and Polynomial Mutation (PM, `eta_m=20.0`).
   - The entire parent population is replaced by the newly created offspring (strictly non-elitist).
   - An external non-dominated front is tracked via `extract_non_dominated` for reporting.

#### 3. Task C Hand Trace Reproduction and Parameter Defense
- **Task C Hand Trace:** Verified on a 4-individual instance over 2 generations on ZDT2 (reproduced exactly by unit test `test_task_c_traces.py`).
- **Critical Parameters:**
  - `t_dom`: Tournament comparison set size. If $t_{dom}$ is too small ($t_{dom}=1$), dominance checks are meaningless and selection degrades to pure niche sharing; if too large ($t_{dom} \approx N$), selection pressure is excessively fierce, leading to premature convergence. Default: $t_{dom} = 10$.
  - `sigma_share`: Niche sharing radius. In our implementation, sharing is performed in normalized objective space $[0, 1]^M$ with $\sigma_{share} = 0.1$, which is robust to decision-variable scaling.

---

### Walkthrough 2: Algorithm 25 — MOACO (Multi-Objective Ant Colony Optimization)

#### 1. Code Architecture and Implementation Details
- **Source File:** `moo/algorithms/alg25_moaco.py`
- **Lecture Family:** Swarm Metaheuristics (Part 4, Chapter 5)
- **Original Citation:** Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press; Alaya et al. (2007).
- **Entry Points:**
  - `initialize(problem, seed=42, n_ants=20, alpha=1.0, beta=2.0, rho=0.1, **kwargs) -> dict`
  - `step(state, problem) -> dict`
  - `run(problem, budget=10000, seed=42, **kwargs) -> dict`

#### 2. Key Internal Functions and Execution Flow
1. **Dual Pheromone and Heuristic Representations:**
   - MOACO maintains two independent $N_{cities} \times N_{cities}$ pheromone matrices:
     - $\tau^1_{ij}$: Pheromone trail for distance minimization.
     - $\tau^2_{ij}$: Pheromone trail for travel time minimization.
   - Pre-computes heuristic desirabilities: $\eta^1_{ij} = 1 / d_{ij}$ and $\eta^2_{ij} = 1 / t_{ij}$.
2. **Probabilistic Route Construction (`construct_tour`):**
   - Each ant $k$ is assigned a specific scalar trade-off weight $\lambda_k = k / (n_{ants} - 1)$.
   - Starting from depot city 0, ant $k$ chooses next unvisited city $j$ from current city $i$ with probability:
     $$P_{ij}^k = \frac{\left[\tau_{ij}^1\right]^{\lambda_k \alpha} \left[\tau_{ij}^2\right]^{(1-\lambda_k) \alpha} \left[\eta_{ij}^1\right]^{\lambda_k \beta} \left[\eta_{ij}^2\right]^{(1-\lambda_k) \beta}}{\sum_{l \in \text{unvisited}} [\dots]}$$
3. **Pheromone Evaporation and Dual-Objective Deposit:**
   - Evaporates all trails: $\tau^1_{ij} \leftarrow (1 - \rho) \tau^1_{ij}$ and $\tau^2_{ij} \leftarrow (1 - \rho) \tau^2_{ij}$.
   - Non-dominated tours in the current colony deposit pheromone:
     $$\Delta \tau^1_{ij} = Q / f_1(\text{tour}), \quad \Delta \tau^2_{ij} = Q / f_2(\text{tour})$$

#### 3. Task C Hand Trace Reproduction and Defense
- **Task C Hand Trace:** Verified on a 4-city TSP instance over 2 iterations (reproduced by `test_task_c_traces.py`).
- **Defending Simplifications:** We use symmetric matrix evaporation and bound pheromone levels in $[\tau_{min}, \tau_{max}]$ (MAX-MIN Ant System style) to prevent search stagnation on edge cases.

---

### Walkthrough 3: Algorithm 37 — GradNorm (Gradient Normalization for Multi-Task Learning)

#### 1. Code Architecture and Implementation Details
- **Source File:** `moo/algorithms/alg37_gradnorm.py`
- **Lecture Family:** Gradient-Based Multi-Task Optimization (Bonus Module 5E, Chapter 5)
- **Original Citation:** Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). *"GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks."* ICML 2018.
- **Entry Points:**
  - `initialize(problem, seed=42, lr_weights=0.025, lr_model=0.01, alpha=1.5, **kwargs) -> dict`
  - `step(state, problem) -> dict`
  - `run(problem, budget=10000, seed=42, **kwargs) -> dict`

#### 2. Key Internal Functions and Execution Flow
1. **Multi-Task Loss and Gradient Norm Extraction:**
   - Computes individual task losses $\mathcal{L}_1(x), \dots, \mathcal{L}_M(x)$ and weighted objective:
     $$\mathcal{L}_{total}(x, w) = \sum_{m=1}^M w_m \mathcal{L}_m(x)$$
   - Computes $L_2$ norm of the gradient with respect to shared parameters $W$:
     $$G_W^{(m)}(t) = ||\nabla_W (w_m(t) \mathcal{L}_m(t))||_2 = w_m(t) ||\nabla_W \mathcal{L}_m(t)||_2$$
2. **Relative Inverse Training Rate and Target Computation:**
   - Computes loss ratio $\tilde{\mathcal{L}}_m(t) = \mathcal{L}_m(t) / \mathcal{L}_m(0)$.
   - Computes relative inverse training rate:
     $$r_m(t) = \frac{\tilde{\mathcal{L}}_m(t)}{\frac{1}{M} \sum_{j=1}^M \tilde{\mathcal{L}}_j(t)}$$
   - Target gradient norm for task $m$:
     $$G_{target}^{(m)}(t) = \bar{G}_W(t) \times \left[ r_m(t) \right]^\alpha$$
3. **Weight Update and Re-normalization:**
   - Computes auxiliary loss $L_{grad} = \sum_{m=1}^M |G_W^{(m)}(t) - G_{target}^{(m)}(t)|$.
   - Differentiates $L_{grad}$ with respect to $w_m$, updates weights via gradient descent, and renormalizes to enforce $\sum_{m=1}^M w_m = M$.

#### 3. Task C Hand Trace Reproduction and Defense
- **Task C Hand Trace:** Verified on analytical Problem P13 ($a = (3, 3), b = (-3, -3)$, start $(0, 5)$) over 2 iterations (reproduced by `test_task_c_traces.py`).
- **Defending Simplifications:** For analytical continuous problems (P13), the shared parameters $W$ are the continuous coordinates $x$. We compute exact analytical gradients without requiring external PyTorch/TensorFlow dependencies, strictly adhering to the assignment's dependency restriction.

---

## Part 3: Live Coding and Predicted Code Mutation Scenarios

During the viva examination, the instructor may ask you to modify a single line of code and predict the resulting behavioral effect before running it. Here are four anticipated code mutation scenarios:

### Scenario 1: Changing `sigma_share` in NPGA (`alg13_npga.py`)
- **Modification:** Change `sigma_share = 0.1` to `sigma_share = 1.0` (an order-of-magnitude increase).
- **Predicted Effect:**
  - When $\sigma_{share}$ is extremely large, every individual in the population falls inside the niche radius of every other individual ($d_{ij} < \sigma_{share} \ \forall i, j$).
  - The sharing function $Sh(d) = 1 - d/\sigma_{share}$ remains positive across the entire objective space.
  - As a result, niche counts become approximately identical for all candidates.
  - The tie-breaking mechanism loses all spatial discriminative power; tie-breaks become purely random.
  - Diversity collapses, Schott's Spacing degrades, and the population fails to spread across the Pareto front.

### Scenario 2: Zeroing $\gamma$ (`alpha`) in GradNorm (`alg37_gradnorm.py`)
- **Modification:** Change `alpha = 1.5` to `alpha = 0.0`.
- **Predicted Effect:**
  - In GradNorm, target gradient norms are computed as $G_{target}^{(m)} = \bar{G}_W \times [r_m(t)]^\alpha$.
  - When $\alpha = 0.0$, $[r_m(t)]^0 = 1.0$ for all tasks, regardless of their training rate or loss ratios.
  - The target gradient norm collapses to a fixed average: $G_{target}^{(m)} = \bar{G}_W$.
  - GradNorm no longer balances task learning rates; it strictly equalizes the raw gradient magnitudes of the tasks. If one task is learning much slower than the other, GradNorm will not boost its learning rate, disabling adaptive task balancing.

### Scenario 3: Modifying Evaporation Rate $\rho$ in MOACO (`alg25_moaco.py`)
- **Modification:** Change `rho = 0.1` to `rho = 0.9` (hyper-evaporation).
- **Predicted Effect:**
  - At each iteration, $90\%$ of all existing pheromone is erased: $\tau \leftarrow 0.1 \tau$.
  - Historical memory of discovered tours is effectively annihilated within 2 generations.
  - Ants can no longer accumulate collective intelligence about successful path combinations.
  - The algorithm degenerates into repeated random nearest-neighbor tour construction, losing convergence capability and producing high-IGD tours.

### Scenario 4: Inverting Deb's Feasibility Rule in CNSGA-II (`alg21_cnsga2.py`)
- **Modification:** Change `if cv1 < cv2` to `if cv1 > cv2` in `constrained_dominates`.
- **Predicted Effect:**
  - Between two infeasible designs, the algorithm now prefers the one with *larger* constraint violation.
  - Selection pressure actively drives the population *away* from the feasible boundary and deep into the forbidden infeasible region.
  - The feasible ratio will drop to 0%, and the algorithm will fail to return a single valid engineering design.
