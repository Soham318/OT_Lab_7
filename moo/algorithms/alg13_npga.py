"""
alg13_npga.py: NPGA (Niched Pareto Genetic Algorithm)

Lecture Family: Non-Elitist Evolutionary Algorithms (Part 2, Chapter 5)
Original Citation: Horn, J., Nafpliotis, N., & Goldberg, D. E. (1994).
"A Niched Pareto Genetic Algorithm for Multiobjective Optimization."
Proceedings of the First IEEE Conference on Evolutionary Computation,
IEEE World Congress on Computational Intelligence, Vol. 1, pp. 82-87.

Task C Deep Dive #1:
- Assigned to student roll numbers with remainder 13: R = 13 ==> ((13 - 1) % 37) + 1 = 13.
- Core mechanism to study: Pareto tournament against a comparison set t_dom, niche-count tie-break.

Algorithmic Logic:
1. Population Initialization:
       Maintains a population of size N (e.g. N = 100).
2. Pareto Dominance Tournament Selection:
       To select each parent for reproduction (repeated N times):
       a. Pick two candidate individuals A and B uniformly at random from the population.
       b. Randomly sample a comparison set T_dom of size t_dom (typically ~10% of N, e.g. t_dom = 10)
          from the population.
       c. Test candidate A against each member of T_dom using Deb's constrained dominance.
          Test candidate B against each member of T_dom using Deb's constrained dominance.
       d. Decision Rules:
          - Case 1: A is non-dominated by T_dom, and B is dominated by at least one member in T_dom.
                    ==> A wins the tournament.
          - Case 2: B is non-dominated by T_dom, and A is dominated by at least one member in T_dom.
                    ==> B wins the tournament.
          - Case 3 (Tie): Both A and B are non-dominated by T_dom, OR both are dominated.
                    ==> Tie is resolved using Equivalence Class Fitness Sharing (Niche Count):
                        Compute niche count nc_A and nc_B against the current population
                        in normalized objective space using sharing radius sigma_share:
                            nc_i = sum_{j=1}^N Sh( ||f_i^norm - f_j^norm||_2 )
                            Sh(d) = 1 - (d / sigma_share)  if d < sigma_share else 0.
                        The candidate with the SMALLER niche count (located in the less crowded,
                        more diverse region) wins!
                        If niche counts are equal, a winner is chosen uniformly at random.
3. Genetic Variation:
       - Pair up winners, apply Simulated Binary Crossover (SBX, prob_c = 0.9).
       - Apply Polynomial Mutation (PM, prob_m = 1 / n_var).
4. Generational Replacement:
       - Offspring replaces parent population (non-elitist).
5. Maintain an external non-dominated archive of all feasible solutions evaluated throughout the search.

Key Parameters & Sensitivity Study Target (Task C):
- t_dom: Tournament comparison set size (e.g. 5, 10, 20).
         - Too small (e.g. t_dom = 2): Weak selection pressure; almost all tournaments tie and fall back to sharing.
         - Too large (e.g. t_dom = 50): Strong selection pressure; almost every candidate is dominated by T_dom.
- sigma_share: Objective space niche radius (e.g. 0.05, 0.1, 0.2).
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import constrained_dominates, normalize_objectives
from moo.algorithms.common import (
    random_population,
    sbx_crossover,
    polynomial_mutation,
    order_crossover,
    swap_mutation,
    compute_single_niche_count,
    extract_non_dominated,
)


def pareto_tournament_select(
    candidate_a_idx: int,
    candidate_b_idx: int,
    t_dom_indices: np.ndarray,
    F: np.ndarray,
    CV: np.ndarray,
    F_norm: np.ndarray,
    sigma_share: float,
    feas_tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Executes a single NPGA Pareto Tournament between candidate A and candidate B.
    Exposed as a standalone pure function for 100% reproducible Task C hand traces.

    Returns:
        int: Index of winning candidate (either candidate_a_idx or candidate_b_idx).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Test if candidate A is dominated by ANY member in T_dom
    a_dominated = False
    for t_idx in t_dom_indices:
        if constrained_dominates(F[t_idx], CV[t_idx], F[candidate_a_idx], CV[candidate_a_idx], feas_tol=feas_tol):
            a_dominated = True
            break

    # Test if candidate B is dominated by ANY member in T_dom
    b_dominated = False
    for t_idx in t_dom_indices:
        if constrained_dominates(F[t_idx], CV[t_idx], F[candidate_b_idx], CV[candidate_b_idx], feas_tol=feas_tol):
            b_dominated = True
            break

    # Decision Rules:
    if not a_dominated and b_dominated:
        return candidate_a_idx
    if not b_dominated and a_dominated:
        return candidate_b_idx

    # Tie-break via Niche Count Sharing in normalized objective space
    nc_a = compute_single_niche_count(F_norm[candidate_a_idx], F_norm, sigma_share=sigma_share)
    nc_b = compute_single_niche_count(F_norm[candidate_b_idx], F_norm, sigma_share=sigma_share)

    if nc_a < nc_b:
        return candidate_a_idx
    elif nc_b < nc_a:
        return candidate_b_idx
    else:
        # Exact tie: pick randomly
        return candidate_a_idx if rng.random() < 0.5 else candidate_b_idx


def initialize(
    problem: Problem,
    seed: int = 42,
    pop_size: int = 100,
    t_dom: int = 10,
    sigma_share: float = 0.1,
    budget: int = 10000,
    **kwargs,
) -> dict:
    """Initializes NPGA population and parameters."""
    rng = np.random.default_rng(seed)
    effective_pop_size = max(1, min(pop_size, budget))
    pop = random_population(problem, effective_pop_size, rng)

    F_list = []
    CV_list = []
    for ind in pop:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    state = {
        "seed": seed,
        "rng": rng,
        "pop_size": effective_pop_size,
        "t_dom": int(t_dom),
        "sigma_share": float(sigma_share),
        "X": pop,
        "F": F,
        "CV": CV,
        "archive_X": pop.copy(),
        "archive_F": F.copy(),
        "archive_CV": CV.copy(),
        "n_evals": effective_pop_size,
        "budget": budget,
        "iter": 0,
    }
    return state


def step(
    state: dict,
    problem: Problem,
    prob_c: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
) -> dict:
    """Executes one NPGA generation."""
    budget = state.get("budget", 1000000)
    rem_budget = budget - state["n_evals"]
    if rem_budget <= 0:
        return state

    rng = state["rng"]
    N = state["pop_size"]
    if N < 2:
        return state

    t_dom = state["t_dom"]
    sigma_share = state["sigma_share"]
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]

    # Precompute normalized objectives for niche count sharing
    F_norm = normalize_objectives(F, ideal=problem.ideal_point, nadir=problem.nadir_point)

    # 1. Select N parents via Pareto Tournaments
    mating_pool_indices = []
    for _ in range(N):
        # Pick 2 distinct candidates
        cand_a, cand_b = rng.choice(N, size=2, replace=False)
        # Sample comparison set T_dom of size t_dom
        t_dom_set = rng.choice(N, size=min(t_dom, N), replace=False)

        winner = pareto_tournament_select(
            candidate_a_idx=cand_a,
            candidate_b_idx=cand_b,
            t_dom_indices=t_dom_set,
            F=F,
            CV=CV,
            F_norm=F_norm,
            sigma_share=sigma_share,
            feas_tol=feas_tol,
            rng=rng,
        )
        mating_pool_indices.append(winner)

    mating_pool = X[mating_pool_indices]

    # 2. Crossover and mutation
    offspring_list = []
    is_perm = (problem.var_type == "permutation")

    for i in range(0, N, 2):
        p1 = mating_pool[i]
        p2 = mating_pool[(i + 1) % N]

        if is_perm:
            c1, c2 = order_crossover(p1, p2, rng)
            c1 = swap_mutation(c1, rng)
            c2 = swap_mutation(c2, rng)
        else:
            c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, prob_c=prob_c, rng=rng)
            c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m, rng=rng)
            c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=eta_m, rng=rng)

            if problem.var_type == "integer":
                c1 = np.round(c1)
                c2 = np.round(c2)
            elif problem.var_type == "binary":
                c1 = (c1 >= 0.5).astype(float)
                c2 = (c2 >= 0.5).astype(float)

        offspring_list.append(c1)
        if len(offspring_list) < N:
            offspring_list.append(c2)

    n_to_eval = min(len(offspring_list), min(N, rem_budget))
    X_off = np.array(offspring_list[:n_to_eval])

    # 3. Evaluate offspring
    F_off_list = []
    CV_off_list = []
    for ind in X_off:
        f, cv, _ = problem.evaluate(ind)
        F_off_list.append(f)
        CV_off_list.append(cv)

    F_off = np.array(F_off_list, dtype=float)
    CV_off = np.array(CV_off_list, dtype=float)

    # 4. Generational replacement
    if len(X_off) == N:
        state["X"] = X_off
        state["F"] = F_off
        state["CV"] = CV_off
    else:
        state["X"][:n_to_eval] = X_off
        state["F"][:n_to_eval] = F_off
        state["CV"][:n_to_eval] = CV_off

    state["n_evals"] += n_to_eval
    state["iter"] += 1

    # Update non-dominated archive
    combined_X = np.vstack((state["archive_X"], X_off))
    combined_F = np.vstack((state["archive_F"], F_off))
    combined_CV = np.concatenate((state["archive_CV"], CV_off))

    arch_X, arch_F, arch_CV = extract_non_dominated(combined_X, combined_F, combined_CV, feas_tol=feas_tol)
    if len(arch_X) > 200:
        keep = rng.choice(len(arch_X), size=200, replace=False)
        arch_X, arch_F, arch_CV = arch_X[keep], arch_F[keep], arch_CV[keep]

    state["archive_X"] = arch_X
    state["archive_F"] = arch_F
    state["archive_CV"] = arch_CV

    return state


def run(
    problem: Problem,
    budget: int = 10000,
    seed: int = 42,
    pop_size: int = 100,
    t_dom: int = 10,
    sigma_share: float = 0.1,
    **kwargs,
) -> dict:
    """Runs NPGA algorithm up to the evaluation budget."""
    state = initialize(problem, seed=seed, pop_size=pop_size, t_dom=t_dom, sigma_share=sigma_share, budget=budget, **kwargs)
    state["budget"] = budget

    while state["n_evals"] < budget and state["pop_size"] >= 2:
        prev_evals = state["n_evals"]
        step(state, problem)
        if state["n_evals"] == prev_evals:
            break

    X_opt, F_opt, CV_opt = extract_non_dominated(
        state["archive_X"], state["archive_F"], state["archive_CV"], feas_tol=problem.feas_tol
    )
    F_opt_nat = problem.to_natural(F_opt)

    return {
        "X_opt": X_opt,
        "F_opt": F_opt,
        "F_opt_nat": F_opt_nat,
        "CV_opt": CV_opt,
        "n_evals": state["n_evals"],
        "state": state,
    }
