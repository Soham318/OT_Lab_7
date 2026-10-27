"""
alg28_motlbo.py: MO-TLBO (Multi-Objective Teaching-Learning-Based Optimization)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Rao, R. V., & Patel, V. (2012).
"Multi-Objective Teaching-Learning-Based Optimization Algorithm for Unconstrained and Constrained Optimization Problems."
International Journal of Industrial Engineering Computations, 3(4), 527-544.

Algorithmic Logic:
1. Class Population Initialization:
       Initialize learner positions X_i (class size N) in [xl, xu].
       Maintain an External Non-Dominated Archive.
2. Teacher Selection:
       Teacher X_teacher is selected from the non-dominated archive (or lowest rank/highest crowding).
3. Teacher Phase (Classroom Instruction):
       Compute class mean M_class = mean(X, axis=0).
       Teaching factor T_F in {1, 2}.
       X_new,i = clip(X_i + r * (X_teacher - T_F * M_class), xl, xu).
       Greedy replacement via constrained Pareto dominance.
4. Learner Phase (Peer Interaction):
       For learner i, pick random peer j != i:
           If X_i dominates X_j: X_new,i = clip(X_i + r * (X_i - X_j), xl, xu)
           If X_j dominates X_i: X_new,i = clip(X_i + r * (X_j - X_i), xl, xu)
           Else: X_new,i = clip(X_i + r * (X_i - X_j) * sgn, xl, xu)
       Greedy replacement via constrained Pareto dominance.
5. Archive Pruning:
       External non-dominated archive updated after every phase.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates, crowding_distance
from moo.algorithms.common import random_population, extract_non_dominated, swap_mutation, order_crossover


def select_teacher(
    arch_X: np.ndarray,
    arch_F: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Selects the Teacher from the non-dominated archive using crowding distance."""
    N_arch = len(arch_X)
    if N_arch <= 2:
        return arch_X[rng.choice(N_arch)]

    cd = crowding_distance(arch_F, list(range(N_arch)))
    cd_clean = np.where(cd >= 1e10, 2.0, cd)
    total_cd = np.sum(cd_clean)
    if total_cd <= 1e-12:
        return arch_X[rng.choice(N_arch)]

    probs = cd_clean / total_cd
    chosen_idx = rng.choice(N_arch, p=probs)
    return arch_X[chosen_idx]


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MO-TLBO learners, archive, and evaluations within budget."""
    rng = np.random.default_rng(seed)
    class_size = kwargs.get("class_size", kwargs.get("pop_size", 40))
    budget = kwargs.get("budget", 10000)
    archive_size = kwargs.get("archive_size", 100)

    effective_size = max(2, min(class_size, budget))
    X = random_population(problem, effective_size, rng)

    F_list = []
    CV_list = []
    for ind in X:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)

    arch_X, arch_F, arch_CV = extract_non_dominated(X, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "class_size": effective_size,
        "budget": budget,
        "archive_size": archive_size,
        "X": X,
        "F": F,
        "CV": CV,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_size,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one cycle of MO-TLBO (Teacher Phase + Learner Phase)."""
    budget = state.get("budget", 1000000)
    rng = state["rng"]
    N = state["class_size"]
    n_var = problem.n_var
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    archive_size = state["archive_size"]

    is_perm = (problem.var_type == "permutation")
    new_candidates_X = []
    new_candidates_F = []
    new_candidates_CV = []

    # 1. Teacher Phase
    if len(arch_X) > 0:
        teacher = select_teacher(arch_X, arch_F, rng)
    else:
        teacher = X[0]

    if not is_perm:
        mean_class = np.mean(X, axis=0)

    for i in range(N):
        if state["n_evals"] >= budget:
            break

        if is_perm:
            cand_X, _ = order_crossover(X[i], teacher, rng)
            cand_X = swap_mutation(cand_X, rng, prob_m=0.2)
        else:
            T_F = rng.choice([1, 2])
            r = rng.uniform(0.0, 1.0, size=n_var)
            diff_mean = r * (teacher - T_F * mean_class)
            cand_X = np.clip(X[i] + diff_mean, problem.xl, problem.xu)
            if problem.var_type == "integer":
                cand_X = np.round(cand_X)
            elif problem.var_type == "binary":
                cand_X = (cand_X >= 0.5).astype(float)

        cand_F, cand_CV, _ = problem.evaluate(cand_X)
        state["n_evals"] += 1

        new_candidates_X.append(cand_X)
        new_candidates_F.append(cand_F)
        new_candidates_CV.append(cand_CV)

        # Greedy replacement
        cand_dom_cur = constrained_dominates(cand_F, cand_CV, F[i], CV[i], feas_tol=feas_tol)
        cur_dom_cand = constrained_dominates(F[i], CV[i], cand_F, cand_CV, feas_tol=feas_tol)

        if cand_dom_cur:
            X[i] = cand_X
            F[i] = cand_F
            CV[i] = cand_CV
        elif not cur_dom_cand and rng.random() < 0.5:
            X[i] = cand_X
            F[i] = cand_F
            CV[i] = cand_CV

    # 2. Learner Phase
    for i in range(N):
        if state["n_evals"] >= budget:
            break

        j = rng.choice([idx for idx in range(N) if idx != i])
        i_dom_j = constrained_dominates(F[i], CV[i], F[j], CV[j], feas_tol=feas_tol)
        j_dom_i = constrained_dominates(F[j], CV[j], F[i], CV[i], feas_tol=feas_tol)

        if is_perm:
            partner = X[i] if i_dom_j else X[j]
            cand_X, _ = order_crossover(X[i], partner, rng)
            cand_X = swap_mutation(cand_X, rng, prob_m=0.2)
        else:
            r = rng.uniform(0.0, 1.0, size=n_var)
            if i_dom_j:
                cand_X = X[i] + r * (X[i] - X[j])
            elif j_dom_i:
                cand_X = X[i] + r * (X[j] - X[i])
            else:
                sgn = 1.0 if rng.random() < 0.5 else -1.0
                cand_X = X[i] + r * (X[i] - X[j]) * sgn

            cand_X = np.clip(cand_X, problem.xl, problem.xu)
            if problem.var_type == "integer":
                cand_X = np.round(cand_X)
            elif problem.var_type == "binary":
                cand_X = (cand_X >= 0.5).astype(float)

        cand_F, cand_CV, _ = problem.evaluate(cand_X)
        state["n_evals"] += 1

        new_candidates_X.append(cand_X)
        new_candidates_F.append(cand_F)
        new_candidates_CV.append(cand_CV)

        cand_dom_cur = constrained_dominates(cand_F, cand_CV, F[i], CV[i], feas_tol=feas_tol)
        cur_dom_cand = constrained_dominates(F[i], CV[i], cand_F, cand_CV, feas_tol=feas_tol)

        if cand_dom_cur:
            X[i] = cand_X
            F[i] = cand_F
            CV[i] = cand_CV
        elif not cur_dom_cand and rng.random() < 0.5:
            X[i] = cand_X
            F[i] = cand_F
            CV[i] = cand_CV

    # 3. Update Archive
    if len(new_candidates_X) > 0:
        c_X = np.array(new_candidates_X)
        c_F = np.array(new_candidates_F)
        c_CV = np.array(new_candidates_CV)

        comb_arch_X = np.vstack((arch_X, c_X)) if len(arch_X) > 0 else c_X
        comb_arch_F = np.vstack((arch_F, c_F)) if len(arch_F) > 0 else c_F
        comb_arch_CV = np.concatenate((arch_CV, c_CV)) if len(arch_CV) > 0 else c_CV

        new_arch_X, new_arch_F, new_arch_CV = extract_non_dominated(
            comb_arch_X, comb_arch_F, comb_arch_CV, feas_tol=feas_tol
        )
        if len(new_arch_X) > archive_size:
            cd = crowding_distance(new_arch_F, list(range(len(new_arch_F))))
            order = np.argsort(-cd)
            keep = order[:archive_size]
            new_arch_X = new_arch_X[keep]
            new_arch_F = new_arch_F[keep]
            new_arch_CV = new_arch_CV[keep]

        state["arch_X"] = new_arch_X
        state["arch_F"] = new_arch_F
        state["arch_CV"] = new_arch_CV

    state["X"] = X
    state["F"] = F
    state["CV"] = CV
    state["iter"] += 1

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MO-TLBO until budget is exhausted."""
    state = initialize(problem, seed=seed, budget=budget, **kwargs)

    while state["n_evals"] < budget:
        state = step(state, problem)

    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]

    is_feas = (arch_CV <= problem.feas_tol)
    if np.any(is_feas):
        opt_X = arch_X[is_feas]
        opt_F = arch_F[is_feas]
        opt_CV = arch_CV[is_feas]
    else:
        opt_X = arch_X
        opt_F = arch_F
        opt_CV = arch_CV

    return {
        "X_opt": opt_X,
        "F_opt": opt_F,
        "CV_opt": opt_CV,
        "n_evals": state["n_evals"],
        "history": [],
        "state": state,
    }
