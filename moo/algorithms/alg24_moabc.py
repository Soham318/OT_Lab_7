"""
alg24_moabc.py: MOABC (Multi-Objective Artificial Bee Colony Algorithm)

Lecture Family: Constrained, Swarm & Nature-Inspired Multiobjective Optimization (Part 4, Chapter 5)
Original Citation: Akbari, R., Hedayatzadeh, R., Ziarati, K., & Hassanizadeh, B. (2012).
"A Multi-Objective Artificial Bee Colony Algorithm."
Swarm and Evolutionary Computation, 2, 39-52.

Algorithmic Logic:
1. Colony Initialization:
       Initialize food source positions X_i (employed bees) within [xl, xu].
       Evaluate food sources and initialize trial counters to 0.
       Initialize External Pareto Archive with non-dominated food sources.
2. Employed Bee Phase:
       Each employed bee explores around food source X_i:
           v_{ij} = x_{ij} + phi_{ij} * (x_{ij} - x_{kj}),  k != i, phi in [-1, 1]
       Greedy selection via Pareto constrained dominance:
           If v_i dominates X_i -> X_i = v_i, reset trial_i = 0.
           If X_i dominates v_i -> trial_i += 1.
           If non-dominating -> replace with 50% probability, reset trial.
3. Onlooker Bee Phase:
       Onlooker bees select food sources via roulette wheel based on Pareto dominance rank:
           P_i = (1 / (1 + rank_i)) / sum_j (1 / (1 + rank_j))
       Explores neighbor v_i around chosen food source; updates via constrained dominance.
4. Scout Bee Phase:
       If trial_i exceeds limit, food source i is abandoned and re-initialized randomly by a scout bee.
5. External Pareto Repository:
       Continually updated with all evaluated candidates; non-dominated filtering and grid pruning.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem
from moo.moo_utils import fast_non_dominated_sort, constrained_dominates
from moo.algorithms.common import random_population, extract_non_dominated, swap_mutation, order_crossover
from moo.algorithms.alg22_mopso import prune_repository


def initialize(problem: Problem, seed: int = 42, **kwargs) -> dict:
    """Initializes MOABC food sources, trial counters, and archive within budget."""
    rng = np.random.default_rng(seed)
    n_food = kwargs.get("n_food", kwargs.get("pop_size", 40))
    budget = kwargs.get("budget", 10000)
    archive_size = kwargs.get("archive_size", 100)
    limit = kwargs.get("limit", max(5, int(0.5 * problem.n_var * n_food)))

    effective_food = max(1, min(n_food, budget))
    X = random_population(problem, effective_food, rng)

    F_list = []
    CV_list = []
    for ind in X:
        f, cv, _ = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list, dtype=float)
    CV = np.array(CV_list, dtype=float)
    trials = np.zeros(effective_food, dtype=int)

    arch_X, arch_F, arch_CV = extract_non_dominated(X, F, CV, feas_tol=problem.feas_tol)

    state = {
        "seed": seed,
        "rng": rng,
        "n_food": effective_food,
        "archive_size": archive_size,
        "limit": limit,
        "X": X,
        "F": F,
        "CV": CV,
        "trials": trials,
        "arch_X": arch_X,
        "arch_F": arch_F,
        "arch_CV": arch_CV,
        "n_evals": effective_food,
        "budget": budget,
        "iter": 0,
    }
    return state


def step(state: dict, problem: Problem) -> dict:
    """Executes one cycle of MOABC (Employed, Onlooker, and Scout phases)."""
    budget = state.get("budget", 1000000)
    rng = state["rng"]
    N = state["n_food"]
    n_var = problem.n_var
    feas_tol = problem.feas_tol

    X = state["X"]
    F = state["F"]
    CV = state["CV"]
    trials = state["trials"]
    arch_X = state["arch_X"]
    arch_F = state["arch_F"]
    arch_CV = state["arch_CV"]
    archive_size = state["archive_size"]
    limit = state["limit"]

    is_perm = (problem.var_type == "permutation")
    new_candidates_X = []
    new_candidates_F = []
    new_candidates_CV = []

    # 1. Employed Bee Phase
    for i in range(N):
        if state["n_evals"] >= budget:
            break

        k = rng.choice([idx for idx in range(N) if idx != i]) if N > 1 else i
        if is_perm:
            v_i = swap_mutation(X[i], rng, prob_m=0.3)
        else:
            j = rng.integers(0, n_var)
            phi = rng.uniform(-1.0, 1.0)
            v_i = X[i].copy()
            v_i[j] = X[i][j] + phi * (X[i][j] - X[k][j])
            v_i = np.clip(v_i, problem.xl, problem.xu)
            if problem.var_type == "integer":
                v_i = np.round(v_i)
            elif problem.var_type == "binary":
                v_i = (v_i >= 0.5).astype(float)

        f_v, cv_v, _ = problem.evaluate(v_i)
        state["n_evals"] += 1
        new_candidates_X.append(v_i)
        new_candidates_F.append(f_v)
        new_candidates_CV.append(cv_v)

        # Greedy selection
        v_dom_x = constrained_dominates(f_v, cv_v, F[i], CV[i], feas_tol=feas_tol)
        x_dom_v = constrained_dominates(F[i], CV[i], f_v, cv_v, feas_tol=feas_tol)

        if v_dom_x:
            X[i] = v_i
            F[i] = f_v
            CV[i] = cv_v
            trials[i] = 0
        elif x_dom_v:
            trials[i] += 1
        else:
            if rng.random() < 0.5:
                X[i] = v_i
                F[i] = f_v
                CV[i] = cv_v
            trials[i] = 0

    # 2. Onlooker Bee Phase
    if state["n_evals"] < budget and N > 1:
        # Pareto ranking of current food sources
        fronts = fast_non_dominated_sort(F, CV=CV, use_deb=True, feas_tol=feas_tol)
        ranks = np.zeros(N, dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = r

        fit = 1.0 / (1.0 + ranks)
        probs = fit / np.sum(fit)

        for _ in range(N):
            if state["n_evals"] >= budget:
                break

            i = rng.choice(N, p=probs)
            k = rng.choice([idx for idx in range(N) if idx != i])

            if is_perm:
                v_i = swap_mutation(X[i], rng, prob_m=0.3)
            else:
                j = rng.integers(0, n_var)
                phi = rng.uniform(-1.0, 1.0)
                v_i = X[i].copy()
                v_i[j] = X[i][j] + phi * (X[i][j] - X[k][j])
                v_i = np.clip(v_i, problem.xl, problem.xu)
                if problem.var_type == "integer":
                    v_i = np.round(v_i)
                elif problem.var_type == "binary":
                    v_i = (v_i >= 0.5).astype(float)

            f_v, cv_v, _ = problem.evaluate(v_i)
            state["n_evals"] += 1
            new_candidates_X.append(v_i)
            new_candidates_F.append(f_v)
            new_candidates_CV.append(cv_v)

            v_dom_x = constrained_dominates(f_v, cv_v, F[i], CV[i], feas_tol=feas_tol)
            x_dom_v = constrained_dominates(F[i], CV[i], f_v, cv_v, feas_tol=feas_tol)

            if v_dom_x:
                X[i] = v_i
                F[i] = f_v
                CV[i] = cv_v
                trials[i] = 0
            elif x_dom_v:
                trials[i] += 1
            else:
                if rng.random() < 0.5:
                    X[i] = v_i
                    F[i] = f_v
                    CV[i] = cv_v
                trials[i] = 0

    # 3. Scout Bee Phase
    for i in range(N):
        if state["n_evals"] >= budget:
            break
        if trials[i] > limit:
            # Abandon food source and reinitialize
            new_ind = random_population(problem, 1, rng)[0]
            f_new, cv_new, _ = problem.evaluate(new_ind)
            state["n_evals"] += 1

            X[i] = new_ind
            F[i] = f_new
            CV[i] = cv_new
            trials[i] = 0

            new_candidates_X.append(new_ind)
            new_candidates_F.append(f_new)
            new_candidates_CV.append(cv_new)

    # 4. Update Archive
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
            new_arch_X, new_arch_F, new_arch_CV = prune_repository(
                new_arch_X, new_arch_F, new_arch_CV, archive_size, n_div=10, rng=rng
            )

        state["arch_X"] = new_arch_X
        state["arch_F"] = new_arch_F
        state["arch_CV"] = new_arch_CV

    state["X"] = X
    state["F"] = F
    state["CV"] = CV
    state["trials"] = trials
    state["iter"] += 1

    return state


def run(problem: Problem, budget: int = 10000, seed: int = 42, **kwargs) -> dict:
    """Runs MOABC until budget is exhausted."""
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
