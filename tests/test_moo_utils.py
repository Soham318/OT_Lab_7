"""
test_moo_utils.py: Unit tests for shared multiobjective toolkit (moo_utils.py)
"""

import math
import numpy as np
from moo.moo_utils import (
    dominates,
    constrained_dominates,
    fast_non_dominated_sort,
    crowding_distance,
    hypervolume_2d,
    hypervolume_monte_carlo,
    hypervolume,
    igd,
    spacing,
    normalize_objectives,
    das_dennis,
)


def test_dominates() -> None:
    # f1 strictly dominates f2
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 3.0])
    assert dominates(a, b) is True
    assert dominates(b, a) is False

    # Equal solutions do not strictly dominate
    assert dominates(a, a) is False

    # Weak dominance in one, equal in another
    c = np.array([1.0, 3.0])
    assert dominates(a, c) is True
    assert dominates(c, a) is False

    # Trade-off solutions (incomparable)
    d = np.array([0.5, 4.0])
    assert dominates(a, d) is False
    assert dominates(d, a) is False


def test_constrained_dominates_debs_rules() -> None:
    feas_tol = 1e-4

    # Case 1: Feasible vs Infeasible
    f_feas = np.array([5.0, 5.0])
    cv_feas = 0.0
    f_infeas = np.array([1.0, 1.0])  # Objectively better, but infeasible
    cv_infeas = 2.0

    assert constrained_dominates(f_feas, cv_feas, f_infeas, cv_infeas, feas_tol=feas_tol) is True
    assert constrained_dominates(f_infeas, cv_infeas, f_feas, cv_feas, feas_tol=feas_tol) is False

    # Case 2: Both Infeasible: smaller CV wins
    cv_better = 1.5
    cv_worse = 3.5
    assert constrained_dominates(f_infeas, cv_better, f_infeas, cv_worse, feas_tol=feas_tol) is True
    assert constrained_dominates(f_infeas, cv_worse, f_infeas, cv_better, feas_tol=feas_tol) is False

    # Case 3: Both Feasible: Pareto dominance on objectives
    f_good = np.array([2.0, 2.0])
    f_bad = np.array([3.0, 4.0])
    assert constrained_dominates(f_good, 0.0, f_bad, 0.0, feas_tol=feas_tol) is True
    assert constrained_dominates(f_bad, 0.0, f_good, 0.0, feas_tol=feas_tol) is False

    # Boundary check on feasibility tolerance:
    # cv = 1e-4 is feasible, cv = 2e-4 is infeasible
    assert constrained_dominates(f_bad, 1e-4, f_good, 2e-4, feas_tol=feas_tol) is True


def test_fast_non_dominated_sort() -> None:
    # 6 points benchmark:
    # P0: [1.0, 5.0]
    # P1: [2.0, 3.0]
    # P2: [4.0, 2.0]
    # P3: [3.0, 4.0] (dominated by P1 and P5)
    # P4: [5.0, 5.0] (dominated by P0, P1, P2, P3, P5)
    # P5: [1.5, 4.0]
    F = np.array(
        [
            [1.0, 5.0],  # 0
            [2.0, 3.0],  # 1
            [4.0, 2.0],  # 2
            [3.0, 4.0],  # 3
            [5.0, 5.0],  # 4
            [1.5, 4.0],  # 5
        ]
    )

    fronts = fast_non_dominated_sort(F, use_deb=False)
    # Front 0 should contain {0, 1, 2, 5}
    assert set(fronts[0]) == {0, 1, 2, 5}
    # Front 1 should contain {3}
    assert set(fronts[1]) == {3}
    # Front 2 should contain {4}
    assert set(fronts[2]) == {4}


def test_crowding_distance() -> None:
    # 3 points along f1 + f2 = 6
    F = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
        ]
    )
    front = [0, 1, 2]
    cd = crowding_distance(F, front)
    # Extrema get 1e14
    assert cd[0] >= 1e14
    assert cd[2] >= 1e14
    # Middle point calculation:
    # diff_f1 = 3 - 1 = 2, diff_f2 = 5 - 3 = 2
    # contrib_f1 = (3 - 1) / 2 = 1.0
    # contrib_f2 = (5 - 3) / 2 = 1.0
    # sum = 2.0
    assert math.isclose(cd[1], 2.0, abs_tol=1e-6)


def test_hypervolume_2d_hand_calculated() -> None:
    # Hand-calculated example:
    # Ref point: (10, 10)
    # Points: (2, 8) and (5, 4)
    # Expected rectangular area:
    # Slice 1: x in [2, 5], height = 10 - 8 = 2  ==> Area = 3 * 2 = 6
    # Slice 2: x in [5, 10], height = 10 - 4 = 6 ==> Area = 5 * 6 = 30
    # Total Area = 36.0
    ref_point = np.array([10.0, 10.0])
    F = np.array(
        [
            [2.0, 8.0],
            [5.0, 4.0],
        ]
    )
    hv = hypervolume_2d(F, ref_point)
    assert math.isclose(hv, 36.0, abs_tol=1e-8)


def test_hypervolume_monte_carlo() -> None:
    # Single point at origin in 3D: [0, 0, 0], ref point at [1, 1, 1]
    # Dominated volume is exactly 1.0
    F = np.array([[0.0, 0.0, 0.0]])
    ref_point = np.array([1.0, 1.0, 1.0])
    hv_mc = hypervolume_monte_carlo(F, ref_point, n_samples=50000, seed=42)
    assert math.isclose(hv_mc, 1.0, abs_tol=0.01)

    # Test dispatcher
    hv_disp = hypervolume(F, ref_point, n_samples=50000, seed=42)
    assert math.isclose(hv_disp, hv_mc, abs_tol=1e-9)


def test_igd() -> None:
    # True front: [[0, 1], [1, 0]]
    true_F = np.array([[0.0, 1.0], [1.0, 0.0]])
    # Exact match: IGD should be 0.0
    assert math.isclose(igd(true_F, true_F), 0.0, abs_tol=1e-9)

    # Shifted obtained front by 0.1 along each axis
    obtained_F = np.array([[0.1, 1.0], [1.0, 0.1]])
    # Nearest neighbor distances are both 0.1
    assert math.isclose(igd(obtained_F, true_F), 0.1, abs_tol=1e-8)


def test_spacing() -> None:
    # Perfectly equidistant front
    F = np.array(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
        ]
    )
    # Manhattan distance between each adjacent pair is |1 - 0| + |2 - 3| = 2.0
    # All distances are identical, so variance = 0 ==> Spacing = 0.0
    s = spacing(F)
    assert math.isclose(s, 0.0, abs_tol=1e-8)


def test_normalize_objectives() -> None:
    F = np.array([[2.0, 10.0], [6.0, 20.0], [10.0, 30.0]])
    ideal = np.array([2.0, 10.0])
    nadir = np.array([10.0, 30.0])
    F_norm = normalize_objectives(F, ideal, nadir)
    expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(F_norm, expected, atol=1e-8)


def test_das_dennis() -> None:
    # p = 4, M = 3
    # Number of points = binom(4 + 3 - 1, 3 - 1) = binom(6, 2) = 15
    pts = das_dennis(4, 3)
    assert pts.shape == (15, 3)
    # Check that each vector sums to 1.0 and is non-negative
    sums = np.sum(pts, axis=1)
    np.testing.assert_allclose(sums, np.ones(15), atol=1e-8)
    assert np.all(pts >= 0.0)


def test_dominates_multidimensional_and_edge_cases() -> None:
    # M = 1
    assert dominates(np.array([1.0]), np.array([2.0])) is True
    assert dominates(np.array([2.0]), np.array([1.0])) is False
    assert dominates(np.array([1.0]), np.array([1.0])) is False

    # M = 3
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 2.0, 4.0])
    assert dominates(v1, v2) is True
    assert dominates(v2, v1) is False
    assert dominates(v1, v1) is False

    # M = 5
    a5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b5 = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    assert dominates(a5, b5) is True
    assert dominates(b5, a5) is False

    # M = 10
    a10 = np.ones(10)
    b10 = np.full(10, 2.0)
    assert dominates(a10, b10) is True
    assert dominates(b10, a10) is False

    # Empty vector edge case
    assert dominates(np.array([]), np.array([])) is False

    # Length mismatch raises ValueError
    try:
        dominates(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_constrained_dominates_threshold_and_boundary_cases() -> None:
    feas_tol = 1e-4
    f1 = np.array([10.0, 10.0])
    f2 = np.array([1.0, 1.0])

    # Feasible boundary: CV = 1.0e-4 (feasible) vs 1.0001e-4 (infeasible)
    # Even though f1 has worse objectives, it constrained-dominates f2 because f1 is feasible
    assert constrained_dominates(f1, 1.0e-4, f2, 1.0001e-4, feas_tol=feas_tol) is True
    assert constrained_dominates(f2, 1.0001e-4, f1, 1.0e-4, feas_tol=feas_tol) is False

    # Negative CV inputs are safely clamped to 0.0
    assert constrained_dominates(f1, -5.0, f2, 1.0, feas_tol=feas_tol) is True

    # Both infeasible with exact identical CV: neither dominates
    assert constrained_dominates(f1, 2.0, f2, 2.0, feas_tol=feas_tol) is False
    assert constrained_dominates(f2, 2.0, f1, 2.0, feas_tol=feas_tol) is False

    # Both infeasible with tiny difference: strictly smaller CV wins
    assert constrained_dominates(f1, 0.50000, f2, 0.50001, feas_tol=feas_tol) is True
    assert constrained_dominates(f2, 0.50001, f1, 0.50000, feas_tol=feas_tol) is False


def test_fast_non_dominated_sort_structural_properties() -> None:
    # 1. Full index partition check: every index appears exactly once across fronts
    rng = np.random.default_rng(123)
    F = rng.uniform(0.0, 10.0, size=(30, 3))
    fronts = fast_non_dominated_sort(F, use_deb=False)
    all_indices: list[int] = []
    for front in fronts:
        all_indices.extend(front)
    assert sorted(all_indices) == list(range(30)), "Every index must be assigned to exactly one front"

    # 2. Strict dominance chain: P0 < P1 < P2 < P3 < P4
    chain = np.array([[float(i), float(i)] for i in range(5)])
    chain_fronts = fast_non_dominated_sort(chain, use_deb=False)
    assert len(chain_fronts) == 5
    for i in range(5):
        assert chain_fronts[i] == [i]

    # 3. All identical points: all enter Front 0 together
    identical = np.tile([2.0, 3.0], (6, 1))
    ident_fronts = fast_non_dominated_sort(identical, use_deb=False)
    assert len(ident_fronts) == 1
    assert set(ident_fronts[0]) == set(range(6))

    # 4. Constrained sorting: feasible solutions partitioned first
    F_mix = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
    CV_mix = np.array([0.0, 0.0, 1.0])  # point 2 is objectively best but infeasible
    c_fronts = fast_non_dominated_sort(F_mix, CV=CV_mix, use_deb=True)
    assert 0 in c_fronts[0]  # feasible point 0
    assert 1 in c_fronts[1]  # feasible point 1 (dominated by 0)
    assert 2 in c_fronts[2]  # infeasible point relegated to last front


def test_crowding_distance_edge_cases() -> None:
    # 0 points
    cd_0 = crowding_distance(np.empty((0, 2)), [])
    assert len(cd_0) == 0

    # 1 point
    cd_1 = crowding_distance(np.array([[1.0, 2.0]]), [0])
    assert len(cd_1) == 1
    assert cd_1[0] >= 1e14

    # 2 points
    cd_2 = crowding_distance(np.array([[1.0, 5.0], [2.0, 3.0]]), [0, 1])
    assert len(cd_2) == 2
    assert cd_2[0] >= 1e14
    assert cd_2[1] >= 1e14

    # Constant objective across front: objective 1 is 5.0 for all 4 points
    F_const = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    cd_const = crowding_distance(F_const, [0, 1, 2, 3])
    assert not np.isnan(cd_const).any(), "No NaN values allowed"
    assert cd_const[0] >= 1e14  # Min extremum of objective 0
    assert cd_const[3] >= 1e14  # Max extremum of objective 0
    assert cd_const[1] > 0.0 and cd_const[1] < 1e14
    assert cd_const[2] > 0.0 and cd_const[2] < 1e14

    # All objectives constant across front
    F_all_const = np.tile([2.0, 4.0], (4, 1))
    cd_all_const = crowding_distance(F_all_const, [0, 1, 2, 3])
    assert not np.isnan(cd_all_const).any()


def test_hypervolume_2d_advanced_hand_calculated() -> None:
    ref_point = np.array([10.0, 10.0])

    # 3-point benchmark:
    # P1=(1, 9), P2=(3, 6), P3=(6, 2)
    # Area:
    # [1, 3] x [9, 10]: 2 * 1 = 2
    # [3, 6] x [6, 10]: 3 * 4 = 12
    # [6, 10] x [2, 10]: 4 * 8 = 32
    # Total = 46.0
    pts_clean = np.array([[1.0, 9.0], [3.0, 6.0], [6.0, 2.0]])
    hv_clean = hypervolume_2d(pts_clean, ref_point)
    assert math.isclose(hv_clean, 46.0, abs_tol=1e-8)

    # Candidate set containing dominated points (e.g. (4, 8) dominated by (3, 6))
    pts_with_dominated = np.array([[1.0, 9.0], [3.0, 6.0], [6.0, 2.0], [4.0, 8.0], [5.0, 7.0]])
    hv_dom = hypervolume_2d(pts_with_dominated, ref_point)
    assert math.isclose(hv_dom, 46.0, abs_tol=1e-8)

    # Candidate set containing points outside the reference box
    pts_with_outside = np.array([[1.0, 9.0], [3.0, 6.0], [6.0, 2.0], [11.0, 1.0], [2.0, 12.0]])
    hv_out = hypervolume_2d(pts_with_outside, ref_point)
    assert math.isclose(hv_out, 46.0, abs_tol=1e-8)

    # Single point
    hv_single = hypervolume_2d(np.array([[3.0, 4.0]]), ref_point)
    assert math.isclose(hv_single, (10.0 - 3.0) * (10.0 - 4.0), abs_tol=1e-8)

    # 1D array single point input
    hv_1d = hypervolume_2d(np.array([3.0, 4.0]), ref_point)
    assert math.isclose(hv_1d, 42.0, abs_tol=1e-8)

    # Empty candidate set
    assert hypervolume_2d(np.empty((0, 2)), ref_point) == 0.0


def test_hypervolume_monte_carlo_edge_cases() -> None:
    ref = np.array([1.0, 1.0, 1.0])
    # Empty set
    assert hypervolume_monte_carlo(np.empty((0, 3)), ref) == 0.0

    # Invalid reference point (behind points)
    assert hypervolume_monte_carlo(np.array([[2.0, 2.0, 2.0]]), ref) == 0.0

    # Determinism with fixed seed
    F = np.array([[0.2, 0.3, 0.4], [0.4, 0.2, 0.3]])
    hv1 = hypervolume_monte_carlo(F, ref, n_samples=20000, seed=77)
    hv2 = hypervolume_monte_carlo(F, ref, n_samples=20000, seed=77)
    assert hv1 == hv2


def test_igd_and_spacing_edge_cases() -> None:
    # Empty inputs to IGD
    assert math.isinf(igd(np.empty((0, 2)), np.array([[1.0, 1.0]])))
    assert math.isinf(igd(np.array([[1.0, 1.0]]), np.empty((0, 2))))

    # Empty input to Spacing
    assert spacing(np.empty((0, 2))) == 0.0

    # Singleton input to Spacing
    assert spacing(np.array([[1.0, 2.0]])) == 0.0

    # Two points input to Spacing (equidistant by definition)
    assert spacing(np.array([[1.0, 3.0], [3.0, 1.0]])) == 0.0


def test_normalize_objectives_edge_cases() -> None:
    # Constant objective column (nadir == ideal)
    F = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    F_norm = normalize_objectives(F)
    assert not np.isnan(F_norm).any()
    np.testing.assert_allclose(F_norm[:, 1], [0.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(F_norm[:, 0], [0.0, 0.5, 1.0], atol=1e-8)

    # Empty array
    assert len(normalize_objectives(np.empty((0, 2)))) == 0
