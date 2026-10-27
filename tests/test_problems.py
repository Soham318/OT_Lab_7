"""
test_problems.py: Unit tests and sanity checks for benchmark problems P1 through P14.
"""

import math
import numpy as np

from moo.problems import (
    get_problem,
    FactorySpeedProblem,
    FleetSizingProblem,
    ZDT2Problem,
    PortfolioProblem,
    CantileverBeamProblem,
    BinhKornProblem,
    WaterAllocationProblem,
    DeliveryRoutingTSPProblem,
    FeatureSelectionProblem,
    DTLZ2Problem,
    ProductionPlanningProblem,
    SupplierSelectionProblem,
    MultiTaskToyProblem,
    SharedMultiTaskNetwork,
    LaptopDesignProblem,
)
from moo.problems.p13_multitask import generate_synthetic_multitask_data


def test_p01_factory_speed() -> None:
    prob = FactorySpeedProblem()
    # At x = 5: f1 = (5 - 3)^2 = 4, f2 = (5 - 7)^2 = 4
    f, cv, _ = prob.evaluate(np.array([5.0]))
    assert math.isclose(f[0], 4.0, abs_tol=1e-8)
    assert math.isclose(f[1], 4.0, abs_tol=1e-8)
    assert cv == 0.0

    # At x = 3: f1 = 0, f2 = (3 - 7)^2 = 16
    f, cv, _ = prob.evaluate(np.array([3.0]))
    assert math.isclose(f[0], 0.0, abs_tol=1e-8)
    assert math.isclose(f[1], 16.0, abs_tol=1e-8)
    assert cv == 0.0

    # True front check: f2 = (4 - sqrt(f1))^2
    front = prob.get_true_front(50)
    for f1, f2 in front:
        assert math.isclose(f2, (4.0 - math.sqrt(f1)) ** 2, abs_tol=1e-6)


def test_p02_fleet_sizing() -> None:
    prob = FleetSizingProblem(is_integer=False)
    # x = 2 gives (50, 4)
    f, cv, _ = prob.evaluate(np.array([2.0]))
    assert math.isclose(f[0], 50.0, abs_tol=1e-8)
    assert math.isclose(f[1], 4.0, abs_tol=1e-8)
    assert cv == 0.0

    # x = 10 gives (10, 100)
    f, cv, _ = prob.evaluate(np.array([10.0]))
    assert math.isclose(f[0], 10.0, abs_tol=1e-8)
    assert math.isclose(f[1], 100.0, abs_tol=1e-8)
    assert cv == 0.0

    # Integer variant
    prob_int = FleetSizingProblem(is_integer=True)
    f_int, _, _ = prob_int.evaluate(np.array([2.3]))
    assert math.isclose(f_int[0], 50.0, abs_tol=1e-8)  # Rounds to 2


def test_p03_zdt2() -> None:
    prob = ZDT2Problem(n_var=10)
    # True front has x2..x10 = 0 -> g = 1, f2 = 1 - f1^2
    x = np.zeros(10)
    x[0] = 0.5
    f, cv, _ = prob.evaluate(x)
    assert math.isclose(f[0], 0.5, abs_tol=1e-8)
    assert math.isclose(f[1], 0.75, abs_tol=1e-8)  # 1 - 0.5^2 = 0.75
    assert cv == 0.0


def test_p04_portfolio() -> None:
    prob = PortfolioProblem()
    # w = (0, 0, 1, 0) gives return 0.15 (int: -0.15) and risk 0.09
    w = np.array([0.0, 0.0, 1.0, 0.0])
    f_int, cv, _ = prob.evaluate(w)
    f_nat, _, _ = prob.evaluate_natural(w)

    assert math.isclose(f_nat[0], 0.09, abs_tol=1e-8)   # Risk
    assert math.isclose(f_nat[1], 0.15, abs_tol=1e-8)   # Return
    assert math.isclose(f_int[1], -0.15, abs_tol=1e-8)  # Internal negated return
    assert cv == 0.0

    # Infeasible: sum(w) = 2.0 -> CV = |2.0 - 1.0| - 1e-4
    w_bad = np.array([0.5, 0.5, 0.5, 0.5])
    _, cv_bad, _ = prob.evaluate(w_bad)
    assert cv_bad > 0.9


def test_p05_cantilever() -> None:
    prob = CantileverBeamProblem()
    # (b, h) = (0.01, 0.05) gives mass = 3.925 kg, defl = 0.016 m, stress = 240 MPa
    x = np.array([0.01, 0.05])
    f, cv, _ = prob.evaluate(x)
    assert math.isclose(f[0], 3.925, abs_tol=1e-4)
    assert math.isclose(f[1], 0.016, abs_tol=1e-4)
    assert cv == 0.0  # b*h^2 = 2.5e-5 >= 2.4e-5, so feasible

    # Infeasible design: b = 0.01, h = 0.02 -> b*h^2 = 4e-6 < 2.4e-5
    x_bad = np.array([0.01, 0.02])
    _, cv_bad, _ = prob.evaluate(x_bad)
    assert cv_bad > 1e-5


def test_p06_binh_korn() -> None:
    prob = BinhKornProblem()
    # (0, 0) gives (0, 50) and is feasible
    f0, cv0, _ = prob.evaluate(np.array([0.0, 0.0]))
    assert math.isclose(f0[0], 0.0, abs_tol=1e-8)
    assert math.isclose(f0[1], 50.0, abs_tol=1e-8)
    assert cv0 == 0.0

    # (5, 3) gives (136, 4) and is feasible
    f1, cv1, _ = prob.evaluate(np.array([5.0, 3.0]))
    assert math.isclose(f1[0], 136.0, abs_tol=1e-8)
    assert math.isclose(f1[1], 4.0, abs_tol=1e-8)
    assert cv1 == 0.0

    # Infeasible point: (0, 3) -> g1 = 25 + 9 - 25 = 9 > 0
    _, cv_bad, _ = prob.evaluate(np.array([0.0, 3.0]))
    assert cv_bad >= 9.0


def test_p07_water_allocation() -> None:
    prob = WaterAllocationProblem()
    # (10, 80) -> natural (8, 96), internal (-8, -96), feasible
    f_int0, cv0, _ = prob.evaluate(np.array([10.0, 80.0]))
    f_nat0, _, _ = prob.evaluate_natural(np.array([10.0, 80.0]))
    assert math.isclose(f_nat0[0], 8.0, abs_tol=1e-8)
    assert math.isclose(f_nat0[1], 96.0, abs_tol=1e-8)
    assert math.isclose(f_int0[0], -8.0, abs_tol=1e-8)
    assert math.isclose(f_int0[1], -96.0, abs_tol=1e-8)
    assert cv0 == 0.0

    # (70, 20) -> natural (56, 24), internal (-56, -24), feasible
    f_int1, cv1, _ = prob.evaluate(np.array([70.0, 20.0]))
    f_nat1, _, _ = prob.evaluate_natural(np.array([70.0, 20.0]))
    assert math.isclose(f_nat1[0], 56.0, abs_tol=1e-8)
    assert math.isclose(f_nat1[1], 24.0, abs_tol=1e-8)
    assert cv1 == 0.0

    # Infeasible: (60, 60) -> x1 + x2 = 120 > 90 -> CV >= 30
    _, cv_bad, _ = prob.evaluate(np.array([60.0, 60.0]))
    assert cv_bad >= 30.0


def test_p08_delivery_routing_tsp() -> None:
    prob = DeliveryRoutingTSPProblem(seed=7)
    # Valid closed tour
    valid_tour = np.arange(15)
    f, cv, _ = prob.evaluate(valid_tour)
    assert cv == 0.0
    assert f[0] > 0.0  # Total distance
    assert f[1] > 0.0  # Total travel time

    # Invalid tour (duplicates)
    invalid_tour = np.zeros(15, dtype=int)
    _, cv_bad, _ = prob.evaluate(invalid_tour)
    assert cv_bad > 0.0


def test_p09_feature_selection() -> None:
    prob = FeatureSelectionProblem()
    # Baseline all-ones mask
    all_ones = np.ones(30, dtype=int)
    f, cv, _ = prob.evaluate(all_ones)
    assert cv == 0.0
    assert f[1] == 30.0  # 30 features selected
    # Accuracy should be strong on breast cancer dataset (error < 0.15)
    assert f[0] < 0.15

    # All-zeros mask: violates constraint (at least 1 feature)
    all_zeros = np.zeros(30, dtype=int)
    _, cv_bad, _ = prob.evaluate(all_zeros)
    assert cv_bad > 0.0


def test_p10_dtlz2() -> None:
    # Test M = 3
    prob3 = DTLZ2Problem(M=3)
    # When all x_i for i >= M are 0.5 -> g = 0 -> sum(f_m^2) = 1.0
    x = np.full(12, 0.5)
    f, cv, _ = prob3.evaluate(x)
    assert cv == 0.0
    sum_sq = float(np.sum(f ** 2))
    assert math.isclose(sum_sq, 1.0, abs_tol=1e-6)

    # Test M = 5
    prob5 = DTLZ2Problem(M=5)
    x5 = np.full(14, 0.5)
    f5, cv5, _ = prob5.evaluate(x5)
    assert cv5 == 0.0
    sum_sq5 = float(np.sum(f5 ** 2))
    assert math.isclose(sum_sq5, 1.0, abs_tol=1e-6)


def test_p11_production_planning() -> None:
    prob = ProductionPlanningProblem()
    # (300, 200) -> profit = 4(300) + 6(200) = 2400, units = 500
    x_opt = np.array([300.0, 200.0])
    f_int, cv, _ = prob.evaluate(x_opt)
    f_nat, _, _ = prob.evaluate_natural(x_opt)
    assert math.isclose(f_nat[0], 2400.0, abs_tol=1e-8)
    assert math.isclose(f_nat[1], 500.0, abs_tol=1e-8)
    assert math.isclose(f_int[0], -2400.0, abs_tol=1e-8)
    assert math.isclose(f_int[1], -500.0, abs_tol=1e-8)
    assert cv == 0.0

    # Infeasible: (400, 200) -> 2(400) + 3(200) = 1400 > 1200
    _, cv_bad, _ = prob.evaluate(np.array([400.0, 200.0]))
    assert cv_bad > 0.0


def test_p12_supplier_selection() -> None:
    prob = SupplierSelectionProblem()
    # S1 (index 0): [120, 8, 10, 92]
    f_nat0, cv0, _ = prob.evaluate_natural(np.array([0]))
    f_int0, _, _ = prob.evaluate(np.array([0]))
    np.testing.assert_allclose(f_nat0, [120.0, 8.0, 10.0, 92.0], atol=1e-8)
    np.testing.assert_allclose(f_int0, [120.0, -8.0, 10.0, -92.0], atol=1e-8)
    assert cv0 == 0.0

    # S4 (index 3): [90, 6, 18, 80]
    f_nat3, _, _ = prob.evaluate_natural(np.array([3]))
    np.testing.assert_allclose(f_nat3, [90.0, 6.0, 18.0, 80.0], atol=1e-8)


def test_p13_multitask() -> None:
    # Part (a): Toy continuous
    prob_toy = MultiTaskToyProblem()
    # Point at (0, 0): dist to (3, 3) is 9+9=18, dist to (-3, -3) is 18
    f, cv, _ = prob_toy.evaluate(np.array([0.0, 0.0]))
    assert math.isclose(f[0], 18.0, abs_tol=1e-8)
    assert math.isclose(f[1], 18.0, abs_tol=1e-8)
    assert cv == 0.0

    # Analytical gradients at (0, 0)
    g1, g2 = prob_toy.gradients(np.array([0.0, 0.0]))
    np.testing.assert_allclose(g1, [-6.0, -6.0], atol=1e-8)
    np.testing.assert_allclose(g2, [6.0, 6.0], atol=1e-8)

    # Part (b): NumPy shared network
    X, y_cls, y_reg = generate_synthetic_multitask_data(n_samples=50, seed=42)
    net = SharedMultiTaskNetwork(seed=42)
    L_cls, L_reg, g_W1_t1, g_W1_t2 = net.compute_losses_and_shared_gradients(X, y_cls, y_reg)
    assert L_cls > 0.0
    assert L_reg > 0.0
    assert g_W1_t1.shape == (10, 16)
    assert g_W1_t2.shape == (10, 16)


def test_p14_laptop_design() -> None:
    prob = LaptopDesignProblem()
    # (13, 40) gives 1.53 kg and ~9.0 h (8.9888 h)
    f_nat0, cv0, _ = prob.evaluate_natural(np.array([13.0, 40.0]))
    assert math.isclose(f_nat0[0], 1.53, abs_tol=1e-3)
    assert math.isclose(f_nat0[1], 8.9888, abs_tol=1e-3)
    assert cv0 == 0.0

    # (13, 90) gives 2.13 kg and ~20.2 h (20.2247 h)
    f_nat1, cv1, _ = prob.evaluate_natural(np.array([13.0, 90.0]))
    assert math.isclose(f_nat1[0], 2.13, abs_tol=1e-3)
    assert math.isclose(f_nat1[1], 20.2247, abs_tol=1e-3)
    assert cv1 == 0.0

    # Monotonicity check: increasing screen size x1 worsens both objectives
    f_worse, _, _ = prob.evaluate_natural(np.array([17.0, 40.0]))
    assert f_worse[0] > f_nat0[0]  # Heavier
    assert f_worse[1] < f_nat0[1]  # Shorter battery life


def test_p04_portfolio_covariance_psd_and_constraints() -> None:
    prob = PortfolioProblem()
    # Check covariance matrix is symmetric and positive semi-definite (all eigenvalues >= 0)
    assert np.allclose(prob.Sigma, prob.Sigma.T), "Sigma must be strictly symmetric"
    eigvals = np.linalg.eigvalsh(prob.Sigma)
    assert np.all(eigvals >= -1e-12), f"Sigma must be positive semi-definite, got {eigvals}"

    # Feasible allocation (sum(w) = 1.0)
    w_feas = np.array([0.25, 0.25, 0.25, 0.25])
    _, cv_feas, _ = prob.evaluate(w_feas)
    assert cv_feas == 0.0

    # Infeasible allocation (sum(w) = 1.5) -> CV = |1.5 - 1.0| - 1e-4 = 0.4999
    w_infeas = np.array([0.5, 0.5, 0.25, 0.25])
    _, cv_infeas, _ = prob.evaluate(w_infeas)
    assert math.isclose(cv_infeas, 0.5 - 1e-4, abs_tol=1e-8)


def test_p05_cantilever_active_stress_boundary() -> None:
    prob = CantileverBeamProblem()
    # Active boundary: b * h^2 == 2.4e-5 m^3 gives bending stress exactly 250 MPa
    # Choose b = 0.01 m, h = sqrt(2.4e-5 / 0.01) = sqrt(0.0024) approx 0.04898979 m
    b = 0.01
    h_boundary = math.sqrt(2.4e-5 / b)
    _, cv_boundary, _ = prob.evaluate(np.array([b, h_boundary]))
    assert cv_boundary == 0.0, "Active stress boundary must be feasible"

    # Just below boundary: h = h_boundary * 0.999 -> infeasible
    _, cv_sub, _ = prob.evaluate(np.array([b, h_boundary * 0.999]))
    assert cv_sub > 0.0, "Below active boundary must be strictly infeasible"


def test_p06_binh_korn_constraint_boundaries() -> None:
    prob = BinhKornProblem()
    # At (0, 0): g1 = (0 - 5)^2 + 0^2 - 25 = 0 <= 0 (active boundary)
    # g2 = 7.7 - ((0 - 8)^2 + 3^2) = 7.7 - (64 + 9) = -65.3 <= 0
    _, cv0, g0 = prob.evaluate(np.array([0.0, 0.0]))
    assert cv0 == 0.0
    assert math.isclose(g0[0], 0.0, abs_tol=1e-8)
    assert g0[1] == 0.0

    # Test point violating g1: (0, 1) -> g1 = 25 + 1 - 25 = 1 > 0
    _, cv_viol1, _ = prob.evaluate(np.array([0.0, 1.0]))
    assert math.isclose(cv_viol1, 1.0, abs_tol=1e-8)


def test_p07_water_environmental_flow_boundaries() -> None:
    prob = WaterAllocationProblem()
    # Boundary: (10, 80) -> x1 = 10 (g2 active), x1 + x2 = 90 (g1 active)
    _, cv1, g1 = prob.evaluate(np.array([10.0, 80.0]))
    assert cv1 == 0.0
    assert math.isclose(g1[0], 0.0, abs_tol=1e-8)
    assert math.isclose(g1[1], 0.0, abs_tol=1e-8)
    assert g1[2] == 0.0

    # Boundary: (70, 20) -> x2 = 20 (g3 active), x1 + x2 = 90 (g1 active)
    _, cv2, g2 = prob.evaluate(np.array([70.0, 20.0]))
    assert cv2 == 0.0
    assert math.isclose(g2[0], 0.0, abs_tol=1e-8)
    assert g2[1] == 0.0
    assert math.isclose(g2[2], 0.0, abs_tol=1e-8)

    # Infeasible: x1 < 10 (e.g. x1 = 5, x2 = 50) -> g2 = 10 - 5 = 5 > 0
    _, cv3, g3 = prob.evaluate(np.array([5.0, 50.0]))
    assert math.isclose(cv3, 5.0, abs_tol=1e-8)
    assert math.isclose(g3[1], 5.0, abs_tol=1e-8)


def test_p08_tsp_symmetry_and_seed() -> None:
    prob = DeliveryRoutingTSPProblem(seed=7)
    # Verify city 0 coordinates under MT19937 seed 7
    expected_c0 = np.array([7.63082894, 77.99187922])
    np.testing.assert_allclose(prob.coords[0], expected_c0, atol=1e-5)

    # Distance matrix symmetry and diagonal
    assert np.allclose(prob.dist_matrix, prob.dist_matrix.T), "Distance matrix must be symmetric"
    assert np.allclose(np.diag(prob.dist_matrix), 0.0), "Distance diagonal must be zero"

    # Travel time matrix symmetry and diagonal
    assert np.allclose(prob.time_matrix, prob.time_matrix.T), "Time matrix must be symmetric"
    assert np.allclose(np.diag(prob.time_matrix), 0.0), "Time diagonal must be zero"
    assert np.all(prob.time_matrix >= 0.0), "Time values must be non-negative"


def test_p10_dtlz2_hypersphere_multiple_m() -> None:
    for M in [3, 4, 5]:
        prob = DTLZ2Problem(M=M)
        n = prob.n_var
        # Fix all distance variables to 0.5 so g = 0
        x = np.full(n, 0.5)
        # Vary angular variables
        x[: M - 1] = np.linspace(0.1, 0.8, M - 1)
        f, cv, _ = prob.evaluate(x)
        assert cv == 0.0
        sum_sq = float(np.sum(f ** 2))
        assert math.isclose(sum_sq, 1.0, abs_tol=1e-8), f"DTLZ2 with M={M} must satisfy sum(f_m^2) == 1.0"


def test_p13_analytical_and_neural_gradient_verification() -> None:
    # Part (a): Finite difference check for analytical gradients
    prob_toy = MultiTaskToyProblem()
    x0 = np.array([1.5, -2.0])
    g1_ana, g2_ana = prob_toy.gradients(x0)

    eps = 1e-6
    # Numerical gradient of f1
    g1_num = np.zeros(2)
    for i in range(2):
        x_plus = x0.copy()
        x_plus[i] += eps
        x_minus = x0.copy()
        x_minus[i] -= eps
        f_plus, _, _ = prob_toy.evaluate(x_plus)
        f_minus, _, _ = prob_toy.evaluate(x_minus)
        g1_num[i] = (f_plus[0] - f_minus[0]) / (2.0 * eps)
    np.testing.assert_allclose(g1_ana, g1_num, atol=1e-5)

    # Numerical gradient of f2
    g2_num = np.zeros(2)
    for i in range(2):
        x_plus = x0.copy()
        x_plus[i] += eps
        x_minus = x0.copy()
        x_minus[i] -= eps
        f_plus, _, _ = prob_toy.evaluate(x_plus)
        f_minus, _, _ = prob_toy.evaluate(x_minus)
        g2_num[i] = (f_plus[1] - f_minus[1]) / (2.0 * eps)
    np.testing.assert_allclose(g2_ana, g2_num, atol=1e-5)

    # Part (b): Numerical gradient verification for neural network shared layer W1
    X, y_cls, y_reg = generate_synthetic_multitask_data(n_samples=20, seed=42)
    net = SharedMultiTaskNetwork(seed=42)
    L_cls, L_reg, grad_W1_t1, grad_W1_t2 = net.compute_losses_and_shared_gradients(X, y_cls, y_reg)

    # Check finite differences on a sample of W1 weights
    for r, c in [(0, 0), (2, 5), (5, 10), (9, 15)]:
        W1_orig = net.W1[r, c]

        # Check Task 1 (Classification loss BCE)
        net.W1[r, c] = W1_orig + eps
        _, _, p1_plus, _ = net.forward(X)
        p1_safe = np.clip(p1_plus, 1e-12, 1.0 - 1e-12)
        L1_plus = -np.mean(y_cls.reshape(-1, 1) * np.log(p1_safe) + (1.0 - y_cls.reshape(-1, 1)) * np.log(1.0 - p1_safe))

        net.W1[r, c] = W1_orig - eps
        _, _, p1_minus, _ = net.forward(X)
        p1_safe_m = np.clip(p1_minus, 1e-12, 1.0 - 1e-12)
        L1_minus = -np.mean(y_cls.reshape(-1, 1) * np.log(p1_safe_m) + (1.0 - y_cls.reshape(-1, 1)) * np.log(1.0 - p1_safe_m))

        net.W1[r, c] = W1_orig
        num_grad1 = (L1_plus - L1_minus) / (2.0 * eps)
        assert math.isclose(grad_W1_t1[r, c], num_grad1, abs_tol=1e-4)

        # Check Task 2 (Regression loss MSE)
        net.W1[r, c] = W1_orig + eps
        _, _, _, p2_plus = net.forward(X)
        L2_plus = np.mean((p2_plus - y_reg.reshape(-1, 1)) ** 2)

        net.W1[r, c] = W1_orig - eps
        _, _, _, p2_minus = net.forward(X)
        L2_minus = np.mean((p2_minus - y_reg.reshape(-1, 1)) ** 2)

        net.W1[r, c] = W1_orig
        num_grad2 = (L2_plus - L2_minus) / (2.0 * eps)
        assert math.isclose(grad_W1_t2[r, c], num_grad2, abs_tol=1e-4)
