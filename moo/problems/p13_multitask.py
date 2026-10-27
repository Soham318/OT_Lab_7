"""
p13_multitask.py: P13 — Multi-Task Learning (Differentiable)

Scenario:
A shared model serves two tasks whose gradients pull in different directions.

Part (a) — Toy Continuous Problem:
- Decision variable x in R^2, starting at (0, 5).
- f1 = ||x - a||^2, where a = (3, 3)
- f2 = ||x - b||^2, where b = (-3, -3)
- Sanity Check:
  The Pareto set is the line segment from a to b: x1 = x2 with -3 <= x1 <= 3.
  Front: f1 = 2 * (t - 3)^2, f2 = 2 * (t + 3)^2 for t in [-3, 3].

Part (b) — Shared-Body Neural Network (Pure NumPy):
- Shared trunk: 10 input features -> 16 hidden units (ReLU).
- Classification head: 16 -> 1 (Sigmoid, Binary Cross-Entropy Loss L1).
- Regression head: 16 -> 1 (Linear, Mean Squared Error Loss L2).
- Forward and backward passes hand-coded strictly in pure NumPy.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


# ---------------------------------------------------------------------------
# Part (a): Toy Continuous Problem
# ---------------------------------------------------------------------------

class MultiTaskToyProblem(Problem):
    def __init__(self) -> None:
        self.a = np.array([3.0, 3.0], dtype=float)
        self.b = np.array([-3.0, -3.0], dtype=float)
        super().__init__(
            name="P13a_MultiTaskToy",
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=np.array([-10.0, -10.0]),
            xu=np.array([10.0, 10.0]),
            var_type="real",
            ideal_point=np.array([0.0, 0.0]),
            nadir_point=np.array([72.0, 72.0]),
            ref_point=np.array([80.0, 80.0]),
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x_vec = np.asarray(x, dtype=float).ravel()
        f1 = float(np.sum((x_vec - self.a) ** 2))
        f2 = float(np.sum((x_vec - self.b) ** 2))
        return np.array([f1, f2], dtype=float), 0.0, np.array([], dtype=float)

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)

    def gradients(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns exact analytical gradients [grad_f1, grad_f2]."""
        x_vec = np.asarray(x, dtype=float).ravel()
        g1 = 2.0 * (x_vec - self.a)
        g2 = 2.0 * (x_vec - self.b)
        return g1, g2

    def get_true_front(self, n_points: int = 100) -> np.ndarray:
        t_vals = np.linspace(-3.0, 3.0, n_points)
        f1 = 2.0 * ((t_vals - 3.0) ** 2)
        f2 = 2.0 * ((t_vals + 3.0) ** 2)
        return np.column_stack((f1, f2))

    def get_true_front_natural(self, n_points: int = 100) -> np.ndarray:
        return self.get_true_front(n_points)


# ---------------------------------------------------------------------------
# Part (b): Pure NumPy Shared-Body Multi-Task Neural Network
# ---------------------------------------------------------------------------

class SharedMultiTaskNetwork:
    """
    Two-task shared-body MLP implemented strictly in NumPy:
    - Input: X in R^(N x 10)
    - Shared hidden layer: H = relu(X @ W1 + b1), W1 in R^(10 x 16)
    - Head 1 (classification): y_cls_hat = sigmoid(H @ W_cls + b_cls)
    - Head 2 (regression): y_reg_hat = H @ W_reg + b_reg
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.d_in = 10
        self.d_hid = 16

        # He initialization for shared trunk
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / self.d_in), size=(self.d_in, self.d_hid))
        self.b1 = np.zeros(self.d_hid)

        # Task heads
        self.W_cls = rng.normal(0.0, np.sqrt(2.0 / self.d_hid), size=(self.d_hid, 1))
        self.b_cls = np.zeros(1)

        self.W_reg = rng.normal(0.0, np.sqrt(2.0 / self.d_hid), size=(self.d_hid, 1))
        self.b_reg = np.zeros(1)

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass.
        Returns: (Z1, H, y_cls_pred, y_reg_pred)
        """
        Z1 = X @ self.W1 + self.b1  # (N, 16)
        H = np.maximum(0.0, Z1)     # ReLU

        Z_cls = H @ self.W_cls + self.b_cls
        # Numerical clip before sigmoid to prevent overflow
        Z_cls_clipped = np.clip(Z_cls, -30.0, 30.0)
        y_cls_pred = 1.0 / (1.0 + np.exp(-Z_cls_clipped))

        y_reg_pred = H @ self.W_reg + self.b_reg

        return Z1, H, y_cls_pred, y_reg_pred

    def compute_losses_and_shared_gradients(
        self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        Computes task losses and gradients of each task loss with respect to shared weights W1.
        Returns: (L_cls, L_reg, grad_W1_task1, grad_W1_task2)
        """
        N = X.shape[0]
        y_cls = y_cls.reshape(-1, 1)
        y_reg = y_reg.reshape(-1, 1)

        Z1, H, p_cls, p_reg = self.forward(X)

        # Loss 1: Binary Cross Entropy
        eps = 1e-12
        p_cls_safe = np.clip(p_cls, eps, 1.0 - eps)
        L_cls = float(
            -np.mean(y_cls * np.log(p_cls_safe) + (1.0 - y_cls) * np.log(1.0 - p_cls_safe))
        )

        # Loss 2: Mean Squared Error
        L_reg = float(np.mean((p_reg - y_reg) ** 2))

        # Backward for Task 1 (Classification)
        dZ_cls = (p_cls - y_cls) / float(N)          # (N, 1)
        dH_task1 = dZ_cls @ self.W_cls.T            # (N, 16)
        dZ1_task1 = dH_task1 * (Z1 > 0.0)           # ReLU backprop
        grad_W1_task1 = X.T @ dZ1_task1             # (10, 16)

        # Backward for Task 2 (Regression)
        dZ_reg = 2.0 * (p_reg - y_reg) / float(N)   # (N, 1)
        dH_task2 = dZ_reg @ self.W_reg.T            # (N, 16)
        dZ1_task2 = dH_task2 * (Z1 > 0.0)           # ReLU backprop
        grad_W1_task2 = X.T @ dZ1_task2             # (10, 16)

        return L_cls, L_reg, grad_W1_task1, grad_W1_task2


def generate_synthetic_multitask_data(
    n_samples: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic multi-task dataset for P13(b)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, 10))
    # True underlying linear functions with non-overlapping feature importance
    y_cls_latent = X[:, 0] * 1.5 - X[:, 1] * 2.0 + X[:, 2] * 0.8
    y_cls = (y_cls_latent > 0.0).astype(int)
    y_reg = X[:, 3] * 3.0 + X[:, 4] * 1.2 - X[:, 5] * 2.5 + rng.normal(0.0, 0.1, size=n_samples)
    return X, y_cls, y_reg
