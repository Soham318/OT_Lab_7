"""
p09_feature_selection.py: P9 — Feature Selection (Binary, medium)

Scenario:
A diagnostic classifier should be accurate while using as few measurements as possible.

Data:
- Wisconsin breast-cancer dataset (30 features, 569 samples).
- Loaded strictly with scikit-learn (load_breast_cancer).
- The 5-NN classifier and 5-fold cross-validation are hand-coded from scratch in pure NumPy.

Solution:
- A 0/1 binary mask of length 30 with at least one 1.

Objectives:
- Minimize error = 1 - cross-validated accuracy
- Minimize number of selected features = sum(x_i)

Constraints:
- At least one feature selected: g(x) = 1 - sum(x_i) <= 0
- Standardized violation: CV = max(0.0, 1.0 - sum(x_i))

Sanity Check:
- All-ones mask is the accuracy baseline.
"""

from __future__ import annotations
import numpy as np
from moo.problems.base import Problem


def _load_wdbc_data() -> tuple[np.ndarray, np.ndarray]:
    """Loads and standardizes Wisconsin Breast Cancer dataset using permitted scoped scikit-learn call."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)

    # Standardize features (zero mean, unit variance)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_scaled = (X - mean) / std

    return X_scaled, y


class FeatureSelectionProblem(Problem):
    def __init__(self) -> None:
        self.X, self.y = _load_wdbc_data()
        self.n_samples, self.n_features = self.X.shape  # 569, 30
        self.k_neighbors = 5
        self.n_folds = 5

        # Create deterministic 5-fold split indices
        self.folds: list[tuple[np.ndarray, np.ndarray]] = []
        indices = np.arange(self.n_samples)
        # Deterministic shuffle with fixed seed 42
        rng = np.random.default_rng(42)
        shuffled = rng.permutation(indices)
        fold_sizes = np.full(self.n_folds, self.n_samples // self.n_folds, dtype=int)
        fold_sizes[: self.n_samples % self.n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            test_idx = shuffled[current : current + fold_size]
            train_idx = np.setdiff1d(shuffled, test_idx)
            self.folds.append((train_idx, test_idx))
            current += fold_size

        super().__init__(
            name="P9_FeatureSelection",
            n_var=self.n_features,
            n_obj=2,
            n_constr=1,
            xl=np.zeros(self.n_features),
            xu=np.ones(self.n_features),
            var_type="binary",
            ideal_point=np.array([0.0, 1.0]),
            nadir_point=np.array([1.0, float(self.n_features)]),
            ref_point=np.array([1.1, float(self.n_features) * 1.1]),
        )

    def _hand_coded_5nn_accuracy(self, selected_cols: np.ndarray) -> float:
        """Evaluates hand-coded 5-NN classifier across the 5 deterministic folds."""
        X_sub = self.X[:, selected_cols]
        total_correct = 0

        for train_idx, test_idx in self.folds:
            X_tr, y_tr = X_sub[train_idx], self.y[train_idx]
            X_te, y_te = X_sub[test_idx], self.y[test_idx]

            # Fast squared Euclidean distances via BLAS: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            te_norm_sq = np.sum(X_te ** 2, axis=1, keepdims=True)
            tr_norm_sq = np.sum(X_tr ** 2, axis=1, keepdims=True).T
            dist_sq = np.maximum(0.0, te_norm_sq + tr_norm_sq - 2.0 * (X_te @ X_tr.T))

            # Find k nearest neighbors (monotonic with distance)
            k = min(self.k_neighbors, len(train_idx))
            nn_indices = np.argpartition(dist_sq, k, axis=1)[:, :k]

            # Majority voting
            nn_labels = y_tr[nn_indices]
            votes = np.sum(nn_labels == 1, axis=1)
            y_pred = (votes >= (k / 2.0)).astype(int)

            total_correct += int(np.sum(y_pred == y_te))

        return float(total_correct / self.n_samples)

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        mask = (np.asarray(x, dtype=float).ravel() >= 0.5)
        n_selected = int(np.sum(mask))

        if n_selected == 0:
            # Violation: at least one feature must be selected
            cv = 1.0
            return np.array([1.0, 0.0], dtype=float), cv, np.array([cv], dtype=float)

        accuracy = self._hand_coded_5nn_accuracy(mask)
        error = 1.0 - accuracy
        cv = 0.0
        g = np.array([0.0], dtype=float)

        return np.array([error, float(n_selected)], dtype=float), cv, g

    def evaluate_natural(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return self.evaluate(x)
