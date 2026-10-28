"""
plot_generator.py: Matplotlib Visualization Module for MOO Lab 7

Features:
- Headless non-interactive backend (Agg) for reliable script execution.
- 2D Pareto front visualization with true front reference curve (where known).
- 3D Pareto front scatter visualization for tri-objective benchmarks.
- Parallel coordinates visualization for many-objective spaces (M >= 4).
- Metric summary annotation boxes (HV, Schott's Spacing, IGD).
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_pareto_front(
    F_opt_nat: np.ndarray,
    problem_name: str,
    alg_name: str,
    save_path: Path | str,
    true_front_nat: np.ndarray | None = None,
    hv: float | None = None,
    spacing_val: float | None = None,
    igd_val: float | None = None,
    budget: int | None = None,
    evals_used: int | None = None,
    seed: int | None = None,
    obj_labels: Sequence[str] | None = None,
) -> Path:
    """
    Renders and saves a Pareto front plot to disk.
    
    Parameters:
        F_opt_nat: Obtained front in natural coordinates (shape: (N, M)).
        problem_name: Name of the benchmark problem (e.g. 'P1_FactorySpeed').
        alg_name: Name of the algorithm (e.g. 'NSGA-II').
        save_path: Destination file path for PNG image.
        true_front_nat: True Pareto front in natural coordinates (if available).
        hv: Computed Hypervolume score.
        spacing_val: Computed Schott's Spacing score.
        igd_val: Computed IGD score.
        budget: Configured evaluation budget.
        evals_used: Actual evaluations used.
        seed: Random seed used.
        obj_labels: Custom names for each objective axis.

    Returns:
        Path: Resolved path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    F = np.atleast_2d(np.asarray(F_opt_nat, dtype=float))
    N_pts, M = F.shape if F.size > 0 else (0, 2)

    # Format metric annotation text
    metric_lines = []
    if evals_used is not None:
        b_str = f" / {budget}" if budget is not None else ""
        metric_lines.append(f"Evals: {evals_used}{b_str}")
    if seed is not None:
        metric_lines.append(f"Seed: {seed}")
    metric_lines.append(f"Points: {N_pts}")
    if hv is not None:
        metric_lines.append(f"HV: {hv:.4f}")
    if spacing_val is not None:
        metric_lines.append(f"Spacing: {spacing_val:.4f}")
    elif N_pts <= 1:
        metric_lines.append("Spacing: N/A (N < 2)")
    else:
        metric_lines.append("Spacing: N/A")
    if igd_val is not None:
        metric_lines.append(f"IGD: {igd_val:.4f}")
    else:
        metric_lines.append("IGD: N/A")
    annot_text = "\n".join(metric_lines)

    # 1. Bi-objective Visualization (M == 2)
    if M == 2:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        # Plot true front if available
        if true_front_nat is not None and len(true_front_nat) > 0:
            TF = np.atleast_2d(np.asarray(true_front_nat, dtype=float))
            if len(TF) == 1:
                ax.scatter(
                    TF[:, 0],
                    TF[:, 1],
                    color="#DC2626",
                    edgecolors="#7F1D1D",
                    marker="*",
                    s=160,
                    label="True Pareto Point",
                    zorder=2,
                )
            else:
                # Sort for clean continuous line
                order = np.argsort(TF[:, 0])
                ax.plot(
                    TF[order, 0],
                    TF[order, 1],
                    color="#DC2626",
                    linestyle="--",
                    linewidth=1.8,
                    label="True Pareto Front",
                    zorder=2,
                )

        # Plot obtained front
        if N_pts > 0:
            ax.scatter(
                F[:, 0],
                F[:, 1],
                color="#2563EB",
                edgecolors="#1E3A8A",
                s=45,
                alpha=0.85,
                label=f"Obtained ({N_pts} pts)",
                zorder=3,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No Feasible Non-Dominated Solutions Found",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#DC2626",
                fontsize=12,
                weight="bold",
            )

        x_label = obj_labels[0] if (obj_labels and len(obj_labels) > 0) else "Objective 1 ($f_1$)"
        y_label = obj_labels[1] if (obj_labels and len(obj_labels) > 1) else "Objective 2 ($f_2$)"
        ax.set_xlabel(x_label, fontsize=11, fontweight="semibold")
        ax.set_ylabel(y_label, fontsize=11, fontweight="semibold")
        ax.set_title(f"{alg_name} on {problem_name}", fontsize=13, fontweight="bold", pad=12)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", framealpha=0.9)

        # Annotate box
        ax.text(
            0.97,
            0.03,
            annot_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
            zorder=4,
        )

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    # 2. Tri-objective Visualization (M == 3)
    elif M == 3:
        fig = plt.figure(figsize=(9, 7), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        if true_front_nat is not None and len(true_front_nat) > 0:
            TF = np.atleast_2d(np.asarray(true_front_nat, dtype=float))
            if len(TF) == 1:
                ax.scatter(
                    TF[:, 0],
                    TF[:, 1],
                    TF[:, 2],
                    color="#DC2626",
                    edgecolors="#7F1D1D",
                    marker="*",
                    s=160,
                    label="True Pareto Point",
                )
            else:
                ax.scatter(
                    TF[:, 0],
                    TF[:, 1],
                    TF[:, 2],
                    color="#94A3B8",
                    s=15,
                    alpha=0.35,
                    label="True Front Sample",
                )

        if N_pts > 0:
            ax.scatter(
                F[:, 0],
                F[:, 1],
                F[:, 2],
                color="#2563EB",
                edgecolors="#1E3A8A",
                s=40,
                alpha=0.85,
                label=f"Obtained ({N_pts} pts)",
            )

        x_label = obj_labels[0] if (obj_labels and len(obj_labels) > 0) else "$f_1$"
        y_label = obj_labels[1] if (obj_labels and len(obj_labels) > 1) else "$f_2$"
        z_label = obj_labels[2] if (obj_labels and len(obj_labels) > 2) else "$f_3$"
        ax.set_xlabel(x_label, fontsize=10, fontweight="semibold")
        ax.set_ylabel(y_label, fontsize=10, fontweight="semibold")
        ax.set_zlabel(z_label, fontsize=10, fontweight="semibold")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", framealpha=0.9)

        fig.text(
            0.02,
            0.02,
            annot_text.replace("\n", "  |  "),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
        )

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    # 3. Many-Objective Visualization (M >= 4) - Parallel Coordinates
    else:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        xs = np.arange(M)

        if N_pts > 0:
            # Normalize for parallel coordinates display to [0, 1] span
            col_min = np.min(F, axis=0)
            col_max = np.max(F, axis=0)
            col_span = np.where(np.abs(col_max - col_min) < 1e-12, 1.0, col_max - col_min)
            F_disp = (F - col_min) / col_span

            for i in range(min(N_pts, 100)):  # Plot up to 100 sample lines
                ax.plot(xs, F_disp[i], color="#2563EB", alpha=0.4, linewidth=1.2)
        else:
            ax.text(0.5, 0.5, "No Non-Dominated Solutions Found", transform=ax.transAxes, ha="center")

        x_names = obj_labels if (obj_labels and len(obj_labels) == M) else [f"Obj {m+1}" for m in range(M)]
        ax.set_xticks(xs)
        ax.set_xticklabels(x_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Normalized Criterion Scale [0, 1]", fontsize=10, fontweight="semibold")
        ax.set_title(f"{alg_name} on {problem_name} (Parallel Coordinates, M={M})", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        ax.text(
            0.98,
            0.03,
            annot_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
        )

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
