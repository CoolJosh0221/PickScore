"""Plotting utilities for calibration analysis."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional


def plot_reliability_diagram(
    bin_data: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Reliability Diagram"
) -> None:
    """
    Generate reliability diagram (confidence vs accuracy per bin).

    Args:
        bin_data: List of bin dictionaries from compute_calibration_bins
        output_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    non_empty = [b for b in bin_data if b["count"] > 0]
    if not non_empty:
        plt.close()
        return

    confidences = [b["confidence"] for b in non_empty]
    accuracies = [b["accuracy"] for b in non_empty]
    gaps = [b["gap"] for b in non_empty]

    # Left plot: Reliability diagram
    ax1.bar(confidences, accuracies, width=0.05, alpha=0.7,
            label="Accuracy", color="blue", edgecolor="black")
    ax1.bar(confidences, gaps, width=0.05, alpha=0.5, bottom=accuracies,
            label="Gap", color="red", edgecolor="black")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Right plot: Confidence histogram
    bin_edges = np.linspace(0, 1, 16)
    bin_counts = np.zeros(15)
    for b in bin_data:
        idx = int(b["lower"] * 15)
        if idx < 15:
            bin_counts[idx] = b["count"]

    ax2.bar(bin_edges[:-1] + 0.033, bin_counts, width=0.066, alpha=0.7,
            color="green", edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Histogram", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_uncertainty_error(
    uncertainty: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
    title: str = "Uncertainty vs Error",
    correlation: Optional[float] = None,
    xlabel: str = "Uncertainty"
) -> None:
    """
    Plot uncertainty vs error scatter with trend line.

    Args:
        uncertainty: [N] uncertainty scores
        errors: [N] binary errors (0=correct, 1=wrong)
        output_path: Path to save the plot
        title: Plot title
        correlation: Spearman correlation to display in legend
        xlabel: X-axis label
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(uncertainty, errors, alpha=0.3, s=10)

    # Trend line
    z = np.polyfit(uncertainty, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(uncertainty.min(), uncertainty.max(), 100)

    if correlation is not None:
        ax.plot(x_line, p(x_line), "r-", linewidth=2,
                label=f"Trend (ρ={correlation:.3f})")
    else:
        ax.plot(x_line, p(x_line), "r-", linewidth=2, label="Trend")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Error (0=correct, 1=wrong)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
