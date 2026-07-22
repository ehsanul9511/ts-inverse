#!/usr/bin/env python3
"""Analyze gender predictions saved by predict_gender_from_reconstruction.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.metrics import classification_report, confusion_matrix


REQUIRED_COLUMNS = {
    "probability_male",
    "prediction",
    "true_gender",
    "correct",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="gender_predictions.csv")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--plot",
        default="gender_probability_histogram.png",
    )
    parser.add_argument(
        "--report",
        default="gender_classification_report.csv",
    )
    parser.add_argument(
        "--confusion-matrix",
        default="gender_confusion_matrix.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    plot_path = Path(args.plot)
    report_path = Path(args.report)
    confusion_path = Path(args.confusion_matrix)

    predictions = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")

    probabilities = predictions["probability_male"].to_numpy(dtype=float)
    y_true = predictions["true_gender"].to_numpy(dtype=int)
    y_pred = predictions["prediction"].to_numpy(dtype=int)

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probability_male values must be between 0 and 1")

    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)
    # right=False assigns 0.0 to bin 0; clip assigns 1.0 to the final bin.
    bin_ids = np.clip(np.digitize(probabilities, bin_edges, right=False) - 1, 0, args.bins - 1)
    counts = np.bincount(bin_ids, minlength=args.bins)
    correct = (y_true == y_pred).astype(float)
    correct_counts = np.bincount(bin_ids, weights=correct, minlength=args.bins)
    accuracies = np.divide(
        correct_counts,
        counts,
        out=np.full(args.bins, np.nan),
        where=counts > 0,
    )

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges) * 0.9
    color_values = np.nan_to_num(accuracies, nan=0.0)
    colors = plt.colormaps["RdYlGn"](color_values)
    colors[counts == 0] = (0.82, 0.82, 0.82, 1.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        centers,
        counts,
        width=widths,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )

    for bar, count, accuracy in zip(bars, counts, accuracies):
        if count == 0:
            continue
        label = f"{accuracy * 100:.1f}%\n(n={count})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.max() * 0.015, 0.15),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Decision threshold")
    ax.set_xticks(bin_edges)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(counts.max() * 1.30, 1))
    ax.set_xlabel("Predicted probability of male")
    ax.set_ylabel("Number of reconstructed samples")
    ax.set_title(
        "Gender prediction confidence and accuracy by probability bucket",
        pad=12,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="RdYlGn"),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("Accuracy within bucket")
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["female", "male"],
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["female", "male"],
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report_dict).transpose()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(report_path, index_label="class_or_average")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_frame = pd.DataFrame(
        matrix,
        index=["true_female", "true_male"],
        columns=["predicted_female", "predicted_male"],
    )
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_frame.to_csv(confusion_path, index_label="true_class")

    print("Classification report\n")
    print(report_text)
    print("Confusion matrix\n")
    print(matrix_frame)
    print(f"\nSaved histogram: {plot_path.resolve()}")
    print(f"Saved report: {report_path.resolve()}")
    print(f"Saved confusion matrix: {confusion_path.resolve()}")


if __name__ == "__main__":
    main()
