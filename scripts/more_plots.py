import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_FILES = {
    "Rand.": "../results/active_learning_random_sampling.json",
    "Indi.": "../results/active_learning_individual_entropy.json",
    "Mix": "../results/active_learning_mixed_entropy.json",
    "Group": "../results/active_learning_group_entropy.json",
    "Vote": "../results/active_learning_vote_variance.json",
}


LINEWIDTHS = {
    "Rand.": 1.8,
    "Indi.": 1.8,
    "Mix": 1.8,
    "Group": 3.2,
    "Vote": 3.2,
}


METRICS = {
    "eval_macro_f1": "Macro-F1",
    "eval_weighted_f1": "Weighted-F1",
    "eval_accuracy": "Accuracy",
    "eval_loss": "Loss",
}


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def smooth(values, window=5):
    if len(values) < window:
        return values

    values = np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def get_xy(results, metric, x_axis="round", use_smoothing=False):
    if x_axis == "round":
        x = [item["round"] for item in results]
        xlabel = "Rounds"
    elif x_axis == "labeled_size":
        x = [item["labeled_size"] for item in results]
        xlabel = "Labeled Data Size"
    else:
        raise ValueError(f"Unknown x_axis: {x_axis}")

    y = [item["test_results"][metric] for item in results]

    if use_smoothing:
        y = smooth(y)

    return x, y, xlabel


def plot_single_metric(metric, x_axis="round", use_smoothing=False):
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))

    for label, file_path in RESULT_FILES.items():
        path = Path(file_path)

        if not path.exists():
            print(f"Missing file: {path}")
            continue

        results = load_results(path)
        x, y, xlabel = get_xy(results, metric, x_axis, use_smoothing)

        plt.plot(
            x,
            y,
            label=label,
            linewidth=LINEWIDTHS[label],
        )

    title_suffix = "Smoothed" if use_smoothing else "Raw"
    plt.title(f"HS-Brexit: {METRICS[metric]} ({title_suffix})", fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(METRICS[metric], fontweight="bold")
    plt.legend()
    plt.tight_layout()

    smooth_tag = "smoothed" if use_smoothing else "raw"
    output_path = output_dir / f"hs_brexit_{metric}_{x_axis}_{smooth_tag}.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_paper_style_three_panel(use_smoothing=False):
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    panel_metrics = [
        ("eval_macro_f1", "Macro-F1"),
        ("eval_weighted_f1", "Weighted-F1"),
        ("eval_accuracy", "Accuracy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric, ylabel) in zip(axes, panel_metrics):
        for label, file_path in RESULT_FILES.items():
            path = Path(file_path)

            if not path.exists():
                print(f"Missing file: {path}")
                continue

            results = load_results(path)
            x, y, _ = get_xy(results, metric, x_axis="round", use_smoothing=use_smoothing)

            ax.plot(
                x,
                y,
                label=label,
                linewidth=LINEWIDTHS[label],
            )

        ax.set_title("HS-Brexit", fontweight="bold")
        ax.set_xlabel("Rounds", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

    axes[0].legend()

    plt.tight_layout()

    smooth_tag = "smoothed" if use_smoothing else "raw"
    output_path = output_dir / f"hs_brexit_three_panel_{smooth_tag}.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    # 1. Individual metric plots over rounds
    for metric in METRICS:
        plot_single_metric(metric, x_axis="round", use_smoothing=False)

    # 2. Individual metric plots over labeled data size
    for metric in METRICS:
        plot_single_metric(metric, x_axis="labeled_size", use_smoothing=False)

    # 3. Smoothed macro-F1 over rounds
    plot_single_metric("eval_macro_f1", x_axis="round", use_smoothing=True)

    # 4. Smoothed macro-F1 over labeled size
    plot_single_metric("eval_macro_f1", x_axis="labeled_size", use_smoothing=True)

    # 5. Paper-style three-panel plot using available metrics
    plot_paper_style_three_panel(use_smoothing=False)
    plot_paper_style_three_panel(use_smoothing=True)


if __name__ == "__main__":
    main()