"""
Plot active learning results for all ACTOR acquisition methods.

This reads the JSON result files produced by the active learning runners
and plots test macro-F1 across rounds, similar to Figure 3 in Wang & Plank.

Run from project root:

    python scripts/plot_active_learning_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_FILES = {
    "Rand.": "results/active_learning_random_sampling.json",
    "Indi.": "results/active_learning_individual_entropy.json",
    "Group": "results/active_learning_group_entropy.json",
    "Vote": "results/active_learning_vote_variance.json",
    "Mix": "results/active_learning_mixed_entropy.json",
}


def extract_metric(round_result, metric_name="eval_macro_f1"):
    """
    Try to extract metric from test_results.

    Change metric_name if your JSON uses another key, e.g.
    'test_macro_f1', 'eval_f1', or 'eval_weighted_f1'.
    """

    test_results = round_result.get("test_results", {})

    if metric_name in test_results:
        return test_results[metric_name]

    possible_keys = [
        "eval_macro_f1",
        "test_macro_f1",
        "macro_f1",
        "eval_f1",
        "f1",
    ]

    for key in possible_keys:
        if key in test_results:
            return test_results[key]

    raise KeyError(
        f"Could not find metric in test_results. Available keys: {list(test_results.keys())}"
    )


def main():
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))

    for label, path in RESULT_FILES.items():
        path = Path(path)

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)

        rounds = []
        scores = []

        for item in results:
            try:
                rounds.append(item["round"])
                scores.append(extract_metric(item))
            except KeyError as e:
                print(f"Skipping one item in {path}: {e}")

        plt.plot(rounds, scores, marker="o", label=label)

    plt.xlabel("Rounds")
    plt.ylabel("Test Macro-F1")
    plt.title("HS-Brexit Active Learning")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "active_learning_macro_f1.png"
    plt.savefig(output_path, dpi=300)

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()