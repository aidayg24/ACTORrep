"""
Plot active learning results for all ACTOR acquisition methods.

This reads the JSON result files produced by the active learning runners
and plots test macro-F1 across rounds, similar to Figure 3 in Wang & Plank.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


RESULT_FILES = {
    "Rand.": "../results/active_learning_random_sampling.json",
    "Indi.": "../results/active_learning_individual_entropy.json",
    "Mix": "../results/active_learning_mixed_entropy.json",
    "Group": "../results/active_learning_group_entropy.json",
    "Vote": "../results/active_learning_vote_variance.json",
    "Bandit" : "../results/active_learning_bandit_ucb.json",
}

LINEWIDTHS = {
    "Rand.": 1.8,
    "Indi.": 1.8,
    "Mix": 1.8,
    "Group": 3.2,
    "Vote": 3.2,
    "Bandit": 3.2,
}


def load_metric(path, metric="eval_macro_f1"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rounds = [item["round"] for item in data]
    scores = [item["test_results"][metric] for item in data]

    return rounds, scores


def main():
    Path("plots").mkdir(exist_ok=True)

    plt.figure(figsize=(6, 4))

    for label, path in RESULT_FILES.items():
        path = Path(path)

        if not path.exists():
            print(f"Missing file: {path}")
            continue

        rounds, scores = load_metric(path)

        plt.plot(
            rounds,
            scores,
            label=label,
            linewidth=LINEWIDTHS[label],
        )

    plt.title("HS-Brexit", fontsize=14, fontweight="bold")
    plt.xlabel("Rounds", fontweight="bold")
    plt.ylabel("F1", fontweight="bold")
    plt.legend()
    plt.tight_layout()

    plt.savefig("plots/hs_brexit_macro_f1.png", dpi=300)
    plt.savefig("plots/hs_brexit_macro_f1.pdf")

    print("Saved plot to plots/hs_brexit_macro_f1.png")


if __name__ == "__main__":
    main()