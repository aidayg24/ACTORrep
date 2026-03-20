"""
Paper-matched random acquisition: round 2.

This script:
1. loads paper round-1 labeled pool
2. loads paper round-1 unlabeled pool
3. randomly samples 60 new annotation pairs
4. adds them to the labeled pool
5. saves round-2 paper pools
"""

import pandas as pd

# ---------------------------
# 1. Load current paper pools
# ---------------------------
labeled_path = "../processed_data/hsbrexit_paper_labeled_pool_round1.csv"
unlabeled_path = "../processed_data/hsbrexit_paper_unlabeled_pool_round1.csv"

labeled_df = pd.read_csv(labeled_path)
unlabeled_df = pd.read_csv(unlabeled_path)

print("Current labeled pool size:", len(labeled_df))
print("Current unlabeled pool size:", len(unlabeled_df))

# ---------------------------
# 2. Random acquisition
# ---------------------------
acquisition_size = 60
random_state = 43

acquired_df = unlabeled_df.sample(n=acquisition_size, random_state=random_state)
remaining_unlabeled_df = unlabeled_df.drop(acquired_df.index)

# ---------------------------
# 3. Update labeled pool
# ---------------------------
updated_labeled_df = pd.concat([labeled_df, acquired_df], ignore_index=True)

print("\nAcquired batch size:", len(acquired_df))
print("Updated labeled pool size:", len(updated_labeled_df))
print("Remaining unlabeled pool size:", len(remaining_unlabeled_df))

print("\nAcquired batch label distribution:")
print(acquired_df["label"].value_counts())

# ---------------------------
# 4. Save round-2 paper pools
# ---------------------------
updated_labeled_df.to_csv(
    "../processed_data/hsbrexit_paper_labeled_pool_round2.csv",
    index=False
)

remaining_unlabeled_df.to_csv(
    "../processed_data/hsbrexit_paper_unlabeled_pool_round2.csv",
    index=False
)

print("\nSaved:")
print("- ../processed_data/hsbrexit_paper_labeled_pool_round2.csv")
print("- ../processed_data/hsbrexit_paper_unlabeled_pool_round2.csv")