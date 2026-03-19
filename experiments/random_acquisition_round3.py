"""
One random acquisition round for ACTOR active learning: round 2.

What this script does
---------------------
This script continues the active learning pipeline.

Input:
- labeled pool after round 1
- unlabeled pool after round 1

Action:
- randomly sample 200 new examples from the unlabeled pool
- add them to the labeled pool
- save the updated round-3 pools
"""

import pandas as pd

# ---------------------------
# 1. Load round-1 pools
# ---------------------------
labeled_path = "../processed_data/hsbrexit_labeled_pool_round2.csv"
unlabeled_path = "../processed_data/hsbrexit_unlabeled_pool_round2.csv"

labeled_df = pd.read_csv(labeled_path)
unlabeled_df = pd.read_csv(unlabeled_path)

print("Current labeled pool size:", len(labeled_df))
print("Current unlabeled pool size:", len(unlabeled_df))

# ---------------------------
# 2. Randomly acquire a new batch
# ---------------------------
random_state = 44
acquisition_size = 200

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
# 4. Save round-2 pools
# ---------------------------
updated_labeled_df.to_csv("../processed_data/hsbrexit_labeled_pool_round3.csv", index=False)
remaining_unlabeled_df.to_csv("../processed_data/hsbrexit_unlabeled_pool_round3.csv", index=False)

print("\nSaved:")
print("- ../processed_data/hsbrexit_labeled_pool_round3.csv")
print("- ../processed_data/hsbrexit_unlabeled_pool_round3.csv")