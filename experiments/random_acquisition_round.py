"""
One random acquisition round for ACTOR active learning.

What this script does
---------------------
This script performs one active learning acquisition step using RANDOM sampling.

Starting point:
- initial labeled pool
- unlabeled pool

Action:
- randomly sample a batch from the unlabeled pool
- add that batch to the labeled pool
- remove that batch from the unlabeled pool

This script only updates the pools.
It does NOT retrain the model yet.
"""

import pandas as pd

# ---------------------------
# 1. Load current pools
# ---------------------------
labeled_path = "../processed_data/hsbrexit_initial_labeled_pool.csv"
unlabeled_path = "../processed_data/hsbrexit_unlabeled_pool.csv"

labeled_df = pd.read_csv(labeled_path)
unlabeled_df = pd.read_csv(unlabeled_path)

print("Current labeled pool size:", len(labeled_df))
print("Current unlabeled pool size:", len(unlabeled_df))

# ---------------------------
# 2. Randomly acquire a batch
# ---------------------------
acquisition_size = 200
random_state = 42

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
# 4. Save updated pools
# ---------------------------
updated_labeled_df.to_csv("../processed_data/hsbrexit_labeled_pool_round1.csv", index=False)
remaining_unlabeled_df.to_csv("../processed_data/hsbrexit_unlabeled_pool_round1.csv", index=False)

print("\nSaved:")
print("- ../processed_data/hsbrexit_labeled_pool_round1.csv")
print("- ../processed_data/hsbrexit_unlabeled_pool_round1.csv")