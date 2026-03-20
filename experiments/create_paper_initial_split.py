"""
Create the paper-matched initial split for HS-Brexit.

Paper setting:
- initial labeled pool = 60
- rest = unlabeled pool

This script does NOT overwrite your old data.
It creates NEW paper-specific files.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Load full training data
# ---------------------------
train_path = "../processed_data/hsbrexit_train_annotations.csv"
train_df = pd.read_csv(train_path)

print("Full training set size:", len(train_df))

print("\nFull label distribution:")
print(train_df["label"].value_counts())

# ---------------------------
# 2. Paper split
# ---------------------------
initial_pool_size = 60
random_state = 42

labeled_df, unlabeled_df = train_test_split(
    train_df,
    train_size=initial_pool_size,
    random_state=random_state,
    stratify=train_df["label"]
)

print("\nInitial labeled pool size:", len(labeled_df))
print("Unlabeled pool size:", len(unlabeled_df))

print("\nLabeled pool distribution:")
print(labeled_df["label"].value_counts())

# ---------------------------
# 3. Save NEW files
# ---------------------------
labeled_df.to_csv(
    "../processed_data/hsbrexit_paper_initial_labeled_pool.csv",
    index=False
)

unlabeled_df.to_csv(
    "../processed_data/hsbrexit_paper_unlabeled_pool.csv",
    index=False
)

print("\nSaved files:")
print("✔ hsbrexit_paper_initial_labeled_pool.csv")
print("✔ hsbrexit_paper_unlabeled_pool.csv")