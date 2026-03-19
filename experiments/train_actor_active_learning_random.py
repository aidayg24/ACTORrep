"""
ACTOR active learning with random acquisition (skeleton).

What this script does
---------------------
This is the first active learning version of the ACTOR replication.

We start with the simplest acquisition strategy: RANDOM SAMPLING.

Why start with random?
----------------------
Before implementing the paper's acquisition strategies, we first need a working
active learning loop:

1. split data into an initial labeled pool and an unlabeled pool
2. train ACTOR on the labeled pool
3. randomly acquire new examples from the unlabeled pool
4. add them to the labeled pool
5. repeat for several rounds

This script only implements the data split for now.
It does NOT train yet.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Load annotation-level training data
# ---------------------------
train_path = "../processed_data/hsbrexit_train_annotations.csv"
train_df = pd.read_csv(train_path)

print("Full training set size:", len(train_df))
print("\nLabel distribution in full training set:")
print(train_df["label"].value_counts())

# ---------------------------
# 2. Create initial labeled pool and unlabeled pool
# For now we start with a simple random split
# ---------------------------
initial_pool_size = 600   # first labeled pool
random_state = 42

initial_labeled_df, unlabeled_df = train_test_split(
    train_df,
    train_size=initial_pool_size,
    random_state=random_state,
    stratify=train_df["label"]
)

print("\nInitial labeled pool size:", len(initial_labeled_df))
print("Unlabeled pool size:", len(unlabeled_df))

print("\nLabel distribution in initial labeled pool:")
print(initial_labeled_df["label"].value_counts())

print("\nLabel distribution in unlabeled pool:")
print(unlabeled_df["label"].value_counts())

# ---------------------------
# 3. Save split datasets
# ---------------------------
initial_labeled_df.to_csv("../processed_data/hsbrexit_initial_labeled_pool.csv", index=False)
unlabeled_df.to_csv("../processed_data/hsbrexit_unlabeled_pool.csv", index=False)

print("\nSaved:")
print("- ../processed_data/hsbrexit_initial_labeled_pool.csv")
print("- ../processed_data/hsbrexit_unlabeled_pool.csv")