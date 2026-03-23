"""
select_paper_round1_group_entropy_weighted.py

Purpose
-------
This script performs the FIRST active-learning acquisition step for the
paper-style weighted ACTOR replication.

It does the following:
1. Loads the initial labeled pool (paper split, 60 items).
2. Loads the corresponding unlabeled pool.
3. Loads the WEIGHTED ACTOR checkpoint trained on that initial pool.
4. Scores each unlabeled item with GROUP-LEVEL ENTROPY.
5. Selects the top 60 most uncertain items.
6. Moves them from unlabeled -> labeled.
7. Saves the new round-1 labeled/unlabeled pools.

Why this script exists
----------------------
We are following the ACTOR paper direction:
- multi-head model with annotator-specific heads
- acquisition based on uncertainty
- specifically: group-level entropy

Important implementation note
-----------------------------
Your current paper CSV files do NOT contain a `text_id` column.
So in this script, we treat each unique `text` as the item identity.

This matches the practical structure of the data you currently have.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 2
BATCH_SIZE = 16
ACQUISITION_SIZE = 60

# Input pools
LABELED_POOL_PATH = "../processed_data/hsbrexit_paper_initial_labeled_pool.csv"
UNLABELED_POOL_PATH = "../processed_data/hsbrexit_paper_unlabeled_pool.csv"

# Weighted checkpoint trained on initial paper pool
MODEL_STATE_PATH = "../outputs/actor_initial_pool_paper_weighted/actor_model_state.pt"

# Output pools after acquisition
OUTPUT_LABELED_PATH = "../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round1.csv"
OUTPUT_UNLABELED_PATH = "../processed_data/hsbrexit_paper_group_weighted_unlabeled_pool_round1.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# ACTOR model definition
# ============================================================
# IMPORTANT:
# This must match the architecture used in your weighted training script.
#
# From your checkpoint errors, we know:
# - encoder is stored under `bert.*`
# - heads are stored under `annotator_heads.*`
#
# So we define the model exactly that way here.
# ============================================================

class ACTORModel(nn.Module):
    def __init__(self, num_annotators, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size

        self.annotator_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)
        ])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Returns:
            logits: tensor of shape [batch_size, num_annotators, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        logits_per_head = []
        for head in self.annotator_heads:
            logits_per_head.append(head(cls_output))  # [batch, num_labels]

        logits = torch.stack(logits_per_head, dim=1)
        return logits


# ============================================================
# Load data
# ============================================================

labeled_df = pd.read_csv(LABELED_POOL_PATH)
unlabeled_df = pd.read_csv(UNLABELED_POOL_PATH)

print(f"Current labeled pool size: {len(labeled_df)}")
print(f"Current unlabeled pool size: {len(unlabeled_df)}")
print()

# Basic column checks
if "text" not in labeled_df.columns:
    raise ValueError(f"`text` column not found in labeled pool: {LABELED_POOL_PATH}")

if "text" not in unlabeled_df.columns:
    raise ValueError(f"`text` column not found in unlabeled pool: {UNLABELED_POOL_PATH}")

# We keep the annotator mapping explicit because ACTOR uses annotator-specific heads.
annotators = ["Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"]
annotator2idx = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

print("Annotator mapping:")
print(annotator2idx)
print()


# ============================================================
# Load tokenizer and trained weighted ACTOR checkpoint
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = ACTORModel(
    num_annotators=num_annotators,
    num_labels=NUM_LABELS
)

# Load checkpoint
state_dict = torch.load(MODEL_STATE_PATH, map_location="cpu")

# Weighted checkpoints may contain class_weights.
# That tensor is not needed for inference-time acquisition scoring.
if "class_weights" in state_dict:
    del state_dict["class_weights"]

# Strict loading is good here because we WANT exact architecture match.
model.load_state_dict(state_dict, strict=True)

model.to(DEVICE)
model.eval()

print(f"Loaded weighted ACTOR model from:\n{MODEL_STATE_PATH}")
print()


# ============================================================
# Build item-level unlabeled dataset
# ============================================================
# Your current CSV has no `text_id`, so we use unique text as the item identity.
#
# Why?
# In ACTOR acquisition, we want to score each ITEM once.
# Since your pool stores repeated annotation rows but lacks text_id,
# unique text is the safest available proxy.
# ============================================================

item_df = unlabeled_df.drop_duplicates(subset=["text"]).copy().reset_index(drop=True)

print(f"Unique unlabeled items to score: {len(item_df)}")
print()


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


dataset = Dataset.from_pandas(item_df[["text"]], preserve_index=False)
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids"]
)


# ============================================================
# Group-level entropy
# ============================================================
# ACTOR paper uses uncertainty from annotator-specific heads.
# For group-level entropy:
#   1. get each head's probability distribution
#   2. average across heads
#   3. compute entropy of that mean distribution
#
# High entropy = high uncertainty = good AL candidate
# ============================================================

def entropy(prob_vector: np.ndarray) -> float:
    prob_vector = np.clip(prob_vector, 1e-12, 1.0)
    return float(-np.sum(prob_vector * np.log(prob_vector)))


scores = []

for start_idx in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Scoring unlabeled pool"):
    batch = dataset[start_idx:start_idx + BATCH_SIZE]

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    token_type_ids = batch["token_type_ids"].to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )  # [batch, num_heads, num_labels]

        probs = torch.softmax(logits, dim=-1)        # [batch, num_heads, num_labels]
        mean_probs = probs.mean(dim=1).cpu().numpy() # [batch, num_labels]

        for i in range(mean_probs.shape[0]):
            scores.append(entropy(mean_probs[i]))

item_df["group_entropy"] = scores


# ============================================================
# Select top-k most uncertain items
# ============================================================

selected_items = item_df.sort_values(
    by="group_entropy",
    ascending=False
).head(ACQUISITION_SIZE).copy()

selected_texts = set(selected_items["text"].tolist())

# Move all rows corresponding to selected texts from unlabeled -> labeled
selected_rows = unlabeled_df[unlabeled_df["text"].isin(selected_texts)].copy()
remaining_unlabeled = unlabeled_df[~unlabeled_df["text"].isin(selected_texts)].copy()
updated_labeled = pd.concat([labeled_df, selected_rows], ignore_index=True)


# ============================================================
# Print debug info
# ============================================================

print(f"Top selected items: {len(selected_items)}")
print(f"Updated labeled pool size: {len(updated_labeled)}")
print(f"Remaining unlabeled pool size: {len(remaining_unlabeled)}")
print()

print("Top 10 entropy scores:")
print(selected_items[["text", "group_entropy"]].head(10).to_string(index=False))
print()

if "label" in selected_rows.columns:
    print("Selected label distribution (debug only):")
    print(selected_rows["label"].value_counts().sort_index())
    print()


# ============================================================
# Save outputs
# ============================================================

updated_labeled.to_csv(OUTPUT_LABELED_PATH, index=False)
remaining_unlabeled.to_csv(OUTPUT_UNLABELED_PATH, index=False)

print("Saved:")
print(f"- {OUTPUT_LABELED_PATH}")
print(f"- {OUTPUT_UNLABELED_PATH}")