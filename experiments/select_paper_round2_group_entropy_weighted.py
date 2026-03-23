"""
select_paper_round2_group_entropy_weighted.py

Purpose
-------
This script performs the NEXT active-learning acquisition step for the
weighted ACTOR replication setup.

What it does
------------
1. Loads the current labeled pool from weighted group round 1.
2. Loads the current unlabeled pool after weighted group round 1.
3. Loads the trained weighted ACTOR checkpoint from weighted group round 1.
4. Scores each UNIQUE unlabeled item using GROUP ENTROPY:
      - get probabilities from all annotator-specific heads
      - average those probabilities across heads
      - compute entropy of the averaged distribution
5. Selects the top-k most uncertain items.
6. Adds ALL annotation rows for selected items to the labeled pool.
7. Removes those items from the unlabeled pool.
8. Saves the new round-2 labeled and unlabeled pools.

Why this matches the replication direction
------------------------------------------
This follows the ACTOR-style active learning direction:
- annotator-specific heads
- uncertainty-based acquisition
- item-level selection
- moving selected items from unlabeled pool to labeled pool

Important
---------
This script is for the WEIGHTED pipeline.
It uses the checkpoint from:
    actor_paper_group_weighted_round1
and produces:
    hsbrexit_paper_group_weighted_labeled_pool_round2.csv
    hsbrexit_paper_group_weighted_unlabeled_pool_round2.csv
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

# Input pools from WEIGHTED GROUP ROUND 1
LABELED_POOL_PATH = "../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round1.csv"
UNLABELED_POOL_PATH = "../processed_data/hsbrexit_paper_group_weighted_unlabeled_pool_round1.csv"

# Trained model from WEIGHTED GROUP ROUND 1
MODEL_STATE_PATH = "../outputs/actor_paper_group_weighted_round1/actor_model_state.pt"

# Output pools for WEIGHTED GROUP ROUND 2
OUTPUT_LABELED_PATH = "../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round2.csv"
OUTPUT_UNLABELED_PATH = "../processed_data/hsbrexit_paper_group_weighted_unlabeled_pool_round2.csv"

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
# IMPORTANT:
# This must match the naming used in your SAVED checkpoint:
#   - self.bert
#   - self.annotator_heads
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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS embedding

        # shape of each head output: [batch_size, num_labels]
        logits_per_head = [head(cls_output) for head in self.annotator_heads]

        # stack to: [batch_size, num_heads, num_labels]
        return torch.stack(logits_per_head, dim=1)

# ============================================================
# Utility functions
# ============================================================
def entropy(prob_vector):
    """
    Compute entropy of a probability vector.

    Entropy is highest when the model is maximally uncertain.
    """
    prob_vector = np.clip(prob_vector, 1e-12, 1.0)
    return -np.sum(prob_vector * np.log(prob_vector))

def detect_item_id_column(df):
    """
    Detect the column that identifies unique items/tweets.

    We prefer a true item id if it exists. Otherwise we fall back to text.
    """
    candidate_cols = ["text_id", "item_id", "tweet_id", "id"]
    for col in candidate_cols:
        if col in df.columns:
            return col
    return "text"

# ============================================================
# Load data
# ============================================================
labeled_df = pd.read_csv(LABELED_POOL_PATH)
unlabeled_df = pd.read_csv(UNLABELED_POOL_PATH)

print(f"Current labeled pool size: {len(labeled_df)}")
print(f"Current unlabeled pool size: {len(unlabeled_df)}")
print()

# Fixed annotator mapping for HS-Brexit ACTOR replication
annotators = ["Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"]
annotator2idx = {ann: i for i, ann in enumerate(annotators)}

print("Annotator mapping:")
print(annotator2idx)
print()

num_annotators = len(annotators)

# ============================================================
# Detect item id column
# ============================================================
item_id_col = detect_item_id_column(unlabeled_df)

if item_id_col == "text":
    print("No explicit item id column found; using text as item identifier.")
else:
    print(f"Using '{item_id_col}' as item identifier.")
print()

# ============================================================
# Load tokenizer and trained model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = ACTORModel(
    num_annotators=num_annotators,
    num_labels=NUM_LABELS
)


state_dict = torch.load(MODEL_STATE_PATH, map_location="cpu")

# Some weighted training checkpoints store loss-function weights.
# These are not model parameters needed for acquisition-time inference.
keys_to_remove = ["loss_fn.weight", "class_weights"]
for k in keys_to_remove:
    if k in state_dict:
        del state_dict[k]

model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()

print(f"Loaded weighted ACTOR model from:\n{MODEL_STATE_PATH}")
print()

# ============================================================
# Build unique unlabeled item table
# We score each UNIQUE item only once.
# ============================================================
item_df = unlabeled_df.drop_duplicates(subset=[item_id_col]).copy()
item_df = item_df.reset_index(drop=True)

print(f"Unique unlabeled items to score: {len(item_df)}")
print()

# We need the text column for tokenization
if "text" not in item_df.columns:
    raise ValueError("Expected a 'text' column in the unlabeled pool, but none was found.")

# ============================================================
# Tokenization
# ============================================================
def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

dataset = Dataset.from_pandas(item_df[[item_id_col, "text"]])
dataset = dataset.map(tokenize_function, batched=True)

# token_type_ids may not always exist for every tokenizer, but BERT uses them
columns_to_keep = ["input_ids", "attention_mask"]
if "token_type_ids" in dataset.column_names:
    columns_to_keep.append("token_type_ids")

dataset.set_format(
    type="torch",
    columns=columns_to_keep
)

# ============================================================
# Group entropy scoring
# ============================================================
scores = []

for start_idx in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Scoring unlabeled pool"):
    batch = dataset[start_idx:start_idx + BATCH_SIZE]

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    if "token_type_ids" in batch:
        token_type_ids = batch["token_type_ids"].to(DEVICE)
    else:
        token_type_ids = None

    with torch.no_grad():
        # logits shape: [batch_size, num_heads, num_labels]
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # convert to probabilities across labels
        probs = torch.softmax(logits, dim=-1)

        # average across annotator heads
        # shape becomes: [batch_size, num_labels]
        mean_probs = probs.mean(dim=1).cpu().numpy()

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

selected_item_ids = set(selected_items[item_id_col].tolist())

# Add ALL rows belonging to the selected items
selected_rows = unlabeled_df[unlabeled_df[item_id_col].isin(selected_item_ids)].copy()
remaining_unlabeled = unlabeled_df[~unlabeled_df[item_id_col].isin(selected_item_ids)].copy()
updated_labeled = pd.concat([labeled_df, selected_rows], ignore_index=True)

print(f"Top selected items: {len(selected_items)}")
print(f"Updated labeled pool size: {len(updated_labeled)}")
print(f"Remaining unlabeled pool size: {len(remaining_unlabeled)}")
print()

print("Top 10 entropy scores:")
cols_to_show = [item_id_col, "text", "group_entropy"] if item_id_col != "text" else ["text", "group_entropy"]
print(selected_items[cols_to_show].head(10).to_string(index=False))
print()

if "label" in selected_rows.columns:
    print("Selected label distribution (debug only):")
    print(selected_rows["label"].value_counts().sort_index())
    print()

# ============================================================
# Save updated pools
# ============================================================
updated_labeled.to_csv(OUTPUT_LABELED_PATH, index=False)
remaining_unlabeled.to_csv(OUTPUT_UNLABELED_PATH, index=False)

print("Saved:")
print(f"- {OUTPUT_LABELED_PATH}")
print(f"- {OUTPUT_UNLABELED_PATH}")