"""
select_paper_round3_group_entropy_weighted.py

Purpose
-------
Round-3 active learning selection for the weighted ACTOR replication.

This script:
1. Loads the Round-2 weighted labeled pool and remaining unlabeled pool.
2. Loads the trained weighted ACTOR model from Round 2.
3. Scores each unique unlabeled item with group entropy.
4. Selects the top 60 most uncertain items.
5. Appends all annotations of those selected items to the labeled pool.
6. Saves the new Round-3 labeled and unlabeled pools.

Why this script exists
----------------------
We are following the same pipeline that worked for:
- Round 1 weighted group entropy
- Round 2 weighted group entropy

So this script is simply the Round-3 continuation of that same setup.

Important
---------
This script assumes:
- unlabeled data is annotation-level
- multiple rows may belong to the same item
- we score unique items once, then move all rows of selected items

Outputs
-------
- ../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round3.csv
- ../processed_data/hsbrexit_paper_group_weighted_unlabeled_pool_round3.csv
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
# Config
# ============================================================
SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 2
BATCH_SIZE = 16
ACQUISITION_SIZE = 60

LABELED_POOL_PATH = "../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round2.csv"
UNLABELED_POOL_PATH = "../processed_data/hsbrexit_paper_group_weighted_unlabeled_pool_round2.csv"
MODEL_STATE_PATH = "../outputs/actor_paper_group_weighted_round2/actor_model_state.pt"

OUTPUT_LABELED_PATH = "../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round3.csv"
OUTPUT_UNLABELED_PATH = "../processed_data/hsbrexit_paper_group_weighted_unlabeled_pool_round3.csv"

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
# Model
# ============================================================
class ACTORModel(nn.Module):
    """
    Weighted ACTOR architecture compatible with the saved Round-2 model.

    Important:
    - keep attribute names compatible with the training script
    - annotator_heads must match saved checkpoint
    """

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
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # [batch, num_heads, num_labels]
        logits_per_head = [head(cls_output) for head in self.annotator_heads]
        return torch.stack(logits_per_head, dim=1)


# ============================================================
# Helpers
# ============================================================
def entropy(prob_vector: np.ndarray) -> float:
    """
    Standard entropy of a probability vector.
    """
    prob_vector = np.clip(prob_vector, 1e-12, 1.0)
    return float(-np.sum(prob_vector * np.log(prob_vector)))


def infer_item_id_column(df: pd.DataFrame) -> str:
    """
    Detect item identifier column.

    We prefer item_id. If not found, we try text_id.
    """
    if "item_id" in df.columns:
        return "item_id"
    if "text_id" in df.columns:
        return "text_id"
    raise ValueError(
        f"Could not find an item identifier column. Available columns: {list(df.columns)}"
    )


def infer_text_column(df: pd.DataFrame) -> str:
    """
    Detect text column.
    """
    for candidate in ["text", "tweet", "sentence", "content"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find text column. Available columns: {list(df.columns)}"
    )


# ============================================================
# Load data
# ============================================================
labeled_df = pd.read_csv(LABELED_POOL_PATH)
unlabeled_df = pd.read_csv(UNLABELED_POOL_PATH)

print(f"Current labeled pool size: {len(labeled_df)}")
print(f"Current unlabeled pool size: {len(unlabeled_df)}")
print()

item_id_col = infer_item_id_column(unlabeled_df)
text_col = infer_text_column(unlabeled_df)

print("Using item identifier column:")
print(item_id_col)
print()

# Fixed annotator mapping used across your ACTOR replication
annotators = ["Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"]
annotator2idx = {ann: i for i, ann in enumerate(annotators)}

print("Annotator mapping:")
print(annotator2idx)
print()

num_annotators = len(annotators)


# ============================================================
# Load tokenizer + model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = ACTORModel(
    num_annotators=num_annotators,
    num_labels=NUM_LABELS,
)

state_dict = torch.load(MODEL_STATE_PATH, map_location="cpu")

# Some weighted checkpoints may contain extra keys such as loss_fn.weight or class_weights.
# Those are not needed for acquisition-time scoring.
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if k in model.state_dict()
}

model.load_state_dict(filtered_state_dict, strict=False)
model.to(DEVICE)
model.eval()

print("Loaded weighted ACTOR model from:")
print(MODEL_STATE_PATH)
print()


# ============================================================
# Build unique item-level unlabeled dataset
# ============================================================
item_df = unlabeled_df.drop_duplicates(subset=[item_id_col]).copy()
item_df = item_df.reset_index(drop=True)

print(f"Unique unlabeled items to score: {len(item_df)}")
print()

dataset = Dataset.from_pandas(item_df[[item_id_col, text_col]], preserve_index=False)

def tokenize_function(batch):
    return tokenizer(
        batch[text_col],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids"]
)


# ============================================================
# Group entropy scoring
# ============================================================
scores = []

for start_idx in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Scoring unlabeled pool"):
    batch = dataset[start_idx:start_idx + BATCH_SIZE]

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    token_type_ids = batch["token_type_ids"].to(DEVICE)

    with torch.no_grad():
        logits_all = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # [batch, num_heads, num_labels]

        probs_all = torch.softmax(logits_all, dim=-1)  # [batch, num_heads, num_labels]

        # Group entropy = entropy of mean predictive distribution across heads
        mean_probs = probs_all.mean(dim=1).cpu().numpy()  # [batch, num_labels]

        for i in range(mean_probs.shape[0]):
            scores.append(entropy(mean_probs[i]))

item_df["group_entropy"] = scores


# ============================================================
# Select top-k items
# ============================================================
selected_items = item_df.sort_values(
    by="group_entropy",
    ascending=False
).head(ACQUISITION_SIZE).copy()

selected_item_ids = set(selected_items[item_id_col].tolist())

selected_rows = unlabeled_df[unlabeled_df[item_id_col].isin(selected_item_ids)].copy()
remaining_unlabeled = unlabeled_df[~unlabeled_df[item_id_col].isin(selected_item_ids)].copy()
updated_labeled = pd.concat([labeled_df, selected_rows], ignore_index=True)

print(f"Top selected items: {len(selected_items)}")
print(f"Updated labeled pool size: {len(updated_labeled)}")
print(f"Remaining unlabeled pool size: {len(remaining_unlabeled)}")
print()

print("Top 10 entropy scores:")
print(
    selected_items[[item_id_col, text_col, "group_entropy"]]
    .head(10)
    .to_string(index=False)
)
print()

print("Selected label distribution (debug only):")
print(selected_rows["label"].value_counts().sort_index())
print()


# ============================================================
# Save
# ============================================================
updated_labeled.to_csv(OUTPUT_LABELED_PATH, index=False)
remaining_unlabeled.to_csv(OUTPUT_UNLABELED_PATH, index=False)

print("Saved:")
print(f"- {OUTPUT_LABELED_PATH}")
print(f"- {OUTPUT_UNLABELED_PATH}")