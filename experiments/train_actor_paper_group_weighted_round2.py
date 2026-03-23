# experiments/train_actor_paper_group_weighted_round2.py

"""
Round 2 weighted ACTOR training for the paper-style replication.

What this script does
---------------------
1. Loads the Round-2 weighted active-learning pool produced by:
   select_paper_round2_group_entropy_weighted.py

2. Loads the dev/test annotation-level splits that you have already been using
   in the rest of the replication.

3. Trains ACTOR with class-weighted cross-entropy so the model does not collapse
   to always predicting the majority class.

4. Evaluates on dev during training and on test after training.

5. Saves a clean model state_dict that can be used by the next acquisition script:
   select_paper_round3_group_entropy_weighted.py

Important note
--------------
This script follows the same structure as your successful
train_actor_paper_group_weighted_round1.py run, just updated for Round 2.

Expected input
--------------
- ../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round2.csv
- your existing dev file
- your existing test file

If your dev/test filenames differ, only change DEV_PATH and TEST_PATH below.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertModel,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Config
# ============================================================
SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 2
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# Round-2 weighted pool produced by your selection script
TRAIN_PATH = "../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round2.csv"

# IMPORTANT:
# Keep these equal to the SAME dev/test files you used before.
# From your logs, dev/test size was 1008 each.
# Replace only if your filenames are different.
DEV_PATH = "../processed_data/hsbrexit_dev_annotations.csv"
TEST_PATH = "../processed_data/hsbrexit_test_annotations.csv"

OUTPUT_DIR = "../outputs/actor_paper_group_weighted_round2"
MODEL_STATE_PATH = os.path.join(OUTPUT_DIR, "actor_model_state.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Model
# ============================================================
class ACTORModel(nn.Module):
    """
    ACTOR model with:
    - shared BERT encoder
    - annotator-specific classification heads
    - weighted cross-entropy loss
    """

    def __init__(self, num_annotators: int, num_labels: int, class_weights: torch.Tensor):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size

        # IMPORTANT:
        # Keep the same naming convention that your weighted checkpoints use.
        self.annotator_heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)]
        )

        # Register class weights as buffer so they move with the model
        self.register_buffer("class_weights", class_weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        annotator_idx=None,
    ):
        """
        Forward pass.

        During training:
            uses annotator_idx to select the correct annotator-specific head
            and computes weighted CE loss.

        During evaluation:
            still returns logits for the provided annotator head.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_repr = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # Build logits from all heads: [batch, num_annotators, num_labels]
        logits_all = torch.stack(
            [head(cls_repr) for head in self.annotator_heads],
            dim=1,
        )

        # Pick the correct annotator head per example
        batch_indices = torch.arange(logits_all.size(0), device=logits_all.device)
        logits = logits_all[batch_indices, annotator_idx]  # [batch, num_labels]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ============================================================
# Metrics
# ============================================================
def compute_metrics(eval_pred):
    """Compute evaluation metrics used throughout the replication."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


# ============================================================
# Data utilities
# ============================================================
def infer_text_column(df: pd.DataFrame) -> str:
    """Infer the text column name from the dataframe."""
    for candidate in ["text", "tweet", "sentence", "content"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find text column. Available columns: {list(df.columns)}"
    )


def infer_annotator_column(df: pd.DataFrame) -> str:
    """Infer annotator column name."""
    for candidate in ["annotator", "worker_id", "annotator_id", "worker"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find annotator column. Available columns: {list(df.columns)}"
    )


def prepare_dataframe(df: pd.DataFrame, text_col: str, annotator_col: str, annotator2idx: dict) -> pd.DataFrame:
    """
    Prepare dataframe for tokenization/training.

    Required output columns:
    - text
    - labels
    - annotator_idx
    """
    out = df.copy()

    out = out.rename(columns={text_col: "text"})
    out["labels"] = out["label"].astype(int)
    out["annotator_idx"] = out[annotator_col].map(annotator2idx)

    if out["annotator_idx"].isna().any():
        bad_values = sorted(out.loc[out["annotator_idx"].isna(), annotator_col].unique())
        raise ValueError(f"Unknown annotators found: {bad_values}")

    out["annotator_idx"] = out["annotator_idx"].astype(int)
    return out


def tokenize_dataframe(df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    """Convert dataframe to HuggingFace Dataset and tokenize it."""
    dataset = Dataset.from_pandas(df[["text", "labels", "annotator_idx"]], preserve_index=False)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(
        type="torch",
        columns=["labels", "annotator_idx", "input_ids", "token_type_ids", "attention_mask"],
    )
    return dataset


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights:
        weight_c = N / (num_classes * count_c)
    """
    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    num_classes = len(label_counts)

    weights = []
    for label_id in range(NUM_LABELS):
        count = label_counts.get(label_id, 0)
        if count == 0:
            raise ValueError(f"Class {label_id} has zero count in training pool.")
        weight = total / (num_classes * count)
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return weights_tensor


# ============================================================
# Load data
# ============================================================
train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Paper group weighted round-2 labeled pool size: {len(train_df)}")
print(f"Dev size: {len(dev_df)}")
print(f"Test size: {len(test_df)}")
print()

print("Round-2 pool label distribution:")
print(train_df["label"].value_counts().sort_index())
print()

# Infer columns
train_text_col = infer_text_column(train_df)
dev_text_col = infer_text_column(dev_df)
test_text_col = infer_text_column(test_df)

train_annotator_col = infer_annotator_column(train_df)
dev_annotator_col = infer_annotator_column(dev_df)
test_annotator_col = infer_annotator_column(test_df)

# Build annotator mapping from full set used in the project
annotators = sorted(train_df[train_annotator_col].unique())
annotator2idx = {ann: i for i, ann in enumerate(annotators)}

print("Annotator mapping:")
print(annotator2idx)
print()

# Class weights
class_weights = compute_class_weights(train_df)
print("Class weights:")
for i, w in enumerate(class_weights.tolist()):
    print(f"class {i}: {w:.4f}")
print()

# Prepare dataframes
train_df = prepare_dataframe(train_df, train_text_col, train_annotator_col, annotator2idx)
dev_df = prepare_dataframe(dev_df, dev_text_col, dev_annotator_col, annotator2idx)
test_df = prepare_dataframe(test_df, test_text_col, test_annotator_col, annotator2idx)

# Tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = tokenize_dataframe(train_df, tokenizer)
dev_dataset = tokenize_dataframe(dev_df, tokenizer)
test_dataset = tokenize_dataframe(test_df, tokenizer)

print("Final columns:")
print(train_dataset.column_names)
print()

# ============================================================
# Model
# ============================================================
model = ACTORModel(
    num_annotators=len(annotator2idx),
    num_labels=NUM_LABELS,
    class_weights=class_weights,
)
model.to(DEVICE)

# ============================================================
# Training args
# ============================================================
# NOTE:
# Your transformers version is older, so we keep this minimal and compatible.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=max(1, len(train_dataset) // BATCH_SIZE),
    eval_steps=max(1, len(train_dataset) // BATCH_SIZE),
    save_steps=max(1, len(train_dataset) // BATCH_SIZE),
    do_eval=True,
    save_strategy="no",
    seed=SEED,
)

# ============================================================
# Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

print("Trainer created successfully.")
print()
print("Starting training on paper group weighted round-2 pool...")

trainer.train()

# ============================================================
# Final evaluation
# ============================================================
print()
print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)

print()
print("Test results:")
print(test_results)
print()

# ============================================================
# Save clean state dict
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save only the model parameters needed later
clean_state_dict = {
    k: v.detach().cpu()
    for k, v in model.state_dict().items()
}

torch.save(clean_state_dict, MODEL_STATE_PATH)

print("Saved trained ACTOR weights:")
print(MODEL_STATE_PATH)