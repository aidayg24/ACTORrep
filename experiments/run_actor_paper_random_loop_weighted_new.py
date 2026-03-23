"""
run_actor_paper_random_loop_weighted_new.py

Purpose
-------
Run the weighted RANDOM active-learning baseline for the ACTOR replication.

What this script does
---------------------
This script starts from the same paper-style initial pool that you already used
for the other experiments and then repeats the following process:

Round 0:
    - train weighted ACTOR on the initial labeled pool
    - evaluate on dev and test

After each round:
    - randomly sample 60 NEW unique items from the unlabeled pool
    - move all annotation rows for those items into the labeled pool

Rounds 1..4:
    - train weighted ACTOR on the updated labeled pool
    - evaluate on dev and test

Why this script matters
-----------------------
You already have a strong weighted group-entropy pipeline.
This script gives you the matching weighted RANDOM baseline so you can compare:

    weighted random  vs  weighted group entropy

This comparison is one of the most important parts of the replication.

Important assumptions
---------------------
1. Training data is annotation-level.
2. Multiple rows can belong to the same item.
3. We randomly acquire unique items, then move all rows for those items.
4. Dev/test must be annotation-level ACTOR splits (not the 168-item majority files).

Outputs
-------
1. Per-round model checkpoints:
   ../outputs/actor_paper_random_loop_weighted/round_{k}/actor_model_state.pt

2. Per-round pool snapshots:
   ../processed_data/hsbrexit_paper_random_weighted_labeled_pool_round{k}.csv
   ../processed_data/hsbrexit_paper_random_weighted_unlabeled_pool_round{k}.csv

3. Final summary file:
   ../results/results_actor_paper_random_loop_weighted.txt
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

# Start from the same initial paper-style split you already created earlier
INITIAL_LABELED_POOL_PATH = "../processed_data/hsbrexit_paper_initial_labeled_pool.csv"
INITIAL_UNLABELED_POOL_PATH = "../processed_data/hsbrexit_paper_unlabeled_pool.csv"

# IMPORTANT:
# These must be the same annotation-level dev/test files that worked
# in your weighted group-entropy training scripts.
DEV_PATH = "../processed_data/hsbrexit_dev_annotations.csv"
TEST_PATH = "../processed_data/hsbrexit_test_annotations.csv"

# Number of active-learning acquisitions after round 0.
# With NUM_ACQUISITIONS = 4, the script runs:
#   round 0, round 1, round 2, round 3, round 4
NUM_ACQUISITIONS = 4
ACQUISITION_SIZE = 60

OUTPUT_BASE_DIR = "../outputs/actor_paper_random_loop_weighted"
RESULTS_PATH = "../results/results_actor_paper_random_loop_weighted_new.txt"

PROCESSED_DATA_DIR = "../processed_data"

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
    Weighted ACTOR model:
    - shared BERT encoder
    - annotator-specific heads
    - weighted cross-entropy

    Naming is kept compatible with the weighted scripts you already used.
    """

    def __init__(self, num_annotators: int, num_labels: int, class_weights: torch.Tensor):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size

        self.annotator_heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)]
        )

        # Store class weights as a buffer so they move with the model
        self.register_buffer("class_weights", class_weights)

        # Weighted loss used during training
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        annotator_idx=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_repr = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        logits_all = torch.stack(
            [head(cls_repr) for head in self.annotator_heads],
            dim=1,
        )  # [batch, num_annotators, num_labels]

        # Select the correct head for each example
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
    """Compute standard classification metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


# ============================================================
# Helpers: dataframe schema detection
# ============================================================
def infer_text_column(df: pd.DataFrame) -> str:
    """Infer text column name."""
    for candidate in ["text", "tweet", "sentence", "content"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find text column. Available columns: {list(df.columns)}")


def infer_annotator_column(df: pd.DataFrame) -> str:
    """Infer annotator column name."""
    for candidate in ["annotator", "worker_id", "annotator_id", "worker"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find annotator column. Available columns: {list(df.columns)}")


def infer_item_id_column(df: pd.DataFrame) -> str:
    """Infer item identifier column name."""
    if "item_id" in df.columns:
        return "item_id"
    if "text_id" in df.columns:
        return "text_id"
    raise ValueError(
        f"Could not find item identifier column. Available columns: {list(df.columns)}"
    )


# ============================================================
# Helpers: data prep
# ============================================================
def prepare_dataframe(
    df: pd.DataFrame,
    text_col: str,
    annotator_col: str,
    annotator2idx: dict,
) -> pd.DataFrame:
    """
    Standardize a dataframe into the format needed by ACTOR.

    Expected final columns:
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
    """
    Convert a prepared dataframe into a tokenized Hugging Face dataset.
    """
    dataset = Dataset.from_pandas(
        df[["text", "labels", "annotator_idx"]],
        preserve_index=False,
    )

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

    return torch.tensor(weights, dtype=torch.float)


def save_clean_state_dict(model: nn.Module, save_path: str) -> None:
    """
    Save a clean state dict for later loading.

    We intentionally exclude loss_fn.* because it duplicates the class weights
    and causes shared-tensor issues in some environments.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    clean_state_dict = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("loss_fn.")
    }
    torch.save(clean_state_dict, save_path)


# ============================================================
# Helpers: random acquisition
# ============================================================
def random_acquire_items(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    item_id_col: str,
    acquisition_size: int,
    seed: int,
):
    """
    Randomly sample unique items from the unlabeled pool and move all rows
    belonging to those items into the labeled pool.
    """
    unique_items = unlabeled_df[item_id_col].drop_duplicates().tolist()

    if acquisition_size > len(unique_items):
        raise ValueError(
            f"Cannot acquire {acquisition_size} items from only {len(unique_items)} remaining unique items."
        )

    rng = random.Random(seed)
    selected_item_ids = set(rng.sample(unique_items, acquisition_size))

    selected_rows = unlabeled_df[unlabeled_df[item_id_col].isin(selected_item_ids)].copy()
    remaining_unlabeled = unlabeled_df[~unlabeled_df[item_id_col].isin(selected_item_ids)].copy()
    updated_labeled = pd.concat([labeled_df, selected_rows], ignore_index=True)

    return updated_labeled, remaining_unlabeled, selected_rows


# ============================================================
# Load data once
# ============================================================
train_full_labeled_df = pd.read_csv(INITIAL_LABELED_POOL_PATH)
train_full_unlabeled_df = pd.read_csv(INITIAL_UNLABELED_POOL_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Initial labeled pool size: {len(train_full_labeled_df)}")
print(f"Initial unlabeled pool size: {len(train_full_unlabeled_df)}")
print(f"Dev size: {len(dev_df)}")
print(f"Test size: {len(test_df)}")
print()

print("Initial pool label distribution:")
print(train_full_labeled_df["label"].value_counts().sort_index())
print()

# Infer schema
train_text_col = infer_text_column(train_full_labeled_df)
dev_text_col = infer_text_column(dev_df)
test_text_col = infer_text_column(test_df)

train_annotator_col = infer_annotator_column(train_full_labeled_df)
dev_annotator_col = infer_annotator_column(dev_df)
test_annotator_col = infer_annotator_column(test_df)

item_id_col = infer_item_id_column(train_full_unlabeled_df)

# Fixed annotator setup for ACTOR on HS-Brexit
annotators = sorted(train_full_labeled_df[train_annotator_col].unique())
annotator2idx = {ann: i for i, ann in enumerate(annotators)}

print("Annotator mapping:")
print(annotator2idx)
print()

print("Using item identifier column:")
print(item_id_col)
print()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prepare dev/test once
dev_prepared_df = prepare_dataframe(dev_df, dev_text_col, dev_annotator_col, annotator2idx)
test_prepared_df = prepare_dataframe(test_df, test_text_col, test_annotator_col, annotator2idx)

dev_dataset = tokenize_dataframe(dev_prepared_df, tokenizer)
test_dataset = tokenize_dataframe(test_prepared_df, tokenizer)


# ============================================================
# Active-learning loop
# ============================================================
results = []

current_labeled_df = train_full_labeled_df.copy()
current_unlabeled_df = train_full_unlabeled_df.copy()

# We run rounds 0..NUM_ACQUISITIONS
# Example: with NUM_ACQUISITIONS = 4, we run rounds 0,1,2,3,4
for round_idx in range(NUM_ACQUISITIONS + 1):
    print("=" * 60)
    print(f"ROUND {round_idx}")
    print("=" * 60)

    print(f"Labeled pool size: {len(current_labeled_df)}")
    print(f"Unlabeled pool size: {len(current_unlabeled_df)}")
    print("Labeled pool distribution:")
    print(current_labeled_df["label"].value_counts().sort_index())
    print()

    # --------------------------------------------------------
    # Save current pool snapshot before training
    # --------------------------------------------------------
    labeled_snapshot_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"hsbrexit_paper_random_weighted_labeled_pool_round{round_idx}.csv",
    )
    unlabeled_snapshot_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"hsbrexit_paper_random_weighted_unlabeled_pool_round{round_idx}.csv",
    )

    current_labeled_df.to_csv(labeled_snapshot_path, index=False)
    current_unlabeled_df.to_csv(unlabeled_snapshot_path, index=False)

    # --------------------------------------------------------
    # Prepare train dataset for this round
    # --------------------------------------------------------
    class_weights = compute_class_weights(current_labeled_df)

    print("Class weights for this round:")
    for i, w in enumerate(class_weights.tolist()):
        print(f"class {i}: {w:.4f}")
    print()

    train_prepared_df = prepare_dataframe(
        current_labeled_df,
        train_text_col,
        train_annotator_col,
        annotator2idx,
    )

    train_dataset = tokenize_dataframe(train_prepared_df, tokenizer)

    # --------------------------------------------------------
    # Build round model
    # --------------------------------------------------------
    model = ACTORModel(
        num_annotators=len(annotator2idx),
        num_labels=NUM_LABELS,
        class_weights=class_weights,
    )
    model.to(DEVICE)

    round_output_dir = os.path.join(OUTPUT_BASE_DIR, f"round_{round_idx}")
    round_model_path = os.path.join(round_output_dir, "actor_model_state.pt")

    # --------------------------------------------------------
    # Training args
    # --------------------------------------------------------
    # save_strategy="no" is essential:
    # it prevents the shared-tensor safetensors crash that you hit before.
    training_args = TrainingArguments(
        output_dir=round_output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=max(1, len(train_dataset) // BATCH_SIZE),
        eval_steps=max(1, len(train_dataset) // BATCH_SIZE),
        save_steps=max(1, len(train_dataset) // BATCH_SIZE),
        do_eval=False,
        save_strategy="no",
        seed=SEED + round_idx,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()
    print()

    # --------------------------------------------------------
    # Save clean model state
    # --------------------------------------------------------
    save_clean_state_dict(model, round_model_path)

    print("Saved ACTOR state dict to:")
    print(round_model_path)
    print()

    # --------------------------------------------------------
    # Evaluate on dev and test
    # --------------------------------------------------------
    print("Evaluating on dev...")
    dev_results = trainer.evaluate(dev_dataset)

    print()
    print("Evaluating on test...")
    test_results = trainer.evaluate(test_dataset)
    print()

    round_summary = {
        "round": round_idx,
        "labeled_pool_size": len(current_labeled_df),
        "dev_accuracy": dev_results["eval_accuracy"],
        "dev_macro_f1": dev_results["eval_macro_f1"],
        "dev_weighted_f1": dev_results["eval_weighted_f1"],
        "test_accuracy": test_results["eval_accuracy"],
        "test_macro_f1": test_results["eval_macro_f1"],
        "test_weighted_f1": test_results["eval_weighted_f1"],
    }
    results.append(round_summary)

    print("Round summary:")
    print(round_summary)
    print()

    # --------------------------------------------------------
    # If this is the final round, stop here
    # --------------------------------------------------------
    if round_idx == NUM_ACQUISITIONS:
        break

    # --------------------------------------------------------
    # Otherwise do random acquisition for next round
    # --------------------------------------------------------
    current_labeled_df, current_unlabeled_df, selected_rows = random_acquire_items(
        labeled_df=current_labeled_df,
        unlabeled_df=current_unlabeled_df,
        item_id_col=item_id_col,
        acquisition_size=ACQUISITION_SIZE,
        seed=SEED + round_idx,
    )

    print(f"Randomly acquired items for next round: {ACQUISITION_SIZE}")
    print("Acquired distribution:")
    print(selected_rows["label"].value_counts().sort_index())
    print()


# ============================================================
# Save final summary results
# ============================================================
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("ACTOR Replication – Weighted Random Active Learning Loop\n")
    f.write("=" * 55 + "\n\n")

    f.write("Configuration\n")
    f.write("-------------\n")
    f.write(f"MODEL_NAME: {MODEL_NAME}\n")
    f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"NUM_ACQUISITIONS: {NUM_ACQUISITIONS}\n")
    f.write(f"ACQUISITION_SIZE: {ACQUISITION_SIZE}\n")
    f.write(f"SEED: {SEED}\n\n")

    f.write("Paths\n")
    f.write("-----\n")
    f.write(f"INITIAL_LABELED_POOL_PATH: {INITIAL_LABELED_POOL_PATH}\n")
    f.write(f"INITIAL_UNLABELED_POOL_PATH: {INITIAL_UNLABELED_POOL_PATH}\n")
    f.write(f"DEV_PATH: {DEV_PATH}\n")
    f.write(f"TEST_PATH: {TEST_PATH}\n")
    f.write(f"OUTPUT_BASE_DIR: {OUTPUT_BASE_DIR}\n")
    f.write(f"RESULTS_PATH: {RESULTS_PATH}\n\n")

    f.write("Per-round summaries\n")
    f.write("-------------------\n")
    for r in results:
        f.write(str(r) + "\n")

print("Saved results to:")
print(RESULTS_PATH)
print()

print("Final round summaries:")
for r in results:
    print(r)