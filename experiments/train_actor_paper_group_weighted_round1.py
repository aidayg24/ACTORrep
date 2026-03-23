"""
train_actor_paper_group_weighted_round1.py

Goal
----
Train a weighted ACTOR model on the pool produced by the weighted
group-entropy acquisition script for paper round 1.

Why this file exists
--------------------
We are trying to replicate the ACTOR paper pipeline as closely as possible,
but we also need a clean and documented record of what we actually ran.

This script:
1. loads the newly expanded labeled pool
2. loads dev/test data
3. prepares ACTOR-style annotation-level datasets
4. computes class weights from the current labeled pool
5. trains weighted ACTOR
6. evaluates on dev and test
7. saves the trained model state dict

Important note
--------------
Your currently saved acquired pool appears to be annotation-level rows and
does not preserve a clean original annotator column in a guaranteed way.
So this script includes a documented fallback:
- if an "annotator" column exists -> use it
- otherwise -> assign annotators cyclically across Ann1..Ann6

That fallback is not ideal for a final scientific replication, but it is the
correct practical continuation of the exact files you currently generated.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import AutoTokenizer, BertModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# Configuration
# ============================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 2
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5

LABELED_POOL_PATH = "../processed_data/hsbrexit_paper_group_weighted_labeled_pool_round1.csv"
DEV_PATH = "../processed_data/hsbrexit_dev_annotations.csv"
TEST_PATH = "../processed_data/hsbrexit_test_annotations.csv"

OUTPUT_DIR = "../outputs/actor_paper_group_weighted_round1"
MODEL_STATE_SAVE_PATH = os.path.join(OUTPUT_DIR, "actor_model_state.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# ACTOR Model
# ============================================================

class ACTORModel(nn.Module):
    """
    Shared BERT encoder + one classification head per annotator.

    Forward behavior:
    - computes logits for all annotator heads
    - if annotator_idx is provided, selects the correct head per row
    - if labels are provided, computes weighted cross-entropy loss
    """
    def __init__(self, num_annotators, num_labels=2, class_weights=None):
        super().__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size

        self.annotator_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)
        ])

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        annotator_idx=None
    ):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # logits for all annotator heads
        all_logits = []
        for head in self.annotator_heads:
            all_logits.append(head(cls_output))

        # [batch, num_annotators, num_labels]
        all_logits = torch.stack(all_logits, dim=1)

        selected_logits = None
        loss = None

        if annotator_idx is not None:
            batch_indices = torch.arange(all_logits.size(0), device=all_logits.device)
            selected_logits = all_logits[batch_indices, annotator_idx]

            if labels is not None:
                loss = self.loss_fn(selected_logits, labels)

        output = {
            "logits": selected_logits if selected_logits is not None else all_logits
        }

        if loss is not None:
            output["loss"] = loss

        return output

# ============================================================
# Load data
# ============================================================

labeled_df = pd.read_csv(LABELED_POOL_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Paper group weighted round-1 labeled pool size: {len(labeled_df)}")
print(f"Dev size: {len(dev_df)}")
print(f"Test size: {len(test_df)}")
print()

print("Round-1 pool label distribution:")
print(labeled_df["label"].value_counts().sort_index())
print()

# ============================================================
# Annotator mapping
# ============================================================

annotators = ["Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"]
annotator2idx = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

print("Annotator mapping:")
print(annotator2idx)
print()

# ============================================================
# Compute class weights from current labeled pool
# ============================================================

label_counts = labeled_df["label"].value_counts().sort_index()
n_samples = len(labeled_df)
n_classes = NUM_LABELS

class_weights = []
for label_id in range(NUM_LABELS):
    count = label_counts.get(label_id, 0)
    if count == 0:
        raise ValueError(f"Class {label_id} has zero instances in the labeled pool.")
    weight = n_samples / (n_classes * count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:")
for i, w in enumerate(class_weights.tolist()):
    print(f"class {i}: {w:.4f}")
print()

# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# Data preparation helper
# ============================================================

def prepare_actor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for ACTOR training/evaluation.

    Expected minimal columns:
    - text
    - label

    Preferred additional column:
    - annotator

    Fallback logic:
    If 'annotator' is missing, assign annotators cyclically.
    This keeps the pipeline runnable with the files we already created.
    """
    df = df.copy()

    if "text" not in df.columns:
        raise ValueError("Expected column 'text' not found.")
    if "label" not in df.columns:
        raise ValueError("Expected column 'label' not found.")

    if "annotator" not in df.columns:
        df["annotator"] = [annotators[i % num_annotators] for i in range(len(df))]

    df["annotator_idx"] = df["annotator"].map(annotator2idx)

    if df["annotator_idx"].isna().any():
        missing = df[df["annotator_idx"].isna()]["annotator"].unique().tolist()
        raise ValueError(f"Unknown annotators found: {missing}")

    df["annotator_idx"] = df["annotator_idx"].astype(int)
    df["labels"] = df["label"].astype(int)

    return df[["text", "labels", "annotator_idx"]]

train_actor_df = prepare_actor_dataframe(labeled_df)
dev_actor_df = prepare_actor_dataframe(dev_df)
test_actor_df = prepare_actor_dataframe(test_df)

# ============================================================
# Convert to Hugging Face Datasets
# ============================================================

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = Dataset.from_pandas(train_actor_df, preserve_index=False)
dev_dataset = Dataset.from_pandas(dev_actor_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_actor_df, preserve_index=False)

train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

final_columns = ["labels", "annotator_idx", "input_ids", "token_type_ids", "attention_mask"]

train_dataset.set_format(type="torch", columns=final_columns)
dev_dataset.set_format(type="torch", columns=final_columns)
test_dataset.set_format(type="torch", columns=final_columns)

print("Final columns:")
print(final_columns)
print()

# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }

# ============================================================
# Build model
# ============================================================

model = ACTORModel(
    num_annotators=num_annotators,
    num_labels=NUM_LABELS,
    class_weights=class_weights
).to(DEVICE)

# ============================================================
# Training arguments
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none",
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
print("Starting training on paper group weighted round-1 pool...")

trainer.train()

# ============================================================
# Evaluate
# ============================================================

print()
print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)

print()
print("Test results:")
print(test_results)
print()

# ============================================================
# Save trained model state dict
# ============================================================

torch.save(model.state_dict(), MODEL_STATE_SAVE_PATH)

print("Saved trained ACTOR weights:")
print(MODEL_STATE_SAVE_PATH)