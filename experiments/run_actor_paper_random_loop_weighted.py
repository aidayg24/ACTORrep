"""
Weighted random active-learning loop for ACTOR on HS-Brexit.

What this script does:
1. Loads the full annotation-level train/dev/test splits.
2. Starts from a small paper-style initial labeled pool (60 examples).
3. Trains ACTOR on the current labeled pool.
4. Evaluates on dev and test.
5. Randomly acquires 60 more examples from the unlabeled pool.
6. Repeats for 4 acquisition rounds (so rounds 0..4 total).
7. Uses class-weighted cross-entropy to reduce majority-class collapse.

Important:
- This script saves a plain PyTorch state_dict after each round:
    actor_model_state.pt
- It also disables safetensors saving to avoid the shared-memory crash
  you got before.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    BertModel,
    Trainer,
    TrainingArguments,
)

# =========================================================
# Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# Paths
# =========================================================
TRAIN_PATH = "../processed_data/hsbrexit_train_annotations.csv"
DEV_PATH = "../processed_data/hsbrexit_dev_annotations.csv"
TEST_PATH = "../processed_data/hsbrexit_test_annotations.csv"

OUTPUT_BASE = "../outputs/actor_paper_random_loop_weighted"
RESULTS_PATH = "../results/results_actor_paper_random_loop_weighted.txt"

# =========================================================
# Experiment settings
# =========================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

INITIAL_POOL_SIZE = 60
ACQUISITION_SIZE = 60
NUM_ACQUISITION_ROUNDS = 4   # gives rounds 0,1,2,3,4 total

NUM_LABELS = 2
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Helper functions
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

def prepare_dataset(df: pd.DataFrame):
    """
    Converts a pandas dataframe into a HuggingFace Dataset
    with tokenized text and torch formatting.
    """
    working_df = df.copy().rename(columns={"label": "labels"})
    dataset = Dataset.from_pandas(working_df)

    dataset = dataset.map(tokenize_function, batched=True)

    removable = [c for c in ["item_id", "text", "annotator_id"] if c in dataset.column_names]
    dataset = dataset.remove_columns(removable)

    dataset.set_format("torch")
    return dataset

def compute_class_weights(labeled_df: pd.DataFrame, num_classes: int = 2):
    """
    Standard inverse-frequency class weights:
        weight_c = N / (K * count_c)
    """
    label_counts = labeled_df["label"].value_counts().sort_index()
    total_examples = len(labeled_df)

    weights = []
    for cls_id in range(num_classes):
        cls_count = label_counts.get(cls_id, 0)

        # safety guard
        if cls_count == 0:
            weight = 1.0
        else:
            weight = total_examples / (num_classes * cls_count)

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)

# =========================================================
# ACTOR model
# =========================================================
class ActorModel(nn.Module):
    """
    ACTOR:
    - shared BERT encoder
    - one classification head per annotator
    - optional class-weighted loss
    """

    def __init__(self, num_annotators: int, num_labels: int = 2, class_weights=None):
        super().__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size

        self.annotator_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)
        ])

        # Store class weights as a buffer.
        # This avoids the safetensors/shared-memory problem you hit earlier.
        if class_weights is not None:
            self.register_buffer("class_weights_buffer", class_weights.clone().detach())
        else:
            self.class_weights_buffer = None

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        annotator_idx=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        batch_size = cls_output.size(0)
        num_labels = self.annotator_heads[0].out_features

        logits = torch.zeros(batch_size, num_labels, device=cls_output.device)

        for i in range(batch_size):
            head_idx = annotator_idx[i].item()
            logits[i] = self.annotator_heads[head_idx](cls_output[i])

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_buffer)
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# =========================================================
# Load data
# =========================================================
train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Full train size:", len(train_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

print("\nFull train label distribution:")
print(train_df["label"].value_counts().sort_index())

# Build annotator mapping from train split
annotators = sorted(train_df["annotator_id"].unique())
annotator_to_id = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

print("\nAnnotator mapping:")
print(annotator_to_id)

# Add annotator index to all splits
train_df["annotator_idx"] = train_df["annotator_id"].map(annotator_to_id)
dev_df["annotator_idx"] = dev_df["annotator_id"].map(annotator_to_id)
test_df["annotator_idx"] = test_df["annotator_id"].map(annotator_to_id)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prepare fixed eval datasets once
dev_dataset = prepare_dataset(dev_df)
test_dataset = prepare_dataset(test_df)

# =========================================================
# Initial split: paper-style
# =========================================================
# We sample 60 annotation rows from the full train set as initial labeled pool.
# The rest becomes unlabeled pool.
initial_labeled_df = train_df.sample(n=INITIAL_POOL_SIZE, random_state=SEED).copy()
unlabeled_df = train_df.drop(initial_labeled_df.index).copy()

initial_labeled_df = initial_labeled_df.reset_index(drop=True)
unlabeled_df = unlabeled_df.reset_index(drop=True)

print("\nInitial labeled pool size:", len(initial_labeled_df))
print("Initial unlabeled pool size:", len(unlabeled_df))

print("\nInitial pool distribution:")
print(initial_labeled_df["label"].value_counts().sort_index())

# =========================================================
# Main active-learning loop
# =========================================================
all_round_results = []

for round_idx in range(NUM_ACQUISITION_ROUNDS + 1):
    print("\n" + "=" * 60)
    print(f"ROUND {round_idx}")
    print("=" * 60)

    labeled_df = initial_labeled_df.copy().reset_index(drop=True)

    print("Labeled pool size:", len(labeled_df))
    print("Labeled pool distribution:")
    print(labeled_df["label"].value_counts().sort_index())

    # -------------------------
    # Compute class weights
    # -------------------------
    class_weights = compute_class_weights(labeled_df, num_classes=NUM_LABELS)

    print("\nClass weights for this round:")
    for i, w in enumerate(class_weights):
        print(f"class {i}: {w:.4f}")

    # -------------------------
    # Prepare training dataset
    # -------------------------
    train_dataset = prepare_dataset(labeled_df)

    # -------------------------
    # Create model
    # -------------------------
    set_seed(SEED + round_idx)

    model = ActorModel(
        num_annotators=num_annotators,
        num_labels=NUM_LABELS,
        class_weights=class_weights,
    ).to(device)

    round_output_dir = os.path.join(OUTPUT_BASE, f"round_{round_idx}")
    os.makedirs(round_output_dir, exist_ok=True)

    logging_steps = max(1, len(train_dataset) // TRAIN_BATCH_SIZE)

    training_args = TrainingArguments(
        output_dir=round_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED + round_idx,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    # Save ACTOR weights manually in plain PyTorch format
    manual_state_path = os.path.join(round_output_dir, "actor_model_state.pt")
    torch.save(model.state_dict(), manual_state_path)
    print(f"\nSaved ACTOR state dict to:\n{manual_state_path}")

    # -------------------------
    # Evaluate on dev and test
    # -------------------------
    print("\nEvaluating on dev...")
    dev_metrics = trainer.evaluate(dev_dataset)

    print("\nEvaluating on test...")
    test_metrics = trainer.evaluate(test_dataset)

    round_summary = {
        "round": round_idx,
        "labeled_pool_size": len(labeled_df),
        "dev_accuracy": dev_metrics["eval_accuracy"],
        "dev_macro_f1": dev_metrics["eval_macro_f1"],
        "dev_weighted_f1": dev_metrics["eval_weighted_f1"],
        "test_accuracy": test_metrics["eval_accuracy"],
        "test_macro_f1": test_metrics["eval_macro_f1"],
        "test_weighted_f1": test_metrics["eval_weighted_f1"],
    }
    all_round_results.append(round_summary)

    # -------------------------
    # Stop after final round
    # -------------------------
    if round_idx == NUM_ACQUISITION_ROUNDS:
        break

    # -------------------------
    # Random acquisition
    # -------------------------
    acquired_df = unlabeled_df.sample(
        n=ACQUISITION_SIZE,
        random_state=SEED + round_idx
    ).copy()

    print(f"\nRandomly acquired examples: {len(acquired_df)}")
    print("Acquired distribution:")
    print(acquired_df["label"].value_counts().sort_index())

    # Add acquired examples to labeled pool
    initial_labeled_df = pd.concat([initial_labeled_df, acquired_df], ignore_index=True)

    # Remove them from unlabeled pool
    unlabeled_df = unlabeled_df.drop(acquired_df.index).reset_index(drop=True)

# =========================================================
# Save results
# =========================================================
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("Weighted ACTOR paper-style random active learning loop results\n")
    f.write("=" * 70 + "\n\n")

    for r in all_round_results:
        f.write(f"Round {r['round']}\n")
        f.write(f"Labeled pool size: {r['labeled_pool_size']}\n")
        f.write(f"Dev accuracy: {r['dev_accuracy']:.4f}\n")
        f.write(f"Dev macro-F1: {r['dev_macro_f1']:.4f}\n")
        f.write(f"Dev weighted-F1: {r['dev_weighted_f1']:.4f}\n")
        f.write(f"Test accuracy: {r['test_accuracy']:.4f}\n")
        f.write(f"Test macro-F1: {r['test_macro_f1']:.4f}\n")
        f.write(f"Test weighted-F1: {r['test_weighted_f1']:.4f}\n")
        f.write("\n")

print("\nSaved results to:")
print(RESULTS_PATH)

print("\nFinal round summaries:")
for r in all_round_results:
    print(r)