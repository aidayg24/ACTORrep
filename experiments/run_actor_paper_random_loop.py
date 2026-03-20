"""
Run a clean paper-style random active learning loop for ACTOR on HS-Brexit.

What this script does
---------------------
1. Load the full annotation-level train/dev/test datasets.
2. Create a paper-style initial labeled pool of size 60.
3. Train ACTOR on the current labeled pool.
4. Evaluate on dev and test after each round.
5. Randomly acquire 60 more annotation rows from the unlabeled pool.
6. Repeat for multiple rounds.
7. Save all round-wise results to a text file.

Why this script is useful
-------------------------
This gives you a much cleaner random baseline than maintaining many separate
round scripts manually.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, BertModel, TrainingArguments, Trainer

# =========================================================
# 1. Configuration
# =========================================================
TRAIN_PATH = "../processed_data/hsbrexit_train_annotations.csv"
DEV_PATH = "../processed_data/hsbrexit_dev_annotations.csv"
TEST_PATH = "../processed_data/hsbrexit_test_annotations.csv"

OUTPUT_DIR = "../outputs/actor_paper_random_loop"
RESULTS_PATH = "../results/results_actor_paper_random_loop.txt"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

INITIAL_POOL_SIZE = 60
QUERY_SIZE = 60
NUM_ROUNDS = 4   # round 0 = initial pool, then 4 acquisitions -> 5 evaluations total

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 2. Reproducibility
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================================================
# 3. Load data
# =========================================================
train_full_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Full train size:", len(train_full_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

print("\nFull train label distribution:")
print(train_full_df["label"].value_counts())

# =========================================================
# 4. Annotator mapping
# =========================================================
annotators = sorted(train_full_df["annotator_id"].unique())
annotator_to_id = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

print("\nAnnotator mapping:")
print(annotator_to_id)

train_full_df["annotator_idx"] = train_full_df["annotator_id"].map(annotator_to_id)
dev_df["annotator_idx"] = dev_df["annotator_id"].map(annotator_to_id)
test_df["annotator_idx"] = test_df["annotator_id"].map(annotator_to_id)

# =========================================================
# 5. Tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# =========================================================
# 6. ACTOR model
# =========================================================
class ActorModel(nn.Module):
    """
    Shared BERT encoder + one linear head per annotator.
    """

    def __init__(self, num_annotators: int, num_labels: int = 2):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size

        self.annotator_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)
        ])

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        annotator_idx=None,
        labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
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
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# =========================================================
# 7. Metrics
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }

# =========================================================
# 8. Dataset preparation helper
# =========================================================
def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """
    Prepare a dataframe for Hugging Face Trainer.
    """
    working_df = df.copy().rename(columns={"label": "labels"})
    dataset = Dataset.from_pandas(working_df)
    dataset = dataset.map(tokenize_function, batched=True)

    removable = [col for col in ["item_id", "text", "annotator_id"] if col in dataset.column_names]
    dataset = dataset.remove_columns(removable)
    dataset.set_format("torch")
    return dataset

# =========================================================
# 9. Train + evaluate one round
# =========================================================
def run_one_round(round_id: int, labeled_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train ACTOR on the current labeled pool, then evaluate on dev and test.
    """
    print("\n" + "=" * 60)
    print(f"ROUND {round_id}")
    print("=" * 60)

    print("Labeled pool size:", len(labeled_df))
    print("Labeled pool distribution:")
    print(labeled_df["label"].value_counts())

    train_dataset = prepare_dataset(labeled_df)
    dev_dataset = prepare_dataset(dev_df)
    test_dataset = prepare_dataset(test_df)

    model = ActorModel(num_annotators=num_annotators, num_labels=2)

    round_output_dir = os.path.join(OUTPUT_DIR, f"round_{round_id}")

    training_args = TrainingArguments(
        output_dir=round_output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=False,
        report_to="none"
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

    print("\nEvaluating on dev...")
    dev_results = trainer.evaluate(dev_dataset)

    print("\nEvaluating on test...")
    test_results = trainer.evaluate(test_dataset)

    return model, dev_results, test_results

# =========================================================
# 10. Initialize labeled / unlabeled pools
# =========================================================
initial_labeled_df = train_full_df.sample(
    n=INITIAL_POOL_SIZE,
    random_state=SEED
).copy()

unlabeled_df = train_full_df.drop(initial_labeled_df.index).copy()

# reset index to keep things clean
initial_labeled_df = initial_labeled_df.reset_index(drop=True)
unlabeled_df = unlabeled_df.reset_index(drop=True)

print("\nInitial labeled pool size:", len(initial_labeled_df))
print("Initial unlabeled pool size:", len(unlabeled_df))

print("\nInitial pool distribution:")
print(initial_labeled_df["label"].value_counts())

# =========================================================
# 11. Run active learning loop
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../results", exist_ok=True)

all_results = []

labeled_df = initial_labeled_df.copy()

for round_id in range(NUM_ROUNDS + 1):
    # Train + evaluate current pool
    model, dev_results, test_results = run_one_round(round_id, labeled_df, dev_df, test_df)

    round_summary = {
        "round": round_id,
        "labeled_pool_size": len(labeled_df),
        "dev_accuracy": dev_results["eval_accuracy"],
        "dev_macro_f1": dev_results["eval_macro_f1"],
        "dev_weighted_f1": dev_results["eval_weighted_f1"],
        "test_accuracy": test_results["eval_accuracy"],
        "test_macro_f1": test_results["eval_macro_f1"],
        "test_weighted_f1": test_results["eval_weighted_f1"],
    }
    all_results.append(round_summary)

    # Stop after final evaluation round
    if round_id == NUM_ROUNDS:
        break

    # Random acquisition for next round
    acquire_n = min(QUERY_SIZE, len(unlabeled_df))
    acquired_df = unlabeled_df.sample(
        n=acquire_n,
        random_state=SEED + round_id + 1
    ).copy()

    unlabeled_df = unlabeled_df.drop(acquired_df.index).reset_index(drop=True)
    acquired_df = acquired_df.reset_index(drop=True)

    print("\nRandomly acquired examples:", len(acquired_df))
    print("Acquired distribution:")
    print(acquired_df["label"].value_counts())

    labeled_df = pd.concat([labeled_df, acquired_df], ignore_index=True)

# =========================================================
# 12. Save results
# =========================================================
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("ACTOR Paper-Style Random Active Learning Loop\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Initial pool size: {INITIAL_POOL_SIZE}\n")
    f.write(f"Query size: {QUERY_SIZE}\n")
    f.write(f"Number of rounds after seed: {NUM_ROUNDS}\n")
    f.write(f"Epochs per round: {NUM_EPOCHS}\n")
    f.write(f"Batch size: {TRAIN_BATCH_SIZE}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Weight decay: {WEIGHT_DECAY}\n\n")

    for res in all_results:
        f.write(f"Round {res['round']}\n")
        f.write(f"  Labeled pool size: {res['labeled_pool_size']}\n")
        f.write(f"  Dev accuracy: {res['dev_accuracy']:.4f}\n")
        f.write(f"  Dev macro-F1: {res['dev_macro_f1']:.4f}\n")
        f.write(f"  Dev weighted-F1: {res['dev_weighted_f1']:.4f}\n")
        f.write(f"  Test accuracy: {res['test_accuracy']:.4f}\n")
        f.write(f"  Test macro-F1: {res['test_macro_f1']:.4f}\n")
        f.write(f"  Test weighted-F1: {res['test_weighted_f1']:.4f}\n")
        f.write("\n")

print("\nSaved results to:")
print(RESULTS_PATH)

print("\nFinal round summaries:")
for res in all_results:
    print(res)