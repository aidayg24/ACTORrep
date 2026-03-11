"""
Majority-vote BERT baseline for HS-Brexit.

This script implements the first and simplest replication baseline for the
ACTOR paper: a standard single-head BERT classifier trained on majority-vote
labels.

For this purpose, we use:
    - the processed CSV files created from the original JSON files (see processed_data)
    - one row per text item
    - the `hard_label` field as the majority-vote label

We previously verified that:
    - `annotations` contains the individual annotator labels
    - `hard_label` matches the majority vote of those annotations

So this script corresponds to the "Single-Majority" style baseline:
    text -> BERT -> CLS embedding -> classifier -> majority label

We use `bert-base-uncased` with Hugging Face's
`AutoModelForSequenceClassification`.
    - the text is encoded by BERT
    - the sentence representation is based on the standard CLS-based
      classification setup
    - the classifier predicts one of two labels:
          0 = non-hate
          1 = hate

Notes
-----
- We intentionally do NOT use class weights yet.
  This keeps the first baseline simple and closer to a vanilla BERT baseline.
- Because the dataset is imbalanced, macro-F1 is especially important.
- This script is the first baseline only; it is NOT the ACTOR multi-head model.
"""

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


# ---------------------------
# Load processed CSV splits
# ---------------------------
train_path = "../processed_data/HS-brexit_train_majority.csv"
dev_path = "../processed_data/HS-brexit_dev_majority.csv"
test_path = "../processed_data/HS-brexit_test_majority.csv"

train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)

print("Train size:", len(train_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

print("\nTrain labels:")
print(train_df["hard_label"].value_counts())

print("\nDev labels:")
print(dev_df["hard_label"].value_counts())

print("\nTest labels:")
print(test_df["hard_label"].value_counts())


# ---------------------------
# Rename label column
# ---------------------------
train_df = train_df.rename(columns={"hard_label": "labels"})
dev_df = dev_df.rename(columns={"hard_label": "labels"})
test_df = test_df.rename(columns={"hard_label": "labels"})


# ---------------------------
# Convert pandas -> Hugging Face Dataset
# ---------------------------
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)


# ---------------------------
# Load tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# Tokenize all splits
train_tokenized = train_dataset.map(tokenize_function, batched=True)
dev_tokenized = dev_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)


# ---------------------------
# Remove columns not needed by the model
# Keep only label + tokenized inputs
# ---------------------------
train_tokenized = train_tokenized.remove_columns(["item_id", "text"])
dev_tokenized = dev_tokenized.remove_columns(["item_id", "text"])
test_tokenized = test_tokenized.remove_columns(["item_id", "text"])


# ---------------------------
# Set PyTorch format
# ---------------------------
train_tokenized.set_format("torch")
dev_tokenized.set_format("torch")
test_tokenized.set_format("torch")

print("\nFinal columns:", train_tokenized.column_names)
print("First label:", train_tokenized[0]["labels"])


# ---------------------------
# Load model
# Standard BERT classifier with 2 output labels
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# ---------------------------
# Define evaluation metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


# ---------------------------
# Training configuration
# We evaluate on the dev set once per epoch and keep the best model
# according to macro-F1
# ---------------------------
training_args = TrainingArguments(
    output_dir="../outputs/majority_baseline",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none"
)


# ---------------------------
# Create Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    compute_metrics=compute_metrics,
)

print("\nSetup complete. Trainer created successfully.")


# ---------------------------
# Train the model
# ---------------------------
print("\nStarting training...")
trainer.train()


# ---------------------------
# Final evaluation on the test set
# ---------------------------
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_tokenized)

print("\nTest results:")
print(test_results)