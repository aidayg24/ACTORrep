"""
Single-annotation BERT baseline for HS-Brexit.

This script trains a standard single-head BERT classifier on all individual
annotations in the HS-Brexit dataset.

Unlike the majority-vote baseline, where each tweet has only one label,
this dataset contains one row per annotation:

    item_id, text, annotator_id, label

So if a tweet was labeled by 6 annotators, it appears 6 times in the training
data, potentially with different labels.


This baseline keeps human disagreement in the training data, but it does NOT
model annotator identity explicitly.

This makes it an important comparison point between:
1. majority-vote baseline
2. single-annotation baseline
3. ACTOR multi-head annotator-specific model

Notes
-----
- We do not use annotator_id as an input feature here.
- We do not use class weights yet.
- We evaluate on annotation-level dev/test sets.
- Because the dataset is imbalanced, macro-F1 is important.
"""

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


# ---------------------------
# Load annotation-level CSV splits
# ---------------------------
train_path = "../processed_data/hsbrexit_train_annotations.csv"
dev_path = "../processed_data/hsbrexit_dev_annotations.csv"
test_path = "../processed_data/hsbrexit_test_annotations.csv"

train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)

print("Train size:", len(train_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

print("\nTrain label distribution:")
print(train_df["label"].value_counts())

print("\nDev label distribution:")
print(dev_df["label"].value_counts())

print("\nTest label distribution:")
print(test_df["label"].value_counts())

print("\nUnique annotators in train:", train_df["annotator_id"].nunique())


# ---------------------------
# Rename label column for Hugging Face
# ---------------------------
train_df = train_df.rename(columns={"label": "labels"})
dev_df = dev_df.rename(columns={"label": "labels"})
test_df = test_df.rename(columns={"label": "labels"})


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


train_tokenized = train_dataset.map(tokenize_function, batched=True)
dev_tokenized = dev_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)


# ---------------------------
# Remove unused columns
# annotator_id is kept in the dataset file for analysis,
# but not used by this baseline model
# ---------------------------
train_tokenized = train_tokenized.remove_columns(["item_id", "text", "annotator_id"])
dev_tokenized = dev_tokenized.remove_columns(["item_id", "text", "annotator_id"])
test_tokenized = test_tokenized.remove_columns(["item_id", "text", "annotator_id"])


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
# Standard BERT classifier, CLS-based
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
# ---------------------------
training_args = TrainingArguments(
    output_dir="../outputs/single_annotation_baseline",
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
# Train
# ---------------------------
print("\nStarting training...")
trainer.train()


# ---------------------------
# Final evaluation on test set
# ---------------------------
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_tokenized)

print("\nTest results:")
print(test_results)

# Test results:
# {'eval_loss': 0.32936447858810425, 'eval_accuracy': 0.876984126984127, 'eval_macro_f1': 0.5692307692307692, 'eval_weighted_f1': 0.837973137973138, 'eval_runtime': 99.6685, 'eval_samples_per_second': 10.114, 'eval_steps_per_second': 1.264, 'epoch': 3.0}
#
# Process finished with exit code 0