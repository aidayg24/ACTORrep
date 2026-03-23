"""
Debug script for checking whether ACTOR collapses to the majority class.

This evaluates a trained ACTOR checkpoint on the annotation-level test set
and prints:
- prediction label distribution
- gold label distribution
- confusion matrix
- classification report
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, BertModel, Trainer, TrainingArguments

# =========================================================
# Paths
# =========================================================
TEST_PATH = "../processed_data/hsbrexit_test_annotations.csv"
TRAIN_PATH = "../processed_data/hsbrexit_train_annotations.csv"

# choose one checkpoint to inspect
MODEL_STATE_PATH = "../outputs/actor_paper_round3/actor_model_state.pt"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Load data
# =========================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

annotators = sorted(train_df["annotator_id"].unique())
annotator_to_id = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

test_df["annotator_idx"] = test_df["annotator_id"].map(annotator_to_id)

print("Test size:", len(test_df))
print("\nGold label distribution:")
print(test_df["label"].value_counts().sort_index())

# =========================================================
# Tokenizer
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
# Model
# =========================================================
class ActorModel(nn.Module):
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
# Prepare dataset
# =========================================================
working_df = test_df.copy().rename(columns={"label": "labels"})
test_dataset = Dataset.from_pandas(working_df)
test_dataset = test_dataset.map(tokenize_function, batched=True)

remove_cols = [c for c in ["item_id", "text", "annotator_id"] if c in test_dataset.column_names]
test_dataset = test_dataset.remove_columns(remove_cols)
test_dataset.set_format("torch")

print("\nFinal columns:")
print(test_dataset.column_names)

# =========================================================
# Load model
# =========================================================
model = ActorModel(num_annotators=num_annotators, num_labels=2)
state_dict = torch.load(MODEL_STATE_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)

print("\nLoaded model from:")
print(MODEL_STATE_PATH)

# =========================================================
# Predict
# =========================================================
args = TrainingArguments(
    output_dir="../outputs/tmp_debug_eval",
    per_device_eval_batch_size=32,
    report_to="none"
)

trainer = Trainer(model=model, args=args)

pred_output = trainer.predict(test_dataset)
logits = pred_output.predictions
gold = pred_output.label_ids
preds = np.argmax(logits, axis=-1)

# =========================================================
# Print diagnostics
# =========================================================
unique_preds, pred_counts = np.unique(preds, return_counts=True)
pred_dist = dict(zip(unique_preds.tolist(), pred_counts.tolist()))

print("\nPrediction distribution:")
print(pred_dist)

print("\nConfusion matrix:")
print(confusion_matrix(gold, preds))

print("\nClassification report:")
print(classification_report(gold, preds, digits=4))