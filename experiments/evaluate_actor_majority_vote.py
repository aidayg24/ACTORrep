"""
Evaluate a trained ACTOR model using majority vote over annotator heads.

What this script does
---------------------
For each unique test item:
1. take the text once
2. run it through the shared BERT encoder
3. get one prediction from each annotator head
4. take majority vote across the 6 head predictions
5. compare that against the gold majority-vote label

Why this matters
----------------
This is closer to the paper's ACTOR evaluation than annotation-level evaluation.
"""

from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, BertModel

# ---------------------------
# 1. Paths
# ---------------------------
test_annotations_path = "../processed_data/hsbrexit_test_annotations.csv"
test_majority_path = "../processed_data/HS-brexit_test_majority.csv"
checkpoint_path = "../outputs/actor_paper_round3/actor_model_state.pt"

model_name = "bert-base-uncased"
max_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Load data
# ---------------------------
test_ann_df = pd.read_csv(test_annotations_path)
test_majority_df = pd.read_csv(test_majority_path)

print("Annotation-level test size:", len(test_ann_df))
print("Majority-vote test item size:", len(test_majority_df))

# ---------------------------
# 3. Build annotator mapping
# IMPORTANT: must match training mapping
# ---------------------------
annotators = sorted(test_ann_df["annotator_id"].unique())
annotator_to_id = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

print("\nAnnotator mapping:")
print(annotator_to_id)

# ---------------------------
# 4. Define ACTOR model
# ---------------------------
class ActorModel(nn.Module):
    def __init__(self, num_annotators, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.annotator_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)
        ])

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output

# ---------------------------
# 5. Load tokenizer + model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = ActorModel(num_annotators=num_annotators, num_labels=2)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

print("\nLoaded ACTOR checkpoint:")
print(checkpoint_path)

# ---------------------------
# 6. Helper: majority vote
# ---------------------------
def majority_vote(labels):
    counts = Counter(labels)
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    return min(winners)  # deterministic tie break

# ---------------------------
# 7. Predict per unique item
# ---------------------------
gold_labels = []
pred_labels = []

for _, row in test_majority_df.iterrows():
    item_id = row["item_id"]
    text = row["text"]
    gold = int(row["hard_label"])

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        cls_output = model.encode(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids", None)
        )

        head_predictions = []
        for head_idx, head in enumerate(model.annotator_heads):
            logits = head(cls_output)                 # shape (1, 2)
            pred = torch.argmax(logits, dim=-1).item()
            head_predictions.append(pred)

    final_pred = majority_vote(head_predictions)

    gold_labels.append(gold)
    pred_labels.append(final_pred)

    if len(gold_labels) <= 5:
        print(f"\nItem {item_id}")
        print("Head predictions:", head_predictions)
        print("Majority prediction:", final_pred)
        print("Gold majority label:", gold)

# ---------------------------
# 8. Metrics
# ---------------------------
accuracy = accuracy_score(gold_labels, pred_labels)
macro_f1 = f1_score(gold_labels, pred_labels, average="macro")
weighted_f1 = f1_score(gold_labels, pred_labels, average="weighted")

print("\nMajority-vote evaluation results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")

print("\nClassification report:")
print(classification_report(gold_labels, pred_labels, digits=4))