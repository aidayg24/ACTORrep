"""
Select new paper round-4 samples using group-level entropy.

What this script does
---------------------
This script scores each unlabeled example using GROUP-LEVEL ENTROPY
from the trained ACTOR round-3 model.

Idea:
- For each text, run it through the shared BERT encoder.
- Pass the same text representation through all annotator heads.
- Convert each head's logits to probabilities.
- Average the probabilities across annotator heads.
- Compute entropy of the averaged probability distribution.
- Select the top-K highest entropy examples.

Why this matches the paper
--------------------------
The paper compares active-learning acquisition strategies for ACTOR, including
group-level entropy. This script implements a practical version of that idea
for the HS-Brexit paper-matched setting.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, BertModel

# ---------------------------
# 1. Paths
# ---------------------------
labeled_path = "../processed_data/hsbrexit_paper_labeled_pool_round3.csv"
unlabeled_path = "../processed_data/hsbrexit_paper_unlabeled_pool_round3.csv"

# IMPORTANT:
# This should point to the best model checkpoint from your latest paper round-3 training.
# For now we load directly from bert-base-uncased and assume you will later replace this
# with your saved ACTOR round-3 model loading if needed.
#
# Since your current training scripts do not yet save/load a custom ACTOR checkpoint
# cleanly, we first implement the full scoring pipeline structure.
#
# In the next refinement step, we will connect this to the trained round-3 ACTOR weights.
model_name = "bert-base-uncased"
checkpoint_path = "../outputs/actor_paper_round3/actor_model_state.pt"

acquisition_size = 60
max_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Load data
# ---------------------------
labeled_df = pd.read_csv(labeled_path)
unlabeled_df = pd.read_csv(unlabeled_path)

print("Current labeled pool size:", len(labeled_df))
print("Current unlabeled pool size:", len(unlabeled_df))

# ---------------------------
# 3. Annotator mapping from labeled data
# ---------------------------
annotators = sorted(labeled_df["annotator_id"].unique())
annotator_to_id = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

print("\nAnnotator mapping:")
print(annotator_to_id)

# ---------------------------
# 4. Define ACTOR model
# ---------------------------
class ActorModel(nn.Module):
    """
    Shared BERT encoder + one linear head per annotator.
    """

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
# 5. Load model
# ---------------------------
model = ActorModel(num_annotators=num_annotators, num_labels=2)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

print("\nLoaded trained ACTOR model weights from:")
print(checkpoint_path)

# ---------------------------
# 6. Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ---------------------------
# 7. Helper: entropy
# ---------------------------
def entropy_from_probs(probs):
    """
    probs: numpy array of shape (num_labels,)
    """
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))

# ---------------------------
# 8. Score each unlabeled example
# ---------------------------
scores = []

for idx, row in unlabeled_df.iterrows():
    text = row["text"]

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

        head_probs = []
        for head in model.annotator_heads:
            logits = head(cls_output)                  # shape: (1, 2)
            probs = torch.softmax(logits, dim=-1)     # shape: (1, 2)
            head_probs.append(probs.cpu().numpy()[0])

        head_probs = np.array(head_probs)             # shape: (num_heads, 2)
        group_probs = head_probs.mean(axis=0)         # average over annotators
        group_entropy = entropy_from_probs(group_probs)

    scores.append(group_entropy)

    if idx % 500 == 0:
        print(f"Scored {idx} examples...")

# ---------------------------
# 9. Attach scores and select top-K
# ---------------------------
unlabeled_df = unlabeled_df.copy()
unlabeled_df["group_entropy"] = scores

selected_df = unlabeled_df.sort_values(
    by="group_entropy",
    ascending=False
).head(acquisition_size)

remaining_unlabeled_df = unlabeled_df.drop(selected_df.index)

print("\nTop selected examples:", len(selected_df))
print("Remaining unlabeled examples:", len(remaining_unlabeled_df))

print("\nSelected label distribution (for debugging only, in real AL this would be hidden):")
print(selected_df["label"].value_counts())

# ---------------------------
# 10. Update labeled pool
# ---------------------------
selected_df = selected_df.drop(columns=["group_entropy"])
updated_labeled_df = pd.concat([labeled_df, selected_df], ignore_index=True)

# ---------------------------
# 11. Save outputs
# ---------------------------
updated_labeled_df.to_csv(
    "../processed_data/hsbrexit_paper_group_labeled_pool_round4.csv",
    index=False
)

remaining_unlabeled_df = remaining_unlabeled_df.drop(columns=["group_entropy"])
remaining_unlabeled_df.to_csv(
    "../processed_data/hsbrexit_paper_group_unlabeled_pool_round4.csv",
    index=False
)

print("\nSaved:")
print("- ../processed_data/hsbrexit_paper_group_labeled_pool_round4.csv")
print("- ../processed_data/hsbrexit_paper_group_unlabeled_pool_round4.csv")