"""
Train ACTOR on the paper-matched initial labeled pool
using class-weighted cross-entropy.

Why this version exists
-----------------------
In the paper-matched low-budget setting for HS-Brexit,
the model is currently collapsing toward the majority class.
This version adds class weights so minority-class errors
are penalized more strongly.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, BertModel
from transformers import TrainingArguments, Trainer

# ---------------------------
# 1. Load datasets
# ---------------------------
train_path = "../processed_data/hsbrexit_paper_initial_labeled_pool.csv"
dev_path = "../processed_data/hsbrexit_dev_annotations.csv"
test_path = "../processed_data/hsbrexit_test_annotations.csv"

train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)

print("Paper initial labeled pool size:", len(train_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

print("\nPaper initial pool label distribution:")
print(train_df["label"].value_counts())

# ---------------------------
# 2. Create annotator mapping
# ---------------------------
annotators = sorted(train_df["annotator_id"].unique())
annotator_to_id = {ann: i for i, ann in enumerate(annotators)}
num_annotators = len(annotators)

train_df["annotator_idx"] = train_df["annotator_id"].map(annotator_to_id)
dev_df["annotator_idx"] = dev_df["annotator_id"].map(annotator_to_id)
test_df["annotator_idx"] = test_df["annotator_id"].map(annotator_to_id)

print("\nAnnotator mapping:")
print(annotator_to_id)

# ---------------------------
# 3. Compute class weights from training data
# ---------------------------
label_counts = train_df["label"].value_counts().sort_index()
num_classes = 2
total = len(train_df)

class_weights = []
for label_id in range(num_classes):
    count = label_counts.get(label_id, 0)
    weight = total / (num_classes * count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float)

print("\nClass weights:")
for i, w in enumerate(class_weights):
    print(f"class {i}: {w:.4f}")

# ---------------------------
# 4. Rename label column
# ---------------------------
train_df = train_df.rename(columns={"label": "labels"})
dev_df = dev_df.rename(columns={"label": "labels"})
test_df = test_df.rename(columns={"label": "labels"})

# ---------------------------
# 5. Convert to HF Dataset
# ---------------------------
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# ---------------------------
# 6. Tokenize
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

train_tokenized = train_tokenized.remove_columns(["item_id", "text", "annotator_id"])
dev_tokenized = dev_tokenized.remove_columns(["item_id", "text", "annotator_id"])
test_tokenized = test_tokenized.remove_columns(["item_id", "text", "annotator_id"])

train_tokenized.set_format("torch")
dev_tokenized.set_format("torch")
test_tokenized.set_format("torch")

print("\nFinal columns:")
print(train_tokenized.column_names)

# ---------------------------
# 7. Define ACTOR model
# ---------------------------
class ActorModel(nn.Module):
    """
    Shared BERT encoder + one linear head per annotator.
    Uses class-weighted cross-entropy.
    """

    def __init__(self, num_annotators, class_weights, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        self.annotator_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)
        ])

        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids, attention_mask, token_type_ids=None, annotator_idx=None, labels=None):
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
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# ---------------------------
# 8. Create model
# ---------------------------
model = ActorModel(
    num_annotators=num_annotators,
    class_weights=class_weights,
    num_labels=2
)

# ---------------------------
# 9. Metrics
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
# 10. Training arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="../outputs/actor_initial_pool_paper_weighted",
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none"
)

# ---------------------------
# 11. Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    compute_metrics=compute_metrics,
)

print("\nTrainer created successfully.")

# ---------------------------
# 12. Train
# ---------------------------
print("\nStarting weighted ACTOR training on paper initial pool...")
trainer.train()

# ---------------------------
# 13. Evaluate on test set
# ---------------------------
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_tokenized)

print("\nTest results:")
print(test_results)

# ---------------------------
# 14. Save trained weights
# ---------------------------
torch.save(
    model.state_dict(),
    "../outputs/actor_initial_pool_paper_weighted/actor_model_state.pt"
)

print("\nSaved trained ACTOR weights:")
print("../outputs/actor_initial_pool_paper_weighted/actor_model_state.pt")