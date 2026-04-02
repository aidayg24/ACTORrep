"""
Training the soft label model
"""
from transformers import TrainingArguments, Trainer

from src.data.softlabel_dataset import prepare_softlabel_datasets
from src.evaluation.metrics_softlabel import compute_metrics_softlabel
from src.models.softlabel_baseline import build_softlabel_model

# data path
train_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_train_softlabel.csv"
dev_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_dev_softlabel.csv"
test_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_test_softlabel.csv"
# load annotation datasets
train_tokenized, dev_tokenized, test_tokenized = prepare_softlabel_datasets(
    train_path,
    dev_path,
    test_path)

# debug check
print(train_tokenized[0].keys())
print(train_tokenized[0]["labels"])
print(train_tokenized[0]["labels"].shape)

# build the model
model = build_softlabel_model()

# Training configuration
training_args = TrainingArguments(
    output_dir="../../outputs/softlabel_baseline",
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
    save_total_limit=2,
    report_to="none"
)
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    compute_metrics=compute_metrics_softlabel,
)

# Train the model
print("\nStarting training...")
trainer.train()

# Final evaluation on the test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_tokenized)

print("\nTest results:")
print(test_results)
