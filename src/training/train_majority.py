"""
Training the majority model
"""
from transformers import TrainingArguments, Trainer

from src.evaluation.majority_metrics import compute_metrics
from src.models.majority_baseline import build_majority_model
from src.data.majority_dataset import prepare_majority_datasets

# data path
train_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_train_majority.csv"
dev_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_dev_majority.csv"
test_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_test_majority.csv"

train_tokenized, dev_tokenized, test_tokenized = prepare_majority_datasets(train_path, dev_path, test_path)

# Training configuration
training_args = TrainingArguments(
    output_dir="../../outputs/majority_baseline",
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
# Create trainer
trainer = Trainer(
    model=build_majority_model(),
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    compute_metrics=compute_metrics,
)

# Train the model
print("\nStarting training...")
trainer.train()

# Final evaluation on the test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_tokenized)

print("\nTest results:")
print(test_results)
