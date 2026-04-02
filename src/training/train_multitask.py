"""
Training the multi-task model
"""
from transformers import TrainingArguments, Trainer

from src.data.annotation_dataset import prepare_annotation_datasets
from src.evaluation.metrics import compute_metrics
from src.models.multitask_baseline import build_multitask_model

# data path
train_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv"
dev_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv"
test_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv"

# load annotation datasets
train_tokenized, dev_tokenized, test_tokenized, annotator_to_id = prepare_annotation_datasets(
    train_path,
    dev_path,
    test_path)

# build the model
model = build_multitask_model(num_labels=2,
                              num_annotators=len(annotator_to_id))

# use dataloader
# train_loader = DataLoader(train_tokenized, batch_size=8, shuffle=True)
# batch = next(iter(train_loader))


# Training configuration
training_args = TrainingArguments(
    output_dir="../../outputs/multitask_baseline",
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
    model=model,
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