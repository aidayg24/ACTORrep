"""
Training utilities for the multi-task model.

This file supports two usages:

1. Standard training on the full train/dev/test annotation splits
   when the file is run directly.

2. Reusable training function for active learning, where the current
   labeled training dataset changes at each round.
"""
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from src.data.annotation_dataset import prepare_annotation_datasets
from src.evaluation.metrics import compute_metrics
from src.models.multitask_baseline import build_multitask_model


def train_multitask_model(
        train_dataset,
        dev_dataset,
        test_dataset,
        num_annotators,
        output_dir,
        model_name="bert-base-uncased",
        num_labels=2,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        early_stopping_patience=2,
):
    """
    Train and evaluate the multi-task model.

    This function is reusable for active learning, where the training set
    changes from round to round.

    :param train_dataset: tokenized training dataset
    :param dev_dataset: tokenized development dataset
    :param test_dataset: tokenized test dataset
    :param num_annotators: number of annotators / classifier heads
    :param output_dir: where checkpoints and logs should be saved
    :param model_name: transformer backbone name
    :param num_labels: number of label classes
    :param num_train_epochs: maximum number of training epochs
    :param per_device_train_batch_size: training batch size
    :param per_device_eval_batch_size: evaluation batch size
    :param learning_rate: optimizer learning rate
    :param weight_decay: weight decay
    :param early_stopping_patience: stop if dev metric does not improve
    :return:
        trainer: Hugging Face trainer object
        dev_results: evaluation results on dev set
        test_results: evaluation results on test set
    """

    # build the model
    model = build_multitask_model(
        model_name=model_name,
        num_labels=num_labels,
        num_annotators=num_annotators
    )

    # training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none"
    )

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    # train the model
    print("\nStarting training...")
    trainer.train()

    # evaluate on dev set
    print("\nEvaluating on dev set...")
    dev_results = trainer.evaluate(dev_dataset)

    # final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print("\nDev results:")
    print(dev_results)

    print("\nTest results:")
    print(test_results)

    return trainer, dev_results, test_results


def main():
    """
    Old usage: run full multi-task training on the fixed annotation splits.
    """

    # data path
    train_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv"
    dev_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv"
    test_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv"

    # load annotation datasets
    train_tokenized, dev_tokenized, test_tokenized, annotator_to_id = prepare_annotation_datasets(
        train=train_path,
        dev=dev_path,
        test=test_path
    )

    # call reusable training function
    train_multitask_model(
        train_dataset=train_tokenized,
        dev_dataset=dev_tokenized,
        test_dataset=test_tokenized,
        num_annotators=len(annotator_to_id),
        output_dir="../../outputs/multitask_baseline",
        model_name="bert-base-uncased",
        num_labels=2,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        early_stopping_patience=2,
    )


if __name__ == "__main__":
    main()
