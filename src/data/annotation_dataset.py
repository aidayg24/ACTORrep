"""
Load annotation-level CSV files, tokenize the text, and create
numeric annotator IDs for multi-task training.
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def prepare_annotation_datasets(train_path, dev_path, test_path, model_name="bert-base-uncased",
    max_length=128):
    """
    this function loads the datasets and return the torch format of them
    :param train_path: path to training data
    :param dev_path: path to dev data
    :param test_path: path to test data
    :return:  train_tokenized, dev_tokenized, test_tokenized
    """
    # # Load csv files
    # train_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv"
    # dev_path = "../../data/HS-Brexit_dataset_processed/processed_data/HS-brexit_dev_annotations.csv"
    # test_path = "../../data/HS-Brexit_dataset_processed/processed_data/HS-brexit_test_annotations.csv"

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    # Build annotator mapping from train split only
    annotators = sorted(train_df["annotator_id"].unique())
    annotator_to_id = {ann: idx for idx, ann in enumerate(annotators)}

    # Apply mapping to all splits
    for df, split_name in [(train_df, "train"), (dev_df, "dev"), (test_df, "test")]:
        unknown = set(df["annotator_id"].unique()) - set(annotator_to_id.keys())
        if unknown:
            raise ValueError(f"Unknown annotators in {split_name}: {unknown}")
        df["annotator_idx"] = df["annotator_id"].map(annotator_to_id)

    # Convert to huggingFace dataset
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # Tokenize all splits
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    dev_tokenized = dev_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    # Removed unused columns
    columns_to_keep = ["input_ids", "attention_mask", "labels", "annotator_idx"]

    # Set format to torch
    train_tokenized.set_format(type="torch", columns=columns_to_keep)
    dev_tokenized.set_format(type="torch", columns=columns_to_keep)
    test_tokenized.set_format(type="torch", columns=columns_to_keep)

    return train_tokenized, dev_tokenized, test_tokenized, annotator_to_id
