"""
Load annotation-level CSV files, tokenize the text, and create
numeric annotator IDs for multi-task training.
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def prepare_annotation_datasets(train,
                                dev,
                                test,
                                model_name="bert-base-uncased",
                                max_length=128,
                                annotator_to_id=None):
    """
    Prepare annotation-level datasets for training.

    This function accepts either:
    - file paths (str) OR
    - pandas DataFrames

    :param train: path or DataFrame for training data
    :param dev: path or DataFrame for dev data
    :param test: path or DataFrame for test data
    :param annotator_to_id: optional annotator mapping
    """

    # Load or use existing DataFrames
    train_df = train if isinstance(train, pd.DataFrame) else pd.read_csv(train)
    dev_df = dev if isinstance(dev, pd.DataFrame) else pd.read_csv(dev)
    test_df = test if isinstance(test, pd.DataFrame) else pd.read_csv(test)


    # Build annotator mapping from train split only
    if annotator_to_id is None:
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
