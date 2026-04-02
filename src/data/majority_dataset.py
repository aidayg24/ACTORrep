"""
load the majority_processed data  and prepare them for training

"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def prepare_majority_datasets(train_path, dev_path, test_path, model_name="bert-base-uncased",
    max_length=128):
    """
    this function loads the datasets and return the torch format of them
    :param train_path: path to training data
    :param dev_path: path to dev data
    :param test_path: path to test data
    :return:  train_tokenized, dev_tokenized, test_tokenized
    """
    # # Load csv files
    # train_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_train_majority.csv"
    # dev_path = "../../data/HS-Brexit_dataset_processed/processed_data/HS-brexit_dev_majority.csv"
    # test_path = "../../data/HS-Brexit_dataset_processed/processed_data/HS-brexit_test_majority.csv"

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

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
    train_tokenized = train_tokenized.remove_columns(["item_id", "text"])
    dev_tokenized = dev_tokenized.remove_columns(["item_id", "text"])
    test_tokenized = test_tokenized.remove_columns(["item_id", "text"])

    # Set format to torch
    train_tokenized.set_format("torch")
    dev_tokenized.set_format("torch")
    test_tokenized.set_format("torch")

    return train_tokenized, dev_tokenized, test_tokenized
