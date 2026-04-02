import json
import os

import pandas as pd

from src.acquisition_methods.random_sampling import random_sampling
from src.data.annotation_dataset import prepare_annotation_datasets
from src.training.train_multitask import train_multitask_model
from src.utils.pool_utils import initialize_pools, update_pools

# data path
train_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv"
dev_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv"
test_path = "../../data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv"

train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)

# get all unique annotators from the full training data
unique_annotators = sorted(train_df["annotator_id"].unique())

# map each annotator to an integer index
annotator2idx = {annotator: idx for idx, annotator in enumerate(unique_annotators)}

# create a list of train row indices
all_indices = list(range(len(train_df)))

# using pool to initialize the labeled and unlabeled
labeled_indices, unlabeled_indices = initialize_pools(
    all_indices,
    initial_size=100,
    seed=42)

# active learning settings
num_rounds = 2
acquisition_size = 30
output_path = "../../results/active_learning_random_sampling.json"

# store all round results here
results = []

for round_id in range(num_rounds):
    print(f"\n===== ROUND {round_id} =====")
    print(f"Labeled pool size: {len(labeled_indices)}")
    print(f"Unlabeled pool size: {len(unlabeled_indices)}")

    # create the current labeled training set for this round
    current_train_df = train_df.iloc[labeled_indices].reset_index(drop=True)

    # prepare tokenized datasets
    train_dataset, dev_dataset, test_dataset, _ = prepare_annotation_datasets(
        train=current_train_df,
        dev=dev_df,
        test=test_df,
        annotator_to_id=annotator2idx
    )

    # train and evaluate the model
    trainer, dev_results, test_results = train_multitask_model(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        num_annotators=len(annotator2idx),
        output_dir=f"../../outputs/activelearning_random/round_{round_id}"
    )

    # save this round's results
    round_result = {
        "round": round_id,
        "labeled_size": len(labeled_indices),
        "unlabeled_size": len(unlabeled_indices),
        "dev_results": dev_results,
        "test_results": test_results
    }

    results.append(round_result)

    # make sure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save results after each round
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # stop after the last round
    if round_id == num_rounds - 1:
        break

    # randomly select new samples from the unlabeled pool
    new_indices = random_sampling(
        unlabeled_indices=unlabeled_indices,
        n_samples=acquisition_size,
        seed=42 + round_id
    )

    # update labeled and unlabeled pools
    labeled_indices, unlabeled_indices = update_pools(
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        new_indices=new_indices
    )

print("\nActive learning finished.")
print(f"Results saved to: {output_path}")
