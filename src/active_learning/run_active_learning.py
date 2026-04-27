"""
Reusable active learning runner.

This file contains one general function for running active learning
with different acquisition methods.

The goal is to avoid having almost identical scripts for random sampling,
entropy sampling, individual entropy, group entropy, vote variance, etc.
Each experiment only needs to pass:
1. method name,
2. acquisition function,
3. whether the acquisition function needs a dataloader,
4. whether it needs annotator_idx.
"""

import json
import os
import pandas as pd

from src.data.annotation_dataset import prepare_annotation_datasets
from src.data.unlabeled_dataset import build_unlabeled_loader
from src.training.train_multitask import train_multitask_model
from src.utils.pool_utils import initialize_pools, update_pools


def run_active_learning(
    method_name,
    acquisition_fn,
    train_path,
    dev_path,
    test_path,
    output_path,
    output_dir_base,
    initial_size=100,
    acquisition_size=30,
    num_rounds=1,
    seed=42,
    text_column="text",
    needs_dataloader=True,
    include_annotator=False,
):
    """
    Run one active learning experiment.

    Args:
        method_name: name stored in the results file.
        acquisition_fn: function used to select new unlabeled examples.
        train_path/dev_path/test_path: data paths.
        output_path: JSON result path.
        output_dir_base: checkpoint/output folder base.
        initial_size: seed labeled pool size.
        acquisition_size: number of new rows selected per round.
        num_rounds: number of AL rounds.
        seed: random seed.
        text_column: text column name.
        needs_dataloader: True for entropy-style methods, False for random.
        include_annotator: True for ACTOR individual/group/mix/vote methods.

    Returns:
        results: list of round-level result dictionaries.
    """

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    unique_annotators = sorted(train_df["annotator_id"].unique())
    annotator2idx = {annotator: idx for idx, annotator in enumerate(unique_annotators)}

    all_indices = list(range(len(train_df)))

    labeled_indices, unlabeled_indices = initialize_pools(
        all_indices,
        initial_size=initial_size,
        seed=seed,
    )

    results = []

    for round_id in range(num_rounds):
        print(f"\n===== ROUND {round_id} | {method_name} =====")
        print(f"Labeled pool size: {len(labeled_indices)}")
        print(f"Unlabeled pool size: {len(unlabeled_indices)}")

        current_train_df = train_df.iloc[labeled_indices].reset_index(drop=True)

        train_dataset, dev_dataset, test_dataset, _ = prepare_annotation_datasets(
            train=current_train_df,
            dev=dev_df,
            test=test_df,
            annotator_to_id=annotator2idx,
        )

        trainer, dev_results, test_results = train_multitask_model(
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            num_annotators=len(annotator2idx),
            output_dir=f"{output_dir_base}/round_{round_id}",
        )

        round_result = {
            "round": round_id,
            "method": method_name,
            "labeled_size": len(labeled_indices),
            "unlabeled_size": len(unlabeled_indices),
            "dev_results": dev_results,
            "test_results": test_results,
        }

        results.append(round_result)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if round_id == num_rounds - 1:
            break

        if needs_dataloader:
            current_unlabeled_df = train_df.iloc[unlabeled_indices].copy()
            current_unlabeled_df["original_index"] = current_unlabeled_df.index

            if include_annotator:
                current_unlabeled_df["annotator_idx"] = (
                    current_unlabeled_df["annotator_id"].map(annotator2idx)
                )

            current_unlabeled_df = current_unlabeled_df.reset_index(drop=True)

            unlabeled_loader = build_unlabeled_loader(
                df=current_unlabeled_df,
                text_column=text_column,
                id_column="original_index",
                annotator_column="annotator_idx",
                tokenizer_name="bert-base-uncased",
                max_length=128,
                batch_size=32,
                include_annotator=include_annotator,
            )

            new_indices = acquisition_fn(
                model=trainer.model,
                dataloader=unlabeled_loader,
                count=acquisition_size,
            )

        else:
            new_indices = acquisition_fn(
                unlabeled_indices=unlabeled_indices,
                n_samples=acquisition_size,
                seed=seed + round_id,
            )

        results[-1]["selected_indices"] = new_indices

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        labeled_indices, unlabeled_indices = update_pools(
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            new_indices=new_indices,
        )

    print("\nActive learning finished.")
    print(f"Results saved to: {output_path}")

    return results