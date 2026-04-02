from __future__ import annotations

import random


def initialize_pools(all_indices, initial_size =100, seed=42):
    """
    Create initial labeled and unlabeled pools.

    This function randomly selects a subset of indices to form the initial
    labeled pool. All remaining indices are placed in the unlabeled pool.

    :param all_indices: List of all available indices (e.g. range(len(train_df)))
    :param initial_size: Number of samples to include in the initial labeled pool
    :param seed: Random seed for reproducibility
    :return:
        labeled_indices: List of indices selected for initial training
        unlabeled_indices: List of remaining indices
    """
    random.seed(seed)

    # randomly pick initial labeled examples
    labeled_indices = random.sample(all_indices, initial_size)

    # everything else stays unlabeled
    unlabeled_indices = [
        i for i in all_indices if i not in labeled_indices
    ]

    return labeled_indices, unlabeled_indices


def update_pools(labeled_indices, unlabeled_indices, new_indices):
    """
    Move newly selected samples from the unlabeled pool to the labeled pool.

    This function updates both pools after an acquisition step:
    - adds new indices to the labeled pool
    - removes them from the unlabeled pool

    :param labeled_indices: Current labeled pool (list of indices)
    :param unlabeled_indices: Current unlabeled pool (list of indices)
    :param new_indices: Indices selected by an acquisition method
    :return:
        updated_labeled_indices: Updated labeled pool
        updated_unlabeled_indices: Updated unlabeled pool
    """

    # add new samples to labeled pool
    updated_labeled_indices = labeled_indices + new_indices

    # remove them from unlabeled pool
    updated_unlabeled_indices = [
        i for i in unlabeled_indices if i not in new_indices
    ]

    return updated_labeled_indices, updated_unlabeled_indices
