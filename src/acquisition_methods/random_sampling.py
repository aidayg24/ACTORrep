from __future__ import annotations

import random


def random_sampling(unlabeled_indices, n_samples, seed=42):
    """
    Uniformly sample indices from the unlabeled pool.
    :param unlabeled_indices: Sequence of candidate indices still available for acquisition.
    :param n_samples: Number of examples to acquire this round.
    :param seed: Optional random seed for reproducibility.
    :return: A list of selected indices.
    """
    random.seed(seed)
    if n_samples > len(unlabeled_indices):
        n_samples = len(unlabeled_indices)
    selected_indices = random.sample(unlabeled_indices, n_samples)
    return selected_indices
