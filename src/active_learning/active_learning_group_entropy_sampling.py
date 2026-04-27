"""
Run active learning with ACTOR group-level entropy.
"""

from src.active_learning.run_active_learning import run_active_learning
from src.acquisition_methods.group_entropy_sampling import group_entropy_sampling


run_active_learning(
    method_name="actor_group_entropy",
    acquisition_fn=group_entropy_sampling,
    train_path="../../data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv",
    dev_path="../../data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv",
    test_path="../../data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv",
    output_path="../../results/active_learning_group_entropy.json",
    output_dir_base="../../outputs/activelearning_group_entropy",
    text_column="text",
    needs_dataloader=True,
    include_annotator=True,
)