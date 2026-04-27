"""
Run active learning with ACTOR vote variance acquisition.
"""

from src.active_learning.run_active_learning import run_active_learning
from src.acquisition_methods.vote_variance_sampling import vote_variance_sampling


run_active_learning(
    method_name="actor_vote_variance",
    acquisition_fn=vote_variance_sampling,
    train_path="data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv",
    dev_path="data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv",
    test_path="data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv",
    output_path="results/active_learning_vote_variance.json",
    output_dir_base="outputs/activelearning_vote_variance",
    text_column="text",
    needs_dataloader=True,
    include_annotator=True,
)