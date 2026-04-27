from src.active_learning.run_active_learning import run_active_learning
from src.acquisition_methods.random_sampling import random_sampling

run_active_learning(
    method_name="random_sampling",
    acquisition_fn=random_sampling,
    train_path="data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv",
    dev_path="data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv",
    test_path="data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv",
    output_path="results/active_learning_random_sampling.json",
    output_dir_base="outputs/activelearning_random",
    needs_dataloader=False,
)