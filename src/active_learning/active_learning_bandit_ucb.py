from src.active_learning.run_active_learning import run_active_learning
from src.acquisition_methods.bandit_sampling import bandit_ucb_sampling


run_active_learning(
    method_name="bandit_ucb_sampling",
    acquisition_fn=bandit_ucb_sampling,
    train_path="data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv",
    dev_path="data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv",
    test_path="data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv",
    output_path="results/active_learning_bandit_ucb.json",
    output_dir_base="outputs/activelearning_bandit_ucb",
    needs_dataloader=True,
    include_annotator=True,
)