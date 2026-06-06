from src.active_learning.run_active_learning import run_active_learning
from src.acquisition_methods.bandit_sampling import bandit_ucb_sampling


run_active_learning(
    method_name="bandit_ucb_sampling_debug",
    acquisition_fn=bandit_ucb_sampling,
    train_path="data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv",
    dev_path="data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv",
    test_path="data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv",
    output_path="results/debug_bandit_ucb.json",
    output_dir_base="outputs/debug_bandit_ucb",
    initial_size=20,
    acquisition_size=5,
    num_rounds=2,
    needs_dataloader=True,
    include_annotator=True,
)