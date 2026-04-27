from src.active_learning.run_active_learning import run_active_learning
from src.acquisition_methods.mixed_entropy_sampling import mixed_entropy_sampling


run_active_learning(
    method_name="actor_mixed_entropy",
    acquisition_fn=mixed_entropy_sampling,
    train_path="../../data/HS-Brexit_dataset_processed/HS-brexit_train_annotations.csv",
    dev_path="../../data/HS-Brexit_dataset_processed/HS-brexit_dev_annotations.csv",
    test_path="../../data/HS-Brexit_dataset_processed/HS-brexit_test_annotations.csv",
    output_path="../../results/active_learning_mixed_entropy.json",
    output_dir_base="../../outputs/activelearning_mixed_entropy",
    needs_dataloader=True,
    include_annotator=True,
)