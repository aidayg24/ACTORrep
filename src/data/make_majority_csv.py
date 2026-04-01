"""
Convert all HS-Brexit JSON splits (train/dev/test) into CSV files
using the majority-vote label (`hard_label`).

we use hard_label as majority vote

Each example in the dataset contains:
    - `annotations`: the individual labels from each annotator
    - `soft_label`: the distribution of labels (probabilities of 0 and 1)
    - `hard_label`: a single label

Example entry (check the dataset):

{
    "annotations": "0,0,0,0,0,0",
    "soft_label": {"0": 1.0, "1": 0.0},
    "hard_label": "0"
}

"""
import json
import csv
import os

input_folder = "../../data/HS-Brexit_dataset_raw"
output_folder = "../../data/HS-Brexit_dataset_processed"

os.makedirs(output_folder, exist_ok=True)

split_files = [
    "HS-Brexit_train.json",
    "HS-Brexit_dev.json",
    "HS-Brexit_test.json"
]

for filename in split_files:
    input_path = os.path.join(input_folder, filename)

    split_name = filename.replace("HS-Brexit_", "").replace(".json", "")
    output_path = os.path.join(output_folder, f"HS-brexit_{split_name}_majority.csv")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "text", "labels"])

        for item_id, item in data.items():
            writer.writerow([
                item_id,
                item["text"],
                int(item["hard_label"])
            ])

    print(f"Saved: {output_path}")