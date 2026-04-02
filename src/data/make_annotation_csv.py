"""
Create annotation-level datasets from the HS-Brexit JSON files.

The original dataset contains multiple annotator labels per tweet.
For example:

annotations: "0,0,0,1,0,0"
annotators: "Ann1,Ann2,Ann3,Ann4,Ann5,Ann6"

Instead of collapsing these into a majority label, we expand them so that
each annotation becomes its own training example.

Example transformation:

Input:
tweet_id = 1
annotations = "0,0,0,1,0,0"

Output rows:
1, text, Ann1, 0
1, text, Ann2, 0
1, text, Ann3, 0
1, text, Ann4, 1
1, text, Ann5, 0
1, text, Ann6, 0

This dataset will be used for the multi-task model baseline,
which trains a standard BERT classifier on all individual annotations.
"""

import json
import csv
import os

INPUT_FOLDER = "../../data/HS-Brexit_dataset_raw"
OUTPUT_FOLDER = "../../data/HS-Brexit_dataset_processed"


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

splits = ["train", "dev", "test"]

for split in splits:

    input_path = f"{INPUT_FOLDER}/HS-Brexit_{split}.json"
    output_path = f"{OUTPUT_FOLDER}/HS-brexit_{split}_annotations.csv"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["item_id", "text", "annotator_id", "labels"])

        for item_id, item in data.items():

            text = item["text"]
            annotations = [a.strip() for a in item["annotations"].split(",")]
            annotators = [a.strip() for a in item["annotators"].split(",")]

            for annotator, label in zip(annotators, annotations):

                writer.writerow([
                    item_id,
                    text,
                    annotator,
                    int(label)
                ])

    print(f"Saved {output_path}")