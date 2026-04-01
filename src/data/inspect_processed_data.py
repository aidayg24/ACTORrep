import pandas as pd

path = "../processed_data/HS-brexit_train_majority.csv"

df = pd.read_csv(path)

print("Number of samples:", len(df))
print("\nLabel distribution:")
print(df["hard_label"].value_counts())

print("\nExample rows:")
print(df.head())