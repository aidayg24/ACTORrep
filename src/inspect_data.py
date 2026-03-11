import json

path = "../HS-Brexit_dataset/HS-Brexit_train.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Type:", type(data))
print("Number of items:", len(data))

first_key = list(data.keys())[0]

print("\nFirst key:", first_key)
print("\nFirst item:")
print(data[first_key])