from datasets import load_dataset

# Load the TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories")

# Open one file to write all stories
with open("tinystories_combined.txt", "w", encoding="utf-8") as f:
    for split_name in dataset.keys():  # train, validation, test
        print(f"Processing split: {split_name}")
        for item in dataset[split_name]:
            f.write(item['text'] + "\n")  # write each story as a line

print("Combined file saved as tinystories_combined.txt")
