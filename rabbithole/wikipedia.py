"""rabbithole.wikipedia module"""

from datasets import load_dataset

wikipedia_dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=False)

if __name__ == "__main__":
    print(wikipedia_dataset.info)
