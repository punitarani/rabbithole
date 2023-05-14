"""rabbithole.wikipedia module"""

import chromadb.api
from datasets import load_dataset
from tqdm import tqdm

from rabbithole.vecstore import client

wikipedia_dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=False)


def get_wikipedia_collection() -> chromadb.api.Collection:
    """
    Get the wikipedia collection
    """
    return client.get_collection("wikipedia")


def prepare_wikipedia_collection(batch_size: int = 10000):
    """
    Prepare the wikipedia collection

    NOTE: Only needs to be run once to prepare the collection for the first time
    """
    collection = client.create_collection("wikipedia")

    total_rows = len(wikipedia_dataset)
    for i in tqdm(range(0, total_rows, batch_size), desc='Processing batches'):
        batch_data = wikipedia_dataset[i: i + batch_size]

        collection.add(
            ids=[str(id) for id in batch_data["id"]],
            embeddings=batch_data["emb"],
            documents=batch_data["text"],
            metadatas=[{"title": title} for title in batch_data["title"]],
        )


if __name__ == "__main__":
    print(wikipedia_dataset.info)
    prepare_wikipedia_collection()
