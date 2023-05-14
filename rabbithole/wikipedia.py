"""rabbithole.wikipedia module"""

from time import time

from chromadb.api import Collection
from datasets import load_dataset
from tqdm import tqdm

from rabbithole.vecstore import client

wikipedia_dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=False)


def get_wikipedia_collection() -> Collection:
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
    with tqdm(total=total_rows, desc='Processing batches', unit='vectors') as pbar:
        for i in range(0, total_rows, batch_size):
            start_time = time()

            batch_data = wikipedia_dataset[i: i + batch_size]
            collection.add(
                ids=[str(id) for id in batch_data["id"]],
                embeddings=batch_data["emb"],
                documents=batch_data["text"],
                metadatas=[{"title": title} for title in batch_data["title"]],
            )

            elapsed_time = time() - start_time
            vectors_per_second = batch_size / elapsed_time
            pbar.set_postfix({'vectors/sec': vectors_per_second}, refresh=True)
            pbar.update(batch_size)


if __name__ == "__main__":
    print(wikipedia_dataset.info)
    prepare_wikipedia_collection()
