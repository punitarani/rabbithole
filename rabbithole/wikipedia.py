"""rabbithole.wikipedia module"""

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
    for collection in client.list_collections():
        if collection.name == "wikipedia":
            print("Wikipedia collection already exists. Deleting and recreating...")
            client.delete_collection("wikipedia")
            break
    collection = client.create_collection("wikipedia")

    total_rows = len(wikipedia_dataset)
    with tqdm(total=total_rows, desc='Processing batches', unit='vectors') as pbar:
        for i in range(0, total_rows, batch_size):
            batch_data = wikipedia_dataset[i: i + batch_size]
            collection.add(
                ids=[str(emb_id) for emb_id in batch_data["id"]],
                embeddings=batch_data["emb"],
                documents=batch_data["text"],
                metadatas=[{"title": title, "url": url} for title, url in zip(batch_data["title"], batch_data["url"])],
            )

            pbar.update(batch_size)


if __name__ == "__main__":
    print(wikipedia_dataset.info)
    prepare_wikipedia_collection()
