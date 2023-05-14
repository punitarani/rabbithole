"""rabbithole.wikipedia module"""

from datasets import load_dataset
from tqdm import tqdm

from rabbithole.vecstore import index


def prepare_wikipedia_collection(batch_size: int = 500):
    """
    Prepare the wikipedia collection
    :param batch_size: Batch size to use when adding documents to the collection
    :return: The wikipedia collection

    NOTE: Only needs to be run once to prepare the collection for the first time
    """
    wikipedia_dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=False)
    print(f"Loaded Wikipedia dataset: {wikipedia_dataset.info}\n")

    total_rows = len(wikipedia_dataset)
    with tqdm(total=total_rows, desc='Processing batches', unit='vectors') as pbar:
        for i in range(0, total_rows, batch_size):
            batch_data = wikipedia_dataset[i: i + batch_size]
            vectors = [
                (
                    str(emb_id),  # Vector ID
                    emb,  # Dense vector values
                    {"title": title, "url": url}  # Vector metadata
                )
                for emb_id, emb, title, url in
                zip(batch_data["id"], batch_data["emb"], batch_data["title"], batch_data["url"])
            ]
            index.upsert(vectors=vectors, namespace="wikipedia")
            pbar.update(batch_size)


if __name__ == "__main__":
    prepare_wikipedia_collection()
