"""rabbithole.wikipedia module"""

from datasets import load_dataset
from tqdm import tqdm

from vecstore import client

wikipedia_dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=False)
wikipedia_collection = client.create_collection("wikipedia")


def prepare_wikipedia_collection(batch_size: int = 10000):
    """
    Prepare the wikipedia collection

    NOTE: Only needs to be run once to prepare the collection for the first time
    """
    total_rows = len(wikipedia_dataset)
    for i in tqdm(range(0, total_rows, batch_size), desc='Processing batches'):
        batch_data = wikipedia_dataset[i: i+batch_size]
        ids, embeddings, documents, metadatas = zip(
            *[(i, row['emb'], row['text'], row['title']) for i, row in enumerate(batch_data)])

        wikipedia_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )


if __name__ == "__main__":
    print(wikipedia_dataset.info)

    print("Preparing wikipedia collection...")
    prepare_wikipedia_collection()
    print("Done!")
