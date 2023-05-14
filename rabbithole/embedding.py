"""rabbithole.embedding module"""

from langchain.embeddings import CohereEmbeddings

cohere = CohereEmbeddings()
cohere.model = "multilingual-22-12"


def embed_document(texts: list[str]) -> list[list[float]]:
    """
    Embed a document using the cohere multilingual-22-12 model
    :param texts: Document texts to embed
    :return: List of embeddings
    """
    return cohere.embed_documents(texts=texts)
