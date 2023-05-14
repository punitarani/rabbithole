"""rabbithole.keywords module"""

from collections import Counter
from heapq import nlargest
from math import log

from chromadb.api import QueryResult

from rabbithole.wikipedia import get_wikipedia_collection

wikipedia_collection = get_wikipedia_collection()


def get_document_keywords(embeddings: list[list[float]], n: int = 10) -> list[str]:
    """
    Get keywords from the text embeddings of a document
    :param embeddings: Text embeddings to get keywords from
    :param n: Number of keywords to return
    :return List of keywords

    NOTE: This requires the embeddings to use cohere multilingual-22-12 model
    """
    # Check input types
    if not isinstance(embeddings, list):
        raise TypeError(f"embeddings must be a list. Got {type(embeddings)}")
    if not isinstance(embeddings[0], list):
        raise TypeError(f"embeddings must be a list of lists. Got list[{type(embeddings[0])}]")

    n_mult = 3  # multiplier to get more results

    # Query the Wikipedia collection with the embeddings
    results: QueryResult = wikipedia_collection.query(
        query_embeddings=embeddings,
        n_results=n * n_mult,
    )

    keywords = []  # to store the keywords
    metadatas = results.get("metadatas", [])

    # Loop over the metadatas to extract the titles as keywords
    for metadata in metadatas:
        keyword_set = {titles.get("title") for titles in metadata if titles.get("title")}
        keywords.append(keyword_set)

    # Initialize counters for keyword count, weight, and document frequency
    keyword_count = Counter()
    keyword_weight = Counter()
    document_frequency = Counter()

    num_documents = len(keywords)

    # Count the document frequency for each unique keyword
    for kw_list in keywords:
        unique_keywords = set(kw_list)
        document_frequency.update(unique_keywords)

    # Calculate the weight of each keyword
    for kw_list in keywords:
        num_keywords = len(kw_list)
        for keyword in kw_list:
            keyword_count[keyword] += 1
            tf = 1 / num_keywords
            idf = log(num_documents / document_frequency[keyword])
            keyword_weight[keyword] += tf * idf

    # Get the n largest keywords by weight
    top_keywords = nlargest(n, keyword_weight, key=keyword_weight.get)

    return top_keywords
