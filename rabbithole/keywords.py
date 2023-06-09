"""rabbithole.keywords module"""

from collections import Counter
from heapq import nlargest
from math import log

import streamlit as st
from pinecone import QueryResponse

from rabbithole.vecstore import index


@st.cache_data
def get_document_keywords(embeddings: list[list[float]], n: int = 10, n_mult=3) -> list[str]:
    """
    Get keywords from the text embeddings of a document
    :param embeddings: Text embeddings to get keywords from
    :param n: Number of keywords to return
    :param n_mult: n-multiplier to query and filter more keywords
    :return List of keywords

    NOTE: This requires the embeddings to use cohere multilingual-22-12 model
    """
    # Check input types
    if not isinstance(embeddings, list):
        raise TypeError(f"embeddings must be a list. Got {type(embeddings)}")
    if not isinstance(embeddings[0], list):
        raise TypeError(f"embeddings must be a list of lists. Got list[{type(embeddings[0])}]")

    results: list[list[dict]] = []

    # Query the Wikipedia collection with the embeddings
    for emb in embeddings:
        result: QueryResponse = index.query(
            vector=emb,
            top_k=n * n_mult,
            include_values=False,
            include_metadata=True,
            namespace="wikipedia"
        )
        results.append(result.get("matches", []))

    # Loop over the metadatas to extract the titles as keywords
    keywords = [
        [
            vector.get("metadata").get("title")
            for vector in result if vector.get("metadata")
        ]
        for result in results
    ]

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
