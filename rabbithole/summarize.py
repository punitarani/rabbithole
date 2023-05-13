"""rabbithole.summarize module"""

from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document


def summarize_document(document: list[Document]) -> str:
    """
    Summarize a document using the langchain summarize chain

    :param document: Document to summarize.
    It must be a list of langchain.schema.Document objects

    :return: Summarized text
    """

    llm = OpenAI()
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    summary = chain.run(document)

    return summary
