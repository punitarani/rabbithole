"""rabbithole.summarize module"""

from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


def summarize_text(text: str) -> str:
    """
    Summarize text using the langchain summarize chain
    :param text: Text to summarize
    :return: Summarized text
    """

    text_splitter = CharacterTextSplitter()
    llm = OpenAI(temperature=0.5)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts[:3]]
    summary = chain.run(docs)

    return summary
