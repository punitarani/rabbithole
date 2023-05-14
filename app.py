"""Streamlit App"""

import streamlit as st
from langchain.schema import Document

from rabbithole import summarize_document
from rabbithole.embedding import embed_document
from rabbithole.keywords import get_document_keywords
from rabbithole.loader import load_file, SUPPORTED_IMG_FILE_TYPES
from rabbithole.mp3 import SUPPORTED_AV_FILE_TYPES

# Global variables
global_documents = {}
global_embeddings = {}
global_keywords = {}
global_summaries = {}


def load_files_with_spinner(files: list) -> dict[str, list[Document]]:
    """
    Load a list of files and return a list of dictionaries of Document objects.
    Display a loading animation while loading each file.
    :param files: List of files to load.
    :return: List of dictionaries of Document objects.
    """
    # Combine the results into a single dictionary
    documents = {}
    for file in files:
        with st.spinner(f'Loading {file.name}...'):
            documents[file.name] = load_file(file)
    return documents


def embed_documents_with_spinner(documents: dict[str, list[Document]]) -> dict[str, list[list[float]]]:
    """
    Embed a list of documents and return a list of dictionaries of embeddings.
    Display a loading animation while embedding each document.
    :param documents: List of documents to embed.
    :return: List of dictionaries of embeddings.
    """
    # Combine the results into a single dictionary
    embeddings = {}
    for doc_name, doc_text in documents.items():
        with st.spinner(f'Embedding {doc_name}...'):
            embeddings[doc_name] = embed_document([doc.page_content for doc in doc_text])
    return embeddings


def extract_keywords_with_spinner(embeddings: dict[str, list[list[float]]]):
    """
    Extract keywords from a list of embeddings and return a list of keywords.
    Display a loading animation while extracting each keyword.
    :param embeddings: List of embeddings to extract keywords from.
    :return: List of keywords.
    """
    # Combine the results into a single dictionary
    keywords = {}
    for doc_name, doc_embeddings in embeddings.items():
        with st.spinner(f'Extracting keywords from {doc_name}...'):
            keywords[doc_name] = get_document_keywords(doc_embeddings)
    return keywords


def generate_summary_with_spinner(documents: dict[str, list[Document]]) -> dict[str, list[list[float]]]:
    """
    Embed a list of documents and return a list of dictionaries of embeddings.
    Display a loading animation while embedding each document.
    :param documents: List of documents to embed.
    :return: List of dictionaries of embeddings.
    """
    summaries = {}
    for doc_name, doc_text in documents.items():
        with st.spinner(f'Summarizing {doc_name}...'):
            summaries[doc_name] = summarize_document(doc_text)
    return summaries


st.set_page_config(page_title="RabbitHole", page_icon="🐇", layout="wide")

st.title("RabbitHole")

uploaded_files = st.file_uploader("Upload content",
                                  type=["docx", "pdf", "txt", *SUPPORTED_IMG_FILE_TYPES, *SUPPORTED_AV_FILE_TYPES],
                                  accept_multiple_files=True)

if st.button("Dive in"):
    if not uploaded_files:
        st.warning("Please upload a file first.")
        st.stop()

    # Load the text from the uploaded PDF files
    global_documents = load_files_with_spinner(uploaded_files)
    global_embeddings = embed_documents_with_spinner(global_documents)
    global_keywords = extract_keywords_with_spinner(global_embeddings)
    global_summaries = generate_summary_with_spinner(global_documents)

    # Display the keywords and summaries
    for doc_name, doc_keywords in global_keywords.items():
        st.header(doc_name)
        st.caption("Keywords: " + ", ".join(doc_keywords))
        st.write(global_summaries[doc_name])
        st.divider()

    st.success('Summarization completed.')
