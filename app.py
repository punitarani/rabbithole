"""Streamlit App"""

import streamlit as st
from langchain.schema import Document

from rabbithole import summarize_document
from rabbithole.loader import load_file
from rabbithole.mp3 import SUPPORTED_FILE_TYPES

# Global variables
results = {}


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


def run_summarization(document: list[Document], doc_name: str):
    """Execute the text summarization"""
    with st.spinner(f'Summarizing {doc_name}...'):
        summary = summarize_document(document[:2])
        results[doc_name] = summary
        st.write(f"'{doc_name}' Summary:\n{summary}")


st.title("RabbitHole")

uploaded_files = st.file_uploader("Upload content", type=["pdf", "txt", *SUPPORTED_FILE_TYPES],
                                  accept_multiple_files=True)

if st.button("Summarize"):
    if not uploaded_files:
        st.warning("Please upload a file first.")
        st.stop()

    # Load the text from the uploaded PDF files
    texts = load_files_with_spinner(uploaded_files)

    # Run the summarization for each document
    for doc_name, doc_text in texts.items():
        run_summarization(doc_text, doc_name)

    st.success('Summarization completed.')
