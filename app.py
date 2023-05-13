"""Streamlit App"""

import tempfile
import threading
from queue import Queue

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rabbithole import summarize_document

# Global variables
q = Queue()


def load_file(file: UploadedFile) -> list[Document]:
    """
    Load a file and return a list of Document objects
    :param file: File to load.
    Supported file types: PDF
    :return: List of Document objects
    """
    if file.type == "application/pdf":
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.read())
        temp_file.close()
        # Load the file using PyMuPDF
        return PyMuPDFLoader(file_path=temp_file.name).load_and_split()
    else:
        raise ValueError(f"Unsupported file type: {file.type}")


def load_files(files: list[UploadedFile]) -> dict[str, list[Document]]:
    """
    Load a list of files and return a dictionary of Document objects
    :param files: List of files to load.
    Supported file types: PDF
    :return: Dictionary of Document objects
    """
    documents = {}
    for file in files:
        documents[file.name] = load_file(file)
    return documents


def run_summarization(q: Queue, document: list[Document]):
    """Execute the text summarization"""
    q.put(summarize_document(document))


st.title("RabbitHole")

uploaded_files = st.file_uploader("Upload content", type=["pdf"], accept_multiple_files=True)

if st.button("Summarize"):
    if not uploaded_files:
        st.warning("Please upload a file first.")
        st.stop()

    # Load the text from the uploaded PDF files
    texts = load_files(uploaded_files)
    text = [doc_text for doc_name, doc_text in texts.items() for doc_text in doc_text]

    # Start the summarization in a separate thread
    # TODD: Display error instead of showing the loading animation forever
    thread = threading.Thread(target=run_summarization, args=(q, text))
    thread.start()

    # Display a loading animation while waiting for the summarization to complete
    with st.spinner('Summarizing...'):
        thread.join()

        # Retrieve the result from the queue
        summarized_text = q.get()
        st.success('Summarization completed.')
        st.write(summarized_text)
