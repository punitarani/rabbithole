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


def run_summarization(q: Queue, document: list[Document]):
    """Execute the text summarization"""
    q.put(summarize_document(document))


st.title("RabbitHole")

uploaded_file = st.file_uploader("Upload a text file", type=["pdf"], accept_multiple_files=False)

if st.button("Summarize"):
    if uploaded_file is None:
        st.warning("Please upload a file first.")
        st.stop()

    # Load the text from the uploaded PDF file
    text = load_file(uploaded_file)

    # Start the summarization in a separate thread
    thread = threading.Thread(target=run_summarization, args=(q, text))
    thread.start()

    # Display a loading animation while waiting for the summarization to complete
    with st.spinner('Summarizing...'):
        thread.join()

        # Retrieve the result from the queue
        summarized_text = q.get()
        st.success('Summarization completed.')
        st.write(summarized_text)
