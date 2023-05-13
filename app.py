"""Streamlit App"""

import tempfile
import threading
from queue import Queue

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document

from rabbithole import summarize_document

# Global variables
q = Queue()


def run_summarization(q: Queue, document: list[Document]):
    """Execute the text summarization"""
    q.put(summarize_document(document))


st.title("RabbitHole")

uploaded_file = st.file_uploader("Upload a text file", type=["pdf"], accept_multiple_files=False)

if st.button("Summarize"):
    if uploaded_file is None:
        st.warning("Please upload a file first.")
        st.stop()

    # Save file as temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Load the text from the uploaded PDF file
    text = PyMuPDFLoader(file_path=temp_file.name).load_and_split()

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
