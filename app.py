"""Streamlit App"""

import threading
from queue import Queue

import streamlit as st
from langchain.schema import Document

from rabbithole import summarize_document
from rabbithole.loader import load_files

# Global variables
q = Queue()


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
