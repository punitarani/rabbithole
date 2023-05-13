"""Streamlit App"""

import threading
from queue import Queue

import streamlit as st
from langchain.schema import Document

from rabbithole import summarize_document
from rabbithole.loader import load_files

# Global variables
q = Queue()
results = {}


def run_summarization(q: Queue, document: list[Document], doc_name: str):
    """Execute the text summarization"""
    result = summarize_document(document)
    results[doc_name] = result
    q.put(True)


st.title("RabbitHole")

uploaded_files = st.file_uploader("Upload content", type=["pdf"], accept_multiple_files=True)

if st.button("Summarize"):
    if not uploaded_files:
        st.warning("Please upload a file first.")
        st.stop()

    # Load the text from the uploaded PDF files
    texts = load_files(uploaded_files)

    # Start the summarization in a separate thread for each document
    threads = []
    for doc_name, doc_text in texts.items():
        thread = threading.Thread(target=run_summarization, args=(q, doc_text, doc_name))
        thread.start()
        threads.append(thread)

    # Display a loading animation while waiting for the summarization to complete
    with st.spinner('Summarizing...'):
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    st.success('Summarization completed.')

    # Display the results
    for doc_name, summarized_text in results.items():
        st.subheader(f"'{doc_name}' Summary:")
        st.write(summarized_text)
