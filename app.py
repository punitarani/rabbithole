"""Streamlit App"""

import queue
import threading

import streamlit as st

from rabbithole import summarize_text


def run_summarization(q, input_text):
    """Execute the text summarization"""
    summarized_text = summarize_text(input_text)
    q.put(summarized_text)


st.title("RabbitHole")

input_text = st.text_area("Enter your text (up to 1000 words)", height=200)

if st.button("Summarize"):
    # Create a queue to hold the result
    q = queue.Queue()

    # Start the summarization in a separate thread
    thread = threading.Thread(target=run_summarization, args=(q, input_text))
    thread.start()

    with st.spinner('Summarizing...'):
        thread.join()
        # Retrieve the result from the queue
        summarized_text = q.get()
        st.success('Summarization completed.')
        st.write(summarized_text)
