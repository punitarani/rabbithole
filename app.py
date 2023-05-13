"""Streamlit App"""

import streamlit as st

from rabbithole import summarize_text

st.title("RabbitHole")

input_text = st.text_area("Enter your text (up to 1000 words)", height=200)

if st.button("Summarize"):
    # TODO: Add loading animation while running
    st.write("Summarized Text:")
    st.write(summarize_text(input_text))
