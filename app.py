"""Streamlit App"""

import openai
import streamlit as st
from langchain.schema import Document
from streamlit_chat import message

from rabbithole import summarize_document
from rabbithole.embedding import embed_document
from rabbithole.keywords import get_document_keywords
from rabbithole.loader import load_file, SUPPORTED_IMG_FILE_TYPES
from rabbithole.mp3 import SUPPORTED_AV_FILE_TYPES
from rabbithole.planner import generate_plan

# Session variables
for state_var in ["uploaded_files", "documents", "embeddings", "keywords", "summaries"]:
    if state_var not in st.session_state:
        st.session_state[state_var] = {}
if "plan" not in st.session_state:
    st.session_state.plan = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "bot_messages" not in st.session_state:
    st.session_state.bot_messages = [
        "Hello, I am here to help you learn more efficiently"
    ]
if "user_messages" not in st.session_state:
    st.session_state.user_messages = []


def generate_response(prompt):
    """Generate a response to a prompt using the GPT-3.5 Turbo model"""
    st.session_state['user_messages'].append(prompt)

    messages = []
    # Alternate between bot and the assistant until the conversation is over
    msg_idx = 0
    while True:
        if len(st.session_state['bot_messages']) > msg_idx:
            messages.append({"role": "assistant", "content": st.session_state['bot_messages'][msg_idx]})
        else:
            break
        if len(st.session_state['user_message']) > msg_idx:
            messages.append({"role": "user", "content": st.session_state['user_messages'][msg_idx]})
        else:
            break
        msg_idx += 1
    print(messages)
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    response = completion.choices[0].message.content
    st.session_state['bot_messages'].append(response)
    print(response)


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
            print(file.name, [len(doc.page_content) for doc in documents[file.name]])
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


def generate_plan_with_spinner() -> dict:
    """Generate a logical plan to study the uploaded documents."""
    with st.spinner("Generating plan..."):
        plan = generate_plan(st.session_state.summaries, st.session_state.keywords)
    return plan


st.set_page_config(page_title="RabbitHole", page_icon="🐇", layout="wide")

st.title("RabbitHole")

if not st.session_state.processed:
    uploaded_files = st.file_uploader("Upload content",
                                      type=["docx", "pdf", "txt", *SUPPORTED_IMG_FILE_TYPES, *SUPPORTED_AV_FILE_TYPES],
                                      accept_multiple_files=True)

    if st.button("Dive in"):
        if not uploaded_files:
            st.warning("Please upload a file first.")
            st.stop()

        # Check if uploaded files have changed
        uploaded_files_changed = False
        if len(uploaded_files) != len(st.session_state.uploaded_files):
            uploaded_files_changed = True
        else:
            for new_file, old_file in zip(uploaded_files, st.session_state.uploaded_files):
                if new_file != old_file:
                    uploaded_files_changed = True
                    break

        if uploaded_files_changed:
            st.session_state.uploaded_files = uploaded_files

            # Load the text from the uploaded PDF files
            st.session_state.documents = load_files_with_spinner(st.session_state.uploaded_files)
            st.session_state.embeddings = embed_documents_with_spinner(st.session_state.documents)
            st.session_state.keywords = extract_keywords_with_spinner(st.session_state.embeddings)
            st.session_state.summaries = generate_summary_with_spinner(st.session_state.documents)

        # Display the keywords and summaries
        for doc_name, doc_keywords in st.session_state.keywords.items():
            st.header(doc_name)
            st.caption("Keywords: " + ", ".join(doc_keywords))
            st.write(st.session_state.summaries[doc_name])
            st.divider()

        st.session_state.processed = True
        st.success('Summarization completed.')

if st.session_state.processed:
    st.header("Loaded Files")
    for file in st.session_state.uploaded_files:
        st.write(file.name)

    # Display the plan
    st.header("Study Plan")
    if st.session_state.plan is None:
        plan = generate_plan_with_spinner()
        st.session_state.plan = plan
    else:
        plan = st.session_state.plan
    for data in plan.get("plan", []):
        for doc_name, doc_data in data.items():
            st.subheader(doc_name)
            st.write(f"**Background Concepts**")
            for concept in doc_data.get("Background Concepts", []):
                st.write(f"- {concept}")
            st.write(f"**Key Concepts**")
            for concept in doc_data.get("Key Concepts", []):
                st.write(f"- {concept}")
            st.write(f"**Further Reading**")
            for concept in doc_data.get("Further Reading", []):
                st.write(f"- {concept}")
        st.write("")

    st.header("Chat")
    # Iterate through the bot and user message and print them alternatively
    message_i = 0
    while True:
        if len(st.session_state.bot_messages) > message_i:
            message(st.session_state.bot_messages[message_i])
        else:
            break
        if len(st.session_state.user_messages) > message_i:
            message(st.session_state.user_messages[message_i], is_user=True)
        else:
            break
        message_i += 1

    user_input = st.text_input("What do you want to learn more about?", key="user_message")
    if st.button("Send"):
        with st.spinner("Generating response..."):
            generate_response(user_input)
        st.experimental_rerun()
