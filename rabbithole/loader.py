"""rabbithole.loader module"""
import tempfile

from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile


def load_file(file: UploadedFile) -> list[Document]:
    """
    Load a file and return a list of Document objects
    :param file: File to load.
    Supported file types: PDF
    :return: List of Document objects
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    if file.type == "application/pdf":
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.read())
        temp_file.close()
        # Load the file using PyMuPDF
        return PyMuPDFLoader(file_path=temp_file.name).load_and_split(text_splitter=text_splitter)
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
