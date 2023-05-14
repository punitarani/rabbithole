"""rabbithole.loader module"""
import tempfile

from langchain.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader, UnstructuredImageLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rabbithole.mp3 import SUPPORTED_AV_FILE_TYPES, convert_to_mp3
from rabbithole.transcribe import transcribe

SUPPORTED_IMG_FILE_TYPES = (".jpg", ".jpeg", ".png")


def save_to_temp_file(file: UploadedFile) -> str:
    """
    Save an uploaded file to a temporary file and return the path.
    :param file: File to save.
    :return: Path to the temporary file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{file.name.split('.')[-1]}", delete=False)
    temp_file.write(file.read())
    temp_file.close()
    return temp_file.name


def load_file(file: UploadedFile) -> list[Document]:
    """
    Load a file and return a list of Document objects
    :param file: File to load.
    Supported file types: PDF
    :return: List of Document objects
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Handle .docx files
    if file.name.endswith(".docx"):
        temp_file = save_to_temp_file(file)
        return Docx2txtLoader(file_path=temp_file).load_and_split(text_splitter=text_splitter)

    # Handle .pdf files
    elif file.name.endswith(".pdf"):
        temp_file = save_to_temp_file(file)
        return PyMuPDFLoader(file_path=temp_file).load_and_split(text_splitter=text_splitter)

    # Handle .txt files
    elif file.name.endswith(".txt"):
        temp_file = save_to_temp_file(file)
        return TextLoader(file_path=temp_file).load_and_split(text_splitter=text_splitter)

    # Handle image files
    elif file.name.endswith(SUPPORTED_IMG_FILE_TYPES):
        temp_file = save_to_temp_file(file)
        return UnstructuredImageLoader(file_path=temp_file).load_and_split(text_splitter=text_splitter)

    # Handle Audio and Video files
    elif file.name.endswith(SUPPORTED_AV_FILE_TYPES):
        temp_file = save_to_temp_file(file)

        # Convert to mp3 and transcribe
        mp3_file = convert_to_mp3(temp_file)
        transcription = transcribe(mp3_file)

        # Save transcription to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        temp_file.write(transcription.encode())
        temp_file.close()

        # Load the file using TextLoader
        return TextLoader(file_path=temp_file.name).load_and_split(text_splitter=text_splitter)

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
