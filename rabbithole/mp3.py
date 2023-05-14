"""rabbithole.mp3 module"""

import tempfile

from moviepy.editor import AudioFileClip
from pydub import AudioSegment

SUPPORTED_FILE_TYPES = (
    # Video formats
    "mp4", "mkv", "webm", "flv", "avi", "mov", "wmv",
    # Audio formats
    "mp3", "wav", "m4a", "flac", "ogg",
)


def convert_to_mp3(filepath: str) -> str:
    """
    Convert a video or audio file to mp3
    :param filepath: File to convert
    :return: Path to converted file
    """

    # Check if the file type is supported
    if filepath.rsplit('.', 1)[-1] not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {filepath.rsplit('.', 1)[-1]}")

    # Load video or audio file
    clip = AudioFileClip(filepath)

    # Make sure the file name ends with .mp3
    if not filepath.endswith('.mp3'):
        filepath = filepath.rsplit('.', 1)[0] + '.mp3'

    # Write audio to file
    clip.write_audiofile(filepath)
    print(f"File converted successfully to {filepath}")

    return filepath


def chunk_mp3(filepath: str, chunk_length: int = 10) -> list[str]:
    """
    Chunk an mp3 file into smaller mp3 files
    :param filepath: File to chunk
    :param chunk_length: Length of each chunk in seconds
    :return: List of chunked filepaths

    Saves the chunked files in a temporary directory
    """

    audio = AudioSegment.from_mp3(filepath)

    # Get the total length of the audio file
    length = len(audio)

    # PyDub handles time in milliseconds
    chunk_length *= 60 * 1000

    # Get the number of chunks and start and end times for each chunk
    chunks = []
    for i in range(0, length, chunk_length):
        start = i
        end = min(length, i + chunk_length)
        chunks.append(audio[start:end])

    # Write each chunk to a file
    chunked_filepaths = []
    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
            chunk.export(temp.name, format="mp3")
            chunked_filepaths.append(temp.name)

    return chunked_filepaths
