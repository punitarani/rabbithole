"""rabbithole.mp3 module"""

import tempfile

from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm

SUPPORTED_AV_FILE_TYPES = (
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

    # If mp3 file, return the filepath
    if filepath.endswith('.mp3'):
        return filepath

    # Check if the file type is supported
    if filepath.rsplit('.', 1)[-1] not in SUPPORTED_AV_FILE_TYPES:
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
    Chunk a mp3 file into smaller mp3 files
    :param filepath: File to chunk
    :param chunk_length: Length of each chunk in seconds
    :return: List of chunked filepaths

    Saves the chunked files in a temporary directory
    """

    audio = AudioSegment.from_mp3(filepath)

    # Get the total length of the audio file
    length = len(audio)

    # PyDub handles time in milliseconds
    chunk_length *= 60 * 1000 * 1.01  # Add 1% to the chunk length to account for overlap
    chunk_length = int(chunk_length)

    if length < chunk_length:
        return [filepath]

    # Get the number of chunks and start and end times for each chunk
    chunks = []
    for i in range(0, length, chunk_length):
        start = i
        end = min(length, i + chunk_length)
        audio_chunk = audio[start:end]

        # Ignore chunks that are too small
        if len(audio_chunk) < 1000:
            continue

        chunks.append(audio_chunk)

    # Write each chunk to a file
    chunked_filepaths = []
    for i, chunk in enumerate(tqdm(chunks, desc="Saving audio chunks", unit="chunk")):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
            chunk.export(temp.name, format="mp3")
            chunked_filepaths.append(temp.name)

    return chunked_filepaths
