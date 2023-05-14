"""rabbithole.transcribe module"""

import openai
from tqdm import tqdm

from rabbithole.mp3 import chunk_mp3


def transcribe(filepath: str) -> str:
    """
    Transcribe a mp3 file using OpenAI's Whisper API
    :param filepath: Path to mp3 file
    :return: Transcription
    """

    # Split the file into chunks
    files = chunk_mp3(filepath)

    transcripts: list[str] = []

    for file in tqdm(files, desc="Transcribing audio", unit="chunk"):
        with open(file, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcripts.append(transcript.get("text", ""))

    return " ".join(transcripts)
