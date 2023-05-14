"""rabbithole.mp3 module"""

from moviepy.editor import AudioFileClip

SUPPORTED_FILE_TYPES = [
    # Video formats
    "mp4", "mkv", "webm", "flv", "avi", "mov", "wmv",
    # Audio formats
    "mp3", "wav", "m4a", "flac", "ogg",
]


def convert_to_mp3(filepath: str) -> None:
    """
    Convert a video or audio file to mp3
    :param filepath: File to convert
    :return: None
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
