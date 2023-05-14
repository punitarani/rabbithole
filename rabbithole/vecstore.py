"""rabbithole.vecstore module"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

DATA_DIR = Path(__file__).parent.parent.joinpath("data")

client = chromadb.Client(settings=Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(DATA_DIR)))
