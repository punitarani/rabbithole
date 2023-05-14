"""rabbithole.vecstore module"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

CHROMADB_DIR = Path(__file__).resolve().parent.parent.joinpath("data", "chromadb")

client = chromadb.Client(settings=Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMADB_DIR)))
