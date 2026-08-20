import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "guidelines_docs"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
CHROMA_PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "diabetes_guidelines"
