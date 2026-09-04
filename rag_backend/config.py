from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import os


load_dotenv()
groq_api_key = os.getenv("brim_api_key")

groq_client = Groq(
    api_key = groq_api_key
)

LLM_MODEL = "openai/gpt-oss-120b"


# Embedding_model
embedding_model =SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Qdrant client
PROJECT_ROOT= Path(__file__).resolve().parent.parent
QDRANT_PATH = PROJECT_ROOT / "qdrant" / "db"
qdrant_client = QdrantClient(path = str(QDRANT_PATH))


# Collection_name
COLLECTION_NAME = "study_assistant"

# Retrieval Settings
TOP_K = 3
