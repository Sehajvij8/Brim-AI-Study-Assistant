from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("brim_api_key")

groq_client = Groq(
    api_key = groq_api_key
)

LLM_MODEL = "llama-3.3-70b-versatile"


# Embedding_model
embedding_model =SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Qdrant client
qdrant_client = QdrantClient(path = "./qdrant/db")

# Collection_name
COLLECTION_NAME = "study_assistant"

# Retrieval Settings
TOP_K = 3
