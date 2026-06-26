from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Embedding_model
embedding_model =SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Qdrant client
qdrant_client = QdrantClient(path = "./qdrant/db")

# Collection_name
COLLECTION_NAME = "study_assistant"

# Retrieval Settings
TOP_K = 3
