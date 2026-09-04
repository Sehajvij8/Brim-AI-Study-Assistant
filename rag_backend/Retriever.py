from .Chunking import chunker
from .Embeddings import generate_embeddings
from .vector_store import store_embeddings
from .config import (
    TOP_K,
    embedding_model,
    qdrant_client,
    COLLECTION_NAME
)

def retrieve_chunks(query):
    try:
        query_embedding = embedding_model.encode(query)
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query = query_embedding.tolist(),
            limit=TOP_K
        )

        retrieved_chunks = []
        for point in results.points:
            retrieved_chunks.append(
                {
                    "text" : point.payload["text"],
                    "source" : point.payload.get("source" , "unknown"),
                    "score" : point.score,
                }
            )

        return retrieved_chunks
    except Exception as e:
        print(f"Retrieval_error {e}")
        return None
    
    
    

# Testing Code
if __name__ == "__main__":
    query =  input("Ask your Question:")
    results = retrieve_chunks(query)

    if results:
        print("\nRetrieved Chunks:\n")

        for i , result in enumerate(results , start = 1):
            print(f"\n--------- Chunk {i} ---------")
            print(f"Source : {result['source']}")
            print(f"Score  : {result['score']:.4f}")
            print(result["text"])
    else:
        print("No relevant Chunks Found 😕")
