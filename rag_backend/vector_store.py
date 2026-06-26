from config import (qdrant_client , COLLECTION_NAME)
from Chunking import chunker
from Embeddings import generate_embeddings
# UUID = Universally Unique Identifier, used to generate unique IDs for each point in the Qdrant collection.
import uuid
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

def store_embeddings(embeddings , chunks):

    try:
        client = qdrant_client
        try:

            client.get_collection(COLLECTION_NAME)
        except Exception:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
        )
        points = []
        for chunk , embedding in zip(chunks , embeddings):
            point = PointStruct(
                id=uuid.uuid4().hex,
                vector=embedding.tolist(),
                payload={"text" : chunk.page_content,
                         "source" : chunk.metadata.get("source" , "unknown"),}
            )
            points.append(point)
        client.upsert(
            collection_name=COLLECTION_NAME, 
            points=points
            )
        print("Embeddings stored successfully in Qdrant.")
        return True
    except Exception as e:
        print(f"Error storing embeddings in Qdrant: {e}")
        return False


if __name__ == "__main__":
     pdf_path = input("Enter the path of the PDF: ").strip()
     chunks = chunker(pdf_path)
     embeddings = generate_embeddings(chunks)
     success = store_embeddings(embeddings , chunks)

     if success:
         print("Embeddings stored successfully in Qdrant.")
     else:
         print("Failed to store embeddings in Qdrant.")  
print(qdrant_client.get_collections())   
