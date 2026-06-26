from Chunking import chunker
from config import embedding_model

def generate_embeddings(chunks):
    """
    Generates embeddings for the given text chunks using a pre-trained model.
    """

    try:
        # Generate embeddings for each chunk
        embeddings = [embedding_model.encode(chunk.page_content) for chunk in chunks]

        return embeddings

    except Exception as e:
        print(f" uh oh 😕 error in generate_embeddings: {e}")
        return None
    
# Testing the Embeddings generation
if __name__ == "__main__":
        
    pdf_path = input("Enter the path of the PDF: ").strip()
    chunks = chunker(pdf_path)

    if chunks is None:
        print("😕Failed to create chunks. Exiting.")
    else:
        embeddings = generate_embeddings(chunks)

        if embeddings is None:
            print("😕Failed to generate embeddings. Exiting.")
        else:
            print("\n✅ Embeddings generated successfully for all chunks.\n")
            print(f"Total Chunks: {len(chunks)}")
           
            
            print(f"Total Embeddings: {len(embeddings)}")
            print(f"Shape of first embedding: {embeddings[0].shape if embeddings else 'N/A'}")

            #First embedding
            print("\n 😊 First 10 values of the first embedding:")
            print(embeddings[0][:10] if embeddings else 'N/A')

            

            




    



        