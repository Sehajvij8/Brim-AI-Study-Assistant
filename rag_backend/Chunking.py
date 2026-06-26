from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import pdf_to_text
from pathlib import Path


def chunker(pdf_path):
    """
    Reads a PDF and converts it into text chunks.
    """

    try:
        # Extract text from PDF
        text = pdf_to_text(pdf_path)
        pdf_name = Path(pdf_path).name
        if not text.strip():
            raise ValueError("The PDF does not contain any readable text.")

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Create chunks
        chunks = text_splitter.create_documents(
            [text],
            metadatas = [
                {
                    "source": pdf_name
                }
            ]
            )

        return chunks

    except FileNotFoundError:
        print("❌ Error: PDF file not found.")

    except ValueError as e:
        print(f"❌ {e}")

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

    return None


if __name__ == "__main__":

    pdf_path = input("Enter the path of the PDF: ").strip()

    chunks = chunker(pdf_path)

    if chunks:
        print(f"\n✅ Total Chunks Created: {len(chunks)}\n")

        for i, chunk in enumerate(chunks, start=1):
            print(f"---------- Chunk {i} ----------")
            print(chunk.page_content[:300])   # First 300 characters
            print()