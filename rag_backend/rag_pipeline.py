from .Retriever import retrieve_chunks
from .llm import generate_answers
from .config import qdrant_client

def rag_pipeline(question , chat_history):
    """
    Executes the complete RAG pipeline.
    """

    try:

        print("\n🔎 Retrieving relevant chunks...\n")

        retrieved_chunks = retrieve_chunks(question)

        if not retrieved_chunks:
            print("❌ No relevant chunks found.")
            return None

        print(f"✅ {len(retrieved_chunks)} relevant chunk(s) retrieved.\n")

        print("🤖 Generating answer...\n")

        answer = generate_answers(
            question,
            retrieved_chunks,
            chat_history
        )

        return {
            "answer" : answer,
            "chunks" : retrieved_chunks
        }

    except Exception as e:

        print(f"Pipeline Error: {e}")

        return None


# For testing the rag_pipeline
if __name__ == "__main__":

    while True:

        question = input("\nAsk your Question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("\n👋 Thank you for using Brim.")
            break

        answer = rag_pipeline(question)

        if answer:

            print("\n==============================")
            print("🤖 Brim Answer\n")
            print(answer)
            print("==============================")

        else:

            print("\n❌ Failed to generate answer.")


