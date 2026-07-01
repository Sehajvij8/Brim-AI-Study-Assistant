# LLM Integration
from .config import (
    LLM_MODEL,
    groq_client,
    qdrant_client
    
)



def generate_answers(question , retrieved_chunks):
    """
    Generates an answer using the retrieved chunks and Groq LLM.
    """
    try:
        # Combine all the retrieved chunks into one context
        context = ""
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context += f"""
    ==================================================
    Source Document : {chunk["source"]}
    
    Chunk Number : {i}

    Content :
    {chunk["text"]}

    ==================================================

    """



        # System Prompt
        system_prompt = """
        You are Brim, an intelligent multilingual AI Study Assistant.

        Rules:

        1. Answer ONLY from the provided context.

        2. Never make up information.

        3. If the answer is not present in the context, reply:

        "I couldn't find this information in the uploaded documents."

        4. Reply in the same language as the user.

        5. If the user writes in Hinglish,
        reply naturally in Hinglish.

        6. Explain concepts in a simple and student-friendly way.

        7. Use bullet points whenever helpful.

        """

        # User prompt
        user_prompt = f"""
        You are provided with the retrieved context from one or more uploaded documents.

        Instructions:

        - Use ONLY the retrieved context.
        - Never use your own knowledge.
        - If the answer is not available, clearly say:
        "I couldn't find this information in the uploaded documents."

        -------------------- Retrieved Context --------------------

        {context}

        -----------------------------------------------------------

        User Question:

        {question}
        """

        response = groq_client.chat.completions.create(
            model = LLM_MODEL,
            temperature = 0.2,
            max_tokens = 1024,
            messages = [
                {
                    "role" : "system",
                    "content" : system_prompt
                },
                {
                    "role" : "user",
                    "content" : user_prompt
                },
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"LLM ERROR : {e}")
        return None
    

    

# -----Testing------ 
if __name__ == "__main__":
    retrieved_chunks = [
        {
            "text": """
Artificial Intelligence (AI) enables machines
to perform tasks that normally require human intelligence.

Machine Learning is a subset of AI.

Deep Learning uses neural networks.
""",
        "source" : "test2.pdf"
        }
    ]
    question = input("Ask your Question:")
    answer = generate_answers(question , retrieved_chunks)

    if answer:
        print("\nBrim Answer🤖:\n")
        print(answer)
    else:
        print("Failed to generate answer.😢")




