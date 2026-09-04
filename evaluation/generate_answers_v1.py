import json
from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT BRIM RAG COMPONENTS
# ============================================================

from rag_backend.Retriever import retrieve_chunks
from rag_backend.llm import generate_answers

# ============================================================
# FILE PATHS
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "brim_generation_evaluation_dataset_v1.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_results_v1.json"
)

# ============================================================
# CONFIGURATION
# ============================================================

K = 3

# ============================================================
# LOAD GENERATION DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


print("\n")
print("=" * 60)
print("BRIM GENERATION EVALUATION - ANSWER GENERATION")
print("=" * 60)

print(
    f"Total Questions: {len(dataset)}"
)

print(
    f"Top-K Context: {K}"
)

# ============================================================
# GENERATE ANSWERS
# ============================================================

results = []


for index, item in enumerate(
    dataset,
    start=1
):

    question_id = item["id"]

    question = item["question"]

    answerable = item["answerable"]


    print("\n")
    print("-" * 60)

    print(
        f"[{index}/{len(dataset)}] {question_id}"
    )

    print(
        "Question:",
        question
    )

# ========================================================
    # RETRIEVE CONTEXT
    # ========================================================

    print(
        "Retrieving context..."
    )


    retrieved_chunks = retrieve_chunks(
        question
    )


    if retrieved_chunks is None:

        retrieved_chunks = []


    # Keep Top-K
    retrieved_chunks = (
        retrieved_chunks[:K]
    )


    print(
        "Retrieved chunks:",
        len(retrieved_chunks)
    )

    # ========================================================
    # PREPARE CHAT HISTORY
    # ========================================================

    # IMPORTANT:
    # Every evaluation question starts with
    # an empty conversation.
    #
    # This prevents previous questions from
    # affecting the next answer.

    chat_history = []

    # ========================================================
    # GENERATE ACTUAL BRIM ANSWER
    # ========================================================

    print(
        "Generating Brim answer..."
    )


    try:

        generated_answer = generate_answers(

            question,

            retrieved_chunks,

            chat_history

        )


    except Exception as e:

        print(
            "❌ Generation failed:"
        )

        print(
            str(e)
        )

        generated_answer = None

    # ========================================================
    # PRINT GENERATED ANSWER
    # ========================================================

    print(
        "\nGenerated Answer:"
    )

    if generated_answer:

        print(
            generated_answer
        )

    else:

        print(
            "❌ No answer generated."
        )
   
    # ========================================================
    # SAVE RETRIEVED CONTEXT
    # ========================================================

    context = []


    for rank, chunk in enumerate(

        retrieved_chunks,

        start=1

    ):

        context.append({

            "rank":
                rank,

            "score":
                chunk.get(
                    "score"
                ),

            "source":
                chunk.get(
                    "source",
                    "Unknown"
                ),

            "text":
                chunk.get(
                    "text",
                    ""
                )

        })


    # ========================================================
    # STORE RESULT
    # ========================================================

    result = {

        "id":
            question_id,

        "category":
            item["category"],

        "question":
            question,

        "answerable":
            answerable,

        "reference_answer":
            item.get(
                "reference_answer"
            ),

        "expected_behavior":
            item.get(
                "expected_behavior"
            ),

        "retrieved_context":
            context,

        "generated_answer":
            generated_answer

    }


    results.append(
        result
    )

# ============================================================
# SAVE RESULTS
# ============================================================

with open(

    RESULTS_PATH,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        results,

        f,

        indent=2,

        ensure_ascii=False

    )

# ============================================================
# FINAL SUMMARY
# ============================================================

successful = sum(

    1

    for result in results

    if result["generated_answer"]

)


failed = (

    len(results)

    -
    successful

)


print("\n")
print("=" * 60)

print(
    "GENERATION DATA COLLECTION COMPLETE"
)

print("=" * 60)


print(
    f"Total Questions : {len(results)}"
)

print(
    f"Successful      : {successful}"
)

print(
    f"Failed          : {failed}"
)

print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)


print(
    "\nNext step:"
)

print(
    "Review generation_results_v1.json"
)

print(
    "before calculating generation metrics."
)


print(
    "\nEvaluation data generation completed successfully."
)