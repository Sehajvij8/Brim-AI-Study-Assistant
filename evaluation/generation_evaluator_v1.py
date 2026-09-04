import os
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# 1. PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Brim's .env file
load_dotenv(PROJECT_ROOT / "rag_backend" / ".env")

# Input: generated answers collected earlier
INPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_results_v1.json"
)

# Output: evaluation results
OUTPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_evaluation_results_v1.json"
)


# Brim's actual generation model
GENERATOR_MODEL = "openai/gpt-oss-120b"

# Separate model used only as evaluator/judge
JUDGE_MODEL = "openai/gpt-oss-20b"


# ============================================================
# 2. API CONFIGURATION
# ============================================================

BRIM_API_KEY = os.getenv("BRIM_API_KEY")

if not BRIM_API_KEY:
    raise ValueError(
        "BRIM_API_KEY not found in rag_backend/.env"
    )


# Create Groq client
client = Groq(api_key=BRIM_API_KEY)


# ============================================================
# 3. LOAD GENERATION RESULTS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)


print("\n" + "=" * 60)
print("BRIM GENERATION EVALUATION V1")
print("=" * 60)

print(f"Generator Model : {GENERATOR_MODEL}")
print(f"Judge Model     : {JUDGE_MODEL}")
print(f"Questions       : {len(evaluation_data)}")


# ============================================================
# 4. CREATE JUDGE PROMPT
# ============================================================

def create_judge_prompt(item):
    """
    Creates the evaluation prompt for one question.
    """

    question = item["question"]
    reference_answer = item["reference_answer"]
    generated_answer = item["generated_answer"]
    expected_behavior = item["expected_behavior"]

    # Combine retrieved chunks
    retrieved_context = "\n\n".join(
        [
            f"[Rank {chunk['rank']}]\n{chunk['text']}"
            for chunk in item["retrieved_context"]
        ]
    )

    # --------------------------------------------------------
    # CASE 1: ANSWERABLE QUESTION
    # --------------------------------------------------------

    if item["answerable"]:

        prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

Your job is to evaluate the GENERATED ANSWER using the QUESTION,
REFERENCE ANSWER, and RETRIEVED CONTEXT.

Do not use outside knowledge to justify the evaluation.

============================================================
QUESTION
============================================================

{question}


============================================================
REFERENCE ANSWER
============================================================

{reference_answer}


============================================================
GENERATED ANSWER
============================================================

{generated_answer}


============================================================
RETRIEVED CONTEXT
============================================================

{retrieved_context}


============================================================
EVALUATION CRITERIA
============================================================

1. CORRECTNESS

Does the generated answer correctly answer the question and capture
the meaning of the reference answer?

Score from 1 to 5:

5 = Completely correct
4 = Mostly correct, only minor omission or wording issue
3 = Partially correct
2 = Mostly incorrect
1 = Completely incorrect


2. FAITHFULNESS

Are the claims in the generated answer supported by the retrieved context?

Score from 1 to 5:

5 = Fully supported by retrieved context
4 = Mostly supported, with only minor unsupported detail
3 = Partially supported
2 = Contains significant unsupported claims
1 = Mostly or completely unsupported


3. RELEVANCE

Does the generated answer directly address the question?

Score from 1 to 5:

5 = Direct and highly relevant
4 = Relevant with minor unnecessary information
3 = Partially relevant
2 = Mostly irrelevant
1 = Does not answer the question


============================================================
IMPORTANT
============================================================

Evaluate the actual answer.

Do not reward an answer simply because it is detailed.

A concise answer can receive 5/5 if it is correct, grounded,
and directly answers the question.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "correctness": 5,
    "faithfulness": 5,
    "relevance": 5,
    "abstention": null,
    "reason": "Brief explanation of the evaluation."
}}
"""

    # --------------------------------------------------------
    # CASE 2: UNANSWERABLE QUESTION
    # --------------------------------------------------------

    else:

        prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

This question is intentionally UNANSWERABLE from the provided documents.

Your main task is to check whether the system correctly refuses
to invent information.

============================================================
QUESTION
============================================================

{question}


============================================================
EXPECTED BEHAVIOR
============================================================

{expected_behavior}


============================================================
GENERATED ANSWER
============================================================

{generated_answer}


============================================================
RETRIEVED CONTEXT
============================================================

{retrieved_context}


============================================================
EVALUATION CRITERIA
============================================================

1. ABSTENTION

PASS = The answer clearly acknowledges that the requested information
is unavailable and does not invent or assume an answer.

FAIL = The answer invents, assumes, or presents unsupported information
as a factual answer.


2. FAITHFULNESS

Does the generated answer avoid making claims unsupported by
the retrieved context?

Score from 1 to 5:

5 = Fully grounded / no unsupported claims
4 = Mostly grounded
3 = Some minor unsupported content
2 = Significant unsupported claims
1 = Hallucinates the requested information


3. RELEVANCE

Does the response appropriately address the unanswerable question?

Score from 1 to 5:

5 = Clearly and directly explains that the information is unavailable
4 = Correct refusal with minor unnecessary content
3 = Partially appropriate
2 = Weak refusal
1 = Does not appropriately address the question


============================================================
IMPORTANT
============================================================

Correctness is NOT scored because there is no reference answer.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "correctness": null,
    "faithfulness": 5,
    "relevance": 5,
    "abstention": "PASS",
    "reason": "Brief explanation of the evaluation."
}}
"""

    return prompt


# ============================================================
# 5. CALL JUDGE MODEL
# ============================================================

def evaluate_answer(item):
    """
    Sends one question to the judge model
    and returns the evaluation as a Python dictionary.
    """

    prompt = create_judge_prompt(item)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluation judge. "
                    "Evaluate only the information provided in the prompt. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # --------------------------------------------------------
    # Remove markdown code fences if the model adds them
    # --------------------------------------------------------

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    # Convert JSON string into Python dictionary
    evaluation = json.loads(content)

    return evaluation


# ============================================================
# 6. VALIDATE JUDGE OUTPUT
# ============================================================

def validate_evaluation(evaluation, answerable):
    """
    Makes sure the judge returned sensible scores.
    """

    required_fields = [
        "correctness",
        "faithfulness",
        "relevance",
        "abstention",
        "reason",
    ]

    for field in required_fields:

        if field not in evaluation:
            raise ValueError(
                f"Missing field in judge response: {field}"
            )

    # --------------------------------------------------------
    # Answerable questions
    # --------------------------------------------------------

    if answerable:

        for metric in [
            "correctness",
            "faithfulness",
            "relevance",
        ]:

            score = evaluation[metric]

            if not isinstance(score, (int, float)):
                raise ValueError(
                    f"{metric} must be a number."
                )

            if score < 1 or score > 5:
                raise ValueError(
                    f"{metric} must be between 1 and 5."
                )

        if evaluation["abstention"] is not None:
            raise ValueError(
                "Abstention must be null for answerable questions."
            )

    # --------------------------------------------------------
    # Unanswerable questions
    # --------------------------------------------------------

    else:

        if evaluation["correctness"] is not None:
            raise ValueError(
                "Correctness must be null for unanswerable questions."
            )

        for metric in [
            "faithfulness",
            "relevance",
        ]:

            score = evaluation[metric]

            if not isinstance(score, (int, float)):
                raise ValueError(
                    f"{metric} must be a number."
                )

            if score < 1 or score > 5:
                raise ValueError(
                    f"{metric} must be between 1 and 5."
                )

        if evaluation["abstention"] not in [
            "PASS",
            "FAIL",
        ]:
            raise ValueError(
                "Abstention must be PASS or FAIL."
            )


# ============================================================
# 7. RUN GENERATION EVALUATION
# ============================================================

results = []


for index, item in enumerate(
    evaluation_data,
    start=1
):

    print(
        f"\n[{index}/{len(evaluation_data)}] "
        f"Evaluating {item['id']}..."
    )

    try:

        # Ask judge model to evaluate the answer
        evaluation = evaluate_answer(item)

        # Validate judge response
        validate_evaluation(
            evaluation,
            item["answerable"]
        )

        # Store final result
        result = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],

            "correctness": evaluation["correctness"],
            "faithfulness": evaluation["faithfulness"],
            "relevance": evaluation["relevance"],

            "abstention": evaluation["abstention"],

            "reason": evaluation["reason"],
        }

        results.append(result)

        print(
            f"   Correctness  : {evaluation['correctness']}"
        )

        print(
            f"   Faithfulness : {evaluation['faithfulness']}"
        )

        print(
            f"   Relevance    : {evaluation['relevance']}"
        )

        if not item["answerable"]:
            print(
                f"   Abstention   : {evaluation['abstention']}"
            )

    except Exception as e:

        print(f"   ERROR: {e}")

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answerable": item["answerable"],

                "correctness": None,
                "faithfulness": None,
                "relevance": None,

                "abstention": "ERROR",

                "reason": str(e),
            }
        )


# ============================================================
# 8. SEPARATE ANSWERABLE / UNANSWERABLE RESULTS
# ============================================================

answerable_results = [
    result
    for result in results
    if result["answerable"] is True
]

unanswerable_results = [
    result
    for result in results
    if result["answerable"] is False
]


# ============================================================
# 9. CALCULATE AVERAGE SCORE
# ============================================================

def average_score(results, metric):
    """
    Calculates average score for a metric.
    """

    scores = [
        result[metric]
        for result in results
        if isinstance(
            result[metric],
            (int, float)
        )
    ]

    if not scores:
        return None

    return round(
        mean(scores),
        2
    )


correctness_average = average_score(
    answerable_results,
    "correctness"
)

faithfulness_average = average_score(
    results,
    "faithfulness"
)

relevance_average = average_score(
    results,
    "relevance"
)


# ============================================================
# 10. CONVERT SCORE TO PERCENTAGE
# ============================================================

def score_percentage(score):

    if score is None:
        return None

    return round(
        (score / 5) * 100,
        2
    )


correctness_percentage = score_percentage(
    correctness_average
)

faithfulness_percentage = score_percentage(
    faithfulness_average
)

relevance_percentage = score_percentage(
    relevance_average
)


# ============================================================
# 11. CALCULATE ABSTENTION PERFORMANCE
# ============================================================

abstention_pass = sum(
    1
    for result in unanswerable_results
    if result["abstention"] == "PASS"
)

abstention_total = len(
    unanswerable_results
)


if abstention_total > 0:

    abstention_accuracy = round(
        (
            abstention_pass
            / abstention_total
        ) * 100,
        2
    )

else:

    abstention_accuracy = None


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

summary = {

    "total_questions": len(results),

    "answerable_questions": len(
        answerable_results
    ),

    "unanswerable_questions": len(
        unanswerable_results
    ),

    "correctness_average": correctness_average,

    "correctness_percentage": correctness_percentage,

    "faithfulness_average": faithfulness_average,

    "faithfulness_percentage": faithfulness_percentage,

    "relevance_average": relevance_average,

    "relevance_percentage": relevance_percentage,

    "abstention_pass": abstention_pass,

    "abstention_total": abstention_total,

    "abstention_accuracy_percentage": (
        abstention_accuracy
    ),
}


# ============================================================
# 13. SAVE RESULTS
# ============================================================

output = {

    "evaluation": (
        "Brim Generation Evaluation V1"
    ),

    "generator_model": GENERATOR_MODEL,

    "judge_model": JUDGE_MODEL,

    "summary": summary,

    "results": results,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 14. PRINT FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("GENERATION EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Total Questions        : "
    f"{summary['total_questions']}"
)

print(
    f"Answerable Questions   : "
    f"{summary['answerable_questions']}"
)

print(
    f"Unanswerable Questions : "
    f"{summary['unanswerable_questions']}"
)


print("\nGeneration Quality")
print("-" * 60)

print(
    f"Correctness   : "
    f"{correctness_average}/5 "
    f"({correctness_percentage}%)"
)

print(
    f"Faithfulness  : "
    f"{faithfulness_average}/5 "
    f"({faithfulness_percentage}%)"
)

print(
    f"Relevance     : "
    f"{relevance_average}/5 "
    f"({relevance_percentage}%)"
)


print("\nAbstention")
print("-" * 60)

print(
    f"Correct Abstentions : "
    f"{abstention_pass}/{abstention_total}"
)

print(
    f"Abstention Accuracy : "
    f"{abstention_accuracy}%"
)


print("\n" + "=" * 60)
print("RESULT FILE")
print("=" * 60)

print(OUTPUT_FILE)