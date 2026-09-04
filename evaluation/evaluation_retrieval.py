import json
import csv
import re
from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
from rag_backend.config import (
    qdrant_client,
    COLLECTION_NAME
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "brim_evaluation_dataset_v2_gold_evidence.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_results_v5.csv"
)

# We evaluate Top-1, Top-3 and Top-5
K_VALUES = [1, 3, 5]

# Questions we want detailed debugging for
DEBUG_QUESTIONS = [
    "Q13",
    "Q19"
]

# Semantic similarity threshold
#
# 0.50 means:
# if a gold evidence phrase and a retrieved sentence
# have cosine similarity >= 0.50,
# we consider the evidence semantically present.
#
# Exact string matches are accepted regardless of threshold.
SEMANTIC_THRESHOLD = 0.50


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLOADING EMBEDDING MODEL...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded successfully."
)


# ============================================================
# QDRANT DEBUG
# ============================================================

print("\nQDRANT DEBUG")

print(
    "Collection:",
    COLLECTION_NAME
)

print(
    "Collections:",
    qdrant_client.get_collections().collections
)


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


print("\nDATASET DEBUG")

print(
    "Dataset type:",
    type(dataset)
)

print(
    "Number of questions:",
    len(dataset)
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    """
    Normalize whitespace so that PDF line breaks
    do not interfere with matching.
    """

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text):

    """
    Convert a retrieved chunk into smaller sentences.

    This is important because comparing a short gold
    evidence phrase with an entire paragraph can produce
    artificially low similarity.
    """

    text = normalize_text(text)

    if not text:

        return []


    # Split on sentence-ending punctuation
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )


    # Remove empty sentences
    sentences = [

        sentence.strip()

        for sentence in sentences

        if sentence.strip()

    ]


    return sentences


# ============================================================
# EXACT MATCH CHECK
# ============================================================

def exact_match(
    evidence,
    sentence
):

    """
    Check whether the gold evidence phrase
    appears directly inside the retrieved sentence.
    """

    return (
        evidence.lower()
        in
        sentence.lower()
    )


# ============================================================
# SEMANTIC MATCHING FOR ONE CHUNK
# ============================================================

def match_evidence_to_chunk(
    chunk_text,
    gold_evidence
):

    """
    Compare every gold evidence phrase with every
    sentence inside a retrieved chunk.

    Returns the best semantic match for each
    evidence phrase.
    """

    sentences = split_into_sentences(
        chunk_text
    )


    if not sentences:

        return []


    # --------------------------------------------------------
    # Encode all sentences in this chunk once
    # --------------------------------------------------------

    sentence_embeddings = (
        embedding_model.encode(
            sentences,
            normalize_embeddings=True
        )
    )


    matches = []


    # --------------------------------------------------------
    # Compare every gold evidence phrase
    # --------------------------------------------------------

    for evidence in gold_evidence:

        # ----------------------------------------------------
        # Exact matching first
        # ----------------------------------------------------

        exact_sentence = None

        for sentence in sentences:

            if exact_match(
                evidence,
                sentence
            ):

                exact_sentence = sentence

                break


        if exact_sentence is not None:

            matches.append({

                "evidence":
                    evidence,

                "score":
                    1.0,

                "sentence":
                    exact_sentence,

                "match_type":
                    "exact"

            })

            continue


        # ----------------------------------------------------
        # Semantic matching
        # ----------------------------------------------------

        evidence_embedding = (
            embedding_model.encode(
                [evidence],
                normalize_embeddings=True
            )
        )


        similarities = cosine_similarity(
            evidence_embedding,
            sentence_embeddings
        )[0]


        best_index = similarities.argmax()

        best_score = float(
            similarities[best_index]
        )

        best_sentence = sentences[
            best_index
        ]


        # ----------------------------------------------------
        # Accept semantic match if threshold reached
        # ----------------------------------------------------

        if (
            best_score
            >= SEMANTIC_THRESHOLD
        ):

            matches.append({

                "evidence":
                    evidence,

                "score":
                    best_score,

                "sentence":
                    best_sentence,

                "match_type":
                    "semantic"

            })


    return matches


# ============================================================
# FIND RELEVANT EVIDENCE IN TOP-K
# ============================================================

def evaluate_retrieved_chunks(
    retrieved_chunks,
    gold_evidence
):

    """
    Evaluate retrieved chunks one by one.

    Returns:

    matched evidence
    first relevant rank
    detailed match information
    """

    matched_evidence = []

    first_relevant_rank = None

    detailed_matches = []


    # --------------------------------------------------------
    # Go through retrieved chunks in ranking order
    # --------------------------------------------------------

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        chunk_text = chunk.get(
            "text",
            ""
        )


        chunk_matches = (
            match_evidence_to_chunk(
                chunk_text,
                gold_evidence
            )
        )


        # ----------------------------------------------------
        # Store detailed matches
        # ----------------------------------------------------

        for match in chunk_matches:

            detailed_matches.append({

                "rank":
                    rank,

                "evidence":
                    match["evidence"],

                "score":
                    match["score"],

                "sentence":
                    match["sentence"],

                "match_type":
                    match["match_type"]

            })


            if (
                match["evidence"]
                not in
                matched_evidence
            ):

                matched_evidence.append(
                    match["evidence"]
                )


        # ----------------------------------------------------
        # First relevant chunk
        # ----------------------------------------------------

        if (
            chunk_matches
            and
            first_relevant_rank is None
        ):

            first_relevant_rank = rank


    return (
        matched_evidence,
        first_relevant_rank,
        detailed_matches
    )


# ============================================================
# EVALUATION
# ============================================================

results = []


for item in dataset:

    question_id = item["id"]

    question = item["question"]

    answerable = item["answerable"]

    gold_evidence = item["gold_evidence"]


    # ========================================================
    # RETRIEVE TOP CHUNKS
    # ========================================================

    retrieved_chunks = retrieve_chunks(
        question
    )


    if retrieved_chunks is None:

        retrieved_chunks = []


    # We need Top-5 for Hit@5
    retrieved_chunks = (
        retrieved_chunks[:5]
    )


    # ========================================================
    # EVALUATE TOP-5
    # ========================================================

    (
        matched_evidence,
        first_rank,
        detailed_matches

    ) = evaluate_retrieved_chunks(

        retrieved_chunks,
        gold_evidence

    )


    # ========================================================
    # DEBUG Q13 / Q19
    # ========================================================

    if question_id in DEBUG_QUESTIONS:

        print("\n")

        print(
            "=" * 70
        )

        print(
            f"{question_id} V5 SEMANTIC RETRIEVAL DEBUG"
        )

        print(
            "=" * 70
        )


        print("\nQuestion:")

        print(question)


        print("\nGold Evidence:")

        for evidence in gold_evidence:

            print(
                f" - {evidence}"
            )


        print(
            "\nRetrieved Chunks:",
            len(retrieved_chunks)
        )


        # ----------------------------------------------------
        # Retrieved chunks
        # ----------------------------------------------------

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"TOP {rank}"
            )

            print(
                "-" * 60
            )


            print(
                "Qdrant Score:",
                chunk.get(
                    "score",
                    "N/A"
                )
            )


            print(
                "Source:",
                chunk.get(
                    "source",
                    "Unknown"
                )
            )


            print(
                "\nChunk Text:"
            )


            print(
                chunk.get(
                    "text",
                    ""
                )
            )


        # ----------------------------------------------------
        # Semantic matches
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "\nEvidence Matching Results:"
        )


        if detailed_matches:

            for match in detailed_matches:

                print(
                    "\nEvidence:"
                )

                print(
                    match["evidence"]
                )


                print(
                    "Rank:",
                    match["rank"]
                )


                print(
                    "Similarity:",
                    f'{match["score"]:.4f}'
                )


                print(
                    "Match Type:",
                    match["match_type"]
                )


                print(
                    "Matched Sentence:"
                )

                print(
                    match["sentence"]
                )


        else:

            print(
                "❌ No evidence matched."
            )


        print(
            "\nFirst Relevant Rank:"
        )


        if first_rank is None:

            print(
                "No relevant chunk"
            )

        else:

            print(
                first_rank
            )


        print(
            "\nSemantic Threshold:"
        )

        print(
            SEMANTIC_THRESHOLD
        )


        print(
            "\nDiagnosis:"
        )


        if first_rank == 1:

            print(
                "✅ Relevant evidence found at Rank 1."
            )

        elif first_rank is not None:

            print(
                f"⚠️ Relevant evidence first found "
                f"at Rank {first_rank}."
            )

        else:

            print(
                "❌ No relevant evidence found "
                "in Top-5."
            )


        print(
            "=" * 70
        )


    # ========================================================
    # HIT@1
    # ========================================================

    if answerable:

        (
            matches_at_1,
            _,
            _

        ) = evaluate_retrieved_chunks(

            retrieved_chunks[:1],
            gold_evidence

        )

        hit_at_1 = int(
            len(matches_at_1) > 0
        )

    else:

        hit_at_1 = None


    # ========================================================
    # HIT@3
    # ========================================================

    if answerable:

        (
            matches_at_3,
            _,
            _

        ) = evaluate_retrieved_chunks(

            retrieved_chunks[:3],
            gold_evidence

        )

        hit_at_3 = int(
            len(matches_at_3) > 0
        )

    else:

        hit_at_3 = None


    # ========================================================
    # HIT@5
    # ========================================================

    if answerable:

        (
            matches_at_5,
            _,
            _

        ) = evaluate_retrieved_chunks(

            retrieved_chunks[:5],
            gold_evidence

        )

        hit_at_5 = int(
            len(matches_at_5) > 0
        )

    else:

        hit_at_5 = None


    # ========================================================
    # RECIPROCAL RANK
    # ========================================================

    if (
        answerable
        and
        first_rank is not None
    ):

        reciprocal_rank = (
            1 / first_rank
        )

    else:

        reciprocal_rank = None


    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({

        "id":
            question_id,

        "category":
            item["category"],

        "question":
            question,

        "answerable":
            answerable,

        "gold_evidence_count":
            len(gold_evidence),

        "matched_evidence_count":
            len(matched_evidence),

        "matched_evidence":
            " | ".join(
                matched_evidence
            ),

        "first_relevant_rank":
            first_rank,

        "hit_at_1":
            hit_at_1,

        "hit_at_3":
            hit_at_3,

        "hit_at_5":
            hit_at_5,

        "reciprocal_rank":
            reciprocal_rank,

        "retrieved_chunks":
            len(retrieved_chunks),

        "retrieved_sources":
            " | ".join(

                chunk.get(
                    "source",
                    "Unknown"
                )

                for chunk in retrieved_chunks

            )

    })


# ============================================================
# SAVE CSV
# ============================================================

if results:

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=results[0].keys()

        )

        writer.writeheader()

        writer.writerows(results)


# ============================================================
# ANSWERABLE RESULTS
# ============================================================

answerable_results = [

    result

    for result in results

    if result["answerable"]

]


total_answerable = len(
    answerable_results
)


# ============================================================
# HIT COUNTS
# ============================================================

hits_at_1 = sum(

    result["hit_at_1"]

    for result in answerable_results

)


hits_at_3 = sum(

    result["hit_at_3"]

    for result in answerable_results

)


hits_at_5 = sum(

    result["hit_at_5"]

    for result in answerable_results

)


# ============================================================
# HIT RATES
# ============================================================

if total_answerable > 0:

    hit_rate_at_1 = (
        hits_at_1
        /
        total_answerable
    )

    hit_rate_at_3 = (
        hits_at_3
        /
        total_answerable
    )

    hit_rate_at_5 = (
        hits_at_5
        /
        total_answerable
    )

else:

    hit_rate_at_1 = 0.0

    hit_rate_at_3 = 0.0

    hit_rate_at_5 = 0.0


# ============================================================
# MRR
# ============================================================

reciprocal_ranks = [

    result["reciprocal_rank"]

    for result in answerable_results

    if result["reciprocal_rank"]
    is not None

]


if reciprocal_ranks:

    mrr = (

        sum(
            reciprocal_ranks
        )
        /
        len(
            reciprocal_ranks
        )

    )

else:

    mrr = 0.0


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print(
    "=" * 60
)

print(
    "BRIM RETRIEVAL EVALUATION - V5"
)

print(
    "=" * 60
)


print(
    f"Total Questions       : "
    f"{len(results)}"
)


print(
    f"Answerable Questions  : "
    f"{total_answerable}"
)


print(
    f"Hits @ 1              : "
    f"{hits_at_1}"
)


print(
    f"Hit Rate @ 1          : "
    f"{hit_rate_at_1:.2%}"
)


print(
    f"Hits @ 3              : "
    f"{hits_at_3}"
)


print(
    f"Hit Rate @ 3          : "
    f"{hit_rate_at_3:.2%}"
)


print(
    f"Hits @ 5              : "
    f"{hits_at_5}"
)


print(
    f"Hit Rate @ 5          : "
    f"{hit_rate_at_5:.2%}"
)


print(
    f"MRR                   : "
    f"{mrr:.4f}"
)


print(
    "\nSemantic Threshold:"
)

print(
    SEMANTIC_THRESHOLD
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)


# ============================================================
# QUESTION LEVEL RESULTS
# ============================================================

print("\n")

print(
    "=" * 60
)

print(
    "QUESTION LEVEL RESULTS"
)

print(
    "=" * 60
)


for result in results:

    # --------------------------------------------------------
    # Unanswerable
    # --------------------------------------------------------

    if not result["answerable"]:

        print(

            f'{result["id"]} → '
            f'UNANSWERABLE'

        )

        continue


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if result["hit_at_3"]:

        status = "✅ HIT"

    else:

        status = "❌ MISS"


    # --------------------------------------------------------
    # First relevant rank
    # --------------------------------------------------------

    if (
        result["first_relevant_rank"]
        is None
    ):

        rank_text = (
            "No relevant chunk"
        )

    else:

        rank_text = (

            "First relevant rank: "
            +
            str(
                result[
                    "first_relevant_rank"
                ]
            )

        )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(

        f'{result["id"]} → '
        f'{status} | '
        f'Hit@1={result["hit_at_1"]} | '
        f'Hit@3={result["hit_at_3"]} | '
        f'Hit@5={result["hit_at_5"]} | '
        f'{rank_text}'

    )


# ============================================================
# END
# ============================================================

print(
    "\nEvaluation completed successfully."
)