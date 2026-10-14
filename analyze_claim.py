
import argparse
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_SENTENCES_PER_ARTICLE = 500
TOP_MATCHING_SENTENCES = 3


def split_sentences(text):
    """Split article text into simple sentence candidates."""
    text = re.sub(r"\s+", " ", text or "").strip()

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.split()) >= 5
    ][:MAX_SENTENCES_PER_ARTICLE]


def main():
    parser = argparse.ArgumentParser(
        description="Compare a claim with clustered news evidence."
    )
    parser.add_argument(
        "--input",
        default="clustered_evidence.json",
        help="Path to the clustered evidence JSON file."
    )
    parser.add_argument(
        "--output",
        default="claim_evidence_analysis.json",
        help="Path to save the claim analysis."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    try:
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"Could not read input file: {error}")

    claim = (data.get("claim") or "").strip()
    articles = data.get("articles", [])

    if not claim:
        raise SystemExit("No claim was found in the input file.")

    if not isinstance(articles, list):
        raise SystemExit("The articles field must be a list.")

    usable_articles = [
        article for article in articles
        if isinstance(article, dict)
        and (article.get("article_text") or "").strip()
    ]

    if not usable_articles:
        raise SystemExit("No usable article text was found.")

    prepared = []

    for article in usable_articles:
        prepared.append({
            "article": article,
            "title": article.get("title", ""),
            "sentences": split_sentences(
                article.get("article_text", "")
            )
        })

    # Keep a mapping between every document vector and its purpose.
    corpus = [claim]
    document_map = []

    for article_index, item in enumerate(prepared):
        corpus.append(item["title"] or "untitled article")
        document_map.append((article_index, "title", None))

        for sentence_index, sentence in enumerate(item["sentences"]):
            corpus.append(sentence)
            document_map.append((
                article_index, "sentence", sentence_index
            ))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )

    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError as error:
        raise SystemExit(
            f"Unable to compare the text: {error}\n"
            "Check whether the claim and articles contain readable words."
        )

    claim_vector = matrix[0]

    # Save the vector positions for each article and sentence.
    vector_positions = {}

    for vector_index, (article_index, kind, sentence_index) in enumerate(
        document_map, start=1
    ):
        vector_positions.setdefault(
            article_index, {"title": None, "sentences": {}}
        )

        if kind == "title":
            vector_positions[article_index]["title"] = vector_index
        else:
            vector_positions[article_index]["sentences"][
                sentence_index
            ] = vector_index

    results = []

    for article_index, item in enumerate(prepared):
        positions = vector_positions[article_index]

        title_vector = matrix[positions["title"]]
        title_score = float(
            cosine_similarity(claim_vector, title_vector)[0][0]
        )

        sentence_records = []

        for sentence_index, sentence in enumerate(item["sentences"]):
            sentence_vector = matrix[
                positions["sentences"][sentence_index]
            ]

            score = float(
                cosine_similarity(
                    claim_vector, sentence_vector
                )[0][0]
            )

            sentence_records.append({
                "text": sentence,
                "similarity_score": round(score, 4)
            })

        sentence_records.sort(
            key=lambda record: record["similarity_score"],
            reverse=True
        )

        best_sentence_score = (
            sentence_records[0]["similarity_score"]
            if sentence_records else 0.0
        )

        relevance = max(title_score, best_sentence_score)

        if relevance >= 0.25:
            category = "higher_relevance_candidate"
        elif relevance >= 0.10:
            category = "possible_relevance"
        else:
            category = "low_textual_match"

        article = item["article"]

        results.append({
            "cluster_id": article.get("cluster_id"),
            "title": item["title"],
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "url": (
                article.get("url")
                or article.get("resolved_url", "")
            ),
            "title_similarity": round(title_score, 4),
            "best_sentence_similarity": round(
                best_sentence_score, 4
            ),
            "relevance_category": category,
            "candidate_evidence_sentences": sentence_records[
                :TOP_MATCHING_SENTENCES
            ],
            "stance": "not_assessed",
            "truth_status": "undetermined"
        })

    results.sort(
        key=lambda record: max(
            record["title_similarity"],
            record["best_sentence_similarity"]
        ),
        reverse=True
    )

    output = {
        "claim": claim,
        "input_file": str(input_path),
        "articles_analyzed": len(results),
        "method": (
            "TF-IDF and cosine similarity between the claim, "
            "headlines, and article sentences"
        ),
        "warning": (
            "Similarity scores are not probabilities of truth. "
            "A matching sentence may support, contradict, or merely "
            "mention the claim. Stance has not yet been assessed. "
            "Sources and quotations must be checked in context."
        ),
        "articles": results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n=== CLAIM EVIDENCE ANALYSIS ===")
    print(f"Claim: {claim}")
    print(f"Articles analyzed: {len(results)}")

    for index, result in enumerate(results[:10], start=1):
        print(f"\n{index}. {result['title']}")
        print(f"   Cluster: {result['cluster_id']}")
        print(f"   Source: {result['source']}")
        print(f"   Relevance: {result['relevance_category']}")
        print(f"   Title similarity: {result['title_similarity']:.3f}")

        if result["candidate_evidence_sentences"]:
            best = result["candidate_evidence_sentences"][0]
            print(
                "   Best matching sentence: "
                f"{best['text'][:350]}"
            )
        else:
            print("   No suitable sentence was extracted.")

    print("\n=== SUMMARY ===")
    print(f"Articles analyzed: {len(results)}")
    print(f"Saved to: {output_path.resolve()}")
    print("Truth verdict: NOT YET DETERMINED")


if __name__ == "__main__":
    main()
