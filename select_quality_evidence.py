
import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MAX_ARTICLES = 12
MAX_PER_CLUSTER = 3
MAX_SENTENCES_PER_ARTICLE = 4


def normalize_url(url):
    if not url:
        return ""

    try:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower().removeprefix("www.")
        return f"{host}{parsed.path.rstrip('/')}"
    except (ValueError, AttributeError):
        return url.strip()


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text or "")
        if sentence.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="clustered_evidence.json")
    parser.add_argument("--quality", default="evidence_quality_review.json")
    parser.add_argument("--output", default="quality_selected_evidence.json")
    args = parser.parse_args()

    cluster_file = Path(args.input)
    quality_file = Path(args.quality)
    output_file = Path(args.output)

    for path in (cluster_file, quality_file):
        if not path.exists():
            print(f"ERROR: Missing file: {path}")
            return

    clustered = load_json(cluster_file)
    quality = load_json(quality_file)
    claim = clustered.get("claim") or quality.get("claim", "")

    if not claim.strip():
        print("ERROR: Claim is empty.")
        return

    # Match quality-review records to article URLs.
    quality_by_url = {}

    for article in quality.get("articles", []):
        url = normalize_url(
            article.get("resolved_url")
            or article.get("url")
            or article.get("google_news_url")
            or ""
        )

        if url:
            quality_by_url[url] = article

    candidates = []
    seen_urls = set()

    for article in clustered.get("articles", []):
        original_url = (
            article.get("resolved_url")
            or article.get("url")
            or article.get("google_news_url")
            or ""
        )
        normalized_url = normalize_url(original_url)

        if not normalized_url or normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)
        quality_article = quality_by_url.get(normalized_url, {})
        screened = quality_article.get("screened_sentences", [])

        category_counts = Counter(
            sentence.get("screening_category", "unknown")
            for sentence in screened
        )

        # Use sentences explicitly marked for further review.
        # This category does not mean the sentence is verified.
        sentence_candidates = []

        for record in screened:
            sentence = record.get("sentence", "").strip()
            category = record.get("screening_category", "unknown")

            if not sentence:
                continue

            if category not in {
                "relevant_sentence_for_review",
                "uncertain_or_early_stage_evidence",
            }:
                continue

            sentence_candidates.append({
                "sentence": sentence,
                "screening_category": category,
                "keyword_relevance": record.get("keyword_relevance", 0.0),
                "contains_uncertainty_cue": record.get(
                    "contains_uncertainty_cue", False
                ),
                "contains_contradiction_cue": record.get(
                    "contains_contradiction_cue", False
                ),
            })

        # Keep article text as a fallback when screening found no sentences.
        if not sentence_candidates:
            sentence_candidates = [
                {
                    "sentence": sentence,
                    "screening_category": "not_screened",
                    "keyword_relevance": 0.0,
                    "contains_uncertainty_cue": False,
                    "contains_contradiction_cue": False,
                }
                for sentence in split_sentences(
                    article.get("article_text", "")[:5000]
                )
            ]

        total = sum(category_counts.values())

        candidates.append({
            "title": article.get("title", ""),
            "url": original_url or normalized_url,
            "normalized_url": normalized_url,
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "cluster_id": article.get("cluster_id", "unknown"),
            "article_text": article.get("article_text", ""),
            "category_counts": dict(category_counts),
            "relevant_share": (
                category_counts["relevant_sentence_for_review"] / total
                if total else 0.0
            ),
            "early_stage_share": (
                category_counts["uncertain_or_early_stage_evidence"] / total
                if total else 0.0
            ),
            "low_relevance_share": (
                category_counts["low_relevance_candidate"] / total
                if total else 0.0
            ),
            "sentence_candidates": sentence_candidates,
        })

    if not candidates:
        print("ERROR: No articles with usable URLs were found.")
        return

    # Calculate similarity for the claim and screened sentences,
    # rather than relying only on whole-article similarity.
    sentence_locations = []

    for article_index, item in enumerate(candidates):
        for sentence_index, record in enumerate(
            item["sentence_candidates"]
        ):
            sentence_locations.append((
                article_index,
                sentence_index,
                record["sentence"],
            ))

    documents = [claim] + [
        sentence for _, _, sentence in sentence_locations
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(documents)
        scores = cosine_similarity(
            matrix[0:1], matrix[1:]
        ).flatten()
    except ValueError as error:
        print("ERROR: Could not calculate text similarity:", error)
        return

    best_sentence_score = [0.0] * len(candidates)

    for location, similarity in zip(sentence_locations, scores):
        article_index, sentence_index, sentence = location
        item = candidates[article_index]
        record = item["sentence_candidates"][sentence_index]

        record["tfidf_similarity"] = round(float(similarity), 4)

        # A small lexical boost helps recognize important claim terms.
        lower = sentence.lower()
        claim_lower = claim.lower()

        claim_terms = set(re.findall(r"\b[a-z0-9]+\b", claim_lower))
        sentence_terms = set(re.findall(r"\b[a-z0-9]+\b", lower))
        stop_words = {
            "the", "a", "an", "is", "at", "about", "of",
            "and", "or", "to", "in", "on", "for", "that",
        }
        claim_terms -= stop_words

        overlap = (
            len(claim_terms & sentence_terms) / len(claim_terms)
            if claim_terms else 0.0
        )
        record["claim_term_overlap"] = round(overlap, 4)

        # Relevance only — never interpret this as truth probability.
        record["sentence_relevance_score"] = round(
            0.70 * float(similarity) + 0.30 * overlap, 4
        )

        # Measure claim-term overlap while ignoring common words.
        lower = sentence.lower()
        claim_lower = claim.lower()

        stop_words = {
            "the", "a", "an", "is", "at", "about", "of",
            "and", "or", "to", "in", "on", "for", "that",
            "it", "this", "was", "were", "are", "be", "by",
            "as", "from", "with", "which",
        }

        claim_terms = {
            term for term in re.findall(r"\b[a-z0-9]+\b", claim_lower)
            if term not in stop_words
        }
        sentence_terms = {
            term for term in re.findall(r"\b[a-z0-9]+\b", lower)
            if term not in stop_words
        }

        overlap = (
            len(claim_terms & sentence_terms) / len(claim_terms)
            if claim_terms else 0.0
        )

        record["claim_term_overlap"] = round(overlap, 4)

        # Require meaningful overlap as well as semantic similarity.
        # This is a retrieval score, NOT a truth probability.
        record["sentence_relevance_score"] = round(
            0.55 * float(similarity) + 0.45 * overlap, 4
        )

        best_sentence_score[article_index] = max(
            best_sentence_score[article_index],
            record["sentence_relevance_score"],
        )

    for index, item in enumerate(candidates):
        item["best_sentence_relevance"] = round(
            best_sentence_score[index], 4
        )

        # A direct phrase match is only a retrieval hint.
        all_text = (
            item.get("title", "") + " " +
            " ".join(
                r["sentence"] for r in item["sentence_candidates"]
            )
        ).lower()

        item["direct_claim_match"] = bool(
            re.search(
                r"\bwater\b.{0,100}\bboil(?:s|ing|ed)?\b"
                r".{0,100}\b100\b",
                all_text,
            )
            or re.search(
                r"\b100\b.{0,100}\bdegrees?\s+celsius\b",
                all_text,
            )
        )

        item["selection_score"] = round(
            0.85 * item["best_sentence_relevance"]
            + 0.10 * item["relevant_share"]
            - 0.05 * item["early_stage_share"]
            - 0.02 * item["low_relevance_share"],
            4,
        )

        # Save only the highest-ranking screened sentences.
        item["selected_sentences"] = sorted(
            item["sentence_candidates"],
            key=lambda record: record.get(
                "sentence_relevance_score", 0.0
            ),
            reverse=True,
        )[:MAX_SENTENCES_PER_ARTICLE]

        del item["sentence_candidates"]

    # Remove articles that have no meaningful sentence-level match.
    # The cutoff is a retrieval safeguard, not a factual judgment.
    candidates = [
        item for item in candidates
        if item["best_sentence_relevance"] > 0
    ]

    candidates.sort(
        key=lambda item: (
            item["selection_score"],
            item["direct_claim_match"],
            item["best_sentence_relevance"],
        ),
        reverse=True,
    )

    selected = []
    cluster_counts = defaultdict(int)

    for item in candidates:
        cluster = str(item["cluster_id"])

        if cluster_counts[cluster] >= MAX_PER_CLUSTER:
            continue

        selected.append(item)
        cluster_counts[cluster] += 1

        if len(selected) >= MAX_ARTICLES:
            break

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "claim": claim,
        "method": (
            "Screened sentence-level TF-IDF and claim-term overlap, "
            "preliminary screening features, URL deduplication, "
            "and cluster diversity limits."
        ),
        "warning": (
            "Relevance scores, direct phrase matches, and screening labels "
            "do not establish truth or source credibility. Review the full "
            "context, source authority, publication date, and source "
            "independence before making an assessment."
        ),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "clusters_represented": len(cluster_counts),
        "articles": selected,
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("=== IMPROVED EVIDENCE SELECTION ===")
    print(f"Claim: {claim}")
    print(f"Candidates with nonzero sentence relevance: {len(candidates)}")
    print(f"Selected: {len(selected)}")
    print(f"Clusters represented: {len(cluster_counts)}")
    print(f"Saved to: {output_file.resolve()}")
    print("\nTOP SELECTED ARTICLES:")

    for index, item in enumerate(selected, start=1):
        print(
            f"{index}. [{item['selection_score']:.3f}] "
            f"[best sentence: {item['best_sentence_relevance']:.3f}] "
            f"{item['title']} (cluster {item['cluster_id']})"
        )

        for sentence in item["selected_sentences"][:2]:
            print("   -", sentence["sentence"][:220])


if __name__ == "__main__":
    main()
