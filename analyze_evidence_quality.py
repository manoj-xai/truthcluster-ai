
import argparse
import json
import re
from pathlib import Path

REPORTING_CUES = re.compile(
    r"\b(said|stated|claimed|announced|predicted|according to|"
    r"reported|expects|wants|plans|found|discovered|shows|"
    r"suggests|revealed|confirmed|denied)\b",
    re.IGNORECASE,
)

UNCERTAINTY_CUES = re.compile(
    r"\b(potential|possible|possibly|may|might|could|early-stage|"
    r"preclinical|pre-clinical|preliminary|unproven|experimental)\b",
    re.IGNORECASE,
)

CONTRADICTION_CUES = re.compile(
    r"\b(false|fabricated|debunked|retracted|hoax|"
    r"no evidence|did not happen|not true)\b",
    re.IGNORECASE,
)


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def classify_sentence(sentence, claim):
    """Screen a sentence for review; this is not a truth detector."""
    text = normalize(sentence)
    claim = normalize(claim)

    claim_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", claim)
        if len(word) > 2
    }
    sentence_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", text)
        if len(word) > 2
    }

    overlap = claim_words & sentence_words
    relevance = (
        len(overlap) / len(claim_words)
        if claim_words else 0.0
    )

    years = re.findall(r"\b(?:19|20|21)\d{2}\b", text)
    mentions_reporting = bool(REPORTING_CUES.search(text))
    mentions_uncertainty = bool(UNCERTAINTY_CUES.search(text))
    mentions_contradiction = bool(CONTRADICTION_CUES.search(text))

    if relevance < 0.15:
        category = "low_relevance_candidate"
        explanation = (
            "The sentence shares few keywords with the claim. "
            "Check semantic relevance before using it."
        )
    elif mentions_contradiction:
        category = "contradiction_review_candidate"
        explanation = (
            "The sentence contains language that may indicate a "
            "contradiction. Read the full context before drawing a conclusion."
        )
    elif mentions_uncertainty:
        category = "uncertain_or_early_stage_evidence"
        explanation = (
            "The sentence includes uncertainty or early-stage research "
            "language. It does not establish that the claim is proven."
        )
    elif mentions_reporting:
        category = "attribution_or_reporting_candidate"
        explanation = (
            "The sentence contains a reporting or attribution cue. "
            "Check the original source and the full statement."
        )
    else:
        category = "relevant_sentence_for_review"
        explanation = (
            "The sentence has some keyword overlap with the claim. "
            "Manual or model-assisted review is still required."
        )

    return {
        "sentence": text,
        "screening_category": category,
        "explanation": explanation,
        "keyword_overlap": sorted(overlap),
        "keyword_relevance": round(relevance, 3),
        "years_found": sorted(set(years)),
        "contains_reporting_cue": mentions_reporting,
        "contains_uncertainty_cue": mentions_uncertainty,
        "contains_contradiction_cue": mentions_contradiction,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Screen evidence sentences for quality-review cues."
    )
    parser.add_argument(
        "--input",
        default="final_evidence_report.json",
        help="Integrated evidence report JSON file"
    )
    parser.add_argument(
        "--output",
        default="evidence_quality_review.json",
        help="Output quality review JSON file"
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"ERROR: Missing file: {input_file}")
        return

    with input_file.open("r", encoding="utf-8") as file:
        report = json.load(file)

    claim = normalize(report.get("claim", ""))

    if not claim:
        print(f"ERROR: No claim found in {input_file}")
        return

    articles = report.get("articles", [])
    if not isinstance(articles, list):
        articles = []

    reviewed_articles = []
    total_sentences = 0

    for index, article in enumerate(articles, start=1):
        if not isinstance(article, dict):
            continue

        title = normalize(article.get("title", ""))
        url = normalize(
            article.get("publisher_url")
            or article.get("resolved_url")
            or article.get("url")
            or article.get("google_news_url")
            or ""
        )

        content = normalize(
            article.get("text")
            or article.get("content")
            or article.get("article_text")
            or article.get("summary")
            or article.get("description")
            or ""
        )

        # Use available candidate sentences from the integrated report.
        candidate_sentences = []
        relevance_data = article.get("relevance", {})

        if isinstance(relevance_data, dict):
            for candidate in relevance_data.get(
                "candidate_sentences", []
            ):
                if isinstance(candidate, dict):
                    candidate_sentences.append(
                        normalize(candidate.get("text", ""))
                    )
                elif isinstance(candidate, str):
                    candidate_sentences.append(normalize(candidate))

        # Prefer original article text when present; otherwise use candidates.
        if content:
            combined_text = ". ".join(
                part for part in [title, content] if part
            )
            sentences = [
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+", combined_text)
                if len(part.strip()) >= 25
            ]
        else:
            sentences = [
                sentence for sentence in candidate_sentences
                if len(sentence) >= 25
            ]

        screened = [
            classify_sentence(sentence, claim)
            for sentence in sentences
        ]
        total_sentences += len(screened)

        category_counts = {}
        for item in screened:
            category = item["screening_category"]
            category_counts[category] = category_counts.get(category, 0) + 1

        reviewed_articles.append({
            "article_number": index,
            "title": title,
            "url": url,
            "extraction_status": article.get(
                "extraction_status", "not_specified"
            ),
            "sentences_reviewed": len(screened),
            "category_counts": category_counts,
            "screened_sentences": screened,
        })

    result = {
        "claim": claim,
        "method": "keyword-overlap and language-cue screening",
        "warning": (
            "These heuristic categories are review suggestions, not truth "
            "labels. Keyword overlap, reporting language, and contradiction "
            "cues cannot establish whether a claim is true or false."
        ),
        "article_count": len(reviewed_articles),
        "sentence_count": total_sentences,
        "articles": reviewed_articles,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\n=== EVIDENCE QUALITY REVIEW ===")
    print(f"Input: {input_file.resolve()}")
    print(f"Claim: {claim}")
    print(f"Articles reviewed: {len(reviewed_articles)}")
    print(f"Sentences screened: {total_sentences}")
    print(f"Saved to: {output_file.resolve()}")
    print("Final truth verdict: NOT DETERMINED")


if __name__ == "__main__":
    main()
