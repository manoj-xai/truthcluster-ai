import argparse
import json
import re
from pathlib import Path

SUPPORT_CUES = [
    "said", "says", "stated", "announced",
    "plans to", "expects to", "predicts",
    "believes", "confirmed"
]

CONTRADICTION_CUES = [
    "denied", "never said", "did not say",
    "didn't say", "no evidence that",
    "false claim", "fact check", "debunked",
    "is incorrect", "was incorrect"
]

NEGATION_CUES = [
    "not", "never", "no", "neither", "without",
    "denied", "false", "incorrect"
]

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "and", "or",
    "in", "on", "at", "by", "for", "with", "from",
    "that", "this", "it", "as", "about", "said",
    "say", "says", "will", "would", "could", "should",
    "has", "have", "had", "do", "does", "did",
    "claim", "claims", "according"
}


def words(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def important_words(text):
    return {
        word for word in words(text)
        if word not in STOP_WORDS and len(word) > 1
    }


def classify_sentence(claim, sentence):
    """Screen a sentence for possible stance; does not verify truth."""
    claim_words = important_words(claim)
    sentence_words = important_words(sentence)

    if not claim_words or not sentence_words:
        return {
            "stance_candidate": "unclear",
            "reason": "Insufficient text for comparison."
        }

    overlap = claim_words.intersection(sentence_words)
    overlap_ratio = len(overlap) / len(claim_words)

    claim_years = set(
        re.findall(r"\b(?:19|20|21)\d{2}\b", claim)
    )
    sentence_years = set(
        re.findall(r"\b(?:19|20|21)\d{2}\b", sentence)
    )

    sentence_lower = sentence.lower()

    has_contradiction_cue = any(
        cue in sentence_lower for cue in CONTRADICTION_CUES
    )
    has_support_cue = any(
        cue in sentence_lower for cue in SUPPORT_CUES
    )
    has_negation = any(
        re.search(
            r"\b" + re.escape(cue) + r"\b",
            sentence_lower
        )
        for cue in NEGATION_CUES
    )

    if overlap_ratio >= 0.30 and has_contradiction_cue:
        label = "possible_contradiction"
        reason = (
            "Relevant words and a contradiction-related phrase were found."
        )

    elif (
        overlap_ratio >= 0.30
        and has_support_cue
        and not has_negation
        and (
            not claim_years
            or claim_years.intersection(sentence_years)
        )
    ):
        label = "possible_support"
        reason = (
            "Relevant words, a reporting cue, and any required claim year "
            "were found."
        )

    else:
        label = "unclear"
        reason = (
            "The simple rules cannot establish whether this sentence "
            "supports or contradicts the claim."
        )

    return {
        "stance_candidate": label,
        "reason": reason,
        "claim_word_overlap": round(overlap_ratio, 3),
        "overlapping_words": sorted(overlap),
        "claim_years": sorted(claim_years),
        "sentence_years": sorted(sentence_years)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Screen candidate evidence sentences for possible stance."
    )
    parser.add_argument(
        "--input",
        default="claim_evidence_analysis.json",
        help="Input claim-evidence analysis JSON file"
    )
    parser.add_argument(
        "--output",
        default="stance_analysis.json",
        help="Output stance analysis JSON file"
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"Error: {input_file} was not found.")
        return

    with input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    claim = data.get("claim", "").strip()
    articles = data.get("articles", [])

    if not claim:
        print("Error: the input file does not contain a claim.")
        return

    analyzed_articles = []

    for article in articles:
        sentence_results = []

        for candidate in article.get(
            "candidate_evidence_sentences", []
        ):
            sentence = candidate.get("text", "")
            result = classify_sentence(claim, sentence)

            result["sentence"] = sentence
            result["similarity_score"] = candidate.get(
                "similarity_score"
            )
            sentence_results.append(result)

        analyzed_articles.append({
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "url": article.get("url", ""),
            "sentence_analysis": sentence_results,
            "article_stance": "undetermined",
            "truth_status": "undetermined"
        })

    counts = {
        "possible_support": 0,
        "possible_contradiction": 0,
        "unclear": 0
    }

    for article in analyzed_articles:
        for item in article["sentence_analysis"]:
            label = item["stance_candidate"]
            if label in counts:
                counts[label] += 1

    output = {
        "claim": claim,
        "articles_analyzed": len(analyzed_articles),
        "sentence_candidate_counts": counts,
        "method": "Rule-based preliminary sentence screening",
        "warning": (
            "These labels are candidates for review, not verified stance "
            "or truth. Keyword rules can miss paraphrases, sarcasm, "
            "quotations, context, and negation scope. No final verdict "
            "has been produced."
        ),
        "articles": analyzed_articles
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n=== STANCE SCREENING SUMMARY ===")
    print(f"Claim: {claim}")
    print(f"Articles analyzed: {len(analyzed_articles)}")
    print(f"Possible support sentences: {counts['possible_support']}")
    print(
        "Possible contradiction sentences: "
        f"{counts['possible_contradiction']}"
    )
    print(f"Unclear sentences: {counts['unclear']}")
    print(f"Saved to: {output_file.resolve()}")
    print("\nFinal truth verdict: NOT DETERMINED")


if __name__ == "__main__":
    main()