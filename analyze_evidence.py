
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def normalize_text(text):
    """Normalize text for comparison and duplicate detection."""
    text = (text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def normalize_url(url):
    """Remove URL fragments and common tracking parameters."""
    if not url:
        return ""

    try:
        parts = urlsplit(url.strip())

        if parts.scheme not in ("http", "https") or not parts.netloc:
            return url.strip()

        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",
            ""
        ))
    except ValueError:
        return url.strip()


def get_words(text):
    return set(normalize_text(text).split())


def relevance_score(claim, title):
    """
    Measure meaningful word overlap between claim and title.
    This is NOT a truth, source credibility, or evidence-quality score.
    """
    claim_words = get_words(claim)
    title_words = get_words(title)

    if not claim_words or not title_words:
        return 0.0

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "will", "would", "can", "could", "said", "says",
        "that", "to", "of", "in", "on", "for", "by",
        "and", "or", "it", "we", "they", "he", "she",
        "has", "have", "had", "be", "been", "being"
    }

    important_claim_words = claim_words - stop_words
    important_title_words = title_words - stop_words

    if not important_claim_words:
        important_claim_words = claim_words

    overlap = important_claim_words & important_title_words
    score = len(overlap) / len(important_claim_words)

    claim_years = set(re.findall(r"\b(?:19|20)\d{2}\b", claim))
    title_years = set(re.findall(r"\b(?:19|20)\d{2}\b", title))

    if claim_years:
        if claim_years & title_years:
            score += 0.15
        else:
            score -= 0.10

    return round(max(0.0, min(1.0, score)), 3)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate and rank extracted news evidence."
    )
    parser.add_argument(
        "--input",
        default="article_content.json",
        help="Path to the article extraction JSON file."
    )
    parser.add_argument(
        "--output",
        default="analyzed_evidence.json",
        help="Path to save the analyzed evidence JSON file."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"Could not read input file: {error}")

    claim = data.get("claim", "").strip()
    articles = data.get("articles", [])

    if not claim or not isinstance(articles, list):
        raise SystemExit(
            "The input file does not have the expected format."
        )

    unique_articles = []
    seen_urls = set()
    seen_titles = set()
    duplicate_count = 0
    skipped_count = 0

    for article in articles:
        if not isinstance(article, dict):
            skipped_count += 1
            continue

        # Only rank articles where readable text was extracted.
        if article.get("extraction_status") != "success":
            skipped_count += 1
            continue

        title = (article.get("title") or "").strip()
        article_text = (article.get("article_text") or "").strip()

        # Prefer the actual publisher URL, not the Google News redirect.
        url = (
            article.get("resolved_url")
            or article.get("google_news_url")
            or article.get("url")
            or ""
        ).strip()

        title_key = normalize_text(title)
        url_key = normalize_url(url)

        if not title_key:
            skipped_count += 1
            continue

        if (
            (url_key and url_key in seen_urls)
            or title_key in seen_titles
        ):
            duplicate_count += 1
            continue

        if url_key:
            seen_urls.add(url_key)

        seen_titles.add(title_key)

        unique_articles.append({
            "title": title,
            "url": url,
            "source": article.get("source", "Unknown"),
            "date": article.get("date", "Unknown"),
            "provider": article.get("provider", "Unknown"),
            "article_text": article_text,
            "extraction_status": article.get(
                "extraction_status", "unknown"
            ),
            "relevance_score": relevance_score(claim, title),
            "verification_status": "not_verified"
        })

    unique_articles.sort(
        key=lambda article: article["relevance_score"],
        reverse=True
    )

    report = {
        "claim": claim,
        "input_file": str(input_path),
        "extracted_article_count": len(articles),
        "successful_extractions": data.get(
            "successful_extractions", None
        ),
        "unique_count": len(unique_articles),
        "duplicate_count": duplicate_count,
        "skipped_count": skipped_count,
        "method": (
            "Publisher URL/title deduplication and "
            "lexical title relevance ranking"
        ),
        "warning": (
            "Relevance measures title-word overlap, not truth or "
            "source credibility. Only successfully extracted articles "
            "are included. No claim has been verified as true or false."
        ),
        "articles": unique_articles
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== EVIDENCE ANALYZER ===")
    print(f"Claim: {claim}")
    print(f"Articles received: {len(articles)}")
    print(f"Unique extracted articles: {len(unique_articles)}")
    print(f"Duplicates: {duplicate_count}")
    print(f"Skipped or unsuccessful: {skipped_count}")

    print("\nTop 10 relevant article titles:")

    for index, article in enumerate(unique_articles[:10], start=1):
        print(
            f"\n{index}. {article['title']}"
            f"\n   Source: {article['source']}"
            f"\n   Relevance: {article['relevance_score']}"
            f"\n   URL: {article['url']}"
        )

    print(f"\nReport saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
