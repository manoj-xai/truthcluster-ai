
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


TRACKING_PARAMETERS = {
    "fbclid", "gclid", "ref", "source",
    "utm_source", "utm_medium", "utm_campaign",
    "utm_term", "utm_content"
}


def normalize_url(url):
    if not url:
        return ""

    try:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()

        if host.startswith("www."):
            host = host[4:]

        path = parsed.path.rstrip("/") or "/"

        query = urlencode([
            (key, value)
            for key, value in parse_qsl(
                parsed.query, keep_blank_values=True
            )
            if key.lower() not in TRACKING_PARAMETERS
        ])

        return urlunparse((
            parsed.scheme.lower(),
            host,
            path,
            "",
            query,
            ""
        ))
    except Exception:
        return url.strip()


def article_url(article):
    return (
        article.get("resolved_url")
        or article.get("url")
        or article.get("link")
        or ""
    )


def article_title(article):
    return article.get("title") or article.get("headline") or ""


def load_json(path):
    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_url_index(data, label):
    index = {}
    ambiguous_keys = set()

    for article in data.get("articles", []):
        key = normalize_url(article_url(article))

        if not key:
            continue

        if key in ambiguous_keys:
            continue

        if key in index:
            index.pop(key)
            ambiguous_keys.add(key)
        else:
            index[key] = article

    if ambiguous_keys:
        print(
            f"WARNING: {label} has {len(ambiguous_keys)} "
            "ambiguous URL key(s); those matches were skipped."
        )

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Build an integrated evidence report."
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Folder containing the stage output JSON files"
    )
    parser.add_argument(
        "--output",
        default="final_evidence_report.json",
        help="Output report path"
    )
    args = parser.parse_args()

    folder = Path(args.folder)

    file_names = {
        "content": "article_content.json",
        "clusters": "clustered_evidence.json",
        "relevance": "claim_evidence_analysis.json",
        "stance": "stance_analysis.json",
        "duplicates": "deduplicated_evidence.json",
    }

    data = {
        name: load_json(folder / filename)
        for name, filename in file_names.items()
    }

    content_articles = data["content"].get("articles", [])

    if not content_articles:
        print(
            "ERROR: No articles found in "
            f"{folder / file_names['content']}."
        )
        return

    indexes = {
        name: build_url_index(data[name], name)
        for name in ["clusters", "relevance", "stance", "duplicates"]
    }

    report_articles = []

    for content in content_articles:
        url = article_url(content)
        key = normalize_url(url)

        cluster = indexes["clusters"].get(key)
        relevance = indexes["relevance"].get(key)
        stance = indexes["stance"].get(key)
        duplicate = indexes["duplicates"].get(key)

        # Preserve status fields from extraction.
        extraction_status = content.get(
            "status",
            content.get("extraction_status", "not_specified")
        )

        article_text = (
            content.get("text")
            or content.get("article_text")
            or ""
        )

        report_articles.append({
            "title": article_title(content),
            "publisher_url": url,
            "extraction_status": extraction_status,
            "text_available": bool(article_text),
            "cluster": {
                "cluster_id": (
                    cluster.get("cluster_id") if cluster else None
                ),
                "matched": cluster is not None,
            },
            "relevance": {
                "title_similarity": (
                    relevance.get("title_similarity")
                    if relevance else None
                ),
                "best_sentence_similarity": (
                    relevance.get("best_sentence_similarity")
                    if relevance else None
                ),
                "relevance_category": (
                    relevance.get("relevance_category", "unavailable")
                    if relevance else "unavailable"
                ),
                "candidate_sentences": (
                    relevance.get("candidate_evidence_sentences", [])
                    if relevance else []
                ),
                "matched": relevance is not None,
            },
            "stance_screening": {
                "article_stance": (
                    stance.get("article_stance", "undetermined")
                    if stance else "unavailable"
                ),
                "truth_status": (
                    stance.get("truth_status", "undetermined")
                    if stance else "unavailable"
                ),
                "sentence_analysis": (
                    stance.get("sentence_analysis", [])
                    if stance else []
                ),
                "matched": stance is not None,
            },
            "duplicate_screening": {
                "group_id": (
                    duplicate.get("duplicate_group_id")
                    if duplicate else None
                ),
                "group_size": (
                    duplicate.get("duplicate_group_size")
                    if duplicate else None
                ),
                "duplicate_type": (
                    duplicate.get("duplicate_type", "unavailable")
                    if duplicate else "unavailable"
                ),
                "matched": duplicate is not None,
            },
        })

    matched_clusters = sum(
        item["cluster"]["matched"] for item in report_articles
    )
    matched_relevance = sum(
        item["relevance"]["matched"] for item in report_articles
    )
    matched_stance = sum(
        item["stance_screening"]["matched"] for item in report_articles
    )
    matched_duplicates = sum(
        item["duplicate_screening"]["matched"]
        for item in report_articles
    )

    result = {
        "claim": data["content"].get("claim", ""),
        "article_count": len(report_articles),
        "matched_cluster_records": matched_clusters,
        "matched_relevance_records": matched_relevance,
        "matched_stance_records": matched_stance,
        "matched_duplicate_records": matched_duplicates,
        "overall_assessment": "undetermined",
        "assessment_note": (
            "This report combines retrieval and screening outputs. "
            "Text similarity and preliminary stance labels do not prove "
            "whether the claim is true or false. Verify evidence, dates, "
            "context, and original sources before making a conclusion."
        ),
        "articles": report_articles,
    }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\n=== INTEGRATED EVIDENCE REPORT ===")
    print(f"Input folder: {folder.resolve()}")
    print(f"Articles retained: {len(report_articles)}")
    print(f"Matched cluster records: {matched_clusters}")
    print(f"Matched relevance records: {matched_relevance}")
    print(f"Matched stance records: {matched_stance}")
    print(f"Matched duplicate records: {matched_duplicates}")
    print("Overall assessment: UNDETERMINED")
    print(f"Saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()