import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from difflib import SequenceMatcher

TITLE_SIMILARITY_THRESHOLD = 0.92

TRACKING_PARAMETERS = {
    "fbclid", "gclid", "ref", "source",
    "utm_source", "utm_medium", "utm_campaign",
    "utm_term", "utm_content"
}


def normalize_title(title):
    title = (title or "").lower()
    title = re.sub(r"https?://\S+", " ", title)
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def normalize_url(url):
    if not url:
        return ""

    try:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()

        if host.startswith("www."):
            host = host[4:]

        path = parsed.path.rstrip("/") or "/"

        query_items = [
            (key, value)
            for key, value in parse_qsl(
                parsed.query, keep_blank_values=True
            )
            if key.lower() not in TRACKING_PARAMETERS
        ]
        query = urlencode(query_items)

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


def get_article_title(article):
    return article.get("title") or article.get("headline") or ""


def get_article_url(article):
    return (
        article.get("resolved_url")
        or article.get("url")
        or article.get("link")
        or ""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Group duplicate or potentially duplicate evidence."
    )
    parser.add_argument(
        "--input",
        default="article_content.json",
        help="Input JSON file containing articles"
        )
    parser.add_argument(
        "--output",
        default="deduplicated_evidence.json",
        help="Output JSON file"
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"ERROR: {input_file} was not found.")
        return

    with input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    articles = data.get("articles", [])

    if not articles:
        print(f"No articles were found in {input_file}.")
        print("Check that the input file contains an 'articles' list.")
        return

    count = len(articles)
    parents = list(range(count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        root_first = find(first)
        root_second = find(second)

        if root_first != root_second:
            parents[root_second] = root_first

    normalized_urls = [
        normalize_url(get_article_url(article))
        for article in articles
    ]
    normalized_titles = [
        normalize_title(get_article_title(article))
        for article in articles
    ]

    same_url_pairs = set()
    similar_title_pairs = set()

    # 1. Group exact normalized publisher URLs.
    url_groups = {}

    for index, url in enumerate(normalized_urls):
        if not url:
            continue

        if url in url_groups:
            other = url_groups[url]
            union(index, other)
            same_url_pairs.add((min(index, other), max(index, other)))
        else:
            url_groups[url] = index

    # 2. Group highly similar headlines.
    for first in range(count):
        title_a = normalized_titles[first]

        if len(title_a) < 16:
            continue

        for second in range(first + 1, count):
            title_b = normalized_titles[second]

            if len(title_b) < 16:
                continue

            similarity = SequenceMatcher(
                None, title_a, title_b
            ).ratio()

            if similarity >= TITLE_SIMILARITY_THRESHOLD:
                union(first, second)
                similar_title_pairs.add((first, second))

    # Build groups after all comparisons.
    groups = {}

    for index in range(count):
        root = find(index)
        groups.setdefault(root, []).append(index)

    sorted_groups = sorted(
        groups.values(),
        key=lambda members: min(members)
    )

    output_articles = []

    for group_number, members in enumerate(sorted_groups, start=1):
        group_id = f"DUP-{group_number:03d}"

        has_same_url = any(
            first in members and second in members
            for first, second in same_url_pairs
        )
        has_similar_title = any(
            first in members and second in members
            for first, second in similar_title_pairs
        )

        if len(members) == 1:
            group_type = "unique"
        elif has_same_url and has_similar_title:
            group_type = "same_url_and_similar_title"
        elif has_same_url:
            group_type = "same_url"
        else:
            group_type = "possible_duplicate_headlines"

        for index in members:
            article = dict(articles[index])
            article["duplicate_group_id"] = group_id
            article["duplicate_group_size"] = len(members)
            article["duplicate_type"] = group_type

            if group_type == "possible_duplicate_headlines":
                article["duplicate_warning"] = (
                    "Possible headline match only; copying or shared "
                    "sourcing has not been established."
                )
            else:
                article["duplicate_warning"] = ""

            output_articles.append(article)

    result = {
        "claim": data.get("claim", ""),
        "input_article_count": count,
        "duplicate_group_count": len(sorted_groups),
        "groups_with_multiple_articles": sum(
            1 for members in sorted_groups if len(members) > 1
        ),
        "method": (
            "Normalized URL matching and conservative headline similarity "
            "using SequenceMatcher."
        ),
        "warning": (
            "Duplicate grouping is a screening aid, not proof of shared "
            "ownership, copying, source independence, or claim truth."
        ),
        "articles": output_articles
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\n=== DUPLICATE SCREENING SUMMARY ===")
    print(f"Input: {input_file}")
    print(f"Articles checked: {count}")
    print(f"Groups created: {len(sorted_groups)}")
    print(
        "Groups with multiple articles: "
        f"{result['groups_with_multiple_articles']}"
    )
    print(f"Saved to: {output_file.resolve()}")

    print("\n=== GROUP DETAILS ===")

    for group_number, members in enumerate(sorted_groups, start=1):
        if len(members) <= 1:
            continue

        print(f"\nDUP-{group_number:03d} ({len(members)} articles)")

        for index in members:
            title = get_article_title(articles[index])
            print(f"  - {title or '(No title)'}")


if __name__ == "__main__":
    main()