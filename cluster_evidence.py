
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

MAX_CLUSTERS = 5


def main():
    parser = argparse.ArgumentParser(
        description="Cluster analyzed news evidence using TF-IDF and K-Means."
    )
    parser.add_argument(
        "--input",
        default="analyzed_evidence.json",
        help="Path to the analyzed evidence JSON file."
    )
    parser.add_argument(
        "--output",
        default="clustered_evidence.json",
        help="Path to save the clustering results."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Error: Input file not found: {input_path}")

    try:
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"Could not read input file: {error}")

    articles = data.get("articles", [])

    # Use extracted, deduplicated articles with enough readable text.
    valid_articles = [
        article for article in articles
        if isinstance(article, dict)
        and len((article.get("article_text") or "").split()) >= 20
    ]

    if len(valid_articles) < 2:
        raise SystemExit(
            "Not enough usable articles for clustering. "
            "At least two articles with readable text are required."
        )

    documents = [
        (article.get("title", "") + " " + article["article_text"])
        for article in valid_articles
    ]

    print(f"Claim: {data.get('claim', '')}")
    print(f"Articles available for clustering: {len(documents)}")
    print("Converting article text into TF-IDF features...")

    # For small datasets, max_df=0.95 can discard terms that occur
    # in nearly every document, potentially leaving no usable features.
    max_df_value = 0.95 if len(documents) >= 20 else 1.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        max_df=max_df_value
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as error:
        raise SystemExit(
            f"Could not create TF-IDF features: {error}\n"
            "Check whether the extracted article text contains readable words."
        )

    number_of_clusters = min(MAX_CLUSTERS, len(valid_articles))

    print(f"Running K-Means with {number_of_clusters} clusters...")

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(tfidf_matrix)

    clustered_articles = []

    for article, label in zip(valid_articles, labels):
        clustered_articles.append({
            "cluster_id": int(label),
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "url": article.get("url", ""),
            "article_text": article.get("article_text", ""),
            "relevance_score": article.get("relevance_score", None)
        })

    cluster_sizes = {
        str(cluster_id): sum(
            article["cluster_id"] == cluster_id
            for article in clustered_articles
        )
        for cluster_id in range(number_of_clusters)
    }

    output = {
        "claim": data.get("claim", ""),
        "input_file": str(input_path),
        "algorithm": "K-Means",
        "feature_extraction": "TF-IDF",
        "number_of_clusters": number_of_clusters,
        "articles_clustered": len(clustered_articles),
        "cluster_sizes": cluster_sizes,
        "warning": (
            "Clusters represent textual similarity, not truth, falsity, "
            "credibility, or independent corroboration. Cluster IDs are "
            "arbitrary labels, not confidence scores."
        ),
        "articles": clustered_articles
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n=== CLUSTERING RESULTS ===")

    for cluster_id in range(number_of_clusters):
        group = [
            article for article in clustered_articles
            if article["cluster_id"] == cluster_id
        ]

        print(f"\nCluster {cluster_id}: {len(group)} articles")

        for article in group[:5]:
            print(f"  - {article['title']}")
            print(f"    Source: {article['source']}")
            print(f"    Relevance: {article['relevance_score']}")

        if len(group) > 5:
            print(f"  ... and {len(group) - 5} more articles")

    print("\n=== SUMMARY ===")
    print(f"Articles clustered: {len(clustered_articles)}")
    print(f"Clusters created: {number_of_clusters}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
