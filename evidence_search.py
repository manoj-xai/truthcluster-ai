
import json
import argparse
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from html import unescape
from html.parser import HTMLParser

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_RSS_URL = "https://news.google.com/rss/search"


def search_gdelt(claim):
    params = {
        "query": claim,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 10,
        "timespan": "3months",
        "sort": "hybridrel",
    }

    response = requests.get(
        GDELT_URL,
        params=params,
        headers={"User-Agent": "FakeNewsVerifier/1.0"},
        timeout=20,
    )

    if response.status_code == 429:
        print("GDELT rate-limited this request.")
        return None

    response.raise_for_status()
    data = response.json()

    return [
        {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": article.get("domain", "Unknown"),
            "date": article.get("seendate", "Unknown"),
            "provider": "GDELT",
        }
        for article in data.get("articles", [])
    ]


def search_google_news(query):
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            text = data.strip()
            if text:
                self.parts.append(text)

    response = requests.get(
        GOOGLE_RSS_URL,
        params={
            "q": query,
            "hl": "en-IN",
            "gl": "IN",
            "ceid": "IN:en",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    articles = []

    for item in root.findall(".//item"):
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        description_html = item.findtext("description", default="")

        parser = TextExtractor()
        parser.feed(unescape(description_html))
        description = " ".join(parser.parts)

        source_element = item.find("source")
        source = (
            source_element.text.strip()
            if source_element is not None and source_element.text
            else "Unknown"
        )

        articles.append({
            "title": title,
            "url": link,
            "source": source,
            "date": item.findtext("pubDate", default="").strip(),
            "description": description,
            "provider": "Google News RSS",
        })

    return articles


def search_claim(claim):
    articles = []

    try:
        articles = search_gdelt(claim) or []
    except (requests.RequestException, ValueError, ET.ParseError) as error:
        print(f"GDELT search failed: {error}")

    if not articles:
        print("Trying Google News RSS...")

        try:
            articles = search_google_news(claim)
        except (requests.RequestException, ET.ParseError) as error:
            print(f"Google News search failed: {error}")

    return articles


def save_results(claim, articles, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "claim": claim,
        "searched_at": datetime.now().isoformat(timespec="seconds"),
        "article_count": len(articles),
        "articles": articles,
    }

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Retrieved {len(articles)} articles.")
    print(f"Saved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Search news evidence for a claim."
    )
    parser.add_argument(
        "--claim",
        help="Claim to search. If omitted, interactive input is used.",
    )
    parser.add_argument(
        "--output",
        default="evidence_results.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    claim = (args.claim or input(
        "Enter a news claim to investigate: "
    )).strip()

    if not claim:
        print("Please enter a claim.")
        return

    articles = search_claim(claim)

    if not articles:
        print("No articles retrieved. Try again later or change the claim.")
        return

    save_results(claim, articles, Path(args.output))
    print("Remember: search results alone cannot verify a claim.")


if __name__ == "__main__":
    main()
