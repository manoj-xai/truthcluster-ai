
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
    )
}

TIMEOUT = 15
MINIMUM_WORDS = 50


def is_google_news_url(url):
    """Check whether a URL points to Google News."""
    host = urlparse(url).netloc.lower().split(":")[0]
    return host == "news.google.com" or host.endswith(".news.google.com")


def decode_google_news_url(url):
    """Resolve a Google News redirect to its publisher URL."""
    if not is_google_news_url(url):
        return url, ""

    try:
        result = gnewsdecoder(
            url,
            interval=1,
            timeout=TIMEOUT
        )

        if (
            isinstance(result, dict)
            and result.get("success") is True
            and result.get("decoded_url")
        ):
            decoded_url = result["decoded_url"]

            if (
                decoded_url.startswith(("http://", "https://"))
                and not is_google_news_url(decoded_url)
            ):
                return decoded_url, ""

        return "", "Google News URL could not be decoded."

    except Exception as error:
        return "", f"Decoder error: {error}"


def get_page(url):
    """Download an HTML page."""
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True
        )

        if response.status_code != 200:
            return None, f"HTTP status: {response.status_code}"

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            return None, "The response was not an HTML page."

        return response, ""

    except requests.RequestException as error:
        return None, f"Download error: {error}"


def extract_text(html):
    """Extract readable article text."""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False
    )

    if text and len(text.split()) >= MINIMUM_WORDS:
        return text.strip(), "trafilatura"

    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script", "style", "nav", "footer",
        "header", "aside", "form"
    ]):
        element.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(" ", strip=True)

    if len(text.split()) >= MINIMUM_WORDS:
        return text, "beautifulsoup_fallback"

    return "", "insufficient_text"


def process_article(article):
    """Resolve, download, and extract one article."""
    original_url = article.get("url", "")

    result = {
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "date": article.get("date", ""),
        "google_news_url": original_url,
        "resolved_url": "",
        "description": article.get("description", ""),
        "article_text": "",
        "extraction_status": "not_attempted",
        "extraction_method": "",
        "error": ""
    }

    if not original_url.startswith(("http://", "https://")):
        result["extraction_status"] = "invalid_url"
        result["error"] = "The article URL is invalid."
        return result

    publisher_url, decode_error = decode_google_news_url(original_url)

    if not publisher_url:
        result["extraction_status"] = "publisher_url_unresolved"
        result["error"] = decode_error
        return result

    result["resolved_url"] = publisher_url

    response, download_error = get_page(publisher_url)

    if response is None:
        result["extraction_status"] = "download_failed"
        result["error"] = download_error
        return result

    final_url = response.url
    result["resolved_url"] = final_url

    if is_google_news_url(final_url):
        result["extraction_status"] = "publisher_url_unresolved"
        result["error"] = "The request ended at Google News."
        return result

    soup = BeautifulSoup(response.text, "html.parser")
    canonical = soup.find("link", rel="canonical")

    if canonical and canonical.get("href"):
        canonical_url = requests.compat.urljoin(
            final_url, canonical["href"]
        )

        if canonical_url.startswith(("http://", "https://")):
            result["resolved_url"] = canonical_url

    text, method = extract_text(response.text)
    result["extraction_method"] = method

    if not text:
        result["extraction_status"] = "insufficient_text"
        result["error"] = (
            "The page did not contain enough readable article text."
        )
        return result

    result["article_text"] = text
    result["extraction_status"] = "success"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract article text from collected news evidence."
    )
    parser.add_argument(
        "--input",
        default="evidence_results.json",
        help="Path to the evidence search JSON file."
    )
    parser.add_argument(
        "--output",
        default="article_content.json",
        help="Path to save extracted article content."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(
            f"Error: Input file not found: {input_path}\n"
            "Run evidence_search.py first or check the --input path."
        )

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    articles = data.get("articles", [])

    if not articles:
        raise SystemExit(f"No articles found in {input_path}.")

    results = []

    print(f"Claim: {data.get('claim', '')}")
    print(f"Attempting to extract {len(articles)} articles...\n")

    for index, article in enumerate(articles, start=1):
        result = process_article(article)
        results.append(result)

        print(
            f"[{index}/{len(articles)}] "
            f"{result['extraction_status']} - "
            f"{result['title'][:75]}"
        )

    successful = sum(
        item["extraction_status"] == "success"
        for item in results
    )

    output = {
        "claim": data.get("claim", ""),
        "articles_processed": len(results),
        "successful_extractions": successful,
        "failed_or_incomplete": len(results) - successful,
        "warning": (
            "Extracted article text is evidence to review, not proof of truth. "
            "Descriptions are not treated as full article text. "
            "Failed extraction does not indicate that a claim is false."
        ),
        "articles": results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Articles processed: {len(results)}")
    print(f"Successful extractions: {successful}")
    print(f"Failed or incomplete: {len(results) - successful}")
    print(f"Saved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
