import os
import time
import json
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from readability import Document
from tqdm import tqdm
from collections import OrderedDict

BASE_URL = "https://www.guvi.in"
BLOG_LISTING = "https://www.guvi.in/blogs"
FAQ_LISTING = "https://www.guvi.in/faqs"

OUT_RAW = "raw/guvi_raw_content.json"
OUT_CLEAN = "processed/guvi_clean_text.json"

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def safe_get(url: str, max_retries: int = 3, backoff: float = 1.0) -> requests.Response:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                return resp
            logging.warning("Non-200 [%s] for %s", resp.status_code, url)
        except Exception as e:
            logging.warning("Request error for %s: %s", url, e)
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts")


def find_candidate_links(listing_url: str, keywords: List[str]) -> List[str]:
    """Collect unique absolute links from listing page that contain any keyword."""
    resp = safe_get(listing_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # ignore anchors, mailto, javascript
        if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        # make absolute
        full = urljoin(BASE_URL, href)
        # only same-domain
        parsed = urlparse(full)
        if parsed.netloc not in ("www.guvi.in", "guvi.in"):
            continue
        # include if any keyword present in path
        if any(k.lower() in full.lower() for k in keywords):
            urls.add(full.split("?")[0].rstrip("/"))
    return sorted(urls)


def extract_main_text(html: str) -> List[str]:
    """
    Use readability to get the main article, then break into paragraphs and headings.
    Return list of cleaned paragraphs (no short noise).
    """
    doc = Document(html)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, "html.parser")
    # pick meaningful tags
    elements = soup.find_all(["p", "h1", "h2", "h3", "li"])
    raw_paras = []
    for el in elements:
        text = el.get_text(separator=" ").strip()
        # basic filter: length and not just numbers/links
        if len(text) < 40:
            continue
        raw_paras.append(" ".join(text.split()))
    # deduplicate while preserving order
    seen = set()
    filtered = []
    for p in raw_paras:
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        filtered.append(p)
    return filtered


def crawl_and_extract(blog_listing=BLOG_LISTING, faq_listing=FAQ_LISTING) -> Dict[str, List[str]]:
    """Crawl listings and extract all candidate pages' main paragraphs."""
    logging.info("Collecting blog links from %s", blog_listing)
    blog_links = find_candidate_links(blog_listing, keywords=["blog", "/blog/"])
    logging.info("Found %d blog candidate links", len(blog_links))

    logging.info("Collecting FAQ links from %s", faq_listing)
    faq_links = find_candidate_links(faq_listing, keywords=["faq", "/faq/"])
    logging.info("Found %d faq candidate links", len(faq_links))

    all_links = sorted(set(blog_links + faq_links))
    logging.info("Total unique candidate pages: %d", len(all_links))

    results = OrderedDict()
    for url in tqdm(all_links, desc="Scraping pages"):
        try:
            resp = safe_get(url)
            paras = extract_main_text(resp.text)
            if paras:
                results[url] = paras
            else:
                logging.info("No main text found for %s", url)
        except Exception as e:
            logging.error("Failed to scrape %s: %s", url, e)
        time.sleep(0.6)  # polite pause
    return results


def join_and_clean(paragraphs: List[str]) -> str:
    """
    Join paragraphs into one cleaned text string.
    Removes duplicate sentences (naive) and collapses whitespace.
    """
    if not paragraphs:
        return ""
    joined = " \n\n ".join(paragraphs)
    # basic de-duplication of exact sentences
    sentences = []
    seen = set()
    for part in joined.split("."):
        s = part.strip()
        if not s:
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        sentences.append(s)
    cleaned = ". ".join(sentences).strip()
    if not cleaned.endswith("."):
        cleaned += "."
    return " ".join(cleaned.split())


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    logging.info("Starting GUVI scraping (blogs + faqs)")
    results = crawl_and_extract()
    if not results:
        logging.error("No pages scraped. Exiting.")
        return

    logging.info("Saving raw paragraphs to %s", OUT_RAW)
    save_json(results, OUT_RAW)

    # create cleaned single text per url
    cleaned = {}
    for url, paras in results.items():
        cleaned[url] = join_and_clean(paras)

    logging.info("Saving cleaned texts to %s", OUT_CLEAN)
    save_json(cleaned, OUT_CLEAN)
    logging.info("Done. Pages scraped: %d", len(results))


if __name__ == "__main__":
    main()
