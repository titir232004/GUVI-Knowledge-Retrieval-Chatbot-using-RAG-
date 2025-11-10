import requests
from bs4 import BeautifulSoup
from readability.readability import Document
import json
from urllib.parse import urljoin

BASE_URL = "https://www.guvi.in"

def clean_text(text):
    """Basic cleaning for readability"""
    return text.strip()

def scrape_page(url):
    """Scrape main content from a page"""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return []

    doc = Document(response.text)
    html_content = doc.summary()
    soup = BeautifulSoup(html_content, "html.parser")
    elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    content = [clean_text(el.get_text()) for el in elements if len(clean_text(el.get_text())) > 30]

    seen = set()
    final_content = []
    for c in content:
        if c not in seen:
            seen.add(c)
            final_content.append(c)
    return final_content

def get_all_urls(listing_url, keyword):
    """Get all URLs containing a keyword (like 'blog' or 'faq')"""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(listing_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {listing_url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    urls = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        if keyword in href:
            full_url = urljoin(BASE_URL, href)
            urls.add(full_url)
    return list(urls)

def scrape_guvi_all():
    """Scrape blogs + FAQs and save raw content"""
    blog_listing_url = "https://www.guvi.in/blogs"
    faq_listing_url = "https://www.guvi.in/faqs"  

    blog_urls = get_all_urls(blog_listing_url, "blog")
    faq_urls = get_all_urls(faq_listing_url, "faq")

    all_content = {}

    for url in blog_urls + faq_urls:
        print(f"Scraping: {url}")
        content = scrape_page(url)
        if content:
            all_content[url] = content

    # Save raw content
    with open("guvi_raw_content.json", "w", encoding="utf-8") as f:
        json.dump(all_content, f, ensure_ascii=False, indent=4)
    print(f"Scraped {len(all_content)} pages and saved to guvi_raw_content.json")

if __name__ == "__main__":
    scrape_guvi_all()
