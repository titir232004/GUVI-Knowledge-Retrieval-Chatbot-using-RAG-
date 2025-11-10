import os
import json
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

BASE_BLOG_URL = "https://www.guvi.in/blog/"
FAQ_URL = "https://www.guvi.in/faq/"
COURSES_URL = "https://www.guvi.in/courses"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Starting GUVI Full Content Scraper...")

# ------------------ SETUP SELENIUM ------------------
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-gpu")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)


def get_total_pages():
    """Detect total number of blog pages from pagination"""
    driver.get(BASE_BLOG_URL)
    time.sleep(4)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    pagination = soup.select("a.page-numbers")
    if pagination:
        numbers = [int(re.sub(r'\D', '', a.get_text())) for a in pagination if a.get_text().isdigit()]
        return max(numbers) if numbers else 1
    return 1


def get_blog_links(page_num):
    """Extract blog post URLs from a specific page"""
    url = BASE_BLOG_URL if page_num == 1 else f"{BASE_BLOG_URL}page/{page_num}/"
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "/blog/" in href and href.startswith("https://www.guvi.in/blog/") and href != BASE_BLOG_URL:
            links.append(href)
    return list(set(links))


def parse_full_page(url):
    """Extract full page content including all headings, paragraphs, and lists"""
    try:
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        title = soup.title.string.strip() if soup.title else "No Title"

        content_parts = []

        # Grab headings
        for h in soup.select("h1,h2,h3,h4,h5,h6"):
            text = h.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

        # Grab paragraphs
        for p in soup.select("p"):
            text = p.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

        # Grab list items
        for li in soup.select("li"):
            text = li.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

        # Grab other text containers
        for div in soup.select(".elementor-widget-container, .entry-content, article, .post-content"):
            text = div.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

        full_content = " ".join(content_parts).strip()
        return {"url": url, "title": title, "content": full_content}

    except Exception as e:
        print(f"❌ Error parsing {url}: {e}")
        return {"url": url, "title": "Error", "content": ""}


def get_all_faqs():
    """Scrape all FAQs from GUVI FAQ page"""
    driver.get(FAQ_URL)
    time.sleep(4)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    faqs = []

    for el in soup.select("h2, h3, .faq-question, .elementor-tab-title, .elementor-accordion-title"):
        q_text = el.get_text(strip=True)
        a_el = el.find_next_sibling(["div", "p"])
        a_text = a_el.get_text(" ", strip=True) if a_el else ""
        if q_text:
            faqs.append({"question": q_text, "answer": a_text})

    print(f"✅ Found {len(faqs)} FAQs")
    return faqs


if __name__ == "__main__":
    # Scrape blogs
    total_pages = get_total_pages()
    print(f"📄 Total blog pages: {total_pages}")

    all_links = []
    for i in range(1, total_pages + 1):
        links = get_blog_links(i)
        all_links.extend(links)
    all_links = list(set(all_links))
    print(f"📰 Total unique blog URLs: {len(all_links)}")

    all_blog_content = []
    for idx, link in enumerate(all_links):
        print(f"🧾 [{idx+1}/{len(all_links)}] Scraping: {link}")
        data = parse_full_page(link)
        if data["content"]:
            all_blog_content.append(data)

    faqs = get_all_faqs()
    courses_page = parse_full_page(COURSES_URL)

    # Combine everything
    all_docs = all_blog_content + \
               [{"url": "FAQ", "title": f["question"], "content": f["answer"]} for f in faqs] + \
               [courses_page]

    output_path = os.path.join(OUTPUT_DIR, "guvi_full_data_full.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    driver.quit()
    print(f"🎉 Full scraping completed! Saved → {output_path}")
