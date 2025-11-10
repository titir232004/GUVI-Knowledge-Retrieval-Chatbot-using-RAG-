import json
import re

def clean_text(text):
    """Clean text: remove emails, phone numbers, references, strange chars"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\S+@\S+\.\S+', '', text)  # emails
    text = re.sub(r'\+?\d[\d\s-]{7,}\d', '', text)  # phones
    text = re.sub(r'\[.*?\]', '', text)  # references [1]
    text = re.sub(r'[^A-Za-z0-9.,!?;:()\'"-]', ' ', text)  # strange chars
    return text

def chunk_text(text_list, chunk_size=300, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    for text in text_list:
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += (chunk_size - overlap)
    return chunks

def preprocess_guvi():
    with open("guvi_raw_content.json", "r", encoding="utf-8") as f:
        raw_content = json.load(f)

    all_texts = []
    for url, paragraphs in raw_content.items():
        for para in paragraphs:
            clean_para = clean_text(para)
            if len(clean_para) > 30:
                all_texts.append(clean_para)

    print(f"Total paragraphs before chunking: {len(all_texts)}")

    chunks = chunk_text(all_texts, chunk_size=300, overlap=50)
    print(f"Total chunks before deduplication: {len(chunks)}")

    seen = set()
    dedup_chunks = []
    for chunk in chunks:
        h = hash(chunk)
        if h not in seen:
            seen.add(h)
            dedup_chunks.append(chunk)

    print(f"Total deduplicated chunks: {len(dedup_chunks)}")

    with open("guvi_chunks.json", "w", encoding="utf-8") as f:
        json.dump(dedup_chunks, f, ensure_ascii=False, indent=4)

    print("Preprocessing complete! Saved to guvi_chunks.json")
    return dedup_chunks

if __name__ == "__main__":
    preprocess_guvi()
