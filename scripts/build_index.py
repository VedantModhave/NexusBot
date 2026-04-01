"""Build FAISS index + metadata from FAQs, policy documents, and PDFs."""

import json
import os
import sys
from collections import Counter

import faiss
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.nlp.embedder import MultilingualEmbedder
from src.ingestion.pdf_parser import PDFParser


def infer_category(text: str) -> str:
    """Infer category from policy chunk text."""
    t = text.lower()
    if any(w in t for w in ["placement", "recruiter", "lpa", "package", "internship", "campus drive"]):
        return "placements"
    if any(w in t for w in ["fee", "tuition", "payment", "refund", "challan"]):
        return "fees"
    if any(w in t for w in ["scholarship", "merit", "ebc", "sc/st", "financial aid"]):
        return "scholarship"
    if any(w in t for w in ["exam", "assessment", "revaluation", "hall ticket", "result"]):
        return "exams"
    if any(w in t for w in ["attendance", "leave", "absence", "condonation"]):
        return "attendance"
    if any(w in t for w in ["hostel", "warden", "mess", "room", "accommodation"]):
        return "hostel"
    if any(w in t for w in ["library", "book", "issue", "fine", "digital", "journal"]):
        return "library"
    if any(w in t for w in ["transport", "bus", "route", "pickup", "commute"]):
        return "transport"
    if any(w in t for w in ["grievance", "complaint", "ombudsman", "redressal"]):
        return "grievance"
    if any(w in t for w in ["timetable", "schedule", "calendar", "holiday", "class timing"]):
        return "timetable"
    return "general"


def build_index():
    print("=" * 60)
    print("  NexusBot Index Builder")
    print("=" * 60)

    data_dir = os.path.join(project_root, "data")
    index_dir = os.path.join(data_dir, "faiss_index")
    os.makedirs(index_dir, exist_ok=True)

    all_docs = []

    # 1. Load FAQs
    faqs_path = os.path.join(data_dir, "faqs.json")
    with open(faqs_path, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    faq_count = 0
    for entry in faqs:
        # English doc
        all_docs.append({
            "id": entry["id"],
            "text": f"{entry['question_en']} {entry['answer_en']}",
            "question": entry["question_en"],
            "answer": entry["answer_en"],
            "category": entry["category"],
            "source": "faq",
            "tags": entry.get("tags", []),
        })
        faq_count += 1

        # Hindi doc
        if entry.get("question_hi") and entry.get("answer_hi"):
            all_docs.append({
                "id": entry["id"] + "_hi",
                "text": f"{entry['question_hi']} {entry['answer_hi']}",
                "question": entry["question_hi"],
                "answer": entry["answer_hi"],
                "category": entry["category"],
                "source": "faq_hi",
                "tags": entry.get("tags", []),
            })
            faq_count += 1

    print(f"[1/4] Loaded {faq_count} FAQ documents")

    # 2. Load policy document with smarter chunking
    doc_count = 0
    policy_path = os.path.join(data_dir, "documents", "college_policy.txt")
    if os.path.exists(policy_path):
        paragraphs = []
        current_chunk = []
        with open(policy_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        if len(chunk_text) > 80:
                            paragraphs.append(chunk_text)
                        current_chunk = []
                else:
                    current_chunk.append(line)
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) > 80:
                paragraphs.append(chunk_text)

        for i, chunk in enumerate(paragraphs):
            all_docs.append({
                "id": f"doc_{i}",
                "text": chunk,
                "question": "",
                "answer": chunk,
                "category": infer_category(chunk),
                "source": "policy",
                "tags": [],
            })
            doc_count += 1

    print(f"[2/4] Loaded {doc_count} policy chunks")

    # 2b. Load PDFs from documents folder
    pdf_count = 0
    documents_dir = os.path.join(data_dir, "documents")
    pdf_parser = PDFParser()
    if os.path.exists(documents_dir):
        try:
            pdf_pages = pdf_parser.parse_directory(documents_dir)
            for page in pdf_pages:
                # Skip if already loaded as .txt
                if page["source"].endswith(".pdf"):
                    all_docs.append({
                        "id": f"pdf_{pdf_count}",
                        "text": page["text"],
                        "question": "",
                        "answer": page["text"],
                        "category": infer_category(page["text"]),
                        "source": f"pdf_page_{page['page_number']}",
                        "tags": [],
                    })
                    pdf_count += 1
        except Exception as e:
            print(f"[2b] PDF parsing warning: {e}")

    if pdf_count > 0:
        print(f"[2b] Loaded {pdf_count} PDF pages")
    print(f"       Total documents: {len(all_docs)}")

    # 3. Generate embeddings
    print("[3/4] Generating embeddings...")
    embedder = MultilingualEmbedder()
    texts = [doc["text"] for doc in all_docs]
    embeddings = embedder.embed(texts)
    print(f"       Embeddings shape: {embeddings.shape}")

    # 4. Build and save FAISS index
    print("[4/4] Building FAISS index...")
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    # Category breakdown
    cat_counts = Counter(doc["category"] for doc in all_docs)

    print()
    print("=" * 60)
    print(f"  [OK] Index rebuilt: {len(all_docs)} total documents")
    print(f"  [FAQ] FAQ entries:    {faq_count}")
    print(f"  [DOC] Policy chunks:  {doc_count}")
    print(f"  [PDF] PDF pages:      {pdf_count}")
    print(f"  [TAG] Categories:     {dict(cat_counts)}")
    print("=" * 60)


if __name__ == "__main__":
    build_index()
