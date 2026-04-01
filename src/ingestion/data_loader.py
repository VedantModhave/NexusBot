"""
Data Loader Module
Loads FAQs from JSON and text documents, chunks them for indexing.
"""

import json
import os
import re
from typing import List, Dict, Any


class DataLoader:
    """Loads and chunks FAQ and document data for the RAG pipeline."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self.data_dir = os.path.abspath(data_dir)
        self.faqs_path = os.path.join(self.data_dir, "faqs.json")
        self.documents_dir = os.path.join(self.data_dir, "documents")

    def load_faqs(self) -> List[Dict[str, Any]]:
        """Load FAQ entries from faqs.json."""
        with open(self.faqs_path, "r", encoding="utf-8") as f:
            faqs = json.load(f)
        return faqs

    def load_documents(self) -> List[Dict[str, str]]:
        """Load all .txt documents from the documents directory."""
        documents = []
        if not os.path.exists(self.documents_dir):
            return documents

        for filename in os.listdir(self.documents_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.documents_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append({
                    "filename": filename,
                    "text": text,
                    "source": filepath
                })
        return documents

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks by sentence boundaries."""
        sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Keep overlap
                overlap_chunk = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def prepare_corpus(self) -> List[Dict[str, Any]]:
        """
        Prepare the full corpus for indexing.
        Returns a list of dicts with keys: id, text, category, source, tags
        """
        corpus = []

        # Process FAQs
        faqs = self.load_faqs()
        for faq in faqs:
            # Add English Q+A as one document
            en_text = f"Question: {faq['question_en']}\nAnswer: {faq['answer_en']}"
            corpus.append({
                "id": f"{faq['id']}_en",
                "text": en_text,
                "category": faq["category"],
                "source": "faqs.json",
                "tags": faq.get("tags", []),
                "question": faq["question_en"],
                "answer": faq["answer_en"]
            })

            # Add Hindi Q+A as one document
            hi_text = f"Question: {faq['question_hi']}\nAnswer: {faq['answer_hi']}"
            corpus.append({
                "id": f"{faq['id']}_hi",
                "text": hi_text,
                "category": faq["category"],
                "source": "faqs.json",
                "tags": faq.get("tags", []),
                "question": faq["question_hi"],
                "answer": faq["answer_hi"]
            })

        # Process text documents
        documents = self.load_documents()
        for doc in documents:
            chunks = self.chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                corpus.append({
                    "id": f"doc_{doc['filename']}_{i}",
                    "text": chunk,
                    "category": "policy",
                    "source": doc["filename"],
                    "tags": [],
                    "question": "",
                    "answer": chunk
                })

        return corpus


if __name__ == "__main__":
    loader = DataLoader()
    corpus = loader.prepare_corpus()
    print(f"Loaded {len(corpus)} documents into corpus")
    for item in corpus[:3]:
        print(f"  [{item['id']}] {item['text'][:80]}...")
