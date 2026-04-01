"""Hybrid BM25 + FAISS retriever with combined-score re-ranking.

BUG 2 FIXES:
  - Added CONFIDENCE_THRESHOLD (0.35): low-confidence results rejected
    before they reach the generator â€” prevents wrong-but-confident answers.
  - Added TAG OVERLAP CHECK: if the top chunk's tags share zero nouns with
    the query AND score < 0.50, the chunk is rejected and empty list returned.
  - Added QUERY EXPANSION: short queries expanded with a synonym dict so
    "gym" also searches for "gymnasium fitness exercise workout".
  - Added SCORE TRANSPARENCY logging: every retrieval prints
    [RETRIEVER] query="..." â†’ chunk="..." score=X.XX method=HYBRID
"""

import json
import os

import faiss
import numpy as np
import string
from rank_bm25 import BM25Okapi

from src.nlp.embedder import MultilingualEmbedder

# â”€â”€ Synonym expansion dict (prevents gym-type misses) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SYNONYMS: dict[str, list[str]] = {
    "gym":          ["gymnasium", "fitness", "exercise", "workout", "weights"],
    "gymnasium":    ["gym", "fitness", "exercise", "workout"],
    "library":      ["lib", "books", "reading room", "digital resources"],
    "canteen":      ["food court", "cafeteria", "mess", "dining"],
    "hostel":       ["dormitory", "accommodation", "warden", "room"],
    "sports":       ["ground", "playground", "athletics", "field"],
    "auditorium":   ["hall", "seminar hall", "audi", "amphitheatre"],
    "lab":          ["laboratory", "practical", "workshop"],
    "bus":          ["transport", "route", "pickup", "commute"],
    "fee":          ["fees", "tuition", "payment", "challan"],
    "scholarship":  ["merit", "grant", "waiver", "financial aid"],
    "placement":    ["job", "career", "recruiter", "campus drive", "package"],
    "exam":         ["examination", "test", "assessment", "result"],
    "wifi":         ["internet", "network", "connectivity", "broadband"],
    "clinic":       ["health centre", "doctor", "medical", "sick room"],
    "complaint":    ["grievance", "redressal", "ombudsman"],
}

# â”€â”€ Minimum confidence threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CONFIDENCE_THRESHOLD = 0.35   # below this â†’ return empty (pipeline uses fallback)
TAG_REJECT_THRESHOLD  = 0.50  # if tags don't overlap AND score < this â†’ reject


def _expand_query(query: str) -> str:
    """Append synonym tokens to a short query for better BM25 recall."""
    words = query.lower().split()
    extra: list[str] = []
    for w in words:
        if w in SYNONYMS:
            extra.extend(SYNONYMS[w])
    if extra:
        expanded = query + " " + " ".join(extra)
        print(f"[RETRIEVER] Query expanded: '{query}' -> '{expanded}'")
        return expanded
    return query

def _tokenize(text: str) -> list[str]:
    """Clean text by removing punctuation and common stopwords."""
    STOPWORDS = {
        "is", "are", "the", "a", "an", "in", "on", "at", "for", "of",
        "and", "or", "what", "how", "where", "when", "do", "does", "can",
        "there", "any", "have", "has", "i", "my", "me", "we", "our",
        "please", "tell", "about", "which", "with", "to", "be"
    }
    clean = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return [w for w in clean.split() if w not in STOPWORDS]


def _query_nouns(query: str) -> set[str]:
    """Extract meaningful (non-stopword) tokens from the query."""
    STOPWORDS = {
        "is", "are", "the", "a", "an", "in", "on", "at", "for", "of",
        "and", "or", "what", "how", "where", "when", "do", "does", "can",
        "there", "any", "have", "has", "i", "my", "me", "we", "our",
        "please", "tell", "about", "which", "with", "to", "be"
    }
    return {w for w in query.lower().split() if w not in STOPWORDS and len(w) > 2}


def _tag_overlap(query_nouns: set[str], tags: list[str]) -> int:
    """Count how many query nouns appear in the chunk's tags list."""
    tag_set = {t.lower() for t in tags}
    # also expand tags through synonym dict for reverse lookup
    expanded_tags: set[str] = set(tag_set)
    for t in tag_set:
        if t in SYNONYMS:
            expanded_tags.update(SYNONYMS[t])
    return len(query_nouns & expanded_tags)


class HybridRetriever:
    """Combines keyword (BM25) and semantic (FAISS) retrieval."""

    def __init__(self, embedder: MultilingualEmbedder):
        self.embedder = embedder

        base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "faiss_index")
        index_path = os.path.abspath(os.path.join(base, "index.faiss"))
        meta_path  = os.path.abspath(os.path.join(base, "metadata.json"))

        print(f"[Retriever] Loading FAISS index from {index_path}")
        self.faiss_index = faiss.read_index(index_path)
        print(f"[Retriever] FAISS index loaded: {self.faiss_index.ntotal} vectors")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Build BM25 index
        corpus    = [entry["text"] for entry in self.metadata]
        tokenized = [_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[Retriever] BM25 index built: {len(corpus)} documents")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Retrieve top-k documents with hybrid BM25 + FAISS scoring.
        Returns empty list if best score < CONFIDENCE_THRESHOLD or
        if tag overlap check fails â€” caller (pipeline) then triggers fallback.
        """
        n = len(self.metadata)
        if n == 0:
            return []

        # ── 1. Query expansion for short / keyword queries ──────────────────────────
        expanded_query = _expand_query(query)

        # ── 2. BM25 scores (use expanded query) ─────────────────────────────
        tokenized_query = _tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_max    = max(bm25_scores.max(), 3.0)  # Bound denominator so noise doesn't inflate
        bm25_norm   = bm25_scores / bm25_max

        # ── 3. FAISS scores (original query — semantic) ─────────────────────────────
        query_vec  = self.embedder.embed_single(query).reshape(1, -1).astype("float32")
        distances, indices = self.faiss_index.search(query_vec, n)

        faiss_norm = np.zeros(n)
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < n:
                faiss_norm[idx] = float(distances[0][rank])

        # ── 4. Dynamic BM25/FAISS weighting by query length ─────────────────────────
        query_words = query.lower().split()
        if len(query_words) <= 3:
            bm25_weight, faiss_weight = 0.6, 0.4
            method = "BM25-heavy"
        elif len(query_words) >= 10:
            bm25_weight, faiss_weight = 0.25, 0.75
            method = "FAISS-heavy"
        else:
            bm25_weight, faiss_weight = 0.4, 0.6
            method = "HYBRID"

        combined = bm25_weight * bm25_norm + faiss_weight * faiss_norm

        # ── 5. Boost FAQ entries by 15% ──────────────────────────────────────
        for i, entry in enumerate(self.metadata):
            if entry.get("source") in ("faq", "faq_hi"):
                combined[i] *= 1.15

        # ── 6. Rank ─────────────────────────────────────────────────────────────────
        top_indices   = np.argsort(combined)[::-1][:top_k]
        best_idx      = int(top_indices[0])
        best_score    = float(combined[best_idx])
        best_chunk_id = self.metadata[best_idx].get("id", f"idx_{best_idx}")

        # â”€â”€ 7. Score transparency log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print(
            f'[RETRIEVER] query="{query}" -> chunk="{best_chunk_id}" '
            f"score={best_score:.4f} method={method}"
        )

        # â”€â”€ 8. CONFIDENCE THRESHOLD: reject if too uncertain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if best_score < CONFIDENCE_THRESHOLD:
            print(
                f"[RETRIEVER] REJECTED â€” score {best_score:.4f} < "
                f"threshold {CONFIDENCE_THRESHOLD}. Triggering fallback."
            )
            return []

        # â”€â”€ 9. TAG OVERLAP CHECK: reject semantic false positives â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        query_nouns = _query_nouns(query)
        if query_nouns:
            best_entry = self.metadata[best_idx]
            overlap    = _tag_overlap(query_nouns, best_entry.get("tags", []))
            if overlap == 0 and best_score < TAG_REJECT_THRESHOLD:
                print(
                    f"[RETRIEVER] TAG-REJECTED â€” zero tag overlap for "
                    f'query_nouns={query_nouns} on chunk "{best_chunk_id}" '
                    f"(score={best_score:.4f}). Triggering fallback."
                )
                return []

        # â”€â”€ 10. Build result list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        results = []
        for idx in top_indices:
            entry = self.metadata[idx].copy()
            entry["score"]       = float(combined[idx])
            entry["bm25_score"]  = float(bm25_norm[idx])
            entry["faiss_score"] = float(faiss_norm[idx])
            results.append(entry)
        return results
