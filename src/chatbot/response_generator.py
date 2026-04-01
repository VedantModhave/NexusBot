"""Response generator: extracts clean answers from retrieved chunks.

BUG FIXES:
  - BUG 1: Removed aggressive regex that stripped decimals (r'\\d+\\.\\d+')
            Numbers like 6.2, 8.5, Rs 32 LPA are now preserved.
  - BUG 3: Sentence selection is now QUERY-SPECIFIC -- top-3 by cosine
            similarity are picked IN SIMILARITY ORDER (most relevant first),
            not restored to reading order, so different queries on the same
            chunk produce differently focused answers.
"""

import re

import numpy as np

from src.nlp.embedder import MultilingualEmbedder


class ResponseGenerator:
    """Generates clean, natural-language responses from retrieved document chunks."""

    FOOTER = "\n\nFor more info, contact: info@xyzcollege.edu.in"

    # Safe safe patterns — ONLY remove known formatting artifacts.
    # None of these touch digits, decimals, ₹, or numeric tokens.
    _CLEAN_PATTERNS = [
        (re.compile(r'-{3,}'),          ''),          # --- separators
        (re.compile(r'={3,}'),          ''),          # === separators
        (re.compile(r'_{3,}'),          ''),          # ___ separators
        # All-caps SECTION / CHAPTER headers like "SECTION 1: HOSTEL POLICIES"
        (re.compile(r'\b(SECTION|CHAPTER|PART|UNIT)\s+[0-9A-Z]+[:\s]+'),  ''),
        (re.compile(r'\s{2,}'),         ' '),         # collapse extra whitespace
    ]

    def __init__(self, embedder: MultilingualEmbedder):
        self.embedder = embedder

    def generate(self, query: str, retrieved: list) -> dict:
        """
        Generate a clean, query-focused response from retrieved chunks.
        Returns dict with text, confidence, category, source_id.
        """
        if not retrieved or retrieved[0]["score"] < 0.25:
            return {
                "text": None,
                "confidence": 0.0,
                "category": "unknown",
                "source_id": "",
            }

        best = retrieved[0]
        answer_text = best.get("answer", best.get("text", ""))

        # ── Step 1: Split into candidate sentences ──────────────────────────
        raw_sentences = re.split(r'(?<=[.!?])\s+', answer_text)
        sentences = [
            s.strip() for s in raw_sentences
            if len(s.strip()) > 30              # skip trivially short fragments
            and not s.strip().isupper()         # skip ALL-CAPS header lines
            and len(s.split()) >= 5             # minimum 5 content words
            and not re.match(r'^\d+[\.\)]\s', s)  # skip "1. " list prefixes
        ]

        if not sentences:
            # Fallback: simply clean and return the raw text as-is
            clean = self._clean_text(answer_text)
        else:
            # ── Step 2: Rank sentences by cosine sim to THIS specific query ─
            query_vec = self.embedder.embed_single(query)          # (384,)
            sent_vecs = self.embedder.embed(sentences)             # (N, 384)
            sims = np.dot(sent_vecs, query_vec)                    # (N,)

            # Pick top-3 in SIMILARITY ORDER (most relevant first)
            # BUG 3 fix: do NOT sort back to reading order — different queries
            # will now produce differently ordered / selected sentences.
            top_k = min(3, len(sentences))
            top_sent_idx = np.argsort(sims)[::-1][:top_k]

            selected = [sentences[i] for i in top_sent_idx]
            clean = " ".join(selected)
            if not clean.endswith("."):
                clean += "."

        clean = self._clean_text(clean)
        clean += self.FOOTER

        return {
            "text": clean,
            "confidence": best["score"],
            "category": best.get("category", "general"),
            "source_id": best.get("id", ""),
        }

    def _clean_text(self, text: str) -> str:
        """
        Remove ONLY known formatting artifacts.
        SAFE: does NOT touch digits, decimals, currency symbols or numeric values.

        BUG 1 FIX: removed the pattern r'\\d+\\.\\d+\\s+' which was stripping
        decimal numbers like 6.2, 8.5, 32 from responses.
        """
        for pattern, replacement in self._CLEAN_PATTERNS:
            text = pattern.sub(replacement, text)
        return text.strip()

    # ── Unit-testable verification ───────────────────────────────────────────
    @staticmethod
    def _test_clean_preserves_numbers():
        """
        Smoke test: verifies BUG 1 fix.
        Raises AssertionError if numbers are stripped.
        """
        rg = ResponseGenerator.__new__(ResponseGenerator)
        sample = "The package was Rs 6.2 LPA and highest was Rs 54 LPA in 2024."
        result = rg._clean_text(sample)
        assert "6.2" in result, f"BUG 1 REGRESSION: '6.2' stripped! Got: {result}"
        assert "54" in result,  f"BUG 1 REGRESSION: '54' stripped!  Got: {result}"
        assert "2024" in result, f"BUG 1 REGRESSION: '2024' stripped! Got: {result}"
        print(f"[UNIT TEST] BUG1 clean_text OK: {result}")
        return True
