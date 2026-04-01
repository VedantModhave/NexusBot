"""Tests for the hybrid retriever module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp.retriever import HybridRetriever


@pytest.fixture
def retriever():
    """Create a retriever instance and attempt to load the index."""
    r = HybridRetriever()
    return r


class TestRetrieverLoading:
    """Test retriever initialization and loading."""

    def test_retriever_creation(self):
        r = HybridRetriever()
        assert r is not None
        assert r.is_loaded is False

    def test_load_returns_bool(self, retriever):
        result = retriever.load()
        assert isinstance(result, bool)


class TestRetrieverSearch:
    """Test retrieval functionality (requires built index)."""

    @pytest.fixture(autouse=True)
    def setup_retriever(self, retriever):
        self.retriever = retriever
        self.index_available = retriever.load()

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index", "index.faiss")
        ),
        reason="FAISS index not built. Run scripts/build_index.py first."
    )
    def test_retrieve_returns_list(self):
        results = self.retriever.retrieve("What is the tuition fee?", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index", "index.faiss")
        ),
        reason="FAISS index not built."
    )
    def test_retrieve_top_k_limit(self):
        results = self.retriever.retrieve("scholarship information", top_k=3)
        assert len(results) <= 3

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index", "index.faiss")
        ),
        reason="FAISS index not built."
    )
    def test_retrieve_result_structure(self):
        results = self.retriever.retrieve("hostel facilities", top_k=5)
        if results:
            result = results[0]
            assert "text" in result
            assert "source_id" in result
            assert "category" in result
            assert "score" in result
            assert "bm25_score" in result
            assert "faiss_score" in result

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index", "index.faiss")
        ),
        reason="FAISS index not built."
    )
    def test_retrieve_scores_are_valid(self):
        results = self.retriever.retrieve("exam schedule", top_k=5)
        for r in results:
            assert 0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"
            assert 0 <= r["bm25_score"] <= 1.0
            assert 0 <= r["faiss_score"] <= 1.0

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index", "index.faiss")
        ),
        reason="FAISS index not built."
    )
    def test_retrieve_results_sorted_by_score(self):
        results = self.retriever.retrieve("library timings", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"
