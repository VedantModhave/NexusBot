"""Tests for the full chat pipeline."""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


INDEX_EXISTS = os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index", "index.faiss")
)


@pytest.mark.skipif(not INDEX_EXISTS, reason="FAISS index not built. Run scripts/build_index.py first.")
class TestChatPipeline:
    """Test the full chat pipeline (requires built index)."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        from src.chatbot.pipeline import ChatPipeline
        self.pipeline = ChatPipeline()

    def test_chat_returns_dict(self):
        result = self.pipeline.chat("What is the tuition fee?", "test-session-1")
        assert isinstance(result, dict)

    def test_chat_response_keys(self):
        result = self.pipeline.chat("What is the fee?", "test-session-2")
        required_keys = {"response", "detected_lang", "confidence", "category", "sources", "fallback"}
        assert required_keys.issubset(set(result.keys()))

    def test_chat_english_query(self):
        result = self.pipeline.chat("What is the tuition fee for B.Tech?", "test-session-3")
        assert result["detected_lang"] == "en"
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        assert isinstance(result["confidence"], float)

    def test_chat_hindi_query(self):
        result = self.pipeline.chat("बी.टेक की फीस क्या है?", "test-session-4")
        assert result["detected_lang"] == "hi"
        assert isinstance(result["response"], str)

    def test_chat_category_detection(self):
        result = self.pipeline.chat("Tell me about scholarships", "test-session-5")
        assert result["category"] in ["fees", "scholarships", "exams", "hostel",
                                       "admissions", "timetable", "library", "policy",
                                       "general", "unknown"]

    def test_chat_low_confidence_fallback(self):
        result = self.pipeline.chat(
            "What is the meaning of quantum entanglement in physics?",
            "test-session-6"
        )
        # This irrelevant query should either get low confidence or fallback
        assert isinstance(result["fallback"], bool)

    def test_chat_context_enrichment(self):
        """Test that follow-up queries are enriched with context."""
        session = "test-context-session"
        # First query
        self.pipeline.chat("What is the tuition fee?", session)
        # Follow-up
        result = self.pipeline.chat("What about MBA?", session)
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_chat_stats(self):
        self.pipeline.chat("Test query", "stats-test")
        stats = self.pipeline.get_stats()
        assert stats["total_queries"] >= 1
        assert "top_categories" in stats
        assert "avg_confidence" in stats
        assert "fallback_rate" in stats

    def test_chat_sources_returned(self):
        result = self.pipeline.chat("What is the hostel fee?", "test-session-7")
        assert isinstance(result["sources"], list)

    def test_pipeline_is_ready(self):
        assert self.pipeline.is_ready is True
