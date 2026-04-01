"""Conversation context manager for follow-up query enrichment."""


class ConversationContext:
    """Tracks last 5 turns per session and enriches follow-up queries."""

    PRONOUNS = {
        "it", "this", "that", "they", "these", "those",
        "the deadline", "the fee", "the exam", "the form",
    }

    def __init__(self):
        self.sessions: dict = {}

    def enrich_query(self, query: str, session_id: str) -> str:
        """Treat every query independently as requested."""
        return query.strip()

    def update(self, session_id: str, query: str, response: str, category: str):
        """Add a turn and keep only the last 5."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "query": query,
            "response": response,
            "category": category,
        })
        self.sessions[session_id] = self.sessions[session_id][-5:]
