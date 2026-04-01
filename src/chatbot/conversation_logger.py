"""Conversation logging for ML dataset generation and analytics."""

import json
import os
import datetime
from typing import List, Dict, Any


class ConversationLogger:
    """Logs full conversations including queries, responses, metadata for training."""

    def __init__(self, log_path: str = None):
        if log_path is None:
            log_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "conversations.jsonl")
            )
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_turn(
        self,
        session_id: str,
        turn_num: int,
        query: str,
        query_lang: str,
        response: str,
        response_lang: str,
        intent: str = None,
        category: str = None,
        confidence: float = 0.0,
        method: str = "RAG",
        is_fallback: bool = False,
    ):
        """Log a single conversation turn."""
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "session_id": session_id,
            "turn_number": turn_num,
            "query": query,
            "query_lang": query_lang,
            "response": response,
            "response_lang": response_lang,
            "intent": intent,
            "category": category,
            "confidence": confidence,
            "method": method,
            "is_fallback": is_fallback,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ConversationLogger] Write failed: {e}")

    def export_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Export all turns for a session from log."""
        session_turns = []
        if not os.path.exists(self.log_path):
            return session_turns
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("session_id") == session_id:
                            session_turns.append(entry)
        except Exception as e:
            print(f"[ConversationLogger] Export failed: {e}")
        return session_turns

    def get_unannotated_turns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch turns without confidence or with low confidence for manual annotation."""
        turns = []
        if not os.path.exists(self.log_path):
            return turns
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Low confidence or lack of category = good for annotation
                        if entry.get("confidence", 0) < 0.5 or not entry.get("category"):
                            turns.append(entry)
                            if len(turns) >= limit:
                                break
        except Exception as e:
            print(f"[ConversationLogger] Fetch failed: {e}")
        return turns[:limit]
