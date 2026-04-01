"""Fallback handler for low-confidence queries."""

import json
import os
import datetime


FALLBACK_MESSAGES = {
    "en": "I'm not confident about that answer. Please contact our helpdesk at helpdesk@xyzcollege.edu.in or call +91-9876-543210 during office hours (9 AM \u2013 5 PM).",
    "hi": "मुझे इस प्रश्न का सटीक उत्तर नहीं पता। कृपया helpdesk@xyzcollege.edu.in पर संपर्क करें या +91-9876-543210 पर कॉल करें।",
    "mr": "मला या प्रश्नाचे अचूक उत्तर माहित नाही. कृपया helpdesk@xyzcollege.edu.in वर संपर्क करा.",
    "ta": "என்னால் இந்த கேள்விக்கு சரியான பதில் சொல்ல முடியவில்லை. helpdesk@xyzcollege.edu.in என்ற மின்னஞ்சலில் தொடர்பு கொள்ளுங்கள்.",
    "te": "నాకు ఈ ప్రశ్నకు సరైన సమాధానం తెలియదు. helpdesk@xyzcollege.edu.in కి సంప్రదించండి.",
    "bn": "আমি এই প্রশ্নের সঠিক উত্তর দিতে পারছি না। helpdesk@xyzcollege.edu.in তে যোগাযোগ করুন।",
}


class FallbackHandler:
    """Returns localized fallback messages and logs failed queries."""

    def __init__(self):
        self.log_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "fallback_log.jsonl")
        )
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def get_message(self, lang: str) -> str:
        return FALLBACK_MESSAGES.get(lang, FALLBACK_MESSAGES["en"])

    def log(self, query: str, session_id: str, lang: str):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "query": query,
            "session_id": session_id,
            "lang": lang,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Fallback] Log write failed: {e}")
