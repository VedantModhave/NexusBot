"""Language detection and translation using deep-translator."""

from langdetect import detect as langdetect_detect, DetectorFactory
from deep_translator import GoogleTranslator

DetectorFactory.seed = 0

LANG_CODES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
    "ta": "tamil",
    "te": "telugu",
    "bn": "bengali",
}

class LanguageTranslator:
    """Detects language and translates between supported Indian languages and English."""

    def __init__(self):
        self.supported = {
            "en": "English",
            "hi": "Hindi",
            "mr": "Marathi",
            "ta": "Tamil",
            "te": "Telugu",
            "bn": "Bengali",
        }

    def detect_language(self, text: str) -> str:
        """Detect language of text. Returns ISO code from supported set, defaults to 'en'."""
        if not text or len(text.strip()) < 4:
            return "en"
        try:
            detected = langdetect_detect(text)
            return detected if detected in self.supported else "en"
        except Exception:
            return "en"

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translates query to English for retrieval.
        Handles:
          - Devanagari Hindi/Marathi
          - Transliterated Hindi/Marathi (Romanized)
          - Already-English text (no-op)
        """
        if source_lang == "en":
            # Even for "en" lang selection, user might type transliterated
            # Indian words — use auto-detect to catch this
            try:
                translated = GoogleTranslator(
                    source="auto",   # KEY FIX: auto-detect instead of "english"
                    target="english"
                ).translate(text)
                return translated if translated else text
            except Exception:
                return text

        try:
            translated = GoogleTranslator(
                source="auto",       # KEY FIX: always use auto for Indian scripts
                target="english"
            ).translate(text)
            return translated if translated else text
        except Exception as e:
            print(f"[TRANSLATION ERROR] {e}")
            return text  # fallback: use original, retrieval may still work

    def translate_response(self, text: str, target_lang: str) -> str:
        """Translates an English response to the target language."""
        if target_lang == "en":
            return text  # no translation needed
        try:
            translated = GoogleTranslator(
                source="english",
                target=LANG_CODES.get(target_lang, "english")
            ).translate(text)
            return translated
        except Exception as e:
            print(f"[TRANSLATION ERROR] to {target_lang}: {e}")
            return text  # fallback to English if translation fails
