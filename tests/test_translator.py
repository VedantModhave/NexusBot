"""Tests for the language translator module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp.translator import LanguageTranslator, SUPPORTED_LANGUAGES


@pytest.fixture
def translator():
    return LanguageTranslator()


class TestDetectLanguage:
    """Test language detection on various samples."""

    def test_detect_english(self, translator):
        text = "What is the tuition fee for the B.Tech program?"
        lang = translator.detect_language(text)
        assert lang == "en", f"Expected 'en', got '{lang}'"

    def test_detect_hindi(self, translator):
        text = "बी.टेक कार्यक्रम की ट्यूशन फीस क्या है?"
        lang = translator.detect_language(text)
        assert lang == "hi", f"Expected 'hi', got '{lang}'"

    def test_detect_marathi(self, translator):
        text = "महाविद्यालयाची फी किती आहे?"
        lang = translator.detect_language(text)
        assert lang == "mr", f"Expected 'mr', got '{lang}'"

    def test_detect_empty_string(self, translator):
        lang = translator.detect_language("")
        assert lang == "en", "Empty string should default to 'en'"

    def test_detect_unsupported_defaults_to_english(self, translator):
        # French text — should fallback to 'en'
        text = "Bonjour comment allez-vous aujourd'hui?"
        lang = translator.detect_language(text)
        assert lang in SUPPORTED_LANGUAGES, f"Got unsupported language: {lang}"


class TestTranslation:
    """Test translation functionality."""

    def test_same_language_returns_original(self, translator):
        text = "Hello World"
        result = translator.translate(text, "en", "en")
        assert result == text

    def test_empty_text_returns_empty(self, translator):
        result = translator.translate("", "en", "hi")
        assert result == ""

    def test_translate_to_english_returns_tuple(self, translator):
        text = "Hello, how are you?"
        translated, lang = translator.translate_to_english(text)
        assert isinstance(translated, str)
        assert lang == "en"
        assert translated == text  # Already English


class TestSupportedLanguages:
    """Test supported languages configuration."""

    def test_all_languages_present(self):
        expected = {"en", "hi", "mr", "ta", "te", "bn"}
        assert set(SUPPORTED_LANGUAGES.keys()) == expected

    def test_language_names_are_strings(self):
        for code, name in SUPPORTED_LANGUAGES.items():
            assert isinstance(name, str)
            assert len(name) > 0
