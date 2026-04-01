"""Full RAG pipeline: detect -> translate -> enrich -> retrieve -> generate -> translate back."""

import re

from src.nlp.embedder import MultilingualEmbedder
from src.nlp.translator import LanguageTranslator
from src.nlp.retriever import HybridRetriever
from src.chatbot.context_manager import ConversationContext
from src.chatbot.response_generator import ResponseGenerator
from src.chatbot.fallback import FallbackHandler
from src.chatbot.conversation_logger import ConversationLogger

GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "heya", "howdy",
    "namaste", "namaskar", "kem cho", "kai hal",
    "नमस्ते", "हेलो", "हाय",
    "नमस्कार", "हॅलो", "हाय"
}

GREETING_RESPONSES = {
    "en": "Hello! 👋 I'm NexusBot, your college assistant. You can ask me about admissions, fees, hostel, placements, exams, and more!",
    "hi": "नमस्ते! 👋 मैं NexusBot हूँ। आप मुझसे प्रवेश, शुल्क, छात्रावास, परीक्षा आदि के बारे में पूछ सकते हैं।",
    "mr": "नमस्कार! 👋 मी NexusBot आहे. तुम्ही प्रवेश, शुल्क, वसतिगृह, परीक्षा याबद्दल विचारू शकता.",
    "ta": "வணக்கம்! 👋 நான் NexusBot. நீங்கள் சேர்க்கை, கட்டணம், விடுதி, வேலைவாய்ப்புகள், தேர்வுகள் பற்றி கேட்கலாம்.",
    "te": "నమస్కారం! 👋 నేను NexusBot. మీరు అడ్మిషన్లు, ఫీజులు, హాస్టల్, ప్లేస్‌మెంట్‌లు, పరీక్షల గురించి అడగవచ్చు.",
    "bn": "নমস্কার! 👋 আমি NexusBot. আপনি ভর্তি, ফি, হোস্টেল, প্লেসমেন্ট, পরীক্ষা সম্পর্কে জিজ্ঞাসা করতে পারেন।"
}

GIBBERISH_RESPONSES = {
    "en": "I didn't understand that. Could you please rephrase? You can ask about fees, hostel, placements, exams, or admissions.",
    "hi": "मुझे समझ नहीं आया। कृपया दोबारा पूछें। आप शुल्क, छात्रावास, परीक्षा आदि के बारे में पूछ सकते हैं।",
    "mr": "मला समजले नाही. कृपया पुन्हा विचारा. तुम्ही शुल्क, वसतिगृह, परीक्षा याबद्दल विचारू शकता.",
    "ta": "எனக்கு புரியவில்லை. தயவுசெய்து மீண்டும் கேட்க முடியுமா? நீங்கள் கட்டணம், விடுதி, வேலைவாய்ப்புகள் பற்றி கேட்கலாம்.",
    "te": "నాకు అర్థం కాలేదు. దయచేసి మళ్లీ అడగగలరా? మీరు ఫీజులు, హాస్టల్, పరీక్షల గురించి అడగవచ్చు.",
    "bn": "আমি বুঝতে পারিনি। অনুগ্রহ করে আবার জিজ্ঞাসা করবেন? আপনি ফি, হোস্টেল, পরীক্ষা সম্পর্কে জিজ্ঞাসা করতে পারেন।"
}

FALLBACK_RESPONSES = {
    "en": "I don't have specific information about that. For details, please contact info@xyzcollege.edu.in or call the admin office.",
    "hi": "मुझे इस बारे में जानकारी नहीं है। कृपया info@xyzcollege.edu.in पर संपर्क करें या एडमिन कार्यालय को कॉल करें।",
    "mr": "मला याबद्दल माहिती नाही. कृपया info@xyzcollege.edu.in वर संपर्क करा किंवा प्रशासकीय कार्यालयात कॉल करा.",
    "ta": "இதைப் பற்றிய குறிப்பிட்ட தகவல் என்னிடம் இல்லை. விவரங்களுக்கு, info@xyzcollege.edu.in ஐத் தொடர்பு கொள்ளவும்.",
    "te": "దీని గురించి నాకు నిర్దిష్ట సమాచారం లేదు. వివరాల కోసం దయచేసి info@xyzcollege.edu.in ని సంప్రదించండి.",
    "bn": "এ সম্পর্কে আমার কাছে নির্দিষ্ট কোনো তথ্য নেই। বিস্তারিত তথ্যের জন্য অনুগ্রহ করে info@xyzcollege.edu.in-এ যোগাযোগ করুন।"
}

def is_greeting(query: str) -> bool:
    cleaned = query.strip().lower().rstrip("!.,?")
    words = cleaned.split()
    if not words:
        return False
    return cleaned in GREETINGS or (len(words) <= 2 and words[0] in GREETINGS)

def is_gibberish(query: str) -> bool:
    query = query.strip()
    # Too short to be meaningful
    if len(query) < 3:
        return True
    # No vowels at all (pure consonant jumble)
    vowels = set("aeiouAEIOUअआइईउऊएऐओऔ")
    if len(query) > 4 and not any(c in vowels for c in query):
        return True
    # Abnormally long single word with no spaces (random keysmash)
    words = query.split()
    if len(words) == 1 and len(query) > 12:
        return True
    # High ratio of repeated characters
    for word in words:
        if len(word) > 4:
            unique_ratio = len(set(word)) / len(word)
            if unique_ratio < 0.3:
                return True
    return False

class ChatPipeline:
    """End-to-end multilingual RAG chatbot pipeline. All models loaded once."""

    def __init__(self):
        print("[NexusBot] Loading embedder...")
        self.embedder = MultilingualEmbedder()

        print("[NexusBot] Loading retriever...")
        self.retriever = HybridRetriever(self.embedder)

        print("[NexusBot] Loading translator...")
        self.translator = LanguageTranslator()

        self.response_gen = ResponseGenerator(self.embedder)
        self.context = ConversationContext()
        self.fallback = FallbackHandler()
        self.conv_logger = ConversationLogger()

        self.stats = {
            "total": 0,
            "fallbacks": 0,
            "confidence_sum": 0.0,
            "categories": {},
        }
        print("[NexusBot] All components ready.")

    def chat(self, user_input: str, session_id: str, response_lang: str = "en") -> dict:
        """Process user message through the full RAG pipeline."""
        self.stats["total"] += 1
        turn_num = self.stats["total"]
        
        # Determine fallback default language safely
        lang = response_lang if response_lang in GREETING_RESPONSES else "en"

        # Step 1: Greeting check
        if is_greeting(user_input):
            response_text = GREETING_RESPONSES.get(lang, GREETING_RESPONSES["en"])
            # Log conversation turn
            self.conv_logger.log_turn(
                session_id=session_id,
                turn_num=turn_num,
                query=user_input,
                query_lang="en",
                response=response_text,
                response_lang=response_lang,
                intent="greeting",
                category="greeting",
                confidence=1.0,
                method="greeting",
                is_fallback=False,
            )
            return {
                "response": response_text,
                "detected_language": response_lang,
                "confidence": 1.0,
                "category": "greeting",
                "is_fallback": False,
                "sources": []
            }

        # Step 2: Gibberish check
        if is_gibberish(user_input):
            response_text = GIBBERISH_RESPONSES.get(lang, GIBBERISH_RESPONSES["en"])
            # Log conversation turn
            self.conv_logger.log_turn(
                session_id=session_id,
                turn_num=turn_num,
                query=user_input,
                query_lang="unknown",
                response=response_text,
                response_lang=response_lang,
                intent="gibberish",
                category="unknown",
                confidence=0.0,
                method="gibberish_check",
                is_fallback=True,
            )
            return {
                "response": response_text,
                "detected_language": response_lang,
                "confidence": 0.0,
                "category": "unknown",
                "is_fallback": True,
                "sources": []
            }

        # Step 3: Translate query to English for retrieval
        detected_lang = self.translator.detect_language(user_input)
        if detected_lang != "en":
            query_en = self.translator.translate_to_english(user_input, detected_lang)
        else:
            query_en = user_input

        # Overwrite detected lang with user explicitly selected lang to ensure it translates to the right response lang
        # Auto-detect is ONLY used for translating query TO English
        
        # Step 4: Enrich with conversation context
        enriched_query = self.context.enrich_query(query_en, session_id)

        # Step 5: Retrieve top 5 chunks
        results = self.retriever.retrieve(enriched_query, top_k=5)

        # Step 6: Check confidence — fallback if too low (Threshold applied manually)
        top_score = results[0]["score"] if results else 0.0

        if top_score < 0.35: # CONFIDENCE_THRESHOLD = 0.35
            self.stats["fallbacks"] += 1
            self.stats["confidence_sum"] += top_score
            self.fallback.log(user_input, session_id, response_lang)

            response_final = FALLBACK_RESPONSES.get(lang, FALLBACK_RESPONSES["en"])
            self.context.update(session_id, user_input, response_final, "unknown")
            
            # Log fallback turn
            self.conv_logger.log_turn(
                session_id=session_id,
                turn_num=turn_num,
                query=user_input,
                query_lang=response_lang,
                response=response_final,
                response_lang=response_lang,
                intent="valid_query",
                category="unknown",
                confidence=round(top_score, 3),
                method="RAG_fallback",
                is_fallback=True,
            )

            return {
                "response": response_final,
                "detected_language": response_lang,
                "confidence": round(top_score, 3),
                "category": "unknown",
                "is_fallback": True,
                "sources": [],
            }

        # Step 7: Generate response in English
        gen_result = self.response_gen.generate(enriched_query, results)
        response_en = gen_result["text"]

        # Step 8: Translate response to user's selected language
        response_final = self.translator.translate_response(response_en, response_lang)

        # Step 9: Update context
        cat = gen_result["category"]
        self.context.update(session_id, query_en, response_en, cat)

        # Step 10: Update stats
        self.stats["confidence_sum"] += gen_result["confidence"]
        self.stats["categories"][cat] = self.stats["categories"].get(cat, 0) + 1
        
        # Log successful RAG turn
        self.conv_logger.log_turn(
            session_id=session_id,
            turn_num=turn_num,
            query=user_input,
            query_lang=detected_lang,
            response=response_final,
            response_lang=response_lang,
            intent="valid_query",
            category=cat,
            confidence=round(gen_result["confidence"], 3),
            method="RAG",
            is_fallback=False,
        )

        return {
            "response": response_final,
            "detected_language": response_lang,
            "confidence": round(gen_result["confidence"], 3),
            "category": cat,
            "is_fallback": False,
            "sources": [r.get("id", "") for r in results[:3]],
        }
