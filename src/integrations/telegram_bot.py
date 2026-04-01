"""
NexusBot — Telegram Integration
================================
Features:
  - Auto language detection from message text (EN / HI / MR)
  - User language preference override via /language command
  - Persistent quick-reply keyboard (6 buttons) — shown on EVERY reply
  - Inline language selector keyboard
  - Typing indicator while fetching response
  - Graceful error handling with fallback message

Requirements:
    pip install python-telegram-bot==20.7 langdetect==1.0.9 httpx==0.27.0

Usage:
    python src/integrations/telegram_bot.py

IMPORTANT: Send /start to your bot first to activate the keyboard.
           If buttons still don't show, send /start again.
"""

import httpx
import os
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)

load_dotenv()
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# ─────────────────────────────────────────────
# CONFIG — update these values
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
NEXUSBOT_API   = os.getenv("NEXUSBOT_API_URL", "http://localhost:8000/chat")


# ─────────────────────────────────────────────
# KEYBOARDS
# NOTE: is_persistent=True keeps keyboard visible after bot restarts.
# We also attach PERSISTENT_KEYBOARD to EVERY reply message so it
# never disappears, even if Telegram decides to hide it.
# ─────────────────────────────────────────────
PERSISTENT_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("💰 Fee Structure"),  KeyboardButton("🎓 Scholarships")],
        [KeyboardButton("📅 Exam Schedule"),  KeyboardButton("🏠 Hostel Info")],
        [KeyboardButton("📊 Placements"),     KeyboardButton("🌐 Change Language")],
    ],
    resize_keyboard=True,
    is_persistent=True,                        # survives bot restarts
    input_field_placeholder="Ask anything about the campus...",
)

LANGUAGE_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🇬🇧 English",  callback_data="lang_en"),
        InlineKeyboardButton("🇮🇳 हिंदी",    callback_data="lang_hi"),
        InlineKeyboardButton("🇮🇳 मराठी",    callback_data="lang_mr"),
    ],
    [
        InlineKeyboardButton("🔄 Auto-detect (recommended)", callback_data="lang_auto"),
    ],
])

LANG_LABELS = {
    "en":   "English 🇬🇧",
    "hi":   "हिंदी 🇮🇳",
    "mr":   "मराठी 🇮🇳",
    "auto": "Auto-detect 🔄",
}


# ─────────────────────────────────────────────
# QUICK REPLY BUTTON MAP
# label → (query_to_send, forced_lang or None)
# None value = special handler (e.g. Change Language)
# ─────────────────────────────────────────────
QUICK_REPLIES: dict[str, tuple[str, str | None] | None] = {
    "💰 Fee Structure":   ("What is the fee structure?", None),
    "🎓 Scholarships":    ("How to apply for scholarship?", None),
    "📅 Exam Schedule":   ("When are the end semester exams?", None),
    "🏠 Hostel Info":     ("What are the hostel rules and facilities?", None),
    "📊 Placements":      ("What is the placement statistics?", None),
    "🌐 Change Language": None,
}


# ─────────────────────────────────────────────
# STATIC RESPONSES
# ─────────────────────────────────────────────
ERROR_RESPONSES = {
    "en": "⚠️ I'm having trouble connecting. Please try again or contact info@xyzcollege.edu.in",
    "hi": "⚠️ कनेक्शन में समस्या है। कृपया दोबारा प्रयास करें या info@xyzcollege.edu.in पर संपर्क करें।",
    "mr": "⚠️ कनेक्शन समस्या आहे. कृपया पुन्हा प्रयत्न करा किंवा info@xyzcollege.edu.in वर संपर्क करा.",
}

NON_TEXT_RESPONSES = {
    "en": "I only understand text messages. Please type your question!",
    "hi": "मैं केवल टेक्स्ट संदेश समझता हूँ। कृपया अपना प्रश्न टाइप करें!",
    "mr": "मी फक्त मजकूर संदेश समजतो. कृपया तुमचा प्रश्न टाइप करा!",
}


# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────
def detect_lang_from_text(text: str) -> str:
    """
    Detect language from actual message text.
    Returns "en", "hi", or "mr". Falls back to "en".
    """
    if not text or len(text.strip()) < 2:
        return "en"
    try:
        detected = detect(text)
        if detected == "mr":
            return "mr"
        if detected == "hi":
            return "hi"
        return "en"
    except LangDetectException:
        return "en"


def resolve_lang(text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Resolve final language. User preference overrides auto-detect.
    """
    preferred = context.user_data.get("preferred_lang", "auto")
    if preferred != "auto":
        return preferred
    return detect_lang_from_text(text)


# ─────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start and /help — always sends PERSISTENT_KEYBOARD.
    This is the trigger that makes the keyboard appear for the first time.
    """
    await update.message.reply_text(
        "👋 Hello! I'm NexusBot, your XYZ College assistant!\n\n"
        "Ask me about admissions, fees, hostel, placements, "
        "exams, or scholarships!\n\n"
        "💡 Tap a button below or type your question:",
        reply_markup=PERSISTENT_KEYBOARD,
    )


async def cmd_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /language — show inline language selector.
    Also re-sends PERSISTENT_KEYBOARD so it stays visible.
    """
    current = context.user_data.get("preferred_lang", "auto")
    await update.message.reply_text(
        f"🌐 Select your preferred language:\n"
        f"Current: {LANG_LABELS.get(current, 'Auto-detect 🔄')}\n\n"
        "(Auto-detect works best for most users)",
        reply_markup=LANGUAGE_KEYBOARD,
    )


# ─────────────────────────────────────────────
# CALLBACK HANDLER — inline language keyboard
# ─────────────────────────────────────────────
async def handle_language_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Save language preference when user taps inline keyboard."""
    query = update.callback_query
    await query.answer()

    lang_map = {
        "lang_en":   "en",
        "lang_hi":   "hi",
        "lang_mr":   "mr",
        "lang_auto": "auto",
    }

    lang_code = lang_map.get(query.data, "auto")
    context.user_data["preferred_lang"] = lang_code
    label = LANG_LABELS[lang_code]

    await query.edit_message_text(
        f"✅ Language set to: {label}\n\n"
        f"All future responses will be in {label}.\n"
        f"Use /language to change it anytime."
    )

    # Send a follow-up WITH keyboard so it reappears after inline tap
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="Keyboard is ready! Ask me anything 👇",
        reply_markup=PERSISTENT_KEYBOARD,
    )

    print(f"[TELEGRAM] User {query.from_user.id} set language → {lang_code}")


# ─────────────────────────────────────────────
# MAIN MESSAGE HANDLER
# ─────────────────────────────────────────────
async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle all text messages including quick reply button taps."""
    text = update.message.text.strip()
    if not text:
        return

    # ── Special: Change Language button ──────────────────────────
    if text == "🌐 Change Language":
        await cmd_language(update, context)
        return

    # ── Quick reply button tap ────────────────────────────────────
    if text in QUICK_REPLIES:
        mapped = QUICK_REPLIES[text]
        if mapped is None:
            return
        query_text, forced_lang = mapped
        lang = forced_lang if forced_lang else resolve_lang(query_text, context)
    else:
        # Free-text query typed by user
        query_text = text
        lang = resolve_lang(text, context)

    print(f"[TELEGRAM] query='{query_text}' lang={lang} user={update.effective_user.id}")

    # ── Typing indicator ──────────────────────────────────────────
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    # ── Call NexusBot API ─────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.post(
                NEXUSBOT_API,
                json={"query": query_text, "lang": lang},
            )
            res.raise_for_status()
            data = res.json()

        response   = data.get("response", "Something went wrong.")
        confidence = data.get("confidence", 0)
        lang_out   = data.get("lang", lang).upper()

        reply = f"{response}\n\n🌐 {lang_out} • {confidence}% confidence"

        # KEY FIX: attach PERSISTENT_KEYBOARD to every single reply
        # This guarantees the keyboard is always visible
        await update.message.reply_text(
            reply,
            reply_markup=PERSISTENT_KEYBOARD,
        )

    except httpx.TimeoutException:
        err_lang = resolve_lang(text, context)
        await update.message.reply_text(
            ERROR_RESPONSES.get(err_lang, ERROR_RESPONSES["en"]),
            reply_markup=PERSISTENT_KEYBOARD,
        )
        print(f"[TELEGRAM ERROR] Timeout — query='{query_text}'")

    except httpx.HTTPStatusError as e:
        err_lang = resolve_lang(text, context)
        await update.message.reply_text(
            ERROR_RESPONSES.get(err_lang, ERROR_RESPONSES["en"]),
            reply_markup=PERSISTENT_KEYBOARD,
        )
        print(f"[TELEGRAM ERROR] HTTP {e.response.status_code} — query='{query_text}'")

    except Exception as e:
        err_lang = resolve_lang(text, context)
        await update.message.reply_text(
            ERROR_RESPONSES.get(err_lang, ERROR_RESPONSES["en"]),
            reply_markup=PERSISTENT_KEYBOARD,
        )
        print(f"[TELEGRAM ERROR] Unexpected: {e}")


# ─────────────────────────────────────────────
# NON-TEXT HANDLER (stickers, photos, voice, etc.)
# ─────────────────────────────────────────────
async def handle_non_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    lang = context.user_data.get("preferred_lang", "en")
    if lang == "auto":
        lang = "en"
    await update.message.reply_text(
        NON_TEXT_RESPONSES.get(lang, NON_TEXT_RESPONSES["en"]),
        reply_markup=PERSISTENT_KEYBOARD,
    )


# ─────────────────────────────────────────────
# BOT RUNNER
# ─────────────────────────────────────────────
def run_telegram_bot() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Add it to your environment or .env file.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_start))
    app.add_handler(CommandHandler("language", cmd_language))

    # Inline keyboard callbacks
    app.add_handler(CallbackQueryHandler(handle_language_callback))

    # Text messages
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # Non-text messages
    app.add_handler(
        MessageHandler(~filters.TEXT, handle_non_text)
    )

    print("[TELEGRAM] NexusBot is running...")
    print("[TELEGRAM] Send /start to your bot to activate the keyboard.")
    print("[TELEGRAM] Commands: /start  /help  /language")
    print("[TELEGRAM] Press Ctrl+C to stop.")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_telegram_bot()