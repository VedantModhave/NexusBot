import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.chatbot.pipeline import ChatPipeline
from src.llm.intent_classifier import classify_intent
from src.llm.response_generator import generate_llm_response
from src.constants import RESPONSES

load_dotenv()

# --------------- Pydantic models ---------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lang: str = Field(default="en")


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int = Field(..., ge=-1, le=1)


# --------------- App lifecycle ---------------

pipeline: Optional[ChatPipeline] = None
llm_client: Optional[AsyncOpenAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, llm_client
    print("[Startup] Initializing components...")
    
    # Initialize LLM Client (Groq via OpenAI SDK)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

    llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    
    # Initialize RAG Pipeline
    pipeline = ChatPipeline()
    print("[Startup] Ready.")
    yield


app = FastAPI(title="NexusBot API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEEDBACK_LOG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "feedback_log.jsonl")
)
FRONTEND_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")
)

# --------------- Helpers ---------------

def get_response(key: str, lang: str) -> dict:
    resp_text = RESPONSES.get(key, {}).get(lang, RESPONSES[key]["en"])
    return {
        "response":   resp_text,
        "confidence": 1.0 if key == "greeting" else 0.0,
        "lang":       lang,
        "category":   key,
        "is_fallback": key != "greeting",
        "sources":    []
    }

# --------------- Endpoints ---------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists(FRONTEND_PATH):
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>NexusBot API running.</h1>")


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query.strip()
    lang = request.lang if request.lang in ["en", "hi", "mr"] else "en"
    session_id = request.session_id

    # -- 1. Guard: Empty --
    if not query:
        return get_response("fallback", lang)

    # -- 2. LLM Intent Classifier --
    intent = await classify_intent(query, llm_client)
    
    # LOG every classification for debugging
    print(f"[INTENT] query='{query}' lang={lang} → intent={intent}")

    if intent == "greeting":
        result = get_response("greeting", lang)
        result["message_id"] = str(uuid.uuid4())
        return result

    if intent == "gibberish":
        # Extra safety: if query has 3+ chars and contains letters,
        # fail-over to retrieval anyway in case of misclassification
        if len(query) >= 3 and any(c.isalpha() for c in query):
            print(f"[INTENT WARNING] Gibberish override attempted for: '{query}'")
            # We let it fall through to retrieval by NOT returning here
        else:
            result = get_response("gibberish", lang)
            result["message_id"] = str(uuid.uuid4())
            return result

    if intent == "out_of_scope":
        result = get_response("out_of_scope", lang)
        result["message_id"] = str(uuid.uuid4())
        return result

    # -- 3. Translate to English for retrieval --
    query_en = pipeline.translator.translate_to_english(query, lang)
    print(f"[TRANSLATION] '{query}' ({lang}) → '{query_en}' (en)")

    # -- 4. RAG Retrieval (BM25 + FAISS) --
    results = pipeline.retriever.retrieve(query_en, top_k=5)
    
    # -- 5. Score threshold gate --
    top_score = results[0]["score"] if results else 0.0
    if not results or top_score < 0.40:
        result = get_response("fallback", lang)
        result["message_id"] = str(uuid.uuid4())
        pipeline.context.update(session_id, query_en, result["response"], "unknown")
        return result

    # -- 6. Response Generation via LLM --
    top_chunk = results[0]
    context_text = top_chunk.get("answer_en", top_chunk.get("text", ""))
    
    response_text = await generate_llm_response(
        query=query_en,
        context_chunk=context_text,
        lang=lang,
        client=llm_client
    )

    # -- 7. Update conversation context --
    cat = top_chunk.get("category", "general")
    pipeline.context.update(session_id, query_en, response_text, cat)

    # -- 8. Confidence label (as requested: int score * 100) --
    confidence_pct = min(int(top_score * 100), 100)

    result = {
        "response":   response_text,
        "confidence": confidence_pct,
        "lang":       lang,
        "category":   cat,
        "is_fallback": False,
        "sources":    [r.get("id", "") for r in results[:3]],
        "message_id": str(uuid.uuid4()),
        "method":     "LLM+RAG"
    }
    return result


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_ready": llm_client is not None,
        "pipeline_ready": pipeline is not None
    }


@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(body.model_dump()) + "\n")
    return {"logged": True}
