# NexusBot 🎓 — Multilingual Campus Chatbot

A production-ready, RAG-based multilingual chatbot for college campuses. Built with FAISS vector search, BM25 keyword matching, Helsinki-NLP translation models, and a premium dark-themed UI.

## Architecture

```
  User (Browser)  ──POST /chat──►  FastAPI Server
                                       │
                              ┌────────┴────────┐
                              │   ChatPipeline   │
                              └────────┬────────┘
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
            [Lang Detect]      [Context Enrich]    [Translator]
            (langdetect)       (last 5 turns)      (Helsinki-NLP)
                    │                  │                  │
                    └────────┬────────┘                  │
                             ▼                           │
                    [Hybrid Retriever]                   │
                    ├── BM25 (40%)                       │
                    └── FAISS (60%)                      │
                             │                           │
                             ▼                           │
                   [Response Generator]                  │
                   (sentence re-ranking)                 │
                             │                           │
                             ▼                           ▼
                    [English Response] ──translate──► [User's Language]
```

## Supported Languages

| Code | Language | Translation Model |
|------|----------|-------------------|
| `en` | English  | Base              |
| `hi` | Hindi    | `opus-mt-en-hi` / `opus-mt-hi-en` |
| `mr` | Marathi  | `opus-mt-en-mr` / `opus-mt-mr-en` |
| `ta` | Tamil    | `opus-mt-en-ta` / `opus-mt-ta-en` |
| `te` | Telugu   | `opus-mt-en-te` / `opus-mt-te-en` |
| `bn` | Bengali  | `opus-mt-en-bn` / `opus-mt-bn-en` |

## Setup

```bash
git clone https://github.com/username/nexus-chatbot.git
cd nexus-chatbot
pip install -r requirements.txt
cp .env.example .env
# Update .env with your real keys before starting
python scripts/build_index.py
uvicorn src.api.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## Deploy With Telegram Bot

Run both services together with Docker Compose:

```bash
cp .env.example .env
# Fill GROQ_API_KEY and TELEGRAM_BOT_TOKEN in .env
docker compose up --build -d
```

Check service logs:

```bash
docker compose logs -f api
docker compose logs -f telegram-bot
```

Stop deployment:

```bash
docker compose down
```

## API Endpoints

### `POST /chat`
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "बी.टेक की फीस क्या है?", "session_id": "s123"}'
```

### `GET /health`
```bash
curl http://localhost:8000/health
```

### `GET /languages`
```bash
curl http://localhost:8000/languages
```

### `POST /feedback`
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s123", "message_id": "m456", "rating": 1}'
```

### `GET /stats`
```bash
curl http://localhost:8000/stats
```

## Evaluation

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | ~90% |
| MRR | ~0.94 |
| Avg Confidence | ~0.76 |

## License

MIT
