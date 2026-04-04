# NexusBot 🎓 — Multilingual Campus Chatbot

A production-ready, RAG-based multilingual chatbot for college campuses. Built with FAISS vector search, BM25 keyword matching, Helsinki-NLP translation models, and a premium dark-themed UI.

**Status**: ✅ Production Ready | **Languages**: 6 (EN, HI, MR, TA, TE, BN) | **Deployment**: Web + Telegram + Docker

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Supported Languages](#supported-languages)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment-with-telegram-bot)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

✨ **Multilingual Support**: Automatic language detection and translation across 6 Indian languages
🔍 **Hybrid Retrieval**: Combines FAISS semantic search (60%) + BM25 keyword matching (40%)
📚 **RAG Pipeline**: Context-aware responses grounded in institutional documents
🤖 **LLM Integration**: Groq API for advanced response generation
💬 **Messaging Integration**: Native Telegram Bot support
📊 **Conversation Logging**: Full turn-by-turn logging for analytics and ML datasets
🔤 **Transliteration Support**: IndicBERT embedder for Hinglish, Marathinglish, etc.
🎨 **Premium UI**: Dark-themed responsive web interface
📦 **Docker Ready**: One-command deployment with Docker Compose
🧪 **Fully Tested**: Smoke tests, retrieval tests, pipeline tests included

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

## Project Structure

```
nexus-chatbot/
├── src/                          # Main application code
│   ├── api/                       # FastAPI application
│   │   └── app.py               # REST API endpoints
│   ├── chatbot/
│   │   ├── pipeline.py          # Main ChatPipeline orchestrator
│   │   ├── context_manager.py   # Conversation context handling
│   │   ├── conversation_logger.py # Turn-by-turn logging
│   │   ├── fallback.py          # Fallback handling
│   │   └── response_generator.py # Response generation
│   ├── nlp/
│   │   ├── embedder.py          # Multilingual embeddings
│   │   ├── indic_embedder.py    # IndicBERT for Indic scripts
│   │   ├── retriever.py         # Hybrid FAISS + BM25 retriever
│   │   └── translator.py        # Helsinki-NLP translations
│   ├── ingestion/
│   │   ├── data_loader.py       # FAQ and document loading
│   │   └── pdf_parser.py        # PDF extraction
│   ├── llm/
│   │   └── intent_classifier.py # Intent classification
│   └── integrations/
│       └── telegram_bot.py       # Telegram Bot integration
├── data/
│   ├── faqs.json                # Institutional FAQs
│   ├── documents/               # PDF source files
│   ├── faiss_index/             # Vector index (auto-generated)
│   └── conversations.jsonl      # Conversation logs (auto-generated)
├── frontend/
│   └── index.html               # Web UI
├── scripts/
│   ├── build_index.py           # Build FAISS index
│   ├── evaluate.py              # Performance evaluation
│   ├── smoke_test.py            # Quick validation tests
│   └── docker_entrypoint.sh     # Container startup
├── tests/
│   ├── test_pipeline.py         # Pipeline tests
│   ├── test_retriever.py        # Retriever tests
│   └── test_translator.py       # Translation tests
├── docker-compose.yml           # Multi-service deployment
├── Dockerfile                   # Container image
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── ENHANCEMENTS.md              # Feature documentation
├── INTEGRATION_SUMMARY.md       # Integration details
└── README.md                    # This file
```

## Prerequisites

- **Python**: 3.9+ (tested on 3.11)
- **RAM**: 4GB minimum (8GB+ recommended for FAISS indexing)
- **Disk**: 2GB free (1GB for models, 500MB for index)
- **API Keys**: 
  - Groq API key (get at https://console.groq.com)
  - Telegram Bot Token (get from @BotFather on Telegram)

## Setup

### Local Development

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/yourusername/nexus-chatbot.git
cd nexus-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env and add your API keys:
# - GROQ_API_KEY=your_groq_api_key
# - TELEGRAM_BOT_TOKEN=your_telegram_token

# 5. Build FAISS index (one-time setup)
python scripts/build_index.py

# 6. Start FastAPI server
uvicorn src.api.app:app --reload --port 8000

# 7. Open browser
# http://localhost:8000
```

### First Run Checklist
- [ ] `.env` file created with API keys
- [ ] FAISS index built (~60 seconds)
- [ ] FastAPI server running on port 8000
- [ ] Web UI loads at http://localhost:8000
- [ ] Test endpoint: `curl http://localhost:8000/health`

## Usage

### Web Interface
1. Open http://localhost:8000 in browser
2. Type query in Hindi/English/other supported language
3. Response appears with source documents highlighted

### Example Queries
```
English:   "What is the fee for B.Tech?"
Hindi:     "बी.टेक की फीस क्या है?"
Hinglish:  "B.Sc ke liye scholarship kitni hai?"
Marathi:   "अभियांत्रिकी शिक्षणाचे शुल्क किती आहे?"
```

### REST API (Programmatic Access)
```bash
# Send a query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "बी.टेक की फीस क्या है?",
    "session_id": "s123"
  }'

# Response includes source documents and confidence score
```

## Deployment with Telegram Bot

### Docker Compose (Recommended)

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Build and start services
docker compose up --build -d

# 3. Check logs
docker compose logs -f api      # FastAPI server
docker compose logs -f bot      # Telegram bot

# 4. Stop deployment
docker compose down
```

**Services Running**:
- FastAPI on http://localhost:8000
- Telegram Bot listening for messages
- Shared volume for vector index

### Telegram Bot Usage
1. Find bot: Search for your bot handle in Telegram
2. Send message: Type query in any supported language
3. Get response: Bot replies with answer and source documents

## API Endpoints

All endpoints return JSON responses with metadata (confidence, language, method).

### `POST /chat`
Send a query and get a response.

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "बी.टेक की फीस क्या है?",
    "session_id": "s123"
  }'
```

**Response:**
```json
{
  "response": "बी.टेक की फीस ₹1,25,000 प्रति सेमेस्टर है।",
  "query_lang": "hi",
  "response_lang": "hi",
  "confidence": 0.92,
  "method": "RAG",
  "sources": [
    "Fee Structure Document - Page 3",
    "2024 Fee Circular"
  ]
}
```

### `GET /health`
Health check endpoint.

```bash
curl http://localhost:8000/health
# Returns: {"status": "healthy"}
```

### `GET /languages`
List supported languages.

```bash
curl http://localhost:8000/languages
# Returns: {"langs": ["en", "hi", "mr", "ta", "te", "bn"]}
```

### `POST /feedback`
Log user feedback for response.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "s123",
    "message_id": "m456",
    "rating": 1
  }'
```

### `GET /stats`
Get system statistics.

```bash
curl http://localhost:8000/stats
# Returns: {
#   "total_queries": 1523,
#   "avg_confidence": 0.78,
#   "language_dist": {...},
#   "successful_rate": 0.87
# }
```

## Testing

### Run All Tests
```bash
# Install test dependencies (included in requirements.txt)
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src
```

### Smoke Test (Quick Validation)
```bash
python scripts/smoke_test.py
# Validates: Index loading, API endpoints, basic retrieval
```

### Evaluate Retrieval Performance
```bash
python scripts/evaluate.py
# Outputs: Top-1 Accuracy, MRR, Confidence metrics
```

## Configuration

### Environment Variables (.env)

```env
# LLM Service
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Telegram Integration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
NEXUSBOT_API_URL=http://localhost:8000/chat

# Optional: Model fine-tuning
# LLM_MODEL=mixtral-8x7b-32768  # Default model
# RETRIEVER_TOP_K=5             # Top documents to retrieve
# TRANSLATION_DEVICE=cpu        # or 'cuda'
```

### Customization

**Add New Languages**: Update [src/nlp/translator.py](src/nlp/translator.py) with new language codes and Helsinki-NLP models.

**Customize FAQs**: Edit [data/faqs.json](data/faqs.json), then rebuild index:
```bash
python scripts/build_index.py
```

**Adjust Retrieval Weights**: Edit [src/nlp/retriever.py](src/nlp/retriever.py) BM25/FAISS blend ratio (currently 40/60).

## Evaluation

### Performance Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| Top-1 Accuracy | ~90% | On held-out test set |
| Mean Reciprocal Rank (MRR) | ~0.94 | Retrieval ranking quality |
| Avg Confidence | ~0.76 | System's self-assessment |
| Multilingual Coverage | 6 languages | EN, HI, MR, TA, TE, BN |
| Response Latency | <500ms | Average end-to-end time |

### Evaluation Commands
```bash
# Generate detailed performance report
python scripts/evaluate.py --detailed

# Test specific language
python scripts/evaluate.py --language hi
```

## Troubleshooting

### Common Issues

**1. FAISS Index Not Found**
```
Error: "faiss_index/index.faiss not found"
Solution: Run python scripts/build_index.py
```

**2. Groq API Key Invalid**
```
Error: "Invalid API key"
Solution: Verify GROQ_API_KEY in .env and check quota at https://console.groq.com
```

**3. Slow Response Time**
```
Cause: Large document corpus or slow embedder
Solution: Reduce TOP_K in retriever.py or use GPU (TRANSLATION_DEVICE=cuda)
```

**4. Telegram Bot Not Responding**
```
Check: docker compose logs bot
Verify: TELEGRAM_BOT_TOKEN is correct in .env
Test: curl http://localhost:8000/health
```

**5. Port 8000 Already in Use**
```bash
# Change port in command
uvicorn src.api.app:app --port 8001
```

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python scripts/build_index.py
```

### Getting Help
1. Check [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) for feature details
2. Review [ENHANCEMENTS.md](ENHANCEMENTS.md) for advanced configurations
3. Open an issue on GitHub with error logs

## Contributing

### Development Setup
```bash
git clone <your-fork>
cd nexus-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Code Style
- Python: Follow PEP 8 (max line length 100)
- Use type hints for all functions
- Write docstrings for all classes and functions
- Run tests before submitting PR

### Adding Features
1. Create feature branch: `git checkout -b feature/your-feature`
2. Write tests for new functionality
3. Update documentation
4. Submit pull request with description

### Reporting Bugs
Include:
- Python version
- Error message and stack trace
- Steps to reproduce
- .env configuration (with keys masked)

## License

MIT
