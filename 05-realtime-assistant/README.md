# Project 5: Real-Time RAG Assistant

A real-time conversational AI assistant with **streaming responses** (SSE), **conversation memory**, and **live knowledge ingestion** — powered by PostgreSQL + pgvector and NVIDIA LLM APIs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (UI)                             │
│  ┌──────────┐    ┌─────────────────────┐    ┌───────────────┐  │
│  │ Sidebar   │    │   Chat Panel        │    │ Knowledge     │  │
│  │ Convos    │    │   SSE streaming     │    │ Base Panel    │  │
│  └──────────┘    └─────────────────────┘    └───────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP + SSE
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                 │
│  POST /chat                                                     │
│    │                                                            │
│    ├─ 1. Save user message                                      │
│    ├─ 2. Load conversation history (last 5 msgs)                │
│    ├─ 3. Generate query embedding ──────► NVIDIA Embeddings API │
│    ├─ 4. Vector similarity search ──────► pgvector (PostgreSQL) │
│    ├─ 5. Build augmented prompt with context                    │
│    └─ 6. Stream response via SSE ───────► NVIDIA Chat API       │
│         (token by token)                   (stream: true)       │
│                                                                 │
│  POST /knowledge                                                │
│    ├─ Accept text or URL (auto-scrape)                          │
│    ├─ Generate embedding                                        │
│    └─ Store in pgvector                                         │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PostgreSQL + pgvector                         │
│                                                                 │
│  rt_knowledge       │ rt_conversations  │ rt_messages            │
│  ─────────────      │ ────────────────  │ ───────────           │
│  id, title,         │ id, created_at    │ id, conversation_id,  │
│  content, source,   │                   │ role, content,        │
│  embedding(1024)    │                   │ chunks_used, time     │
│  HNSW index         │                   │                       │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User types question
       │
       ▼
  ┌─────────┐     ┌──────────────┐     ┌─────────────┐
  │  Query   │────►│  Embed query │────►│  pgvector   │
  │  text    │     │  (NVIDIA)    │     │  cosine     │
  └─────────┘     └──────────────┘     │  search     │
                                        └──────┬──────┘
                                               │ top-K chunks
                                               ▼
                                    ┌──────────────────┐
   Conversation ──────────────────► │  Build prompt:   │
   history (5 msgs)                 │  system + context │
                                    │  + history + msg  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  NVIDIA Chat API │
                                    │  stream: true    │
                                    └────────┬─────────┘
                                             │ tokens
                                             ▼
                                    ┌──────────────────┐
                                    │  SSE stream      │
                                    │  data: {token}   │──────► Browser
                                    │  event: done     │        (real-time)
                                    └──────────────────┘
```

## Features

- **🔄 Real-time streaming** — Responses appear token-by-token via Server-Sent Events
- **💬 Conversation memory** — Chat history persisted in PostgreSQL, last 5 messages used as context
- **📚 Live knowledge ingestion** — Add text or URLs to the knowledge base in real-time
- **🔍 Semantic search** — pgvector HNSW index for fast cosine similarity retrieval
- **🎨 Modern dark UI** — ChatGPT-style interface with sidebar, chat bubbles, and knowledge panel

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| Database | PostgreSQL 17 + pgvector |
| Embeddings | NVIDIA `nv-embedqa-e5-v5` (1024 dims) |
| LLM | `moonshotai/kimi-k2.5` via NVIDIA API |
| Streaming | Server-Sent Events (SSE) |
| Frontend | Vanilla HTML/CSS/JS |

## Setup

```bash
cd projects/rag-portfolio/05-realtime-assistant

# Activate venv
source ../venv/bin/activate

# Seed the database with example articles
python seed.py

# Run the server
python app.py
# → http://localhost:8005
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/knowledge` | Add knowledge (text or URL) |
| `GET` | `/knowledge` | List all knowledge items |
| `DELETE` | `/knowledge/{id}` | Remove knowledge item |
| `POST` | `/conversations` | Create new conversation |
| `GET` | `/conversations` | List all conversations |
| `GET` | `/conversations/{id}/messages` | Get conversation messages |
| `POST` | `/chat` | Send message → stream response (SSE) |

### Chat request example

```bash
curl -N -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": 1, "message": "What is RAG?"}'
```

Response stream:
```
event: search
data: [{"id": 3, "title": "RAG", "similarity": 0.92}]

data: {"token": "RAG"}
data: {"token": " stands"}
data: {"token": " for"}
...
event: done
data: {}
```

## SSE vs WebSocket

This project uses **SSE** (Server-Sent Events) instead of WebSocket because:

1. **Simpler** — Unidirectional (server → client), standard HTTP
2. **Native support** — Browser `fetch` + `ReadableStream` handles it
3. **Auto-reconnect** — Built into the EventSource API
4. **Sufficient** — LLM streaming is inherently unidirectional
5. **FastAPI native** — `StreamingResponse` works out of the box

## Database Schema

Tables are prefixed with `rt_` (real-time):

- **`rt_knowledge`** — Knowledge base with vector embeddings (HNSW indexed)
- **`rt_conversations`** — Conversation sessions
- **`rt_messages`** — Chat messages with role, content, and used chunks metadata
