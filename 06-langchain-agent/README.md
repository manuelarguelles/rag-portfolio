# 06 — LangChain RAG Agent 🤖

A **production-ready RAG Agent** built with LangChain that combines vector search, mathematical calculations, and date awareness through an intelligent tool-using agent.

## Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────┐
│        LangChain Agent          │
│   (ReAct / Tool-Calling LLM)   │
│                                 │
│  ┌───────────────────────────┐  │
│  │    Tool Selection Logic   │  │
│  │  "Which tool(s) to use?"  │  │
│  └─────────┬─────────────────┘  │
│            │                    │
│   ┌────────┼────────┐          │
│   ▼        ▼        ▼          │
│ ┌────┐  ┌────┐  ┌────────┐    │
│ │ 🔍 │  │ 🧮 │  │  📅    │    │
│ │Search│ │Calc│  │ Date   │    │
│ └──┬──┘ └──┬─┘  └───┬────┘    │
│    │       │         │         │
│    ▼       ▼         ▼         │
│  PGVector  eval()  datetime    │
│  (1024d)                       │
│                                │
│  ┌───────────────────────────┐ │
│  │  Synthesize Final Answer  │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
     │
     ▼
  Response (with tool usage indicators)
```

## What Makes This Different from Simple RAG?

| Feature | Simple RAG | LangChain Agent |
|---------|-----------|-----------------|
| Query handling | Always searches vectors | **Decides** whether to search |
| Calculations | ❌ | ✅ Built-in calculator |
| Multi-step reasoning | ❌ | ✅ Chain multiple tools |
| Date awareness | ❌ | ✅ Current date tool |
| Memory | Single turn | ✅ 5-turn conversation window |
| Tool composition | ❌ | ✅ Search → Calculate |

### Example: Multi-Tool Query

> "What is China's GDP divided by its population?"

1. 🔍 Agent searches knowledge base → finds GDP ($18.53T) and population (1,425M)
2. 🧮 Agent calculates → `18530000000000 / 1425000000 = 13,003.5`
3. 📝 Agent synthesizes → "China's GDP per capita is approximately $13,003"

## Tech Stack

- **LangChain** — Agent framework with tools
- **LangGraph** — ReAct agent execution
- **NVIDIA NIM** — LLM (kimi-k2.5) + Embeddings (nv-embedqa-e5-v5)
- **PGVector** — Vector storage (1024 dimensions, HNSW index)
- **Flask** — HTTP API
- **PostgreSQL 17** — Database

## Agent Tools

| Tool | Description |
|------|-------------|
| `search_knowledge` | Semantic search across ingested documents via pgvector |
| `calculator` | Safe mathematical expression evaluator (arithmetic, trig, log, etc.) |
| `get_current_date` | Returns current date, time, and day of week |

## Setup

### Prerequisites
- PostgreSQL 17 with `pgvector` extension
- Python venv at `../venv/`
- NVIDIA API key at `~/.config/nvidia/api_key`

### Run

```bash
# Activate venv
source ../venv/bin/activate

# Start the server
python app.py
# → http://localhost:5006

# Seed with example data
python seed.py
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/chat` | Chat with the agent |
| `POST` | `/ingest` | Add a document |
| `GET` | `/documents` | List all documents |
| `GET` | `/tools` | List available tools |

### Chat API

```bash
curl -X POST http://localhost:5006/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuál es la población de India?", "session_id": "test"}'
```

Response:
```json
{
  "answer": "La población de India es de 1,442 millones de habitantes...",
  "tools_used": [
    {
      "name": "search_knowledge",
      "args": {"query": "población India"},
      "result": "[Estadísticas Población Mundial 2024]..."
    }
  ],
  "session_id": "test"
}
```

### Ingest API

```bash
curl -X POST http://localhost:5006/ingest \
  -H "Content-Type: application/json" \
  -d '{"title": "My Document", "content": "Long text content here..."}'
```

## Database Schema

```sql
-- Documents table
lc_documents (id, title, content, metadata, created_at)

-- Chunks with vector embeddings
lc_chunks (id, document_id, content, embedding vector(1024), metadata, created_at)
-- HNSW index for fast cosine similarity search
```

## Key Concepts

### LangChain Agents vs Chains
- **Chain**: Fixed sequence (embed → search → generate)
- **Agent**: Dynamic — LLM decides which tools to use and in what order

### ReAct Pattern
The agent follows the **Reasoning + Acting** loop:
1. **Think**: What does the user need?
2. **Act**: Call appropriate tool(s)
3. **Observe**: Review tool output
4. **Repeat** or **Respond**: Continue reasoning or give final answer

### Conversation Memory
Uses a sliding window of the last 5 exchanges, keeping context without excessive token usage.

## Port

- **5006** (default)
