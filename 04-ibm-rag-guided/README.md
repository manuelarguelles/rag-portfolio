# 04 — IBM RAG Guided: Production Patterns

Advanced RAG pipeline implementing production patterns inspired by IBM's approach to enterprise RAG systems.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Advanced (IBM) Pipeline                       │
│                                                                  │
│  Query ──→ Expansion ──→ Retrieval ──→ Re-ranking ──→ Answer    │
│              │              │             │              │        │
│         LLM generates   pgvector      LLM scores    LLM with    │
│         2-3 variants    top-10 per    relevance     top-5        │
│                         variant       1-10          chunks       │
│                                                         │        │
│                                                    Grounding     │
│                                                    Check         │
│                                                    (faithfulness │
│                                                     evaluation)  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     Simple Pipeline                              │
│                                                                  │
│  Query ──→ Retrieval ──→ Answer                                 │
│              │              │                                    │
│           pgvector       LLM with                               │
│           top-5          chunks                                  │
└──────────────────────────────────────────────────────────────────┘
```

## IBM Production Patterns Implemented

### 1. Query Expansion
The user's question is expanded into 2-3 alternative phrasings using the LLM. This improves recall by capturing different ways the same concept might be expressed in the knowledge base. All variations are searched, and results are deduplicated with the best similarity score retained.

### 2. Re-ranking
After initial retrieval via vector similarity, an LLM re-scores each chunk on a 1-10 relevance scale. This compensates for the limitations of embedding-based similarity by using the LLM's deeper understanding of semantic relevance. Only the top-5 re-ranked chunks proceed to generation.

### 3. Answer Grounding
After generating an answer, a separate LLM call verifies whether each claim in the answer is actually supported by the source chunks. This produces a **faithfulness score** (0.0-1.0) and identifies any unsupported claims. This is critical for enterprise use where hallucination must be detected.

### 4. Evaluation Metrics
Every query is logged with:
- **Relevance score** — How relevant the retrieved context is
- **Faithfulness score** — How grounded the answer is in the sources
- **Chunks used** — Number of source chunks in the answer
- **Latency** — End-to-end pipeline duration

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/ingest` | Ingest document (auto-chunk + embed) |
| `POST` | `/query` | Advanced pipeline (expansion + reranking + grounding) |
| `POST` | `/query/simple` | Simple pipeline (direct retrieval only) |
| `GET` | `/documents` | List all documents |
| `GET` | `/logs` | Query history with metrics |
| `GET` | `/logs/compare` | Compare simple vs advanced metrics |
| `DELETE` | `/documents/{id}` | Delete a document |

## Setup & Run

```bash
cd projects/rag-portfolio/04-ibm-rag-guided
source ../venv/bin/activate

# Seed database with 10 sample documents
python seed.py

# Start the server
python app.py
# → http://localhost:8004
```

## Tech Stack

- **Backend**: FastAPI + psycopg2
- **Vector DB**: PostgreSQL + pgvector (HNSW index, cosine similarity)
- **Embeddings**: NVIDIA `nv-embedqa-e5-v5` (1024 dims)
- **LLM**: `moonshotai/kimi-k2.5` via NVIDIA API
- **Frontend**: Vanilla HTML/CSS/JS, dark theme

## Simple vs Advanced: When to Use Each

| Aspect | Simple | Advanced (IBM) |
|--------|--------|----------------|
| Latency | ~2-4s | ~10-20s |
| Accuracy | Good for clear questions | Better for ambiguous queries |
| Faithfulness | Unknown | Measured (0-1 score) |
| Recall | Single query match | Multi-query expansion |
| Ranking | Embedding similarity only | LLM re-ranked |
| Cost | 2 API calls | 5+ API calls |

The advanced pipeline is ideal for enterprise settings where answer quality and verifiability outweigh latency considerations.
