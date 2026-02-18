# 07 — GraphRAG Pipeline

**Knowledge Graph + Vector Search** — A hybrid retrieval system that combines structured graph traversal with semantic vector search, all powered by PostgreSQL.

## What is GraphRAG?

Traditional RAG (Retrieval-Augmented Generation) finds relevant text chunks using vector similarity and feeds them to an LLM. This works well for factual lookups but struggles with **relationship questions** like:

- *"How are Pizarro and Atahualpa connected?"*
- *"What inventions led to the Industrial Revolution?"*
- *"Which moons orbit Jupiter?"*

**GraphRAG** solves this by adding a **knowledge graph layer**:

```
Traditional RAG:  Question → Vector Search → Chunks → LLM → Answer
GraphRAG:         Question → Vector Search → Entities → Graph Traversal
                                                ↘ Chunks ↗ → LLM → Answer
```

The graph captures **entities** (people, places, concepts) and **relationships** between them, enabling multi-hop reasoning that pure vector search can't do.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                         │
│                                                                 │
│  Document → Chunking → Embeddings → gr_chunks (vector search)  │
│         ↘                                                       │
│          LLM Extraction → Entities → gr_entities (+ embeddings) │
│                        → Relations → gr_relationships           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                            │
│                                                                 │
│  Question → Embed → Similar Entities (vector search)            │
│                   → Graph Traversal (1-hop, 2-hop SQL JOINs)    │
│                   → Similar Chunks (vector search)              │
│                   → Combined Context → LLM → Answer             │
└─────────────────────────────────────────────────────────────────┘
```

## GraphRAG vs Traditional RAG

| Aspect | Traditional RAG | GraphRAG |
|--------|----------------|----------|
| **Retrieval** | Vector similarity on chunks | Vector + graph traversal |
| **Relationships** | Implicit in text | Explicit as graph edges |
| **Multi-hop** | Limited (needs info in same chunk) | Natural (follow edges) |
| **Structure** | Flat chunks | Entities + relationships + chunks |
| **Best for** | Factual lookups | Relationship & reasoning questions |
| **Extraction** | Split text only | LLM extracts structured knowledge |

## Database Schema (PostgreSQL)

Instead of Neo4j, we simulate a knowledge graph using PostgreSQL with pgvector:

```
gr_documents ──┬── gr_chunks (with vector embeddings)
               └── gr_entities (with vector embeddings)
                       │
                   gr_relationships (source ↔ target)
```

- **gr_entities**: Nodes with type (PERSON, PLACE, ORGANIZATION, CONCEPT, DATE)
- **gr_relationships**: Directed edges with typed relationships (CONQUERED, FOUNDED, etc.)
- **gr_chunks**: Text fragments with embeddings for traditional vector search
- Graph traversal via SQL JOINs replaces Cypher/Gremlin queries

## Tech Stack

- **Backend**: Flask + psycopg2 + pgvector
- **LLM**: Kimi K2.5 (via NVIDIA API) for entity extraction and answer generation
- **Embeddings**: NVIDIA NV-EmbedQA-E5-v5 (1024 dimensions)
- **Database**: PostgreSQL 17 with vector extension
- **Frontend**: vis-network for interactive graph visualization

## Setup

```bash
# Activate shared venv
source ../venv/bin/activate

# Start server (creates tables automatically)
python app.py

# In another terminal, seed example data
source ../venv/bin/activate
python seed.py

# Open browser
open http://localhost:5007
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface with graph visualization |
| `POST` | `/ingest` | Ingest document, extract entities & relationships |
| `POST` | `/query` | Hybrid query (vector + graph traversal) |
| `GET` | `/graph` | Full graph as JSON (nodes + edges) |
| `GET` | `/entities` | List all entities with types |
| `GET` | `/documents` | List all documents with stats |
| `GET` | `/stats` | Aggregate statistics |

### Ingest Example

```bash
curl -X POST http://localhost:5007/ingest \
  -H "Content-Type: application/json" \
  -d '{"title": "Test", "content": "Marie Curie discovered radium in Paris..."}'
```

### Query Example

```bash
curl -X POST http://localhost:5007/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What did Pizarro do in Peru?"}'
```

## Features

- **LLM Entity Extraction**: Automatically extracts people, places, organizations, concepts, and dates from text
- **Hybrid Search**: Combines vector similarity (entities + chunks) with graph traversal (1-hop and 2-hop relationships)
- **Interactive Graph**: Force-directed visualization with color-coded entity types
- **Graph Traversal**: SQL JOINs simulate graph queries, following relationships to find connected entities
- **Entity Deduplication**: Existing entities are matched by name to avoid duplicates across documents

## Port

Runs on **port 5007**.
