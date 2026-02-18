# 02 — Document Analysis: PDF Processing with RAG

Upload PDFs, extract text, chunk it, generate embeddings, store in pgvector, and ask questions answered by an LLM using retrieved context.

## Architecture

```
┌──────────┐     ┌────────────┐     ┌──────────┐     ┌────────────┐
│  Upload   │────▶│  PyMuPDF   │────▶│ Chunking │────▶│ NVIDIA API │
│   PDF     │     │  Extract   │     │ 500c/100o│     │ Embeddings │
└──────────┘     └────────────┘     └──────────┘     └─────┬──────┘
                                                           │
                                                           ▼
┌──────────┐     ┌────────────┐     ┌──────────┐     ┌────────────┐
│ Response  │◀───│    LLM     │◀───│  Top-5    │◀───│  pgvector  │
│  + Cites  │    │ kimi-k2.5  │    │  Search   │    │   Store    │
└──────────┘     └────────────┘     └──────────┘     └────────────┘
```

## Flow

1. **Upload** — User uploads a PDF via the web UI (drag & drop or click)
2. **Extract** — PyMuPDF extracts text from each page
3. **Chunk** — Text is split into 500-char chunks with 100-char overlap
4. **Embed** — Each chunk is sent to NVIDIA's `nv-embedqa-e5-v5` model (1024-dim vectors)
5. **Store** — Chunks + embeddings are saved in PostgreSQL with pgvector (HNSW index)
6. **Query** — User asks a question → embedded → cosine similarity search → top 5 chunks retrieved
7. **Answer** — `kimi-k2.5` LLM generates an answer grounded in the retrieved context, citing sources

## Stack

| Component     | Technology                          |
|---------------|-------------------------------------|
| Backend       | FastAPI + Python 3                  |
| PDF Parsing   | PyMuPDF (fitz)                      |
| Vector DB     | PostgreSQL 17 + pgvector (HNSW)     |
| Embeddings    | NVIDIA NV-EmbedQA-E5-v5 (1024-dim) |
| LLM           | Moonshot Kimi K2.5 via NVIDIA API   |
| Frontend      | Vanilla HTML/CSS/JS                 |

## Setup

```bash
# Activate the shared venv
source ../venv/bin/activate

# Install extra dependency
pip install pymupdf

# Run (port 8001)
python app.py
```

Open [http://localhost:8001](http://localhost:8001)

## Endpoints

| Method   | Path                      | Description                          |
|----------|---------------------------|--------------------------------------|
| `GET`    | `/`                       | Web interface                        |
| `POST`   | `/upload`                 | Upload PDF (multipart/form-data)     |
| `POST`   | `/query`                  | Ask a question `{"question": "..."}` |
| `GET`    | `/documents`              | List all uploaded PDFs               |
| `DELETE` | `/documents/{id}`         | Delete a PDF and its chunks          |
| `GET`    | `/documents/{id}/chunks`  | View chunks of a specific PDF        |

## Constraints

- Max 50 pages per PDF
- Chunk size: 500 characters, overlap: 100 characters
- Embeddings generated in batches of 20

## Sample PDFs

The `sample-pdfs/` directory contains test PDFs generated with Python. You can also use any PDF you have:

```bash
# Generate sample PDFs
python generate_samples.py
```
