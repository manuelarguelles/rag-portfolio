"""
03 - Multi-Document RAG
Sistema RAG que maneja múltiples documentos organizados en colecciones,
permite buscar dentro de una colección específica o en todas,
y cita la fuente de cada fragmento relevante.
"""

import os
import json
import httpx
import psycopg2
import psycopg2.extras
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────
NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

app = FastAPI(title="Multi-Document RAG")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── Pydantic models ────────────────────────────────────────────────
class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None

class DocumentCreate(BaseModel):
    title: str
    content: str
    source: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    collection_id: Optional[int] = None
    top_k: int = 5

# ── DB helpers ──────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DB_URL)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS md_collections (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS md_documents (
                    id SERIAL PRIMARY KEY,
                    collection_id INTEGER REFERENCES md_collections(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    source TEXT,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS md_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES md_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(1024),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            # Index – create only if not exists
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE indexname = 'md_chunks_embedding_hnsw_idx'
                    ) THEN
                        CREATE INDEX md_chunks_embedding_hnsw_idx
                        ON md_chunks USING hnsw (embedding vector_cosine_ops);
                    END IF;
                END $$;
            """)
        conn.commit()

# ── Chunking ────────────────────────────────────────────────────────
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

def recursive_split(text: str, max_len: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Recursive character text splitter."""
    if len(text) <= max_len:
        return [text.strip()] if text.strip() else []

    # Find best separator
    for sep in SEPARATORS:
        if sep and sep in text:
            parts = text.split(sep)
            break
    else:
        # Hard split
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_len
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if c]

    # Merge parts into chunks respecting max_len
    chunks = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            # If a single part exceeds max_len, recurse
            if len(part) > max_len:
                chunks.extend(recursive_split(part, max_len, overlap))
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # Apply overlap: prepend tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            merged = prev_tail + " " + chunks[i]
            if len(merged) <= max_len + overlap:
                overlapped.append(merged)
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    return chunks

# ── NVIDIA API helpers ──────────────────────────────────────────────
async def get_embeddings(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Get embeddings from NVIDIA API. input_type: 'passage' for docs, 'query' for queries."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    all_embeddings = []
    # Process in batches of 50
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
            "input_type": input_type,
            "encoding_format": "float",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{NVIDIA_BASE}/embeddings", headers=headers, json=payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Embedding API error: {resp.text}")
            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in sorted_data])
    return all_embeddings

async def chat_completion(system_prompt: str, user_message: str) -> str:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{NVIDIA_BASE}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Chat API error: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

# ── Startup ─────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()

# ── Collections endpoints ───────────────────────────────────────────
@app.post("/collections")
async def create_collection(body: CollectionCreate):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(
                    "INSERT INTO md_collections (name, description) VALUES (%s, %s) RETURNING id, name, description, created_at",
                    (body.name, body.description),
                )
                row = cur.fetchone()
            except psycopg2.errors.UniqueViolation:
                raise HTTPException(status_code=409, detail="Collection name already exists")
        conn.commit()
    return dict(row)

@app.get("/collections")
async def list_collections():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT c.id, c.name, c.description, c.created_at,
                       COUNT(d.id) AS doc_count
                FROM md_collections c
                LEFT JOIN md_documents d ON d.collection_id = c.id
                GROUP BY c.id
                ORDER BY c.created_at DESC
            """)
            rows = cur.fetchall()
    return [dict(r) for r in rows]

@app.delete("/collections/{collection_id}")
async def delete_collection(collection_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM md_collections WHERE id = %s RETURNING id", (collection_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Collection not found")
        conn.commit()
    return {"deleted": collection_id}

# ── Documents endpoints ─────────────────────────────────────────────
@app.post("/collections/{collection_id}/documents")
async def add_document(collection_id: int, body: DocumentCreate):
    # Verify collection exists
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM md_collections WHERE id = %s", (collection_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Collection not found")

    # Chunk the document
    chunks = recursive_split(body.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document content is empty or could not be chunked")

    # Get embeddings for all chunks
    embeddings = await get_embeddings(chunks, input_type="passage")

    # Insert document + chunks in a transaction
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO md_documents (collection_id, title, source, content) VALUES (%s, %s, %s, %s) RETURNING id, title, source, created_at",
                (collection_id, body.title, body.source, body.content),
            )
            doc = cur.fetchone()
            doc_id = doc["id"]

            for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "collection_id": collection_id,
                    "document_id": doc_id,
                    "document_title": body.title,
                    "source": body.source,
                    "chunk_index": i,
                }
                cur.execute(
                    "INSERT INTO md_chunks (document_id, chunk_index, content, embedding, metadata) VALUES (%s, %s, %s, %s::vector, %s)",
                    (doc_id, i, chunk_text, str(emb), json.dumps(metadata)),
                )
        conn.commit()

    return {**dict(doc), "chunk_count": len(chunks)}

@app.get("/collections/{collection_id}/documents")
async def list_documents(collection_id: int):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id FROM md_collections WHERE id = %s", (collection_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Collection not found")
            cur.execute("""
                SELECT d.id, d.title, d.source, d.created_at,
                       COUNT(c.id) AS chunk_count
                FROM md_documents d
                LEFT JOIN md_chunks c ON c.document_id = d.id
                WHERE d.collection_id = %s
                GROUP BY d.id
                ORDER BY d.created_at DESC
            """, (collection_id,))
            rows = cur.fetchall()
    return [dict(r) for r in rows]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM md_documents WHERE id = %s RETURNING id", (document_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Document not found")
        conn.commit()
    return {"deleted": document_id}

# ── Query endpoint ──────────────────────────────────────────────────
@app.post("/query")
async def query(body: QueryRequest):
    # Embed the question
    q_emb = (await get_embeddings([body.question], input_type="query"))[0]

    # Search chunks (optionally filtered by collection)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if body.collection_id:
                cur.execute("""
                    SELECT c.id, c.content, c.chunk_index, c.metadata,
                           d.title AS doc_title, d.source AS doc_source,
                           col.name AS collection_name,
                           1 - (c.embedding <=> %s::vector) AS similarity
                    FROM md_chunks c
                    JOIN md_documents d ON d.id = c.document_id
                    JOIN md_collections col ON col.id = d.collection_id
                    WHERE d.collection_id = %s
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                """, (str(q_emb), body.collection_id, str(q_emb), body.top_k))
            else:
                cur.execute("""
                    SELECT c.id, c.content, c.chunk_index, c.metadata,
                           d.title AS doc_title, d.source AS doc_source,
                           col.name AS collection_name,
                           1 - (c.embedding <=> %s::vector) AS similarity
                    FROM md_chunks c
                    JOIN md_documents d ON d.id = c.document_id
                    JOIN md_collections col ON col.id = d.collection_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                """, (str(q_emb), str(q_emb), body.top_k))
            results = cur.fetchall()

    if not results:
        return {
            "answer": "No se encontraron documentos relevantes. Agrega documentos a una colección primero.",
            "sources": [],
        }

    # Build context with source citations
    context_parts = []
    sources = []
    for i, r in enumerate(results):
        label = f"[{r['collection_name']} → {r['doc_title']}, chunk #{r['chunk_index']}]"
        context_parts.append(f"Fuente {i+1} {label}:\n{r['content']}")
        sources.append({
            "index": i + 1,
            "collection": r["collection_name"],
            "document": r["doc_title"],
            "source": r["doc_source"],
            "chunk_index": r["chunk_index"],
            "similarity": round(float(r["similarity"]), 4),
            "preview": r["content"][:150] + "..." if len(r["content"]) > 150 else r["content"],
        })

    context = "\n\n".join(context_parts)

    system_prompt = """Eres un asistente de RAG multi-documento. Responde la pregunta del usuario basándote ÚNICAMENTE en las fuentes proporcionadas.

REGLAS:
1. Usa SOLO la información de las fuentes. No inventes datos.
2. Cita las fuentes usando el formato [Fuente N] al final de cada afirmación relevante.
3. Si las fuentes no contienen información suficiente, dilo claramente.
4. Responde en el mismo idioma de la pregunta.
5. Sé conciso pero completo."""

    user_msg = f"""Contexto de {len(results)} fuentes:

{context}

Pregunta: {body.question}"""

    answer = await chat_completion(system_prompt, user_msg)

    return {"answer": answer, "sources": sources}

# ── Web UI ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)
