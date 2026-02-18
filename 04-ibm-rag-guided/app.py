"""
IBM RAG Guided — Production Patterns
=====================================
Advanced RAG pipeline with query expansion, re-ranking, answer grounding,
and evaluation metrics inspired by IBM's production RAG approach.
"""

import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import Optional

import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

app = FastAPI(title="IBM RAG Guided")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ── Database ──────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(DB_URL)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ibm_documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ibm_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES ibm_documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector(1024),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                DO $$ BEGIN
                    CREATE INDEX ibm_chunks_embedding_idx
                        ON ibm_chunks USING hnsw (embedding vector_cosine_ops);
                EXCEPTION WHEN duplicate_table THEN NULL;
                END $$;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ibm_query_log (
                    id SERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    expanded_query TEXT,
                    answer TEXT,
                    relevance_score FLOAT,
                    faithfulness_score FLOAT,
                    chunks_used INTEGER,
                    latency_ms INTEGER,
                    mode TEXT DEFAULT 'advanced',
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            # Add mode column if missing (for existing tables)
            cur.execute("""
                DO $$ BEGIN
                    ALTER TABLE ibm_query_log ADD COLUMN mode TEXT DEFAULT 'advanced';
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)
        conn.commit()


# ── NVIDIA API helpers ────────────────────────────────────────────────────────

async def get_embedding(text: str, input_type: str = "passage") -> list[float]:
    """Get embedding from NVIDIA API with retry."""
    for attempt in range(5):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": EMBED_MODEL,
                    "input": [text],
                    "input_type": input_type,
                    "encoding_format": "float",
                },
            )
            if r.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
    r.raise_for_status()
    return []


async def get_embeddings_batch(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Get embeddings for multiple texts (batched)."""
    results = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": EMBED_MODEL,
                    "input": batch,
                    "input_type": input_type,
                    "encoding_format": "float",
                },
            )
            r.raise_for_status()
            data = r.json()["data"]
            results.extend([d["embedding"] for d in sorted(data, key=lambda x: x["index"])])
    return results


async def chat_completion(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Call NVIDIA chat completion API with retry on rate limit."""
    last_error = None
    for attempt in range(8):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": CHAT_MODEL,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                if r.status_code == 429:
                    wait = min(3 * (attempt + 1), 30)
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            await asyncio.sleep(3)
    raise last_error or Exception("chat_completion failed after retries")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks if chunks else [text]


# ── RAG Pipeline Steps ────────────────────────────────────────────────────────

async def expand_query(query: str) -> list[str]:
    """Step 1: Query expansion — generate 2-3 variations of the user query."""
    prompt = f"""You are a search query expander. Given a user question, generate exactly 3 alternative phrasings
that capture different aspects or ways to ask the same question. Return ONLY a JSON array of strings.

User question: {query}

Return format: ["variation1", "variation2", "variation3"]"""

    response = await chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
    )

    try:
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            variations = json.loads(match.group())
            return [query] + [v for v in variations if isinstance(v, str)][:3]
    except (json.JSONDecodeError, TypeError):
        pass
    return [query]


async def retrieve_chunks(query_text: str, top_k: int = 10) -> list[dict]:
    """Step 2: Retrieve top-k chunks using pgvector cosine similarity."""
    embedding = await get_embedding(query_text, input_type="query")
    emb_str = "[" + ",".join(str(x) for x in embedding) + "]"

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.id, c.content, c.document_id, d.title,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM ibm_chunks c
                JOIN ibm_documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (emb_str, emb_str, top_k),
            )
            return [dict(row) for row in cur.fetchall()]


async def retrieve_multi_query(queries: list[str], top_k_per_query: int = 10) -> list[dict]:
    """Retrieve chunks for multiple query variations and deduplicate."""
    all_chunks: dict[int, dict] = {}
    for q in queries:
        chunks = await retrieve_chunks(q, top_k_per_query)
        for c in chunks:
            cid = c["id"]
            if cid not in all_chunks or c["similarity"] > all_chunks[cid]["similarity"]:
                all_chunks[cid] = c
    # Sort by best similarity
    return sorted(all_chunks.values(), key=lambda x: x["similarity"], reverse=True)


async def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Step 3: LLM-based re-ranking of retrieved chunks."""
    if not chunks:
        return []

    # Build prompt for re-ranking
    chunk_texts = "\n\n".join(
        f"[CHUNK {i}] (from: {c['title']})\n{c['content'][:400]}"
        for i, c in enumerate(chunks[:15])  # Limit to 15 for prompt size
    )

    prompt = f"""You are a relevance judge. Score each chunk from 1-10 based on how relevant it is to the query.
Return ONLY a JSON array of objects with "index" and "score" fields.

Query: {query}

{chunk_texts}

Return format: [{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}, ...]
Score ALL chunks listed above."""

    response = await chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )

    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            score_map = {s["index"]: s["score"] for s in scores if isinstance(s, dict)}

            for i, c in enumerate(chunks[:15]):
                c["rerank_score"] = score_map.get(i, 5)

            reranked = sorted(chunks[:15], key=lambda x: x.get("rerank_score", 0), reverse=True)
            return reranked[:top_k]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Fallback: return top-k by similarity
    return chunks[:top_k]


async def generate_answer(query: str, chunks: list[dict]) -> str:
    """Step 4: Generate answer from top re-ranked chunks."""
    if not chunks:
        return "No relevant information found in the knowledge base."

    context = "\n\n---\n\n".join(
        f"[Source: {c['title']}]\n{c['content']}" for c in chunks
    )

    prompt = f"""You are a knowledgeable technical assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly. Cite sources when possible.

Context:
{context}

Question: {query}

Answer:"""

    return await chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )


async def check_grounding(answer: str, chunks: list[dict]) -> dict:
    """Step 5: Grounding check — verify answer faithfulness to source chunks."""
    context = "\n\n".join(c["content"] for c in chunks)

    prompt = f"""You are a faithfulness evaluator. Analyze whether each claim in the answer is supported by the source chunks.

Source chunks:
{context[:3000]}

Answer to evaluate:
{answer}

Evaluate and return a JSON object with:
- "faithfulness_score": float 0.0-1.0 (what fraction of claims are supported)
- "relevance_score": float 0.0-1.0 (how relevant is the answer to the source material)
- "unsupported_claims": list of strings (any claims not found in sources)
- "assessment": brief explanation

Return ONLY valid JSON."""

    response = await chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {
                "faithfulness_score": float(result.get("faithfulness_score", 0.5)),
                "relevance_score": float(result.get("relevance_score", 0.5)),
                "unsupported_claims": result.get("unsupported_claims", []),
                "assessment": result.get("assessment", ""),
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return {
        "faithfulness_score": 0.5,
        "relevance_score": 0.5,
        "unsupported_claims": [],
        "assessment": "Could not evaluate grounding.",
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ingest")
async def ingest_document(request: Request):
    """Ingest a document: chunk it, embed chunks, store in DB."""
    body = await request.json()
    title = body.get("title", "").strip()
    content = body.get("content", "").strip()
    category = body.get("category", "general").strip()

    if not title or not content:
        return JSONResponse({"error": "title and content are required"}, status_code=400)

    # Chunk the document
    chunks = chunk_text(content)

    # Get embeddings for all chunks
    embeddings = await get_embeddings_batch(chunks, input_type="passage")

    # Store in DB
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ibm_documents (title, content, category) VALUES (%s, %s, %s) RETURNING id",
                (title, content, category),
            )
            doc_id = cur.fetchone()[0]

            for chunk_text_item, emb in zip(chunks, embeddings):
                emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                cur.execute(
                    "INSERT INTO ibm_chunks (document_id, content, embedding) VALUES (%s, %s, %s::vector)",
                    (doc_id, chunk_text_item, emb_str),
                )
        conn.commit()

    return {"status": "ok", "document_id": doc_id, "chunks_created": len(chunks)}


@app.post("/query")
async def query_advanced(request: Request):
    """Advanced RAG pipeline: expansion → retrieval → reranking → generation → grounding."""
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    start = time.time()
    pipeline_steps = []

    # Step 1: Query Expansion
    t0 = time.time()
    expanded_queries = await expand_query(query)
    pipeline_steps.append({
        "step": "Query Expansion",
        "detail": expanded_queries,
        "duration_ms": int((time.time() - t0) * 1000),
    })

    # Step 2: Multi-query Retrieval
    t0 = time.time()
    retrieved = await retrieve_multi_query(expanded_queries, top_k_per_query=10)
    pipeline_steps.append({
        "step": "Retrieval",
        "detail": f"{len(retrieved)} unique chunks retrieved",
        "duration_ms": int((time.time() - t0) * 1000),
    })

    # Step 3: Re-ranking
    t0 = time.time()
    reranked = await rerank_chunks(query, retrieved, top_k=5)
    pipeline_steps.append({
        "step": "Re-ranking",
        "detail": [
            {"chunk_id": c["id"], "title": c["title"], "rerank_score": c.get("rerank_score", 0),
             "similarity": round(c["similarity"], 4)}
            for c in reranked
        ],
        "duration_ms": int((time.time() - t0) * 1000),
    })

    # Step 4: Generation
    t0 = time.time()
    answer = await generate_answer(query, reranked)
    pipeline_steps.append({
        "step": "Generation",
        "detail": "Answer generated from top-5 re-ranked chunks",
        "duration_ms": int((time.time() - t0) * 1000),
    })

    # Step 5: Grounding Check
    t0 = time.time()
    grounding = await check_grounding(answer, reranked)
    pipeline_steps.append({
        "step": "Grounding Check",
        "detail": grounding,
        "duration_ms": int((time.time() - t0) * 1000),
    })

    total_ms = int((time.time() - start) * 1000)

    # Log to DB
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO ibm_query_log
                   (query, expanded_query, answer, relevance_score, faithfulness_score, chunks_used, latency_ms, mode)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    query,
                    json.dumps(expanded_queries),
                    answer,
                    grounding["relevance_score"],
                    grounding["faithfulness_score"],
                    len(reranked),
                    total_ms,
                    "advanced",
                ),
            )
        conn.commit()

    return {
        "query": query,
        "answer": answer,
        "mode": "advanced",
        "metrics": {
            "relevance_score": grounding["relevance_score"],
            "faithfulness_score": grounding["faithfulness_score"],
            "chunks_used": len(reranked),
            "latency_ms": total_ms,
            "unsupported_claims": grounding.get("unsupported_claims", []),
            "assessment": grounding.get("assessment", ""),
        },
        "expanded_queries": expanded_queries,
        "pipeline": pipeline_steps,
        "sources": [
            {"id": c["id"], "title": c["title"], "similarity": round(c["similarity"], 4),
             "rerank_score": c.get("rerank_score", 0), "content": c["content"][:200]}
            for c in reranked
        ],
    }


@app.post("/query/simple")
async def query_simple(request: Request):
    """Simple RAG pipeline: direct retrieval → generation (no expansion, reranking, or grounding)."""
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    start = time.time()

    # Direct retrieval
    chunks = await retrieve_chunks(query, top_k=5)

    # Direct generation
    answer = await generate_answer(query, chunks)

    total_ms = int((time.time() - start) * 1000)

    # Compute simple relevance (avg similarity)
    avg_sim = sum(c["similarity"] for c in chunks) / len(chunks) if chunks else 0

    # Log to DB
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO ibm_query_log
                   (query, expanded_query, answer, relevance_score, faithfulness_score, chunks_used, latency_ms, mode)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (query, None, answer, round(avg_sim, 4), None, len(chunks), total_ms, "simple"),
            )
        conn.commit()

    return {
        "query": query,
        "answer": answer,
        "mode": "simple",
        "metrics": {
            "relevance_score": round(avg_sim, 4),
            "faithfulness_score": None,
            "chunks_used": len(chunks),
            "latency_ms": total_ms,
        },
        "sources": [
            {"id": c["id"], "title": c["title"], "similarity": round(c["similarity"], 4),
             "content": c["content"][:200]}
            for c in chunks
        ],
    }


@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT d.id, d.title, d.category, d.created_at,
                       COUNT(c.id) AS chunk_count
                FROM ibm_documents d
                LEFT JOIN ibm_chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
            """)
            docs = cur.fetchall()
    return [
        {**dict(d), "created_at": d["created_at"].isoformat() if d["created_at"] else None}
        for d in docs
    ]


@app.get("/logs")
async def get_logs():
    """Get query log history with metrics."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, query, expanded_query, answer, relevance_score,
                       faithfulness_score, chunks_used, latency_ms, mode, created_at
                FROM ibm_query_log
                ORDER BY created_at DESC
                LIMIT 50
            """)
            logs = cur.fetchall()
    return [
        {**dict(l), "created_at": l["created_at"].isoformat() if l["created_at"] else None}
        for l in logs
    ]


@app.get("/logs/compare")
async def compare_logs():
    """Compare metrics between simple and advanced modes."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT mode,
                       COUNT(*) AS total_queries,
                       ROUND(AVG(relevance_score)::numeric, 4) AS avg_relevance,
                       ROUND(AVG(faithfulness_score)::numeric, 4) AS avg_faithfulness,
                       ROUND(AVG(chunks_used)::numeric, 1) AS avg_chunks,
                       ROUND(AVG(latency_ms)::numeric, 0) AS avg_latency_ms
                FROM ibm_query_log
                GROUP BY mode
            """)
            comparison = [dict(row) for row in cur.fetchall()]

            # Also get per-query pairs (same query, different modes)
            cur.execute("""
                SELECT a.query,
                       a.relevance_score AS adv_relevance,
                       a.faithfulness_score AS adv_faithfulness,
                       a.latency_ms AS adv_latency,
                       s.relevance_score AS simple_relevance,
                       s.latency_ms AS simple_latency
                FROM ibm_query_log a
                JOIN ibm_query_log s ON a.query = s.query AND s.mode = 'simple'
                WHERE a.mode = 'advanced'
                ORDER BY a.created_at DESC
                LIMIT 20
            """)
            pairs = [dict(row) for row in cur.fetchall()]

    return {"summary": comparison, "paired_comparisons": pairs}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document and its chunks."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ibm_documents WHERE id = %s RETURNING id", (doc_id,))
            deleted = cur.fetchone()
        conn.commit()
    if deleted:
        return {"status": "deleted", "id": doc_id}
    return JSONResponse({"error": "not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
