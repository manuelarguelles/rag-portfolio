"""
Project 5 – Real-Time RAG Assistant
SSE streaming · Conversation memory · Live knowledge ingestion
"""

import json, os, re, asyncio
from datetime import datetime
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ── Config ──────────────────────────────────────────────────────────────
NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE    = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL    = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL     = "moonshotai/kimi-k2.5"
DB_URL         = "postgresql://macdenix@localhost/rag_portfolio"
EMBED_DIM      = 1024
TOP_K          = 4

app = FastAPI(title="Real-Time RAG Assistant")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ── DB helpers ──────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DB_URL)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rt_knowledge (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT NOT NULL,
                    source TEXT,
                    embedding vector(1024),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            # safe idempotent index
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE indexname = 'rt_knowledge_embedding_idx'
                    ) THEN
                        CREATE INDEX rt_knowledge_embedding_idx
                            ON rt_knowledge USING hnsw (embedding vector_cosine_ops);
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rt_conversations (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rt_messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES rt_conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunks_used JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        conn.commit()

init_db()

# ── NVIDIA helpers ──────────────────────────────────────────────────────
async def get_embedding(text: str) -> list[float]:
    """Get embedding vector from NVIDIA API."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{NVIDIA_BASE}/embeddings",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            json={
                "model": EMBED_MODEL,
                "input": [text],
                "input_type": "query",
                "encoding_format": "float",
            },
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


async def stream_chat(messages: list[dict]):
    """Yield SSE chunks from NVIDIA chat completions (streaming)."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{NVIDIA_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


def vector_search(embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """Search rt_knowledge by cosine similarity."""
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title, content, source,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM rt_knowledge
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, vec_str, top_k),
            )
            return [dict(r) for r in cur.fetchall()]


async def scrape_url(url: str) -> str:
    """Basic URL text scrape."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "RAG-Bot/1.0"})
        r.raise_for_status()
        text = r.text
        # strip HTML tags crudely
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]  # limit


# ── Knowledge endpoints ─────────────────────────────────────────────────
@app.post("/knowledge")
async def add_knowledge(request: Request):
    body = await request.json()
    content = body.get("content", "").strip()
    title   = body.get("title", "")
    source  = body.get("source", "")
    url     = body.get("url", "").strip()

    if url and not content:
        try:
            content = await scrape_url(url)
            source = url
            if not title:
                title = url[:80]
        except Exception as e:
            raise HTTPException(400, f"Failed to scrape URL: {e}")

    if not content:
        raise HTTPException(400, "content or url required")

    embedding = await get_embedding(content[:2000])
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO rt_knowledge (title, content, source, embedding)
                   VALUES (%s, %s, %s, %s::vector) RETURNING id, created_at""",
                (title, content, source, vec_str),
            )
            row = cur.fetchone()
        conn.commit()

    return {"id": row[0], "created_at": str(row[1]), "title": title}


@app.get("/knowledge")
def list_knowledge():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, source, LEFT(content, 200) as preview, created_at "
                "FROM rt_knowledge ORDER BY created_at DESC"
            )
            return [dict(r) for r in cur.fetchall()]


@app.delete("/knowledge/{kid}")
def delete_knowledge(kid: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rt_knowledge WHERE id=%s RETURNING id", (kid,))
            if not cur.fetchone():
                raise HTTPException(404, "Not found")
        conn.commit()
    return {"deleted": kid}


# ── Conversation endpoints ──────────────────────────────────────────────
@app.post("/conversations")
def create_conversation():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO rt_conversations DEFAULT VALUES RETURNING id, created_at")
            row = cur.fetchone()
        conn.commit()
    return {"id": row[0], "created_at": str(row[1])}


@app.get("/conversations")
def list_conversations():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT c.id, c.created_at,
                       (SELECT content FROM rt_messages
                        WHERE conversation_id = c.id AND role='user'
                        ORDER BY id LIMIT 1) as first_message,
                       (SELECT COUNT(*) FROM rt_messages WHERE conversation_id = c.id) as msg_count
                FROM rt_conversations c
                ORDER BY c.created_at DESC
            """)
            return [dict(r) for r in cur.fetchall()]


@app.get("/conversations/{cid}/messages")
def get_messages(cid: int):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, role, content, chunks_used, created_at "
                "FROM rt_messages WHERE conversation_id=%s ORDER BY id",
                (cid,),
            )
            return [dict(r) for r in cur.fetchall()]


# ── Chat (SSE streaming) ────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    conversation_id = body.get("conversation_id")
    user_message    = body.get("message", "").strip()
    if not conversation_id or not user_message:
        raise HTTPException(400, "conversation_id and message required")

    # Save user message
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rt_messages (conversation_id, role, content) VALUES (%s,'user',%s) RETURNING id",
                (conversation_id, user_message),
            )
        conn.commit()

    # Get conversation history (last 5 messages)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content FROM rt_messages "
                "WHERE conversation_id=%s ORDER BY id DESC LIMIT 10",
                (conversation_id,),
            )
            history = list(reversed([dict(r) for r in cur.fetchall()]))

    # Vector search
    query_embedding = await get_embedding(user_message)
    chunks = vector_search(query_embedding)
    chunks_info = [{"id": c["id"], "title": c["title"], "similarity": float(c["similarity"])} for c in chunks]

    # Build context
    context_parts = []
    for c in chunks:
        sim = float(c["similarity"])
        if sim > 0.3:
            label = c["title"] or c["source"] or f"chunk-{c['id']}"
            context_parts.append(f"[{label} | sim={sim:.2f}]\n{c['content'][:1500]}")

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "(no relevant knowledge found)"

    system_prompt = (
        "You are a helpful Real-Time RAG assistant. Answer the user's question using "
        "the provided knowledge base context when relevant. If the context doesn't contain "
        "relevant information, say so and answer from general knowledge. "
        "Be concise but thorough. Use markdown formatting when helpful.\n\n"
        f"## Knowledge Base Context\n\n{context_block}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    # Add conversation history (skip the system, add as user/assistant pairs)
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    async def event_stream():
        full_response = []
        # Send search results event first
        yield f"event: search\ndata: {json.dumps(chunks_info)}\n\n"
        try:
            async for token in stream_chat(messages):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        # Save assistant response
        assistant_content = "".join(full_response)
        if assistant_content:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO rt_messages (conversation_id, role, content, chunks_used) "
                        "VALUES (%s, 'assistant', %s, %s::jsonb)",
                        (conversation_id, assistant_content, json.dumps(chunks_info)),
                    )
                conn.commit()

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── UI ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8005, reload=True)
