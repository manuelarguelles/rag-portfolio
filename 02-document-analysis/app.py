"""
02 - Document Analysis: PDF Processing with RAG + pgvector
Upload PDFs → extract text → chunk → embed → store → query with LLM
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
import httpx
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ── Config ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"

DB_URL = "postgresql://macdenix@localhost/rag_portfolio"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_PAGES = 50
EMBEDDING_BATCH_SIZE = 20  # NVIDIA API batch limit

app = FastAPI(title="Document Analysis", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Database ─────────────────────────────────────────────────────────────
def get_db():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


# ── NVIDIA API helpers ───────────────────────────────────────────────────
async def get_embeddings(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Get embeddings from NVIDIA API in batches."""
    all_embeddings = []
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            # Truncate very long texts to avoid API errors
            batch = [t[:2048] for t in batch]
            payload = {
                "model": EMBEDDING_MODEL,
                "input": batch,
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END",
            }
            for attempt in range(3):
                resp = await client.post(
                    f"{NVIDIA_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 429:
                    await asyncio.sleep((attempt + 1) * 5)
                    continue
                break
            if resp.status_code != 200:
                raise HTTPException(500, f"Embedding API error: {resp.text}")
            data = resp.json()
            for item in sorted(data["data"], key=lambda x: x["index"]):
                all_embeddings.append(item["embedding"])

    return all_embeddings


async def chat_completion(system_prompt: str, user_message: str) -> str:
    """Call NVIDIA chat completion API with retry on rate limit."""
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
        "max_tokens": 4096,
    }

    max_retries = 3
    async with httpx.AsyncClient(timeout=300.0) as client:
        for attempt in range(max_retries):
            resp = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 429:
                wait = (attempt + 1) * 10  # 10s, 20s, 30s
                await asyncio.sleep(wait)
                continue
            if resp.status_code != 200:
                raise HTTPException(500, f"Chat API error: {resp.text}")
            msg = resp.json()["choices"][0]["message"]
            # kimi-k2.5 is a reasoning model: content may be null if tokens
            # were exhausted on reasoning; fall back to reasoning_content
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning_content", "")
            return content.strip()

    raise HTTPException(503, "Chat API rate limited — please try again in a moment")


# ── PDF Processing ───────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> list[dict]:
    """Extract text from each page of a PDF. Returns list of {page, text}."""
    doc = fitz.open(file_path)
    if doc.page_count > MAX_PAGES:
        doc.close()
        raise HTTPException(
            400, f"PDF has {doc.page_count} pages (max {MAX_PAGES})"
        )
    pages = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks of ~CHUNK_SIZE chars."""
    chunks = []
    chunk_index = 0

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page_num,
                    "content": chunk_content,
                })
                chunk_index += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, chunk, embed, and store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    # Save file
    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    try:
        # Extract text
        pages = extract_text_from_pdf(str(file_path))
        if not pages:
            raise HTTPException(400, "No text could be extracted from this PDF")

        total_pages = max(p["page"] for p in pages)

        # Chunk
        chunks = chunk_text(pages)
        if not chunks:
            raise HTTPException(400, "No chunks generated from PDF text")

        # Generate embeddings
        texts = [c["content"] for c in chunks]
        embeddings = await get_embeddings(texts, input_type="passage")

        # Store in DB
        conn = get_db()
        cur = conn.cursor()

        # Insert document record
        cur.execute(
            """INSERT INTO pdf_documents (filename, total_pages, total_chunks)
               VALUES (%s, %s, %s) RETURNING id""",
            (file.filename, total_pages, len(chunks)),
        )
        doc_id = cur.fetchone()[0]

        # Insert chunks with embeddings
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                """INSERT INTO pdf_chunks (document_id, chunk_index, page_number, content, embedding)
                   VALUES (%s, %s, %s, %s, %s::vector)""",
                (
                    doc_id,
                    chunk["chunk_index"],
                    chunk["page_number"],
                    chunk["content"],
                    json.dumps(emb),
                ),
            )

        cur.close()
        conn.close()

        return {
            "id": doc_id,
            "filename": file.filename,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "message": f"Successfully processed {file.filename}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing PDF: {str(e)}")
    finally:
        # Clean up uploaded file (data is in DB now)
        if file_path.exists():
            file_path.unlink()


@app.post("/query")
async def query_documents(request: Request):
    """Query uploaded documents using semantic search + LLM."""
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(400, "Question is required")

    # Get question embedding
    embeddings = await get_embeddings([question], input_type="query")
    query_embedding = json.dumps(embeddings[0])

    # Search top-5 similar chunks
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT pc.id, pc.content, pc.page_number, pc.chunk_index,
                  pd.filename,
                  1 - (pc.embedding <=> %s::vector) AS similarity
           FROM pdf_chunks pc
           JOIN pdf_documents pd ON pd.id = pc.document_id
           ORDER BY pc.embedding <=> %s::vector
           LIMIT 5""",
        (query_embedding, query_embedding),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    # Build context for LLM
    context_parts = []
    sources = []
    for r in results:
        context_parts.append(
            f"[Source: {r['filename']}, Page {r['page_number']}]\n{r['content']}"
        )
        sources.append({
            "filename": r["filename"],
            "page_number": r["page_number"],
            "chunk_index": r["chunk_index"],
            "similarity": round(float(r["similarity"]), 4),
            "preview": r["content"][:200] + ("..." if len(r["content"]) > 200 else ""),
        })

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a helpful document analysis assistant. Answer the user's question based ONLY on the provided document context. 
If the context doesn't contain enough information to answer, say so clearly.
Always cite which document and page number your answer comes from.
Be concise but thorough. Use the same language as the question."""

    user_message = f"""Context from uploaded documents:

{context}

---

Question: {question}"""

    answer = await chat_completion(system_prompt, user_message)

    return {"answer": answer, "sources": sources}


@app.get("/documents")
async def list_documents():
    """List all uploaded PDF documents."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT id, filename, total_pages, total_chunks, created_at
           FROM pdf_documents ORDER BY created_at DESC"""
    )
    docs = cur.fetchall()
    cur.close()
    conn.close()

    # Serialize datetimes
    for d in docs:
        d["created_at"] = d["created_at"].isoformat()

    return docs


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document and all its chunks."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM pdf_documents WHERE id = %s RETURNING filename", (doc_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(404, "Document not found")

    return {"message": f"Deleted '{row[0]}' and all its chunks"}


@app.get("/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: int):
    """Get all chunks for a specific document."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Verify document exists
    cur.execute("SELECT filename FROM pdf_documents WHERE id = %s", (doc_id,))
    doc = cur.fetchone()
    if not doc:
        cur.close()
        conn.close()
        raise HTTPException(404, "Document not found")

    cur.execute(
        """SELECT id, chunk_index, page_number, content, created_at
           FROM pdf_chunks WHERE document_id = %s
           ORDER BY chunk_index""",
        (doc_id,),
    )
    chunks = cur.fetchall()
    cur.close()
    conn.close()

    for c in chunks:
        c["created_at"] = c["created_at"].isoformat()

    return {"filename": doc["filename"], "chunks": chunks}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
