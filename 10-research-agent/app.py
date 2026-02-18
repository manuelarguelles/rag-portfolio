"""
Proyecto 10 — AI Research Agent
Agente de investigación automatizada con scraping, pgvector y LLM.
"""

import os
import json
import asyncio
import re
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import httpx
from bs4 import BeautifulSoup
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"
NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

llm = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

app = FastAPI(title="AI Research Agent", version="1.0.0")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ── Pydantic models ────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    topic: str

class AddSourceRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

# ── DB helpers ──────────────────────────────────────────────────────────
def get_conn():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ra_research_projects (
        id SERIAL PRIMARY KEY,
        topic TEXT NOT NULL,
        status TEXT DEFAULT 'pending'
            CHECK (status IN ('pending','researching','analyzing','completed','failed')),
        report TEXT,
        sources_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS ra_sources (
        id SERIAL PRIMARY KEY,
        project_id INTEGER REFERENCES ra_research_projects(id) ON DELETE CASCADE,
        url TEXT,
        title TEXT,
        content TEXT NOT NULL,
        embedding vector(1024),
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS ra_findings (
        id SERIAL PRIMARY KEY,
        project_id INTEGER REFERENCES ra_research_projects(id) ON DELETE CASCADE,
        category TEXT,
        finding TEXT NOT NULL,
        confidence TEXT CHECK (confidence IN ('high','medium','low')),
        source_ids INTEGER[],
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    # Index — create if not exists via DO block
    cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'ra_sources'
              AND indexdef LIKE '%hnsw%'
        ) THEN
            CREATE INDEX ON ra_sources USING hnsw (embedding vector_cosine_ops);
        END IF;
    END$$;
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ── Retry helper ────────────────────────────────────────────────────────
def with_retry(fn, max_retries=3, base_delay=5):
    """Retry a callable on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                delay = base_delay * (2 ** attempt)
                print(f"⏳ Rate limited, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
    return fn()  # Last attempt, let it raise

# ── Embedding helper ───────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    text = text[:2048]
    resp = with_retry(lambda: llm.embeddings.create(
        input=[text], model=EMBED_MODEL,
        extra_body={"input_type": "passage", "truncate": "END"}
    ))
    return resp.data[0].embedding

def get_query_embedding(text: str) -> list[float]:
    text = text[:2048]
    resp = with_retry(lambda: llm.embeddings.create(
        input=[text], model=EMBED_MODEL,
        extra_body={"input_type": "query", "truncate": "END"}
    ))
    return resp.data[0].embedding

# ── Text chunking ──────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks

# ── LLM helper ─────────────────────────────────────────────────────────
def chat(system: str, user: str, temperature: float = 0.3) -> str:
    resp = with_retry(lambda: llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=4096,
    ))
    return resp.choices[0].message.content.strip()

# ── Web scraping ───────────────────────────────────────────────────────
async def scrape_url(url: str) -> dict:
    """Scrape a URL and return {url, title, content}."""
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Research Agent Bot)"}
        ) as client:
            r = await client.get(url)
            r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Extract text from article or main or body
        main = soup.find("article") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else ""

        # Clean up
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
        content = "\n".join(lines)

        if len(content) < 100:
            return {"url": url, "title": title, "content": "", "error": "Content too short"}

        return {"url": url, "title": title, "content": content[:15000]}
    except Exception as e:
        return {"url": url, "title": url, "content": "", "error": str(e)}

# ── Research pipeline ──────────────────────────────────────────────────
def update_project_status(project_id: int, status: str, **kwargs):
    conn = get_conn()
    cur = conn.cursor()
    sets = ["status = %s"]
    vals = [status]
    for k, v in kwargs.items():
        sets.append(f"{k} = %s")
        vals.append(v)
    vals.append(project_id)
    cur.execute(f"UPDATE ra_research_projects SET {', '.join(sets)} WHERE id = %s", vals)
    conn.commit()
    cur.close()
    conn.close()

def generate_search_queries(topic: str) -> list[str]:
    raw = chat(
        "Eres un asistente de investigación. Genera exactamente 5 queries de búsqueda web "
        "para investigar el tema dado. Devuelve SOLO un JSON array de strings, sin texto extra.",
        f"Tema: {topic}"
    )
    # Extract JSON array
    try:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    # Fallback: split by lines
    return [l.strip().strip('"').strip("- ") for l in raw.splitlines() if l.strip()][:5]

def store_chunks(project_id: int, url: str, title: str, content: str) -> list[int]:
    """Chunk content, embed, and store. Returns list of source IDs."""
    chunks = chunk_text(content)
    if not chunks:
        return []

    conn = get_conn()
    cur = conn.cursor()
    ids = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        cur.execute(
            "INSERT INTO ra_sources (project_id, url, title, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (project_id, url, title, chunk, emb)
        )
        ids.append(cur.fetchone()[0])

    # Update sources count
    cur.execute(
        "UPDATE ra_research_projects SET sources_count = "
        "(SELECT COUNT(*) FROM ra_sources WHERE project_id = %s) WHERE id = %s",
        (project_id, project_id)
    )
    conn.commit()
    cur.close()
    conn.close()
    return ids

def analyze_findings(project_id: int):
    """LLM analyzes all sources and extracts categorized findings."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT id, content, title FROM ra_sources WHERE project_id = %s ORDER BY id", (project_id,))
    sources = cur.fetchall()
    cur.close()
    conn.close()

    if not sources:
        return

    # Build context (limit to ~12k chars)
    context_parts = []
    total = 0
    for s in sources:
        snippet = f"[Source {s['id']}] {s['title']}:\n{s['content'][:600]}"
        if total + len(snippet) > 12000:
            break
        context_parts.append(snippet)
        total += len(snippet)

    context = "\n\n".join(context_parts)

    raw = chat(
        "Eres un analista de investigación experto. Analiza las fuentes y extrae hallazgos clave. "
        "Devuelve SOLO un JSON array con objetos: "
        '{"category": "string", "finding": "string (2-3 oraciones)", '
        '"confidence": "high|medium|low", "source_ids": [int]}. '
        "Categorías sugeridas: Tendencias, Adopción, Desafíos, Oportunidades, Regulación, Impacto, Tecnología. "
        "Extrae entre 4 y 8 findings.",
        f"Fuentes de investigación:\n\n{context}"
    )

    try:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            findings = json.loads(match.group())
        else:
            findings = []
    except:
        findings = []

    if findings:
        conn = get_conn()
        cur = conn.cursor()
        for f in findings:
            source_ids = f.get("source_ids", [])
            # Validate source_ids exist
            valid_ids = [sid for sid in source_ids if any(s['id'] == sid for s in sources)]
            cur.execute(
                "INSERT INTO ra_findings (project_id, category, finding, confidence, source_ids) "
                "VALUES (%s, %s, %s, %s, %s)",
                (project_id, f.get("category", "General"), f["finding"],
                 f.get("confidence", "medium"), valid_ids if valid_ids else None)
            )
        conn.commit()
        cur.close()
        conn.close()

def generate_report(project_id: int) -> str:
    """Generate a structured research report using LLM."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT topic FROM ra_research_projects WHERE id = %s", (project_id,))
    project = cur.fetchone()

    cur.execute("SELECT * FROM ra_findings WHERE project_id = %s ORDER BY category, id", (project_id,))
    findings = cur.fetchall()

    cur.execute("SELECT DISTINCT url, title FROM ra_sources WHERE project_id = %s AND url IS NOT NULL", (project_id,))
    sources = cur.fetchall()

    cur.close()
    conn.close()

    findings_text = "\n".join(
        f"- [{f['category']}] ({f['confidence']}): {f['finding']}" for f in findings
    ) if findings else "No se encontraron hallazgos específicos."

    sources_text = "\n".join(
        f"- {s['title']}: {s['url']}" for s in sources
    ) if sources else "Sin fuentes."

    report = chat(
        "Eres un redactor de reportes de investigación profesional. "
        "Genera un reporte estructurado en español con formato markdown. "
        "Incluye: título, resumen ejecutivo (2-3 párrafos), hallazgos principales "
        "(organizados por categoría), conclusiones y recomendaciones, y lista de fuentes. "
        "El reporte debe ser claro, profesional y bien estructurado.",
        f"Tema de investigación: {project['topic']}\n\n"
        f"Hallazgos:\n{findings_text}\n\n"
        f"Fuentes:\n{sources_text}",
        temperature=0.4
    )
    return report

# ── Seed URLs for demo (simulated web search) ─────────────────────────
SEED_URLS = {
    "default": [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
    ],
    "inteligencia artificial": [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://es.wikipedia.org/wiki/Inteligencia_artificial",
    ],
    "machine learning": [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
    ],
}

def get_seed_urls(topic: str) -> list[str]:
    """Get seed URLs based on topic keywords."""
    topic_lower = topic.lower()
    for key, urls in SEED_URLS.items():
        if key in topic_lower:
            return urls
    return SEED_URLS["default"]

# ── Background research pipeline ──────────────────────────────────────
async def run_research_pipeline(project_id: int, topic: str):
    """Full async pipeline: scrape → chunk → embed → analyze → report."""
    try:
        # Phase 1: Researching — scrape sources
        update_project_status(project_id, "researching")

        # Generate search queries for context
        queries = generate_search_queries(topic)

        # Scrape seed URLs
        seed_urls = get_seed_urls(topic)
        all_source_ids = []

        for url in seed_urls:
            result = await scrape_url(url)
            if result["content"]:
                ids = store_chunks(project_id, result["url"], result["title"], result["content"])
                all_source_ids.extend(ids)

        if not all_source_ids:
            update_project_status(project_id, "failed",
                                  report="No se pudo obtener contenido de las fuentes.")
            return

        # Phase 2: Analyzing — extract findings
        update_project_status(project_id, "analyzing")
        analyze_findings(project_id)

        # Phase 3: Generate report
        report = generate_report(project_id)
        update_project_status(project_id, "completed",
                              report=report,
                              completed_at=datetime.now())

    except Exception as e:
        update_project_status(project_id, "failed",
                              report=f"Error en la investigación: {str(e)}")

# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/research")
def list_projects():
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT id, topic, status, sources_count, created_at, completed_at "
                "FROM ra_research_projects ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {**r, "created_at": r["created_at"].isoformat() if r["created_at"] else None,
         "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None}
        for r in rows
    ]

@app.post("/research")
async def create_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO ra_research_projects (topic) VALUES (%s) RETURNING id",
        (req.topic,)
    )
    project_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    background_tasks.add_task(run_research_pipeline, project_id, req.topic)
    return {"project_id": project_id, "status": "pending", "topic": req.topic}

@app.get("/research/{project_id}")
def get_project(project_id: int):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM ra_research_projects WHERE id = %s", (project_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(404, "Project not found")
    return {
        **row,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
    }

@app.get("/research/{project_id}/report")
def get_report(project_id: int):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT id, topic, status, report FROM ra_research_projects WHERE id = %s", (project_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(404, "Project not found")
    return {"project_id": row["id"], "topic": row["topic"],
            "status": row["status"], "report": row["report"]}

@app.get("/research/{project_id}/findings")
def get_findings(project_id: int):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT id, category, finding, confidence, source_ids, created_at "
                "FROM ra_findings WHERE project_id = %s ORDER BY category, id", (project_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {**r, "created_at": r["created_at"].isoformat() if r["created_at"] else None}
        for r in rows
    ]

@app.get("/research/{project_id}/sources")
def get_sources(project_id: int):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, url, title, LEFT(content, 200) as preview, created_at "
        "FROM ra_sources WHERE project_id = %s ORDER BY id", (project_id,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {**r, "created_at": r["created_at"].isoformat() if r["created_at"] else None}
        for r in rows
    ]

@app.post("/research/{project_id}/add-source")
async def add_source(project_id: int, req: AddSourceRequest):
    # Verify project exists
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, status FROM ra_research_projects WHERE id = %s", (project_id,))
    proj = cur.fetchone()
    cur.close()
    conn.close()

    if not proj:
        raise HTTPException(404, "Project not found")

    # Scrape
    result = await scrape_url(req.url)
    if not result["content"]:
        raise HTTPException(400, f"No content extracted: {result.get('error', 'Unknown')}")

    ids = store_chunks(project_id, result["url"], result["title"], result["content"])
    return {
        "message": f"Added {len(ids)} chunks from {result['title']}",
        "source_ids": ids,
        "title": result["title"],
        "chunks": len(ids),
    }

@app.post("/research/{project_id}/query")
def query_research(project_id: int, req: QueryRequest):
    # Get relevant chunks via vector search
    q_emb = get_query_embedding(req.question)

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Verify project
    cur.execute("SELECT topic FROM ra_research_projects WHERE id = %s", (project_id,))
    proj = cur.fetchone()
    if not proj:
        cur.close()
        conn.close()
        raise HTTPException(404, "Project not found")

    # Vector search
    cur.execute("""
        SELECT id, url, title, content,
               1 - (embedding <=> %s::vector) as similarity
        FROM ra_sources
        WHERE project_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 5
    """, (q_emb, project_id, q_emb))
    chunks = cur.fetchall()

    # Get findings
    cur.execute("SELECT category, finding, confidence FROM ra_findings WHERE project_id = %s", (project_id,))
    findings = cur.fetchall()

    cur.close()
    conn.close()

    if not chunks:
        return {"answer": "No hay información disponible para este proyecto aún.", "sources": []}

    context = "\n\n".join(
        f"[Fuente: {c['title']}]\n{c['content']}" for c in chunks
    )
    findings_ctx = "\n".join(
        f"- [{f['category']}] {f['finding']}" for f in findings
    ) if findings else ""

    answer = chat(
        "Eres un asistente de investigación. Responde la pregunta basándote SOLO "
        "en el contexto proporcionado de la investigación. Si no hay información "
        "suficiente, dilo. Responde en español, de forma clara y concisa.",
        f"Tema de investigación: {proj['topic']}\n\n"
        f"Hallazgos previos:\n{findings_ctx}\n\n"
        f"Contexto relevante:\n{context}\n\n"
        f"Pregunta: {req.question}"
    )

    return {
        "answer": answer,
        "sources": [
            {"id": c["id"], "title": c["title"], "url": c["url"],
             "similarity": round(float(c["similarity"]), 3)}
            for c in chunks
        ],
    }

# ── Re-analyze endpoint (useful after adding new sources) ─────────────
@app.post("/research/{project_id}/reanalyze")
async def reanalyze(project_id: int, background_tasks: BackgroundTasks):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM ra_research_projects WHERE id = %s", (project_id,))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(404, "Project not found")

    # Clear old findings
    cur.execute("DELETE FROM ra_findings WHERE project_id = %s", (project_id,))
    conn.commit()
    cur.close()
    conn.close()

    async def _reanalyze():
        try:
            update_project_status(project_id, "analyzing")
            analyze_findings(project_id)
            conn2 = get_conn()
            cur2 = conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur2.execute("SELECT topic FROM ra_research_projects WHERE id = %s", (project_id,))
            proj = cur2.fetchone()
            cur2.close()
            conn2.close()
            report = generate_report(project_id)
            update_project_status(project_id, "completed",
                                  report=report, completed_at=datetime.now())
        except Exception as e:
            update_project_status(project_id, "failed",
                                  report=f"Error re-analyzing: {str(e)}")

    background_tasks.add_task(_reanalyze)
    return {"message": "Re-analysis started", "project_id": project_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
