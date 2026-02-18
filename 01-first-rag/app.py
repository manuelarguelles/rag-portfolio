"""
First RAG System — Backend FastAPI
===================================
Sistema RAG básico: ingestar textos → embeddings → pgvector → búsqueda → LLM.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ── Configuración ──────────────────────────────────────────────────────────

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://macdenix@localhost/rag_portfolio")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "moonshotai/kimi-k2.5")
EMBEDDING_DIM = 1024
TOP_K = 5  # Número de documentos similares a recuperar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Base de datos ──────────────────────────────────────────────────────────

def get_db():
    """Obtiene una conexión a PostgreSQL."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


def init_db():
    """Crea la tabla y el índice si no existen."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    # Índice HNSW para búsqueda por similitud coseno
    cur.execute("""
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'documents' AND indexdef LIKE '%hnsw%';
    """)
    if not cur.fetchone():
        cur.execute(f"""
            CREATE INDEX ON documents
            USING hnsw (embedding vector_cosine_ops);
        """)
        logger.info("Índice HNSW creado.")
    cur.close()
    conn.close()
    logger.info("Base de datos inicializada correctamente.")


# ── NVIDIA NIM API ─────────────────────────────────────────────────────────

async def get_embedding(text: str, input_type: str = "query") -> list[float]:
    """Genera un embedding usando NVIDIA NIM API."""
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY no configurada")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{NVIDIA_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json={
                    "model": EMBEDDING_MODEL,
                    "input": [text],
                    "input_type": input_type,
                    "encoding_format": "float",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Error NVIDIA embeddings: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=502, detail=f"Error al generar embedding: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error de conexión con NVIDIA API: {e}")
            raise HTTPException(status_code=502, detail="No se pudo conectar con NVIDIA API")


async def chat_completion(messages: list[dict]) -> str:
    """Genera una respuesta usando el LLM de NVIDIA NIM."""
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY no configurada")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json={
                    "model": CHAT_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Error NVIDIA chat: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=502, detail=f"Error al generar respuesta: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error de conexión con NVIDIA chat API: {e}")
            raise HTTPException(status_code=502, detail="No se pudo conectar con NVIDIA Chat API")


# ── FastAPI App ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la base de datos al arrancar."""
    init_db()
    yield

app = FastAPI(
    title="First RAG System",
    description="Sistema RAG básico con PostgreSQL + pgvector + NVIDIA NIM",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


# ── Modelos Pydantic ───────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    title: str
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K

class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Sirve la interfaz web."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ingest")
async def ingest_document(req: IngestRequest):
    """
    Ingesta un documento: genera embedding y guarda en pgvector.
    """
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="Título y contenido son requeridos")

    # Generar embedding del contenido
    embedding = await get_embedding(req.content, input_type="passage")

    # Guardar en la base de datos
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO documents (title, content, embedding)
               VALUES (%s, %s, %s::vector)
               RETURNING id, created_at""",
            (req.title.strip(), req.content.strip(), str(embedding)),
        )
        row = cur.fetchone()
        doc_id, created_at = row[0], row[1]
        logger.info(f"Documento ingestado: id={doc_id}, title='{req.title}'")
        return {
            "message": "Documento ingestado correctamente",
            "id": doc_id,
            "title": req.title.strip(),
            "created_at": str(created_at),
        }
    except Exception as e:
        logger.error(f"Error al insertar documento: {e}")
        raise HTTPException(status_code=500, detail="Error al guardar en la base de datos")
    finally:
        cur.close()
        conn.close()


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """
    Consulta RAG: embedding de la pregunta → búsqueda vectorial → LLM.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta es requerida")

    # 1. Generar embedding de la pregunta
    query_embedding = await get_embedding(req.question.strip(), input_type="query")

    # 2. Buscar documentos similares en pgvector
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            f"""
            SELECT id, title, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (str(query_embedding), str(query_embedding), req.top_k),
        )
        results = cur.fetchall()
    except Exception as e:
        logger.error(f"Error en búsqueda vectorial: {e}")
        raise HTTPException(status_code=500, detail="Error en búsqueda vectorial")
    finally:
        cur.close()
        conn.close()

    if not results:
        return QueryResponse(
            answer="No hay documentos en la base de datos. Ingesta algunos documentos primero.",
            sources=[],
        )

    # 3. Construir contexto para el LLM
    context_parts = []
    sources = []
    for doc in results:
        context_parts.append(f"[{doc['title']}]\n{doc['content']}")
        sources.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"][:200] + ("..." if len(doc["content"]) > 200 else ""),
            "similarity": round(float(doc["similarity"]), 4),
        })

    context = "\n\n---\n\n".join(context_parts)

    # 4. Generar respuesta con el LLM
    system_prompt = (
        "Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado. "
        "Si la información no está en el contexto, dilo claramente. "
        "Responde en español de forma clara y concisa. "
        "Cita las fuentes cuando sea relevante."
    )
    user_prompt = f"Contexto:\n{context}\n\nPregunta: {req.question.strip()}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = await chat_completion(messages)

    return QueryResponse(answer=answer, sources=sources)


@app.get("/documents")
async def list_documents():
    """Lista todos los documentos almacenados."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            "SELECT id, title, content, created_at FROM documents ORDER BY created_at DESC"
        )
        docs = cur.fetchall()
        return [
            {
                "id": d["id"],
                "title": d["title"],
                "content": d["content"],
                "created_at": str(d["created_at"]),
            }
            for d in docs
        ]
    except Exception as e:
        logger.error(f"Error al listar documentos: {e}")
        raise HTTPException(status_code=500, detail="Error al obtener documentos")
    finally:
        cur.close()
        conn.close()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Elimina un documento por su ID."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM documents WHERE id = %s RETURNING id", (doc_id,))
        deleted = cur.fetchone()
        if not deleted:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        logger.info(f"Documento eliminado: id={doc_id}")
        return {"message": f"Documento {doc_id} eliminado correctamente"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar documento: {e}")
        raise HTTPException(status_code=500, detail="Error al eliminar documento")
    finally:
        cur.close()
        conn.close()


@app.get("/stats")
async def get_stats():
    """Estadísticas del sistema."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("SELECT COUNT(*) as total FROM documents")
        total = cur.fetchone()["total"]
        cur.execute("""
            SELECT COUNT(*) as with_embedding
            FROM documents WHERE embedding IS NOT NULL
        """)
        with_embedding = cur.fetchone()["with_embedding"]
        return {
            "total_documents": total,
            "documents_with_embedding": with_embedding,
            "embedding_dimension": EMBEDDING_DIM,
            "embedding_model": EMBEDDING_MODEL,
            "chat_model": CHAT_MODEL,
            "database": "PostgreSQL + pgvector",
        }
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error al obtener estadísticas")
    finally:
        cur.close()
        conn.close()


# ── Ejecución directa ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
