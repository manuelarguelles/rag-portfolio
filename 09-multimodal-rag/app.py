"""
Proyecto 9: Multimodal RAG — Text + Images
Sistema RAG que combina documentos de texto e imágenes.
Las imágenes se indexan via descripción textual (text-bridge strategy).
"""

import os
import uuid
import json
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file, abort
from openai import OpenAI
import psycopg2
from psycopg2.extras import RealDictCursor
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"
THUMB_MAX = 300  # px

nv = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(DB_URL)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mm_items (
                    id SERIAL PRIMARY KEY,
                    item_type TEXT NOT NULL CHECK (item_type IN ('text', 'image')),
                    title TEXT NOT NULL,
                    content TEXT,
                    description TEXT,
                    image_path TEXT,
                    thumbnail_path TEXT,
                    embedding vector(1024),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                SELECT 1 FROM pg_indexes
                WHERE tablename='mm_items'
                  AND indexdef LIKE '%hnsw%';
            """)
            if not cur.fetchone():
                cur.execute("""
                    CREATE INDEX ON mm_items
                    USING hnsw (embedding vector_cosine_ops);
                """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mm_collections (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mm_item_collections (
                    item_id INTEGER REFERENCES mm_items(id) ON DELETE CASCADE,
                    collection_id INTEGER REFERENCES mm_collections(id) ON DELETE CASCADE,
                    PRIMARY KEY (item_id, collection_id)
                );
            """)
        conn.commit()


# ---------------------------------------------------------------------------
# Embedding & LLM
# ---------------------------------------------------------------------------

def _retry(fn, max_retries=5, base_delay=5):
    """Retry a function with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                delay = base_delay * (2 ** attempt)
                print(f"  ⏳ Rate limited, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
    return fn()  # final attempt


def embed_text(text: str) -> list[float]:
    """Generate a 1024-dim embedding for a piece of text."""
    def _call():
        resp = nv.embeddings.create(
            input=[text],
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return resp.data[0].embedding
    return _retry(_call)


def chat_answer(question: str, contexts: list[dict]) -> str:
    """Ask the LLM to answer using retrieved contexts."""
    ctx_parts = []
    for i, c in enumerate(contexts, 1):
        tag = f"[{'IMAGE' if c['item_type'] == 'image' else 'TEXT'} #{c['id']}] {c['title']}"
        body = c.get("content") or c.get("description") or ""
        ctx_parts.append(f"{tag}\n{body}")
    system = (
        "You are a helpful multimodal assistant. "
        "Answer the user question using ONLY the provided context items. "
        "If an item is an IMAGE, mention what the image depicts based on its description. "
        "If you cannot answer from the context, say so. Be concise."
    )
    user = f"Context:\n{'---'.join(ctx_parts)}\n\nQuestion: {question}"

    def _call():
        resp = nv.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    return _retry(_call)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def save_image(file_storage) -> tuple[str, str]:
    """Save original + thumbnail. Return (image_path, thumbnail_path) relative names."""
    ext = Path(file_storage.filename).suffix.lower() or ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    thumb_name = f"thumb_{name}"

    orig_path = UPLOAD_DIR / name
    thumb_path = UPLOAD_DIR / thumb_name

    file_storage.save(str(orig_path))

    img = Image.open(orig_path)
    img.thumbnail((THUMB_MAX, THUMB_MAX))
    img.save(str(thumb_path))

    return name, thumb_name


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---- Items CRUD ----------------------------------------------------------

@app.post("/items/text")
def add_text_item():
    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    metadata = data.get("metadata", {})
    if not title or not content:
        return jsonify({"error": "title and content required"}), 400

    emb = embed_text(f"{title} {content}")

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO mm_items (item_type, title, content, embedding, metadata)
                   VALUES ('text', %s, %s, %s::vector, %s::jsonb)
                   RETURNING id, item_type, title, content, created_at""",
                (title, content, str(emb), json.dumps(metadata)),
            )
            row = cur.fetchone()
        conn.commit()
    return jsonify(dict(row)), 201


@app.post("/items/image")
def add_image_item():
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    file = request.files["file"]
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    metadata_raw = request.form.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        metadata = {}

    if not title or not description:
        return jsonify({"error": "title and description required"}), 400

    image_name, thumb_name = save_image(file)
    emb = embed_text(f"{title} {description}")

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO mm_items
                   (item_type, title, description, image_path, thumbnail_path, embedding, metadata)
                   VALUES ('image', %s, %s, %s, %s, %s::vector, %s::jsonb)
                   RETURNING id, item_type, title, description, image_path, thumbnail_path, created_at""",
                (title, description, image_name, thumb_name, str(emb), json.dumps(metadata)),
            )
            row = cur.fetchone()
        conn.commit()
    return jsonify(dict(row)), 201


@app.get("/items")
def list_items():
    item_type = request.args.get("type")  # text | image | None
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if item_type in ("text", "image"):
                cur.execute(
                    """SELECT id, item_type, title, content, description,
                              image_path, thumbnail_path, metadata, created_at
                       FROM mm_items WHERE item_type=%s ORDER BY created_at DESC""",
                    (item_type,),
                )
            else:
                cur.execute(
                    """SELECT id, item_type, title, content, description,
                              image_path, thumbnail_path, metadata, created_at
                       FROM mm_items ORDER BY created_at DESC"""
                )
            rows = cur.fetchall()
    return jsonify([dict(r) for r in rows])


@app.delete("/items/<int:item_id>")
def delete_item(item_id):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch paths to clean up files
            cur.execute("SELECT image_path, thumbnail_path FROM mm_items WHERE id=%s", (item_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "not found"}), 404
            cur.execute("DELETE FROM mm_items WHERE id=%s", (item_id,))
        conn.commit()

    # Remove files
    for p in (row.get("image_path"), row.get("thumbnail_path")):
        if p:
            fp = UPLOAD_DIR / p
            if fp.exists():
                fp.unlink()

    return jsonify({"deleted": item_id})


# ---- Image serving -------------------------------------------------------

@app.get("/items/<int:item_id>/image")
def serve_image(item_id):
    thumb = request.args.get("thumb", "0") == "1"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT image_path, thumbnail_path FROM mm_items WHERE id=%s AND item_type='image'",
                (item_id,),
            )
            row = cur.fetchone()
    if not row:
        abort(404)
    fname = row["thumbnail_path"] if thumb else row["image_path"]
    fpath = UPLOAD_DIR / fname
    if not fpath.exists():
        abort(404)
    return send_file(fpath)


# ---- Query ---------------------------------------------------------------

@app.post("/query")
def query():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    top_k = data.get("top_k", 5)
    item_type = data.get("type")  # optional filter

    if not question:
        return jsonify({"error": "question required"}), 400

    emb = embed_text(question)

    type_filter = ""
    params: list = [str(emb), top_k]
    if item_type in ("text", "image"):
        type_filter = "WHERE item_type = %s"
        params = [str(emb), item_type, top_k]

    sql = f"""
        SELECT id, item_type, title, content, description,
               image_path, thumbnail_path, metadata, created_at,
               1 - (embedding <=> %s::vector) AS similarity
        FROM mm_items
        {type_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    if item_type in ("text", "image"):
        sql = f"""
            SELECT id, item_type, title, content, description,
                   image_path, thumbnail_path, metadata, created_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM mm_items
            WHERE item_type = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [str(emb), item_type, str(emb), top_k]
    else:
        sql = f"""
            SELECT id, item_type, title, content, description,
                   image_path, thumbnail_path, metadata, created_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM mm_items
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [str(emb), str(emb), top_k]

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["similarity"] = float(d["similarity"])
        results.append(d)

    # Generate answer
    try:
        answer = chat_answer(question, results)
    except Exception as e:
        answer = f"⚠️ Could not generate LLM answer ({e}). Showing retrieved results below."

    return jsonify({"answer": answer, "results": results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5009, debug=True)
