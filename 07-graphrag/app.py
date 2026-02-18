"""
Project 7 — GraphRAG Pipeline
Knowledge Graph simulated with PostgreSQL + pgvector.

Extracts entities & relationships from text using LLM,
builds a knowledge graph, and combines graph traversal
with vector search to answer questions.
"""

import os, json, re, textwrap, time
from datetime import datetime

import psycopg2
import psycopg2.extras
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"
API_KEY = open(os.path.expanduser("~/.config/nvidia/api_key")).read().strip()
BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"
EMBED_DIM = 1024

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
app = Flask(__name__)

# ── Database helpers ────────────────────────────────────────────────────
def get_db():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gr_documents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gr_entities (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES gr_documents(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            description TEXT,
            embedding vector(1024),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gr_relationships (
            id SERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES gr_entities(id) ON DELETE CASCADE,
            target_id INTEGER REFERENCES gr_entities(id) ON DELETE CASCADE,
            rel_type TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gr_chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES gr_documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding vector(1024),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    # Indexes — CREATE INDEX IF NOT EXISTS
    cur.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'gr_entities_embedding_idx') THEN
                CREATE INDEX gr_entities_embedding_idx ON gr_entities USING hnsw (embedding vector_cosine_ops);
            END IF;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'gr_chunks_embedding_idx') THEN
                CREATE INDEX gr_chunks_embedding_idx ON gr_chunks USING hnsw (embedding vector_cosine_ops);
            END IF;
        END $$;
    """)
    cur.close()
    conn.close()

# ── Embedding helper ────────────────────────────────────────────────────
def _retry(fn, max_retries=5, base_delay=2):
    """Retry a function with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e) or "rate" in str(e).lower():
                delay = base_delay * (2 ** attempt)
                print(f"  ⏳ Rate limited, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
    return fn()  # Last attempt, let it raise

def embed(text: str) -> list[float]:
    """Get 1024-dim embedding for a text string."""
    def _call():
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=[text[:2048]],
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return resp.data[0].embedding
    return _retry(_call)

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in small batches with rate limit handling."""
    if not texts:
        return []
    results = []
    batch_size = 5  # Small batches to avoid rate limits
    for i in range(0, len(texts), batch_size):
        batch = [t[:2048] for t in texts[i:i+batch_size]]
        def _call(b=batch):
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=b,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "END"},
            )
            return [d.embedding for d in resp.data]
        batch_result = _retry(_call)
        results.extend(batch_result)
        if i + batch_size < len(texts):
            time.sleep(1)  # Small delay between batches
    return results

# ── LLM helpers ─────────────────────────────────────────────────────────
ENTITY_EXTRACT_PROMPT = textwrap.dedent("""\
You are an expert at extracting structured information from text.
Given the following text, extract ALL entities and relationships.

For entities, identify:
- PERSON: People, historical figures, characters
- PLACE: Cities, countries, regions, geographical features
- ORGANIZATION: Companies, institutions, governments, armies
- CONCEPT: Abstract ideas, technologies, events, processes
- DATE: Specific dates, time periods, eras

For relationships, identify how entities are connected.

Return ONLY valid JSON with this exact structure:
{
  "entities": [
    {"name": "Entity Name", "type": "PERSON|PLACE|ORGANIZATION|CONCEPT|DATE", "description": "Brief description"}
  ],
  "relationships": [
    {"source": "Entity Name A", "target": "Entity Name B", "type": "RELATIONSHIP_TYPE", "description": "Brief description of the relationship"}
  ]
}

Relationship types should be descriptive verbs/phrases like:
CONQUERED, FOUNDED, LOCATED_IN, LED, PARTICIPATED_IN, CAUSED, PRECEDED, SUCCEEDED, ALLIED_WITH, OPPOSED, CREATED, DISCOVERED, PART_OF, BORN_IN, DIED_IN, RULED, etc.

Extract as many entities and relationships as possible. Be thorough.
Do NOT include any text outside the JSON block.

TEXT:
""")

def extract_entities_and_relations(text: str) -> dict:
    """Use LLM to extract entities and relationships from text."""
    def _call():
        return client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You extract structured knowledge from text. Always respond with valid JSON only."},
                {"role": "user", "content": ENTITY_EXTRACT_PROMPT + text},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
    resp = _retry(_call)
    raw = resp.choices[0].message.content.strip()
    # Try to parse JSON — handle markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to find JSON object in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"entities": [], "relationships": []}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks if chunks else [text]


# ── Ingestion Pipeline ──────────────────────────────────────────────────
def ingest_document(title: str, content: str) -> dict:
    """Full ingestion pipeline: store doc, extract entities, embed everything."""
    conn = get_db()
    cur = conn.cursor()

    # 1. Store document
    cur.execute(
        "INSERT INTO gr_documents (title, content) VALUES (%s, %s) RETURNING id",
        (title, content),
    )
    doc_id = cur.fetchone()[0]

    # 2. Chunk and embed text
    chunks = chunk_text(content)
    chunk_embeddings = embed_batch(chunks)
    for chunk_text_val, chunk_emb in zip(chunks, chunk_embeddings):
        cur.execute(
            "INSERT INTO gr_chunks (document_id, content, embedding) VALUES (%s, %s, %s::vector)",
            (doc_id, chunk_text_val, str(chunk_emb)),
        )

    # 3. Extract entities & relationships via LLM
    extracted = extract_entities_and_relations(content)
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])

    # 4. Store entities with embeddings
    entity_map = {}  # name -> id
    entity_texts = [f"{e['name']}: {e.get('description', e['name'])}" for e in entities]
    entity_embeddings = embed_batch(entity_texts) if entity_texts else []

    for ent, emb in zip(entities, entity_embeddings):
        name = ent["name"].strip()
        etype = ent.get("type", "CONCEPT").strip()
        desc = ent.get("description", "")

        # Check if entity already exists (by name, case-insensitive)
        cur.execute("SELECT id FROM gr_entities WHERE LOWER(name) = LOWER(%s)", (name,))
        existing = cur.fetchone()
        if existing:
            entity_map[name.lower()] = existing[0]
            # Update embedding if better (has description now)
            if desc:
                cur.execute(
                    "UPDATE gr_entities SET description = %s, embedding = %s::vector WHERE id = %s",
                    (desc, str(emb), existing[0]),
                )
        else:
            cur.execute(
                "INSERT INTO gr_entities (document_id, name, entity_type, description, embedding) VALUES (%s, %s, %s, %s, %s::vector) RETURNING id",
                (doc_id, name, etype, desc, str(emb)),
            )
            entity_map[name.lower()] = cur.fetchone()[0]

    # 5. Store relationships
    rel_count = 0
    for rel in relationships:
        src_name = rel.get("source", "").strip().lower()
        tgt_name = rel.get("target", "").strip().lower()
        if src_name in entity_map and tgt_name in entity_map:
            cur.execute(
                "INSERT INTO gr_relationships (source_id, target_id, rel_type, description) VALUES (%s, %s, %s, %s)",
                (entity_map[src_name], entity_map[tgt_name], rel["type"], rel.get("description", "")),
            )
            rel_count += 1

    cur.close()
    conn.close()

    return {
        "document_id": doc_id,
        "chunks": len(chunks),
        "entities": len(entities),
        "relationships": rel_count,
    }


# ── Query Pipeline ──────────────────────────────────────────────────────
def graph_query(question: str, top_k: int = 5) -> dict:
    """
    Hybrid query: vector search for similar entities + graph traversal
    for related entities + chunk search for context.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    q_emb = embed(question)
    q_emb_str = str(q_emb)

    # 1. Find similar entities by vector search
    cur.execute("""
        SELECT id, name, entity_type, description,
               1 - (embedding <=> %s::vector) AS similarity
        FROM gr_entities
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (q_emb_str, q_emb_str, top_k))
    similar_entities = cur.fetchall()

    # 2. Graph traversal: find related entities (1-hop and 2-hop)
    entity_ids = [e["id"] for e in similar_entities]
    related_entities = []
    graph_paths = []

    if entity_ids:
        # 1-hop: direct relationships
        cur.execute("""
            SELECT DISTINCT
                e1.name AS source_name, e1.entity_type AS source_type,
                r.rel_type, r.description AS rel_desc,
                e2.name AS target_name, e2.entity_type AS target_type
            FROM gr_relationships r
            JOIN gr_entities e1 ON r.source_id = e1.id
            JOIN gr_entities e2 ON r.target_id = e2.id
            WHERE r.source_id = ANY(%s) OR r.target_id = ANY(%s)
            LIMIT 30
        """, (entity_ids, entity_ids))
        one_hop = cur.fetchall()
        graph_paths.extend(one_hop)

        # Get IDs of 1-hop neighbors for 2-hop
        neighbor_ids = set()
        for row in one_hop:
            # Get IDs from names
            cur.execute("SELECT id FROM gr_entities WHERE name = %s", (row["source_name"],))
            r = cur.fetchone()
            if r:
                neighbor_ids.add(r["id"])
            cur.execute("SELECT id FROM gr_entities WHERE name = %s", (row["target_name"],))
            r = cur.fetchone()
            if r:
                neighbor_ids.add(r["id"])

        # 2-hop: relationships of neighbors (excluding already found)
        new_ids = list(neighbor_ids - set(entity_ids))
        if new_ids:
            cur.execute("""
                SELECT DISTINCT
                    e1.name AS source_name, e1.entity_type AS source_type,
                    r.rel_type, r.description AS rel_desc,
                    e2.name AS target_name, e2.entity_type AS target_type
                FROM gr_relationships r
                JOIN gr_entities e1 ON r.source_id = e1.id
                JOIN gr_entities e2 ON r.target_id = e2.id
                WHERE r.source_id = ANY(%s) OR r.target_id = ANY(%s)
                LIMIT 20
            """, (new_ids, new_ids))
            two_hop = cur.fetchall()
            graph_paths.extend(two_hop)

    # 3. Vector search on chunks for text context
    cur.execute("""
        SELECT c.content, d.title,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM gr_chunks c
        JOIN gr_documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """, (q_emb_str, q_emb_str, top_k))
    relevant_chunks = cur.fetchall()

    cur.close()
    conn.close()

    # 4. Build context for LLM
    entity_context = "\n".join([
        f"- {e['name']} ({e['entity_type']}): {e['description'] or 'No description'} [similarity: {e['similarity']:.3f}]"
        for e in similar_entities
    ])

    graph_context = "\n".join([
        f"- {p['source_name']} --[{p['rel_type']}]--> {p['target_name']}: {p['rel_desc'] or ''}"
        for p in graph_paths
    ])

    chunk_context = "\n\n".join([
        f"[From: {c['title']}]\n{c['content']}"
        for c in relevant_chunks
    ])

    # 5. Generate answer with LLM
    system_prompt = textwrap.dedent("""\
    You are a knowledge graph-enhanced assistant. You answer questions using
    structured knowledge (entities and relationships) combined with text passages.

    When answering:
    1. Use the graph relationships to understand connections between entities
    2. Use the text chunks for detailed context
    3. Cite specific entities and relationships in your answer
    4. If you mention an entity, wrap it in **bold**
    5. Be precise and informative
    """)

    user_prompt = textwrap.dedent(f"""\
    QUESTION: {question}

    RELEVANT ENTITIES (by vector similarity):
    {entity_context or "No entities found."}

    KNOWLEDGE GRAPH RELATIONSHIPS:
    {graph_context or "No relationships found."}

    TEXT CONTEXT:
    {chunk_context or "No text context found."}

    Answer the question using the above information. Reference specific entities and relationships.
    """)

    def _llm_call():
        return client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
    resp = _retry(_llm_call)
    answer = resp.choices[0].message.content.strip()

    # Collect mentioned entity names
    mentioned_entities = list(set(
        e["name"] for e in similar_entities
    ))

    return {
        "answer": answer,
        "entities_used": [dict(e) for e in similar_entities],
        "graph_paths": [dict(p) for p in graph_paths],
        "chunks_used": len(relevant_chunks),
        "mentioned_entities": mentioned_entities,
    }


# ── Flask Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.json
    title = data.get("title", "Untitled")
    content = data.get("content", "")
    if not content.strip():
        return jsonify({"error": "Content is required"}), 400
    try:
        result = ingest_document(title, content)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question is required"}), 400
    try:
        result = graph_query(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/graph")
def graph():
    """Return full graph as nodes + edges JSON for visualization."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT id, name, entity_type, description
        FROM gr_entities
        ORDER BY id
    """)
    entities = cur.fetchall()

    cur.execute("""
        SELECT r.id, r.source_id, r.target_id, r.rel_type, r.description,
               e1.name AS source_name, e2.name AS target_name
        FROM gr_relationships r
        JOIN gr_entities e1 ON r.source_id = e1.id
        JOIN gr_entities e2 ON r.target_id = e2.id
        ORDER BY r.id
    """)
    relationships = cur.fetchall()

    cur.close()
    conn.close()

    nodes = [
        {"id": e["id"], "label": e["name"], "type": e["entity_type"], "description": e["description"] or ""}
        for e in entities
    ]
    edges = [
        {"id": r["id"], "from": r["source_id"], "to": r["target_id"],
         "label": r["rel_type"], "description": r["description"] or "",
         "source_name": r["source_name"], "target_name": r["target_name"]}
        for r in relationships
    ]

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/entities")
def entities():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT e.id, e.name, e.entity_type, e.description, d.title AS document_title
        FROM gr_entities e
        LEFT JOIN gr_documents d ON e.document_id = d.id
        ORDER BY e.entity_type, e.name
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/documents")
def documents():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT d.id, d.title, d.created_at,
               LENGTH(d.content) AS content_length,
               (SELECT COUNT(*) FROM gr_entities e WHERE e.document_id = d.id) AS entity_count,
               (SELECT COUNT(*) FROM gr_chunks c WHERE c.document_id = d.id) AS chunk_count
        FROM gr_documents d
        ORDER BY d.created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for r in rows:
        rd = dict(r)
        rd["created_at"] = rd["created_at"].isoformat() if rd["created_at"] else None
        result.append(rd)
    return jsonify(result)


@app.route("/stats")
def stats():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT COUNT(*) AS count FROM gr_documents")
    doc_count = cur.fetchone()["count"]

    cur.execute("SELECT entity_type, COUNT(*) AS count FROM gr_entities GROUP BY entity_type ORDER BY count DESC")
    entity_stats = cur.fetchall()

    cur.execute("SELECT COUNT(*) AS count FROM gr_relationships")
    rel_count = cur.fetchone()["count"]

    cur.execute("SELECT COUNT(*) AS count FROM gr_chunks")
    chunk_count = cur.fetchone()["count"]

    cur.close()
    conn.close()

    return jsonify({
        "documents": doc_count,
        "entity_types": [dict(r) for r in entity_stats],
        "total_entities": sum(r["count"] for r in entity_stats),
        "relationships": rel_count,
        "chunks": chunk_count,
    })


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(port=5007, debug=True)
