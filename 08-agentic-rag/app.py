"""
Proyecto 8: Agentic RAG — Multi-Agent Autonomous System
Agentes puros en Python (sin LangChain/CrewAI).
"""

import os, json, time, asyncio, textwrap, logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────
API_KEY = open(os.path.expanduser("~/.config/nvidia/api_key")).read().strip()
BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"

llm = OpenAI(base_url=BASE_URL, api_key=API_KEY)

app = FastAPI(title="Agentic RAG — Multi-Agent System")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# ── Database helpers ─────────────────────────────────────────────────────
def get_conn():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ag_knowledge_bases (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS ag_documents (
            id SERIAL PRIMARY KEY,
            kb_id INTEGER REFERENCES ag_knowledge_bases(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS ag_chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES ag_documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding vector(1024),
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS ag_task_log (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            agent_trace JSONB NOT NULL,
            final_answer TEXT,
            total_steps INTEGER,
            total_latency_ms INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    # Index (idempotent via IF NOT EXISTS on PG 15+, otherwise ignore error)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS ag_chunks_hnsw_idx ON ag_chunks USING hnsw (embedding vector_cosine_ops);")
    except Exception:
        conn.rollback()
    conn.commit()
    cur.close()
    conn.close()

def get_embedding(text: str) -> list[float]:
    """Get embedding vector from NVIDIA API."""
    resp = llm.embeddings.create(input=[text], model=EMBED_MODEL, extra_body={"input_type": "query", "truncate": "END"})
    return resp.data[0].embedding

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks

# ── Agent System ─────────────────────────────────────────────────────────

class Agent:
    """Base autonomous agent that thinks via LLM."""

    def __init__(self, name: str, role: str, instructions: str, icon: str = "🤖"):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.icon = icon

    def think(self, context: str, temperature: float = 0.3) -> dict:
        """Call LLM with role + instructions + context. Returns structured dict."""
        system_prompt = f"""You are {self.name}, a specialized AI agent.
Role: {self.role}
Instructions: {self.instructions}

You MUST respond with valid JSON only. No markdown, no extra text.
"""
        t0 = time.time()
        resp = None
        for attempt in range(8):
            try:
                resp = llm.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = min(3 * (attempt + 1), 20)
                    logging.warning(f"[{self.name}] Rate limit hit, retrying in {wait}s (attempt {attempt+1}/8)")
                    time.sleep(wait)
                else:
                    raise
        if resp is None:
            return {
                "action": "error", "result": "LLM rate limited after retries",
                "reasoning": "Could not get response from LLM",
                "_latency_ms": int((time.time() - t0) * 1000),
                "_agent": self.name, "_icon": self.icon,
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        latency_ms = int((time.time() - t0) * 1000)
        raw = resp.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown code blocks)
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            raw = raw.strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"action": "raw_response", "result": raw, "reasoning": "Could not parse structured JSON"}
        result["_latency_ms"] = latency_ms
        result["_agent"] = self.name
        result["_icon"] = self.icon
        result["_timestamp"] = datetime.now(timezone.utc).isoformat()
        return result


class RouterAgent(Agent):
    """Decides which specialist agents to invoke."""

    def __init__(self):
        super().__init__(
            name="Router",
            role="Query Router & Strategy Planner",
            instructions=textwrap.dedent("""\
                Analyze the user query and decide which specialist agents should handle it.
                Available agents: research, analyst, both.
                - "research": query needs information retrieval from knowledge bases
                - "analyst": query needs data analysis, comparison, or synthesis
                - "both": query needs both retrieval AND analysis

                Also identify which knowledge bases are most relevant.
                
                Respond with JSON:
                {
                    "action": "route",
                    "strategy": "research" | "analyst" | "both",
                    "reasoning": "why this strategy",
                    "target_topics": ["list of relevant topics/keywords"],
                    "sub_queries": ["refined sub-queries for each agent"]
                }
            """),
            icon="🧭",
        )


class ResearchAgent(Agent):
    """Searches pgvector knowledge bases for relevant information."""

    def __init__(self):
        super().__init__(
            name="Research",
            role="Knowledge Base Researcher",
            instructions=textwrap.dedent("""\
                You receive search results from the knowledge base.
                Analyze them, extract the most relevant information, and organize it clearly.
                
                Respond with JSON:
                {
                    "action": "research_complete",
                    "result": "organized findings as clear text",
                    "key_facts": ["list of key facts found"],
                    "confidence": 0.0-1.0,
                    "reasoning": "how you selected and organized this information"
                }
            """),
            icon="🔍",
        )

    def search_knowledge(self, query: str, top_k: int = 8) -> list[dict]:
        """Search pgvector for relevant chunks."""
        embedding = get_embedding(query)
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.id, c.content, d.title AS doc_title, kb.name AS kb_name,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM ag_chunks c
            JOIN ag_documents d ON d.id = c.document_id
            JOIN ag_knowledge_bases kb ON kb.id = d.kb_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, (embedding, embedding, top_k))
        results = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        for r in results:
            r["similarity"] = float(r["similarity"])
        return results

    def execute(self, query: str) -> dict:
        """Full research: search + analyze."""
        results = self.search_knowledge(query)
        if not results:
            return {
                "action": "research_complete",
                "result": "No relevant information found in the knowledge bases.",
                "key_facts": [],
                "confidence": 0.0,
                "reasoning": "No chunks matched the query.",
                "chunks_found": 0,
                "_agent": self.name,
                "_icon": self.icon,
                "_timestamp": datetime.now(timezone.utc).isoformat(),
                "_latency_ms": 0,
            }

        # Build context with search results
        context = f"User query: {query}\n\nSearch results from knowledge base:\n\n"
        for i, r in enumerate(results, 1):
            context += f"[{i}] (KB: {r['kb_name']}, Doc: {r['doc_title']}, Sim: {r['similarity']:.3f})\n{r['content']}\n\n"

        result = self.think(context)
        result["chunks_found"] = len(results)
        result["top_similarity"] = max(r["similarity"] for r in results)
        result["sources"] = [{"kb": r["kb_name"], "doc": r["doc_title"], "similarity": r["similarity"]} for r in results[:5]]
        return result


class AnalystAgent(Agent):
    """Analyzes data, compares information, draws conclusions."""

    def __init__(self):
        super().__init__(
            name="Analyst",
            role="Data Analyst & Synthesizer",
            instructions=textwrap.dedent("""\
                You analyze data provided to you. Look for patterns, comparisons,
                trends, and draw analytical conclusions.
                
                Respond with JSON:
                {
                    "action": "analysis_complete",
                    "result": "your analytical findings",
                    "insights": ["key insights"],
                    "data_points": ["specific data points referenced"],
                    "reasoning": "your analytical methodology"
                }
            """),
            icon="📊",
        )

    def execute(self, query: str, research_data: Optional[str] = None) -> dict:
        """Analyze the query with optional research data."""
        context = f"User query: {query}\n"
        if research_data:
            context += f"\nResearch data provided:\n{research_data}\n"
        context += "\nProvide analytical insights based on the above."
        return self.think(context)


class WriterAgent(Agent):
    """Composes the final, polished response."""

    def __init__(self):
        super().__init__(
            name="Writer",
            role="Response Composer",
            instructions=textwrap.dedent("""\
                You compose the final response for the user based on inputs from other agents.
                Write a clear, well-structured, informative response.
                Use the research findings and analytical insights to create a comprehensive answer.
                Write in the same language as the user's query.
                
                Respond with JSON:
                {
                    "action": "response_complete",
                    "result": "your final composed response (can use markdown)",
                    "summary": "one-line summary",
                    "reasoning": "how you structured the response"
                }
            """),
            icon="✍️",
        )

    def compose(self, query: str, agent_outputs: list[dict]) -> dict:
        """Compose final response from all agent outputs."""
        context = f"Original user query: {query}\n\n"
        for output in agent_outputs:
            agent_name = output.get("_agent", "Unknown")
            context += f"--- {agent_name} Agent Output ---\n"
            context += f"Result: {output.get('result', 'N/A')}\n"
            if "key_facts" in output:
                context += f"Key facts: {json.dumps(output['key_facts'])}\n"
            if "insights" in output:
                context += f"Insights: {json.dumps(output['insights'])}\n"
            if "sources" in output:
                context += f"Sources: {json.dumps(output['sources'])}\n"
            context += "\n"
        context += "Compose a comprehensive, well-structured final response for the user."
        return self.think(context, temperature=0.5)


# ── Agent Orchestrator ───────────────────────────────────────────────────

class AgentOrchestrator:
    """Orchestrates the multi-agent pipeline."""

    def __init__(self):
        self.router = RouterAgent()
        self.researcher = ResearchAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

    def run(self, query: str) -> dict:
        """Execute the full multi-agent pipeline."""
        t0 = time.time()
        trace = []
        agent_outputs = []

        # Step 1: Router decides strategy
        route_result = self.router.think(
            f"User query: {query}\n\nDecide the best strategy to answer this query."
        )
        trace.append({
            "step": 1,
            "agent": "Router",
            "icon": "🧭",
            "action": route_result.get("action", "route"),
            "strategy": route_result.get("strategy", "both"),
            "reasoning": route_result.get("reasoning", ""),
            "latency_ms": route_result.get("_latency_ms", 0),
            "timestamp": route_result.get("_timestamp", ""),
        })

        strategy = route_result.get("strategy", "both")
        step = 2

        # Step 2: Execute specialist agents
        research_result = None
        if strategy in ("research", "both"):
            research_result = self.researcher.execute(query)
            trace.append({
                "step": step,
                "agent": "Research",
                "icon": "🔍",
                "action": research_result.get("action", "research_complete"),
                "chunks_found": research_result.get("chunks_found", 0),
                "top_similarity": research_result.get("top_similarity", 0),
                "confidence": research_result.get("confidence", 0),
                "reasoning": research_result.get("reasoning", ""),
                "latency_ms": research_result.get("_latency_ms", 0),
                "timestamp": research_result.get("_timestamp", ""),
            })
            agent_outputs.append(research_result)
            step += 1

        analyst_result = None
        if strategy in ("analyst", "both"):
            research_text = research_result.get("result", "") if research_result else None
            analyst_result = self.analyst.execute(query, research_text)
            trace.append({
                "step": step,
                "agent": "Analyst",
                "icon": "📊",
                "action": analyst_result.get("action", "analysis_complete"),
                "insights": analyst_result.get("insights", []),
                "reasoning": analyst_result.get("reasoning", ""),
                "latency_ms": analyst_result.get("_latency_ms", 0),
                "timestamp": analyst_result.get("_timestamp", ""),
            })
            agent_outputs.append(analyst_result)
            step += 1

        # Step 3: Writer composes final response
        writer_result = self.writer.compose(query, agent_outputs)
        trace.append({
            "step": step,
            "agent": "Writer",
            "icon": "✍️",
            "action": "response_complete",
            "summary": writer_result.get("summary", ""),
            "reasoning": writer_result.get("reasoning", ""),
            "latency_ms": writer_result.get("_latency_ms", 0),
            "timestamp": writer_result.get("_timestamp", ""),
        })

        total_ms = int((time.time() - t0) * 1000)
        final_answer = writer_result.get("result", "No response generated.")

        # Save trace to DB
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO ag_task_log (query, agent_trace, final_answer, total_steps, total_latency_ms)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (query, json.dumps(trace), final_answer, step, total_ms),
        )
        trace_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return {
            "trace_id": trace_id,
            "query": query,
            "answer": final_answer,
            "strategy": strategy,
            "trace": trace,
            "total_steps": step,
            "total_latency_ms": total_ms,
        }


orchestrator = AgentOrchestrator()

# ── Pydantic Models ──────────────────────────────────────────────────────

class KBCreate(BaseModel):
    name: str
    description: Optional[str] = None

class DocCreate(BaseModel):
    title: str
    content: str

class QueryRequest(BaseModel):
    query: str

# ── Endpoints ────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Knowledge Bases
@app.post("/knowledge-bases")
def create_kb(body: KBCreate):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO ag_knowledge_bases (name, description) VALUES (%s, %s) RETURNING id, name, description, created_at",
            (body.name, body.description),
        )
        row = cur.fetchone()
        conn.commit()
        return {"id": row[0], "name": row[1], "description": row[2], "created_at": str(row[3])}
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(400, f"Knowledge base '{body.name}' already exists")
    finally:
        cur.close()
        conn.close()

@app.get("/knowledge-bases")
def list_kbs():
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT kb.*, 
               COUNT(DISTINCT d.id) AS doc_count,
               COUNT(c.id) AS chunk_count
        FROM ag_knowledge_bases kb
        LEFT JOIN ag_documents d ON d.kb_id = kb.id
        LEFT JOIN ag_chunks c ON c.document_id = d.id
        GROUP BY kb.id
        ORDER BY kb.created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/knowledge-bases/{kb_id}/documents")
def add_document(kb_id: int, body: DocCreate):
    conn = get_conn()
    cur = conn.cursor()
    # Verify KB exists
    cur.execute("SELECT id FROM ag_knowledge_bases WHERE id = %s", (kb_id,))
    if not cur.fetchone():
        cur.close()
        conn.close()
        raise HTTPException(404, "Knowledge base not found")

    # Insert document
    cur.execute(
        "INSERT INTO ag_documents (kb_id, title, content) VALUES (%s, %s, %s) RETURNING id",
        (kb_id, body.title, body.content),
    )
    doc_id = cur.fetchone()[0]

    # Chunk and embed
    chunks = chunk_text(body.content)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        cur.execute(
            "INSERT INTO ag_chunks (document_id, content, embedding) VALUES (%s, %s, %s::vector)",
            (doc_id, chunk, embedding),
        )

    conn.commit()
    cur.close()
    conn.close()
    return {"document_id": doc_id, "chunks_created": len(chunks), "title": body.title}

# Query
@app.post("/query")
def query(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    result = orchestrator.run(body.query)
    return result

# Traces
@app.get("/traces")
def list_traces(limit: int = 20):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, query, total_steps, total_latency_ms, created_at FROM ag_task_log ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/traces/{trace_id}")
def get_trace(trace_id: int):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM ag_task_log WHERE id = %s", (trace_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(404, "Trace not found")
    return dict(row)

# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
