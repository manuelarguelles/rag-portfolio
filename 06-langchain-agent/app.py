"""
Proyecto 6: LangChain RAG Agent — Production Ready
Agent RAG con tools: vector search, calculator, current date.
"""

import os
import json
import math
import datetime
from pathlib import Path
from typing import List, Optional

from flask import Flask, request, jsonify, render_template
import psycopg2
import psycopg2.extras

# --- LangChain imports ---
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings as BaseEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "moonshotai/kimi-k2.5"
EMBED_DIMS = 1024
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"

app = Flask(__name__)

# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


# ─── LLM & Embeddings (NVIDIA NIM via OpenAI-compatible API) ─────────────────

llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    temperature=0.3,
    max_tokens=2048,
)

# Custom Embeddings for NVIDIA NIM (requires input_type for asymmetric models)
class NvidiaEmbeddings(BaseEmbeddings):
    """Custom embeddings wrapper for NVIDIA NIM asymmetric models."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def _embed(self, texts: list, input_type: str) -> list:
        resp = self._client.embeddings.create(
            input=texts,
            model=self._model,
            extra_body={"input_type": input_type},
        )
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: list) -> list:
        # Batch in groups of 50 to avoid API limits
        all_embeddings = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i+50]
            all_embeddings.extend(self._embed(batch, "passage"))
        return all_embeddings

    def embed_query(self, text: str) -> list:
        return self._embed([text], "query")[0]


embeddings = NvidiaEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)

# ─── Text Splitter ────────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information. Use this when the user asks about stored documents, facts, or any topic that might be in the database."""
    try:
        query_embedding = embeddings.embed_query(query)
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT c.content, c.metadata, d.title,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM lc_chunks c
            JOIN lc_documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT 5
        """, (query_embedding, query_embedding))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return "No relevant documents found in the knowledge base."

        results = []
        for r in rows:
            sim = float(r["similarity"])
            results.append(f"[{r['title']}] (similarity: {sim:.3f})\n{r['content']}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /, **), math functions (sqrt, sin, cos, log, etc.), and constants (pi, e). Example: 'sqrt(144) + 3 * 2'"""
    try:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "int": int, "float": float,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "log2": math.log2, "exp": math.exp, "ceil": math.ceil,
            "floor": math.floor, "pi": math.pi, "e": math.e,
            "factorial": math.factorial,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


@tool
def get_current_date() -> str:
    """Get the current date and time. Use this when the user asks about today's date, current time, or anything time-related."""
    now = datetime.datetime.now()
    return now.strftime(
        "Current date: %A, %B %d, %Y\nCurrent time: %H:%M:%S\nTimezone: Local"
    )


# ─── Agent ────────────────────────────────────────────────────────────────────

TOOLS = [search_knowledge, calculator, get_current_date]

SYSTEM_PROMPT = """You are a helpful RAG assistant with access to tools. You can:
1. Search a knowledge base for relevant information
2. Perform mathematical calculations
3. Get the current date and time

Guidelines:
- When the user asks about topics that might be in the knowledge base, use the search_knowledge tool.
- When asked to compute something, use the calculator tool.
- When asked about dates or time, use get_current_date.
- You can combine tools: search for data, then calculate with it.
- Always provide clear, well-formatted answers.
- If you used a tool, mention what you found.
- Respond in the same language the user uses.
"""

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)

# ─── In-memory conversation store (per-session, last 5 exchanges) ─────────────

conversations = {}  # session_id -> list of messages
MAX_HISTORY = 5     # pairs of human/ai messages


def get_conversation(session_id: str) -> list:
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]


def trim_conversation(session_id: str):
    conv = conversations.get(session_id, [])
    # Keep only last MAX_HISTORY pairs (human + ai = 2 msgs each)
    # Count only HumanMessage/AIMessage pairs
    pairs = []
    current_pair = []
    for msg in conv:
        if isinstance(msg, HumanMessage):
            current_pair = [msg]
        elif isinstance(msg, AIMessage) and current_pair:
            current_pair.append(msg)
            pairs.append(current_pair)
            current_pair = []
    # Keep last MAX_HISTORY pairs
    kept = pairs[-MAX_HISTORY:]
    conversations[session_id] = [m for pair in kept for m in pair]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ingest", methods=["POST"])
def ingest():
    """Ingest a document: split into chunks, embed, store."""
    data = request.json
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    metadata = data.get("metadata", {})

    if not title or not content:
        return jsonify({"error": "title and content required"}), 400

    # Split into chunks
    chunks = text_splitter.split_text(content)
    if not chunks:
        return jsonify({"error": "no chunks generated"}), 400

    # Generate embeddings for all chunks
    chunk_embeddings = embeddings.embed_documents(chunks)

    conn = get_db()
    cur = conn.cursor()
    try:
        # Insert document
        cur.execute(
            "INSERT INTO lc_documents (title, content, metadata) VALUES (%s, %s, %s) RETURNING id",
            (title, content, json.dumps(metadata)),
        )
        doc_id = cur.fetchone()[0]

        # Insert chunks with embeddings
        for chunk_text, emb in zip(chunks, chunk_embeddings):
            chunk_meta = {**metadata, "document_id": doc_id, "title": title}
            cur.execute(
                "INSERT INTO lc_chunks (document_id, content, embedding, metadata) VALUES (%s, %s, %s::vector, %s)",
                (doc_id, chunk_text, emb, json.dumps(chunk_meta)),
            )

        return jsonify({
            "id": doc_id,
            "title": title,
            "chunks": len(chunks),
            "message": f"Document '{title}' ingested with {len(chunks)} chunks",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/chat", methods=["POST"])
def chat():
    """Chat with the LangChain agent."""
    data = request.json
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "message required"}), 400

    # Build conversation history
    history = get_conversation(session_id)
    messages = list(history) + [HumanMessage(content=message)]

    try:
        # Invoke the agent
        result = agent.invoke({"messages": messages})

        # Extract info from agent messages
        agent_messages = result.get("messages", [])
        tools_used = []
        final_answer = ""

        for msg in agent_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append({
                            "name": tc["name"],
                            "args": tc["args"],
                        })
                if msg.content and not msg.tool_calls:
                    final_answer = msg.content
            elif isinstance(msg, ToolMessage):
                # Attach tool results to tool info
                for t in tools_used:
                    if t["name"] == msg.name and "result" not in t:
                        t["result"] = msg.content[:500]
                        break

        # If final_answer is empty, check the last AIMessage
        if not final_answer:
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content
                    break

        # Update conversation memory
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=final_answer))
        conversations[session_id] = history
        trim_conversation(session_id)

        return jsonify({
            "answer": final_answer,
            "tools_used": tools_used,
            "session_id": session_id,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/documents", methods=["GET"])
def list_documents():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT d.id, d.title, d.metadata, d.created_at,
               COUNT(c.id) AS chunk_count
        FROM lc_documents d
        LEFT JOIN lc_chunks c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.created_at DESC
    """)
    docs = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify([{
        "id": d["id"],
        "title": d["title"],
        "metadata": d["metadata"],
        "chunk_count": d["chunk_count"],
        "created_at": d["created_at"].isoformat() if d["created_at"] else None,
    } for d in docs])


@app.route("/tools", methods=["GET"])
def list_tools():
    tool_info = []
    for t in TOOLS:
        tool_info.append({
            "name": t.name,
            "description": t.description,
        })
    return jsonify(tool_info)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=True)
