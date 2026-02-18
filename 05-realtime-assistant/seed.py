"""
Seed data for Real-Time RAG Assistant (Project 5)
Inserts sample knowledge articles and a demo conversation.
"""

import json
import psycopg2
import httpx
from pathlib import Path

NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"


def get_embedding(text: str) -> list[float]:
    r = httpx.post(
        f"{NVIDIA_BASE}/embeddings",
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        json={
            "model": EMBED_MODEL,
            "input": [text],
            "input_type": "query",
            "encoding_format": "float",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


ARTICLES = [
    {
        "title": "How Transformers Work",
        "content": (
            "Transformers are a type of neural network architecture introduced in the 2017 paper "
            "'Attention Is All You Need'. They rely on self-attention mechanisms to process "
            "sequences in parallel, unlike RNNs which process sequentially. The key components "
            "are: multi-head attention layers, feed-forward networks, and positional encodings. "
            "Transformers power modern LLMs like GPT, Claude, and LLaMA. The self-attention "
            "mechanism computes attention scores between all pairs of tokens, allowing the model "
            "to capture long-range dependencies efficiently."
        ),
        "source": "AI Fundamentals",
    },
    {
        "title": "PostgreSQL Vector Search with pgvector",
        "content": (
            "pgvector is a PostgreSQL extension that adds vector similarity search capabilities. "
            "It supports storing embeddings as the 'vector' data type and provides operators for "
            "cosine distance (<=>), L2 distance (<->), and inner product (<#>). For efficient "
            "nearest-neighbor search, pgvector supports IVFFlat and HNSW indexes. HNSW (Hierarchical "
            "Navigable Small World) indexes are generally preferred for their better recall/speed "
            "tradeoff. Typical usage: store document embeddings, then query with "
            "'ORDER BY embedding <=> query_vector LIMIT k'."
        ),
        "source": "Database Engineering",
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "RAG is a technique that combines information retrieval with language model generation. "
            "Instead of relying solely on the LLM's training data, RAG first retrieves relevant "
            "documents from a knowledge base using vector similarity search, then includes those "
            "documents as context in the prompt. This approach reduces hallucinations, keeps responses "
            "grounded in factual data, and allows updating the knowledge base without retraining "
            "the model. Key components: document chunking, embedding generation, vector store, "
            "retrieval, and augmented prompt construction."
        ),
        "source": "AI Architecture",
    },
    {
        "title": "Server-Sent Events (SSE) for Streaming",
        "content": (
            "Server-Sent Events (SSE) is a web technology that allows a server to push updates "
            "to the client over a single HTTP connection. Unlike WebSockets, SSE is unidirectional "
            "(server to client only) and uses standard HTTP. The server sends data with "
            "'Content-Type: text/event-stream', and each message is prefixed with 'data: '. "
            "SSE is ideal for streaming LLM responses token-by-token. In JavaScript, the "
            "EventSource API or fetch with ReadableStream can consume SSE endpoints. SSE "
            "automatically handles reconnection and is simpler to implement than WebSockets."
        ),
        "source": "Web Technologies",
    },
    {
        "title": "FastAPI: Modern Python Web Framework",
        "content": (
            "FastAPI is a modern, high-performance Python web framework built on Starlette and "
            "Pydantic. Key features: automatic OpenAPI docs, async/await support, type validation, "
            "and dependency injection. FastAPI supports StreamingResponse for SSE, making it ideal "
            "for real-time AI applications. It handles both sync and async endpoints, supports "
            "WebSocket connections, and provides middleware for CORS, authentication, etc. "
            "Performance is comparable to Node.js and Go frameworks thanks to its ASGI foundation."
        ),
        "source": "Python Frameworks",
    },
    {
        "title": "The History of Coffee",
        "content": (
            "Coffee originated in Ethiopia, where legend says a goat herder named Kaldi discovered "
            "it around 850 AD after noticing his goats became energetic eating certain berries. "
            "By the 15th century, coffee was cultivated in Yemen and spread through the Ottoman "
            "Empire. Coffee houses became centers of social activity and intellectual discussion, "
            "earning the nickname 'schools of the wise'. Today, Brazil is the world's largest "
            "coffee producer, followed by Vietnam and Colombia. The two main species are Arabica "
            "(70% of production, smoother taste) and Robusta (30%, stronger, more caffeine)."
        ),
        "source": "General Knowledge",
    },
    {
        "title": "Kubernetes Container Orchestration",
        "content": (
            "Kubernetes (K8s) is an open-source container orchestration platform originally "
            "developed by Google. It automates deployment, scaling, and management of containerized "
            "applications. Core concepts include Pods (smallest deployable unit), Services (stable "
            "network endpoints), Deployments (desired state management), and ConfigMaps/Secrets "
            "for configuration. Kubernetes uses a declarative model where you specify the desired "
            "state and the control plane works to achieve it. It supports auto-scaling, rolling "
            "updates, self-healing, and service discovery."
        ),
        "source": "DevOps",
    },
    {
        "title": "Climate Change and Renewable Energy",
        "content": (
            "Global average temperatures have risen approximately 1.1°C above pre-industrial levels. "
            "The primary driver is greenhouse gas emissions from burning fossil fuels. Renewable "
            "energy sources—solar, wind, hydroelectric, and geothermal—offer pathways to reduce "
            "emissions. Solar energy costs have dropped 89% since 2010, making it competitive with "
            "fossil fuels. Wind power capacity has grown 4x in the last decade. Battery storage "
            "technology is advancing rapidly, addressing intermittency challenges. The Paris "
            "Agreement aims to limit warming to 1.5°C above pre-industrial levels."
        ),
        "source": "Science & Environment",
    },
]


def seed():
    conn = psycopg2.connect(DB_URL)

    # Clean existing seed data
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rt_messages")
        cur.execute("DELETE FROM rt_conversations")
        cur.execute("DELETE FROM rt_knowledge")
    conn.commit()

    # Insert knowledge articles
    print("📚 Inserting knowledge articles...")
    for i, art in enumerate(ARTICLES, 1):
        print(f"  [{i}/{len(ARTICLES)}] {art['title']}")
        emb = get_embedding(art["content"][:2000])
        vec_str = "[" + ",".join(str(x) for x in emb) + "]"
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rt_knowledge (title, content, source, embedding) "
                "VALUES (%s, %s, %s, %s::vector)",
                (art["title"], art["content"], art["source"], vec_str),
            )
        conn.commit()

    # Create a demo conversation
    print("\n💬 Creating demo conversation...")
    with conn.cursor() as cur:
        cur.execute("INSERT INTO rt_conversations DEFAULT VALUES RETURNING id")
        conv_id = cur.fetchone()[0]

        messages = [
            ("user", "What is RAG and how does it work?"),
            (
                "assistant",
                "**Retrieval-Augmented Generation (RAG)** combines information retrieval with "
                "language model generation. Here's how it works:\n\n"
                "1. **Document Processing**: Documents are chunked and converted to embeddings\n"
                "2. **Vector Storage**: Embeddings are stored in a vector database (like pgvector)\n"
                "3. **Retrieval**: When a user asks a question, the query is embedded and similar "
                "documents are found via vector search\n"
                "4. **Augmented Generation**: Retrieved documents are included as context in the "
                "LLM prompt, grounding the response in factual data\n\n"
                "This approach reduces hallucinations and allows updating knowledge without "
                "retraining the model.",
            ),
            ("user", "What vector database are we using?"),
            (
                "assistant",
                "We're using **pgvector**, a PostgreSQL extension for vector similarity search. "
                "It supports:\n\n"
                "- **Cosine distance** (`<=>`), **L2 distance** (`<->`), and **inner product** (`<#>`)\n"
                "- **HNSW indexes** for fast approximate nearest neighbor search\n"
                "- Standard SQL integration — no separate database needed!\n\n"
                "The typical query pattern is:\n"
                "```sql\n"
                "SELECT * FROM documents\n"
                "ORDER BY embedding <=> query_vector\n"
                "LIMIT 5;\n"
                "```",
            ),
        ]

        chunks_example = json.dumps([
            {"id": 1, "title": "Retrieval-Augmented Generation (RAG)", "similarity": 0.92},
            {"id": 2, "title": "PostgreSQL Vector Search with pgvector", "similarity": 0.78},
        ])

        for role, content in messages:
            cur.execute(
                "INSERT INTO rt_messages (conversation_id, role, content, chunks_used) "
                "VALUES (%s, %s, %s, %s::jsonb)",
                (conv_id, role, content, chunks_example if role == "assistant" else "[]"),
            )
    conn.commit()
    conn.close()
    print(f"\n✅ Seeded {len(ARTICLES)} articles + 1 conversation (id={conv_id})")


if __name__ == "__main__":
    seed()
