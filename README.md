# 🧠 RAG Portfolio — 10 Proyectos

Portfolio de 10 proyectos RAG (Retrieval-Augmented Generation) usando PostgreSQL + pgvector como base de datos vectorial.

## Stack
- **Python 3.9+** — Backend
- **PostgreSQL 17 + pgvector 0.8.1** — Base de datos vectorial
- **FastAPI** — API web
- **HTML/CSS/JS** — Frontend de prueba
- **LLM**: Databricks Foundation Models / NVIDIA NIM (Kimi K2.5)

## Proyectos (de simple a complejo)

| # | Proyecto | Descripción | Estado |
|---|---------|-------------|--------|
| 1 | **First RAG System** | RAG básico desde cero | 🔄 En progreso |
| 2 | **Document Analysis** | Procesamiento de PDFs con LLM | ⏳ Pendiente |
| 3 | **Multi-Document RAG** | RAG sobre múltiples documentos | ⏳ Pendiente |
| 4 | **IBM RAG Guided** | Patrones de producción | ⏳ Pendiente |
| 5 | **Real-Time Assistant** | Pipeline RAG en tiempo real | ⏳ Pendiente |
| 6 | **LangChain RAG Agent** | Agente RAG production-ready | ⏳ Pendiente |
| 7 | **GraphRAG Pipeline** | Knowledge Graph con Neo4j | ⏳ Pendiente |
| 8 | **Agentic RAG** | Agentes autónomos | ⏳ Pendiente |
| 9 | **Multimodal RAG** | Text + Imágenes | ⏳ Pendiente |
| 10 | **AI Research Agent** | Análisis automatizado | ⏳ Pendiente |

## Requisitos
```bash
# PostgreSQL + pgvector
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb rag_portfolio
psql rag_portfolio -c "CREATE EXTENSION vector;"

# Python dependencies (por proyecto)
pip install -r requirements.txt
```

## Estructura
```
rag-portfolio/
├── 01-first-rag/          # RAG básico desde cero
├── 02-document-analysis/  # PDF + LLM
├── 03-multi-document/     # Múltiples documentos
├── 04-ibm-rag-guided/     # Producción
├── 05-realtime-assistant/ # Streaming
├── 06-langchain-agent/    # LangChain
├── 07-graphrag/           # Neo4j Knowledge Graph
├── 08-agentic-rag/        # Agentes autónomos
├── 09-multimodal-rag/     # Text + Images
├── 10-research-agent/     # Investigación automatizada
└── shared/                # Utilidades compartidas
```

## Autor
Manuel Argüelles — Data Engineer / Analytics Engineer
