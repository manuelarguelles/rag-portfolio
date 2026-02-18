# 🧠 RAG Portfolio — 10 Proyectos

Portafolio de 10 proyectos RAG (Retrieval-Augmented Generation) progresivos, de básico a avanzado. Todos implementados con PostgreSQL + pgvector + NVIDIA NIM API. Sin frameworks pagos, 100% open source.

## Stack Común

- **Python 3.9** + FastAPI (excepto P06: Flask)
- **PostgreSQL 17** + pgvector 0.8.1 — búsqueda vectorial con índice HNSW
- **NVIDIA NIM API** (gratis): `nv-embedqa-e5-v5` (embeddings 1024d) + `Kimi K2.5` (LLM)
- **Frontend**: HTML/CSS/JS vanilla (dark theme)

## Proyectos

| # | Proyecto | Tipo de RAG | Puerto | Demo |
|---|---------|-------------|--------|------|
| 01 | [First RAG System](01-first-rag/) | RAG básico desde cero | 8000 | [▶️](recordings/01-first-rag-system.mp4) |
| 02 | [Document Analysis](02-document-analysis/) | PDF Processing + Chunking | 8002 | [▶️](recordings/02-document-analysis.mp4) |
| 03 | [Multi-Document RAG](03-multi-document/) | Colecciones + filtros | 8003 | [▶️](recordings/03-multi-document.mp4) |
| 04 | [IBM RAG Guided](04-ibm-rag-guided/) | Query Expansion + Reranking + Grounding | 8004 | [▶️](recordings/04-ibm-rag-guided.mp4) |
| 05 | [Real-Time Assistant](05-realtime-assistant/) | SSE Streaming + Memoria | 8005 | [▶️](recordings/05-real-time-assistant.mp4) |
| 06 | [LangChain Agent](06-langchain-agent/) | ReAct Agent + Tools | 5006 | [▶️](recordings/06-langchain-agent.mp4) |
| 07 | [GraphRAG](07-graphrag/) | Knowledge Graph + Vector Search | 5007 | [▶️](recordings/07-graphrag.mp4) |
| 08 | [Agentic RAG](08-agentic-rag/) | Multi-Agent (Router, Research, Analyst, Writer) | 8008 | [▶️](recordings/08-agentic-rag.mp4) |
| 09 | [Multimodal RAG](09-multimodal-rag/) | Text + Images unificados | 5009 | [▶️](recordings/09-multimodal-rag.mp4) |
| 10 | [AI Research Agent](10-research-agent/) | Investigación automatizada end-to-end | 8010 | [▶️](recordings/10-ai-research-agent.mp4) |

## Progresión de Complejidad

```
01 Basic → 02 +PDF Documents → 03 +Multi-doc Collections → 04 +Enterprise Pipeline (IBM)
    → 05 +Streaming/Memory → 06 +Framework/Tools → 07 +Knowledge Graph
        → 08 +Multi-Agent → 09 +Multimodal → 10 +Autonomous Research
```

## ¿Cuál usar según el caso?

| Caso de uso | Proyecto recomendado |
|---|---|
| Aprender RAG desde cero | 01 First RAG |
| Documentos internos (PDFs) | 02 Document Analysis |
| Múltiples fuentes organizadas | 03 Multi-Document |
| Producción enterprise | 04 IBM RAG Guided |
| Chatbot conversacional | 05 Real-Time Assistant |
| Múltiples herramientas | 06 LangChain Agent |
| Preguntas sobre relaciones | 07 GraphRAG |
| Consultas complejas | 08 Agentic RAG |
| Contenido mixto (text + images) | 09 Multimodal RAG |
| Investigación autónoma | 10 Research Agent |

## Quickstart

```bash
# 1. Requisitos
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb rag_portfolio
psql rag_portfolio -c "CREATE EXTENSION vector;"

# 2. Configurar API key de NVIDIA NIM (gratis)
# https://build.nvidia.com/ → Get API Key
export NVIDIA_API_KEY="nvapi-..."

# 3. Ejecutar cualquier proyecto
cd 01-first-rag
pip install -r requirements.txt  # o usar venv compartido
python seed.py                    # cargar datos de ejemplo
python app.py                     # iniciar servidor
# Abrir http://localhost:8000
```

## Estructura

```
rag-portfolio/
├── 01-first-rag/           # RAG básico (FastAPI + pgvector)
├── 02-document-analysis/   # PDF processing (PyMuPDF + chunking)
├── 03-multi-document/      # Colecciones + búsqueda filtrada
├── 04-ibm-rag-guided/      # Query expansion + reranking + grounding
├── 05-realtime-assistant/  # SSE streaming + conversación
├── 06-langchain-agent/     # LangChain + LangGraph ReAct agent
├── 07-graphrag/            # Knowledge graph en PostgreSQL
├── 08-agentic-rag/         # 4 agentes autónomos (Python puro)
├── 09-multimodal-rag/      # Text + images (text-bridge strategy)
├── 10-research-agent/      # Web scraping + análisis + reporte
├── recordings/             # Video demos de cada proyecto
└── shared/                 # Utilidades compartidas
```

## Highlights Técnicos

- **Sin Neo4j/Docker**: GraphRAG (P07) implementa grafos con PostgreSQL puro via SQL JOINs
- **Sin LangChain innecesario**: Agentic RAG (P08) usa Python puro — sin CrewAI ni AutoGen
- **Modelo asimétrico**: NVIDIA nv-embedqa-e5-v5 requiere `input_type: "query"` vs `"passage"` — implementación custom en P06
- **Text-bridge para imágenes**: P09 indexa imágenes via descripción textual (sin modelo de visión)
- **Cada proyecto aislado**: Tablas con prefijos únicos (`documents`, `pdf_*`, `md_*`, `ibm_*`, `rt_*`, `lc_*`, `gr_*`, `ag_*`, `mm_*`, `ra_*`)

## Documentación

📄 [Documentación completa en Notion](https://www.notion.so/RAG-Portfolio-10-Proyectos-30b63102b2a381719794e509d1bab9e4)

## Autor

**Manuel Argüelles** — Data Engineer / Analytics Engineer
