# 🤖 Proyecto 8: Agentic RAG — Agentes Autónomos

Sistema de Retrieval-Augmented Generation con múltiples agentes autónomos que colaboran para responder consultas complejas. Implementado con **agentes puros en Python** — sin LangChain, sin CrewAI.

## 🏗️ Arquitectura Multi-Agente

```
                    ┌──────────────┐
                    │   Usuario    │
                    │   (Query)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  🧭 Router   │  ← Analiza la query y decide la estrategia
                    │    Agent     │
                    └──┬───────┬───┘
                       │       │
            ┌──────────▼──┐ ┌─▼───────────┐
            │ 🔍 Research │ │ 📊 Analyst  │  ← Especialistas ejecutan en paralelo
            │    Agent    │ │    Agent    │
            └──────┬──────┘ └─────┬───────┘
                   │              │
                   └──────┬───────┘
                    ┌─────▼────────┐
                    │  ✍️ Writer   │  ← Compone la respuesta final
                    │    Agent     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  📋 Trace    │  ← Se guarda el trace completo
                    │    Log      │
                    └──────────────┘
```

### Agentes

| Agente | Rol | Qué hace |
|--------|-----|----------|
| 🧭 **Router** | Strategy Planner | Analiza la query y decide qué agentes invocar (research, analyst, o ambos) |
| 🔍 **Research** | Knowledge Retriever | Busca información relevante en pgvector, organiza hallazgos |
| 📊 **Analyst** | Data Analyst | Analiza datos, encuentra patrones, genera insights |
| ✍️ **Writer** | Response Composer | Compone la respuesta final integrando todos los inputs |

### Estrategias del Router

- **research**: Solo búsqueda en la base de conocimiento (ej: "¿Cuál es el PIB de Perú?")
- **analyst**: Solo análisis de datos (ej: "Compara las tendencias de IA vs blockchain")
- **both**: Búsqueda + análisis (ej: "¿Cómo afectan las exportaciones mineras al crecimiento?")

## 🆚 Agentic RAG vs RAG Tradicional

| Aspecto | RAG Tradicional | Agentic RAG |
|---------|-----------------|-------------|
| Pipeline | Fijo: Embed → Search → Generate | Dinámico: Router decide el flujo |
| Decisión | No hay decisión, siempre busca | Router evalúa si buscar, analizar, o ambos |
| Especialización | Un solo prompt hace todo | Agentes especializados por tarea |
| Trazabilidad | Limitada | Trace completo de cada agente |
| Complejidad | Simple, predecible | Mayor, pero más capaz |
| Cuándo usar | Queries simples de búsqueda | Queries complejas que requieren razonamiento |

### ¿Cuándo usar agentes vs pipeline fijo?

**Usa pipeline fijo cuando:**
- Las queries son predecibles y simples
- Latencia es crítica (los agentes añaden overhead)
- El dominio es estrecho y bien definido

**Usa agentes cuando:**
- Las queries requieren razonamiento multi-paso
- Necesitas combinar búsqueda con análisis
- Quieres trazabilidad del proceso de decisión
- El dominio es amplio o las queries son variadas

## 📦 Stack

- **Backend**: FastAPI (Python)
- **LLM**: moonshotai/kimi-k2.5 (via NVIDIA API)
- **Embeddings**: nvidia/nv-embedqa-e5-v5 (1024 dims)
- **Vector Store**: PostgreSQL + pgvector (HNSW index)
- **Frontend**: HTML/CSS/JS vanilla (dark theme)
- **Agentes**: Clases Python puras (sin frameworks)

## 🚀 Instrucciones

### 1. Activar entorno

```bash
source /Users/macdenix/clawd/projects/rag-portfolio/venv/bin/activate
cd /Users/macdenix/clawd/projects/rag-portfolio/08-agentic-rag/
```

### 2. Inicializar base de datos y seed

```bash
python seed.py
```

Esto crea 2 knowledge bases con documentos sobre economía peruana y tecnología 2025.

### 3. Ejecutar servidor

```bash
python app.py
```

El servidor corre en `http://localhost:8008`.

### 4. Usar

- Abre `http://localhost:8008` en el navegador
- Haz preguntas como:
  - *"¿Cuál es la situación económica actual de Perú?"*
  - *"Compara las tendencias de IA y blockchain en 2025"*
  - *"¿Cómo afectan las exportaciones mineras al crecimiento del PIB?"*
- Observa el **Agent Trace Panel** a la derecha para ver el flujo de decisión

## 📊 API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Interfaz web |
| `POST` | `/knowledge-bases` | Crear base de conocimiento |
| `GET` | `/knowledge-bases` | Listar bases |
| `POST` | `/knowledge-bases/{id}/documents` | Agregar documento |
| `POST` | `/query` | Procesar query con multi-agentes |
| `GET` | `/traces` | Listar traces |
| `GET` | `/traces/{id}` | Detalle de un trace |

## 🗄️ Tablas

Todas prefijadas con `ag_`:
- `ag_knowledge_bases` — Bases de conocimiento
- `ag_documents` — Documentos
- `ag_chunks` — Chunks con embeddings (vector 1024d)
- `ag_task_log` — Log de ejecución de agentes (trace JSONB)
