# 🔍 First RAG System

Sistema RAG (Retrieval-Augmented Generation) básico construido desde cero con PostgreSQL + pgvector + NVIDIA NIM API.

## ¿Qué es RAG?

**RAG** (Retrieval-Augmented Generation) es un patrón de arquitectura que combina:

1. **Retrieval** — Búsqueda de información relevante en una base de conocimiento
2. **Augmented** — Enriquecimiento del prompt con el contexto encontrado
3. **Generation** — Generación de respuesta por un LLM usando ese contexto

Esto permite que el LLM responda con información específica y actualizada, reduciendo las alucinaciones.

## Arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                     First RAG System                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ Usuario │───▶│   FastAPI     │───▶│  NVIDIA NIM API   │   │
│  │  (Web)  │◀───│   Backend     │◀───│  (Embeddings+LLM) │   │
│  └─────────┘    └──────┬───────┘    └───────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│               ┌────────────────┐                             │
│               │  PostgreSQL    │                             │
│               │  + pgvector    │                             │
│               │  (vectores)    │                             │
│               └────────────────┘                             │
└──────────────────────────────────────────────────────────────┘
```

### Flujo de Ingesta
```
Texto → NVIDIA Embedding API → vector(1024) → INSERT en pgvector
```

### Flujo de Consulta
```
Pregunta → Embedding → Búsqueda coseno en pgvector → Top-K docs
    → Contexto + Pregunta → LLM → Respuesta fundamentada
```

## Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| Backend | Python 3.9 + FastAPI |
| Base de datos | PostgreSQL 17 + pgvector 0.8.1 |
| Embeddings | NVIDIA NIM API (`nvidia/nv-embedqa-e5-v5`, 1024 dims) |
| LLM | NVIDIA NIM API (`moonshotai/kimi-k2.5`) |
| Frontend | HTML + CSS + JavaScript (vanilla) |
| Índice vectorial | HNSW (Hierarchical Navigable Small World) |

## Requisitos

- PostgreSQL 17 con extensión pgvector
- Python 3.9+
- API Key de NVIDIA NIM (gratis: https://build.nvidia.com)

## Cómo ejecutar

### 1. Configurar variables de entorno

Editar `.env` con tus valores:

```env
NVIDIA_API_KEY=tu-api-key-aquí
DATABASE_URL=postgresql://usuario@localhost/rag_portfolio
```

### 2. Crear la base de datos (si no existe)

```bash
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
createdb rag_portfolio
psql rag_portfolio -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Activar el entorno virtual e instalar dependencias

```bash
source /Users/macdenix/clawd/projects/rag-portfolio/venv/bin/activate
pip install psycopg2-binary pgvector fastapi uvicorn python-dotenv httpx jinja2
```

### 4. Cargar datos de ejemplo

```bash
cd /Users/macdenix/clawd/projects/rag-portfolio/01-first-rag
python seed.py
```

Esto carga 8 documentos sobre temas variados: historia de Perú, tecnología, ciencia, gastronomía.

### 5. Iniciar el servidor

```bash
python app.py
# o: uvicorn app:app --reload --port 8000
```

### 6. Abrir la interfaz

Ir a http://localhost:8000 en el navegador.

## Interfaz Web

La interfaz tiene un diseño oscuro moderno con:

- **Header**: Título del proyecto + estadísticas en tiempo real (total docs, dimensión, modelo)
- **Panel izquierdo**: Lista de documentos ingestados con opción de eliminar cada uno
- **Panel derecho superior**: Formulario para ingestar nuevos documentos (título + contenido)
- **Panel derecho inferior**: Campo de consulta RAG con respuesta del LLM y fuentes relevantes con porcentaje de similitud

## Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Interfaz web |
| `POST` | `/ingest` | Ingestar documento (título + contenido) |
| `POST` | `/query` | Consulta RAG (pregunta → respuesta + fuentes) |
| `GET` | `/documents` | Listar todos los documentos |
| `DELETE` | `/documents/{id}` | Eliminar un documento |
| `GET` | `/stats` | Estadísticas del sistema |

### Ejemplos con curl

**Ingestar:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"title": "Test", "content": "Este es un documento de prueba."}'
```

**Consultar:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es Machu Picchu?"}'
```

**Listar documentos:**
```bash
curl http://localhost:8000/documents
```

**Estadísticas:**
```bash
curl http://localhost:8000/stats
```

## Estructura del proyecto

```
01-first-rag/
├── app.py              # Backend FastAPI completo
├── seed.py             # Script para cargar datos de ejemplo
├── .env                # Variables de entorno
├── README.md           # Este archivo
└── templates/
    └── index.html      # Interfaz web (HTML/CSS/JS)
```

## Conceptos clave implementados

1. **Embeddings**: Representación vectorial de texto en 1024 dimensiones
2. **Similitud coseno**: Métrica para comparar cercanía semántica entre vectores
3. **Índice HNSW**: Estructura de datos para búsqueda aproximada eficiente de vecinos más cercanos
4. **Prompt engineering**: Sistema de prompts que instruye al LLM a responder solo con el contexto proporcionado
5. **RAG pipeline**: Flujo completo de retrieval → augment → generate

---

*Proyecto 1 del RAG Portfolio — Construido con ❤️ y pgvector*
