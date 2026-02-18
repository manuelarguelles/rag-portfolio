# 03 — Multi-Document RAG

Sistema RAG que maneja múltiples documentos organizados en **colecciones**, permite buscar dentro de una colección específica o en todas, y cita la fuente de cada fragmento relevante.

## ¿Qué es Multi-Document RAG?

A diferencia de un RAG single-document que trabaja con un único texto, **Multi-Document RAG** organiza el conocimiento en colecciones de documentos, permitiendo:

- **Búsqueda contextualizada**: consultar dentro de una colección específica o en todo el corpus
- **Trazabilidad**: cada respuesta cita exactamente de qué documento y fragmento proviene
- **Organización**: agrupar documentos por tema, proyecto o dominio

```
┌─────────────────────────────────────────────────────────┐
│                     USUARIO                             │
│              "¿Quién proclamó la                        │
│           independencia del Perú?"                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│                  QUERY ENGINE                            │
│  1. Embedding de la pregunta (NVIDIA nv-embedqa-e5-v5)  │
│  2. Filtro opcional por colección                        │
│  3. Búsqueda vectorial (pgvector cosine similarity)     │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│               PostgreSQL + pgvector                      │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Historia de │  │ Tecnología  │  │  Ciencia    │      │
│  │    Perú     │  │             │  │             │      │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │      │
│  │ │ Doc 1   │ │  │ │ Doc 1   │ │  │ │ Doc 1   │ │      │
│  │ │ chunks[]│ │  │ │ chunks[]│ │  │ │ chunks[]│ │      │
│  │ ├─────────┤ │  │ ├─────────┤ │  │ ├─────────┤ │      │
│  │ │ Doc 2   │ │  │ │ Doc 2   │ │  │ │ Doc 2   │ │      │
│  │ │ chunks[]│ │  │ │ chunks[]│ │  │ │ chunks[]│ │      │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└──────────────────┬───────────────────────────────────────┘
                   │ Top-K chunks + metadata
                   ▼
┌──────────────────────────────────────────────────────────┐
│                    LLM (Kimi K2.5)                       │
│  Genera respuesta citando [Fuente N] de cada chunk      │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│                   RESPUESTA                              │
│  "José de San Martín proclamó la independencia el       │
│   28 de julio de 1821 [Fuente 1]. Posteriormente,      │
│   Bolívar consolidó... [Fuente 2]"                      │
│                                                          │
│  📋 Fuentes:                                            │
│  [1] Historia de Perú → Independencia, chunk #2 (94.2%)│
│  [2] Historia de Perú → Independencia, chunk #3 (89.1%)│
└──────────────────────────────────────────────────────────┘
```

## Estructura del Proyecto

```
03-multi-document/
├── app.py                  # Backend FastAPI
├── seed.py                 # Script para datos de ejemplo
├── templates/
│   └── index.html          # Interfaz web (dark theme)
└── README.md               # Este archivo
```

## Requisitos

- PostgreSQL 17 con extensión `pgvector`
- Python con dependencias del venv compartido
- NVIDIA API Key en `~/.config/nvidia/api_key`

## Instrucciones

### 1. Activar entorno virtual

```bash
cd /Users/macdenix/clawd/projects/rag-portfolio
source venv/bin/activate
```

### 2. Iniciar el servidor

```bash
cd 03-multi-document
python app.py
```

El servidor arranca en **http://localhost:8003**. Las tablas se crean automáticamente al inicio.

### 3. Cargar datos de ejemplo

En otra terminal (con el venv activado):

```bash
cd /Users/macdenix/clawd/projects/rag-portfolio/03-multi-document
source ../venv/bin/activate
python seed.py
```

Esto crea 3 colecciones con ~11 documentos sobre Historia de Perú, Tecnología y Ciencia.

### 4. Usar la interfaz

1. Abre **http://localhost:8003**
2. En el **sidebar izquierdo** verás las colecciones
3. Haz clic en una colección para ver/agregar documentos
4. Ve a la pestaña **🔍 Consultar** para hacer preguntas
5. Selecciona una colección específica o "Todas" para buscar en todo el corpus

## API Endpoints

| Método   | Ruta                              | Descripción                                  |
|----------|-----------------------------------|----------------------------------------------|
| `POST`   | `/collections`                    | Crear colección                              |
| `GET`    | `/collections`                    | Listar colecciones con conteo de docs        |
| `DELETE` | `/collections/{id}`               | Eliminar colección + docs + chunks           |
| `POST`   | `/collections/{id}/documents`     | Agregar documento (chunking + embeddings)    |
| `GET`    | `/collections/{id}/documents`     | Listar documentos de una colección           |
| `DELETE` | `/documents/{id}`                 | Eliminar documento + chunks                  |
| `POST`   | `/query`                          | Pregunta → busca chunks → LLM con citas     |
| `GET`    | `/`                               | Interfaz web                                 |

## Estrategia de Chunking

Se usa **recursive character splitting** con:
- **Tamaño máximo**: 500 caracteres
- **Overlap**: 100 caracteres
- **Separadores** (en orden de prioridad): `\n\n`, `\n`, `. `, `, `, ` `, `""`

Cada chunk preserva metadata de origen (colección, documento, fuente, índice).

## Tecnologías

- **FastAPI** — Backend REST API
- **PostgreSQL + pgvector** — Almacenamiento y búsqueda vectorial (HNSW index)
- **NVIDIA NV-EmbedQA-E5-v5** — Modelo de embeddings (1024 dims)
- **Kimi K2.5** — Modelo de chat para generación de respuestas
- **Jinja2** — Templates HTML
