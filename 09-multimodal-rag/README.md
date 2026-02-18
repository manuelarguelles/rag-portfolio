# Proyecto 9: Multimodal RAG — Text + Images

A Retrieval-Augmented Generation system that combines **text documents** and **images** in a unified vector search space. Images are indexed via textual descriptions (text-bridge strategy), enabling semantic search across both modalities.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Text Input  │────▶│  Generate        │────▶│  PostgreSQL  │
│  (title +    │     │  Embedding       │     │  + pgvector  │
│   content)   │     │  (NV-EmbedQA)    │     │              │
└─────────────┘     └──────────────────┘     │  mm_items    │
                                              │  table with  │
┌─────────────┐     ┌──────────────────┐     │  vector(1024)│
│ Image Input  │────▶│  User provides   │────▶│              │
│ (file +      │     │  description     │     └──────┬───────┘
│  title +     │     │  → Embedding of  │            │
│  description)│     │  title + desc    │            │
└─────────────┘     └──────────────────┘            │
                                                     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Question   │────▶│  Query Embedding │────▶│ Vector Search │
│              │     │  + cosine search │     │ (HNSW index)  │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │  LLM Answer   │
                                              │  (Kimi K2.5)  │
                                              └──────────────┘
```

## The Text-Bridge Strategy

Since the NVIDIA NV-EmbedQA-E5-v5 embedding model only supports **text** (not images), we use a "text-bridge" approach:

1. **Upload**: User provides an image along with a **title** and **description**
2. **Embedding**: We generate the embedding from `title + " " + description` (pure text)
3. **Search**: When querying, the question embedding is compared against all items — both text content embeddings and image description embeddings
4. **Results**: Mixed results return both text cards and image cards with thumbnails

This effectively converts images into text-representable vectors, enabling unified multimodal search.

### Limitations

- **No automatic image understanding**: The system relies on human-provided descriptions. A vision-capable model (GPT-4V, LLaVA, etc.) could generate captions automatically.
- **Description quality matters**: Search quality depends on how well the description captures the image content.
- **No image-to-image search**: You can't search using an image as input — only text queries.

### How It Would Work With a Vision Model

```
Image → Vision Model (e.g., GPT-4V) → Auto-caption → Embedding → Vector DB
```

With a vision model, the workflow would be fully automatic: upload an image, the model describes it, and the description is embedded. This removes the need for manual descriptions.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Database | PostgreSQL 17 + pgvector |
| Embeddings | NVIDIA NV-EmbedQA-E5-v5 (1024 dims) |
| Chat LLM | Kimi K2.5 (via NVIDIA API) |
| Image Processing | Pillow |
| Frontend | Vanilla HTML/CSS/JS |

## Setup

### Prerequisites
- PostgreSQL 17 with pgvector extension
- Database: `rag_portfolio`
- NVIDIA API key at `~/.config/nvidia/api_key`

### Install Dependencies

```bash
source /Users/macdenix/clawd/projects/rag-portfolio/venv/bin/activate
pip install pillow  # Additional dependency
```

### Seed Sample Data

```bash
python seed.py
```

This creates:
- 5 text items (art, cities, nature, ML, cuisine)
- 5 generated images with Pillow (map, logo, diagram, palette, schema)

### Run

```bash
python app.py
```

Open http://localhost:5009

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/items/text` | Add text item `{title, content}` |
| `POST` | `/items/image` | Upload image (multipart: file + title + description) |
| `GET` | `/items` | List all items (optional `?type=text\|image`) |
| `DELETE` | `/items/{id}` | Delete item |
| `POST` | `/query` | Multimodal search `{question, top_k?, type?}` |
| `GET` | `/items/{id}/image` | Serve image (optional `?thumb=1`) |

## Database Tables

- **`mm_items`** — Unified table for text and image items with vector embeddings
- **`mm_collections`** — Named collections for organizing items
- **`mm_item_collections`** — Many-to-many relationship between items and collections

## Project Structure

```
09-multimodal-rag/
├── app.py              # Flask backend
├── seed.py             # Sample data generator
├── README.md
├── templates/
│   └── index.html      # Web UI
├── sample-images/      # Generated sample images (in git)
└── uploads/            # User-uploaded images (gitignored)
```
