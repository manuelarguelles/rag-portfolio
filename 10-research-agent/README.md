# Proyecto 10 — AI Research Agent 🔬

Agente de investigación automatizada que, dado un tema, recopila información de la web, la almacena en pgvector, extrae hallazgos categorizados con IA y genera un reporte estructurado.

## Concepto

El AI Research Agent implementa un pipeline completo de investigación automatizada:

```
Tema → Búsqueda → Scraping → Chunking → Embeddings → Análisis → Reporte
```

El usuario define un tema de investigación y el agente:
1. **Genera queries** de búsqueda con LLM
2. **Scraping** de páginas web (URLs manuales o seed URLs)
3. **Extrae y chunkea** el contenido relevante
4. **Embede** cada chunk con NVIDIA NV-EmbedQA y almacena en pgvector
5. **Analiza** los chunks con LLM para extraer hallazgos categorizados
6. **Genera** un reporte estructurado con resumen ejecutivo, hallazgos y conclusiones
7. **Permite Q&A** sobre la investigación usando RAG

## Stack

- **Backend**: FastAPI + psycopg2 + pgvector
- **Scraping**: httpx + BeautifulSoup4
- **Embeddings**: NVIDIA NV-EmbedQA-E5-v5 (1024 dims)
- **LLM**: Moonshot Kimi-K2.5 (vía NVIDIA API)
- **DB**: PostgreSQL 17 + pgvector (HNSW index)
- **Frontend**: HTML/CSS/JS vanilla (dark theme dashboard)

## Tablas

| Tabla | Propósito |
|-------|-----------|
| `ra_research_projects` | Proyectos de investigación (tema, estado, reporte) |
| `ra_sources` | Fuentes scrapeadas, chunkeadas y embebidas |
| `ra_findings` | Hallazgos categorizados extraídos por el LLM |

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Interfaz web |
| `POST` | `/research` | Iniciar investigación (topic) |
| `GET` | `/research` | Listar proyectos |
| `GET` | `/research/{id}` | Estado del proyecto |
| `GET` | `/research/{id}/report` | Reporte generado |
| `GET` | `/research/{id}/findings` | Hallazgos categorizados |
| `GET` | `/research/{id}/sources` | Fuentes recopiladas |
| `POST` | `/research/{id}/add-source` | Agregar URL para scraping |
| `POST` | `/research/{id}/query` | Preguntar sobre la investigación |
| `POST` | `/research/{id}/reanalyze` | Re-analizar con nuevas fuentes |

## Ejecución

```bash
cd projects/rag-portfolio
source venv/bin/activate

# Seed (datos de ejemplo)
python 10-research-agent/seed.py

# Servidor
python 10-research-agent/app.py
# → http://localhost:8010
```

## Pipeline de Investigación

### Fase 1: Recopilación (Researching)
- El LLM genera 3-5 queries de búsqueda relevantes al tema
- Para cada URL disponible: scraping con httpx + BeautifulSoup
- Se eliminan scripts, estilos, nav, footer y se extrae texto limpio
- El contenido se chunkea (800 palabras con 150 de overlap)
- Cada chunk se embede y almacena en pgvector

### Fase 2: Análisis (Analyzing)
- El LLM recibe los chunks como contexto
- Extrae hallazgos categorizados (Tendencias, Adopción, Desafíos, etc.)
- Cada hallazgo incluye nivel de confianza (high/medium/low)
- Los hallazgos se vinculan a sus fuentes

### Fase 3: Reporte (Completed)
- El LLM genera un reporte markdown estructurado
- Incluye: resumen ejecutivo, hallazgos por categoría, conclusiones y fuentes
- El usuario puede hacer preguntas adicionales usando RAG

## Limitaciones

- **Sin búsqueda web real**: La demo usa URLs seed y scraping manual
- **Scraping básico**: No maneja JavaScript-rendered pages ni CAPTCHAs
- **Rate limits**: Las APIs de NVIDIA tienen límites de requests

## Extender con Búsqueda Real

Para agregar búsqueda web real, reemplazar `get_seed_urls()` con:

```python
# Opción 1: Brave Search API
import httpx
async def search_web(query: str) -> list[str]:
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.search.brave.com/res/v1/web/search",
                             headers={"X-Subscription-Token": BRAVE_API_KEY},
                             params={"q": query, "count": 5})
        return [result["url"] for result in r.json()["web"]["results"]]

# Opción 2: SerpAPI
# Opción 3: Google Custom Search API
```

## Datos de Ejemplo

El seed crea el proyecto **"Inteligencia Artificial en Latinoamérica 2025"** con:
- 6 fuentes simuladas (11 chunks) sobre IA en la región
- 5 hallazgos categorizados (Adopción, Tendencias, Desafíos, Regulación, Oportunidades)
- Reporte pre-generado con resumen ejecutivo y conclusiones
