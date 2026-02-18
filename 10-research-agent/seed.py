"""
Seed data for AI Research Agent — Proyecto 10
Creates a sample project: "Inteligencia Artificial en Latinoamérica 2025"
"""

import psycopg2
from pgvector.psycopg2 import register_vector
from pathlib import Path
from openai import OpenAI

DB_URL = "postgresql://macdenix@localhost/rag_portfolio"
NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

llm = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

def get_embedding(text: str) -> list[float]:
    text = text[:2048]
    resp = llm.embeddings.create(input=[text], model=EMBED_MODEL,
                                  extra_body={"input_type": "passage", "truncate": "END"})
    return resp.data[0].embedding


# ── Source data ─────────────────────────────────────────────────────────
SOURCES = [
    {
        "url": "https://ialatam.org/informe-2025",
        "title": "Informe IA LATAM 2025 — Estado de la Inteligencia Artificial",
        "chunks": [
            "La adopción de inteligencia artificial en América Latina ha experimentado un crecimiento "
            "del 45% durante 2024-2025, liderado por Brasil, México y Colombia. Las empresas de la "
            "región están invirtiendo significativamente en soluciones de IA generativa, procesamiento "
            "de lenguaje natural y automatización de procesos. Según encuestas regionales, el 68% de "
            "las empresas medianas y grandes ya utilizan alguna forma de IA en sus operaciones, aunque "
            "la mayoría se encuentra en etapas iniciales de implementación.",

            "Los principales sectores que lideran la adopción de IA en Latinoamérica son fintech y "
            "banca (78% de adopción), salud (52%), agricultura de precisión (47%), manufactura (41%) "
            "y gobierno digital (35%). Brasil concentra el 40% de las startups de IA de la región, "
            "seguido por México con 22% y Colombia con 12%. Argentina y Chile destacan en investigación "
            "académica y desarrollo de talento especializado.",
        ]
    },
    {
        "url": "https://bancointeramericano.org/ai-investment-2025",
        "title": "BID — Inversión en IA en América Latina",
        "chunks": [
            "El Banco Interamericano de Desarrollo estima que la inversión en inteligencia artificial "
            "en América Latina alcanzará los 8.5 mil millones de dólares en 2025, un incremento del "
            "62% respecto a 2023. Los fondos de venture capital han invertido más de 2.3 mil millones "
            "en startups de IA latinoamericanas durante el último año, con rondas significativas en "
            "Brasil (Nubank AI Labs, $400M), México (Clip AI, $180M) y Colombia (Rappi AI, $150M).",

            "Sin embargo, persiste una brecha significativa con respecto a otras regiones. Mientras "
            "que Estados Unidos invierte aproximadamente $67 mil millones anuales en IA, toda "
            "Latinoamérica apenas alcanza el 12% de esa cifra. La fuga de talento sigue siendo un "
            "desafío: se estima que el 35% de los ingenieros de ML formados en la región emigran a "
            "Silicon Valley o Europa dentro de los primeros 5 años de carrera.",
        ]
    },
    {
        "url": "https://regulacionai.gov/latam-framework",
        "title": "Marco Regulatorio de IA en LATAM — Panorama 2025",
        "chunks": [
            "Brasil aprobó su Ley de Inteligencia Artificial en 2024, convirtiéndose en el primer "
            "país de la región con un marco regulatorio integral. La ley establece clasificación de "
            "riesgo para sistemas de IA, requisitos de transparencia y auditoría, y protección contra "
            "sesgos algorítmicos. México y Colombia están en proceso de legislación similar, con "
            "proyectos de ley que se esperan aprobar en 2025.",

            "Chile lanzó su Política Nacional de IA actualizada con enfoque en ética y derechos "
            "humanos. Argentina estableció un sandbox regulatorio para experimentación con IA en "
            "servicios financieros. Uruguay y Costa Rica se destacan por sus programas de gobierno "
            "digital que integran IA en servicios públicos, incluyendo chatbots de atención ciudadana "
            "y sistemas de detección de fraude en aduanas.",
        ]
    },
    {
        "url": "https://talentoia.edu/reporte-latam",
        "title": "Talento en IA — Formación y Capacidades en LATAM",
        "chunks": [
            "Las universidades latinoamericanas han incrementado su oferta de programas en IA y "
            "ciencia de datos en un 180% desde 2022. Brasil lidera con 45 programas de maestría "
            "especializados, seguido por México (28), Argentina (18) y Colombia (15). Destacan "
            "instituciones como USP, UNAM, UBA y la Universidad de los Andes por su producción "
            "académica en machine learning e IA aplicada.",
        ]
    },
    {
        "url": "https://agritech-ai.com/precision-farming-latam",
        "title": "IA en Agricultura de Precisión — Caso LATAM",
        "chunks": [
            "La agricultura de precisión impulsada por IA está transformando el campo latinoamericano. "
            "En Brasil, drones equipados con visión computacional monitorean más de 2 millones de "
            "hectáreas de soja y café. En Argentina, startups como Kilimo usan modelos de ML para "
            "optimizar el riego, logrando ahorros de agua del 30%. Colombia emplea IA para la "
            "detección temprana de la roya del café, reduciendo pérdidas en un 25%.",

            "México está implementando sistemas de IA para predecir rendimientos de maíz y aguacate, "
            "ayudando a más de 50,000 pequeños agricultores a tomar mejores decisiones de siembra. "
            "La FAO estima que la IA aplicada a la agricultura en LATAM podría incrementar la "
            "productividad regional en un 20% para 2030, reduciendo simultáneamente el uso de "
            "pesticidas en un 35% mediante aplicación selectiva con drones autónomos.",
        ]
    },
    {
        "url": "https://healthai.latam/diagnostico-2025",
        "title": "IA en Salud — Diagnóstico Asistido en América Latina",
        "chunks": [
            "Los sistemas de diagnóstico asistido por IA están expandiéndose rápidamente en "
            "hospitales latinoamericanos. Brasil cuenta con más de 120 hospitales usando IA para "
            "análisis de imágenes médicas, incluyendo detección de cáncer de mama con una precisión "
            "del 94%. México implementó un sistema nacional de triaje por IA en 2024 que ha reducido "
            "los tiempos de espera en urgencias en un 40%.",
        ]
    },
]

FINDINGS = [
    {
        "category": "Adopción",
        "finding": "La adopción de IA en LATAM creció 45% en 2024-2025, con el 68% de empresas medianas y grandes utilizando alguna forma de IA. Brasil, México y Colombia lideran el crecimiento regional, concentrando más del 74% de las startups de IA.",
        "confidence": "high",
        "source_ids_offset": [0, 1],
    },
    {
        "category": "Tendencias",
        "finding": "Los sectores fintech/banca (78%), salud (52%) y agricultura de precisión (47%) lideran la adopción de IA en la región. La IA generativa y el procesamiento de lenguaje natural son las tecnologías con mayor demanda empresarial.",
        "confidence": "high",
        "source_ids_offset": [0, 4, 5],
    },
    {
        "category": "Desafíos",
        "finding": "La brecha de inversión con mercados desarrollados sigue siendo significativa: LATAM invierte apenas el 12% de lo que invierte EEUU en IA. La fuga de talento afecta al 35% de ingenieros de ML formados en la región que emigran dentro de 5 años.",
        "confidence": "high",
        "source_ids_offset": [1, 3],
    },
    {
        "category": "Regulación",
        "finding": "Brasil lidera con su Ley de IA aprobada en 2024, estableciendo clasificación de riesgo y requisitos de transparencia. México y Colombia están en proceso de legislación similar, mientras Chile, Uruguay y Costa Rica avanzan con sandboxes y políticas nacionales.",
        "confidence": "medium",
        "source_ids_offset": [2],
    },
    {
        "category": "Oportunidades",
        "finding": "La IA aplicada a agricultura en LATAM podría incrementar la productividad regional en 20% para 2030, según la FAO. Casos como Kilimo (Argentina) muestran ahorros de agua del 30% y reducción de pesticidas del 35% con drones autónomos.",
        "confidence": "medium",
        "source_ids_offset": [4],
    },
]

REPORT = """# Inteligencia Artificial en Latinoamérica 2025

## Resumen Ejecutivo

América Latina está experimentando una transformación acelerada en la adopción de inteligencia artificial, con un crecimiento del 45% en implementaciones durante 2024-2025. La región, liderada por Brasil, México y Colombia, ha visto una inversión estimada de $8.5 mil millones de dólares en tecnologías de IA, aunque persiste una brecha significativa con respecto a mercados más maduros.

El ecosistema de IA latinoamericano se caracteriza por una fuerte presencia en sectores como fintech, agricultura de precisión y salud, donde la tecnología está generando impactos medibles en productividad y eficiencia. Sin embargo, desafíos como la fuga de talento, la desigualdad en el acceso tecnológico y la necesidad de marcos regulatorios robustos requieren atención urgente.

La región muestra un potencial enorme, con casos de éxito que demuestran que la IA puede ser un catalizador de desarrollo sostenible, desde la optimización agrícola hasta la mejora de servicios de salud pública.

## Hallazgos Principales

### Adopción y Crecimiento
- El 68% de empresas medianas y grandes en LATAM utilizan alguna forma de IA
- Brasil concentra el 40% de startups de IA, seguido por México (22%) y Colombia (12%)
- La inversión regional alcanzó $8.5 mil millones en 2025, un 62% más que en 2023
- Los sectores fintech (78%), salud (52%) y agritech (47%) lideran la adopción

### Innovación Sectorial
- **Agricultura**: Drones con IA monitorean 2M+ hectáreas en Brasil; modelos de ML optimizan riego con 30% de ahorro de agua
- **Salud**: 120+ hospitales en Brasil usan IA para diagnóstico; México redujo tiempos de espera en urgencias en 40%
- **Fintech**: Nubank AI Labs ($400M), Clip AI ($180M) y Rappi AI ($150M) lideran las rondas de inversión

### Desafíos
- Brecha de inversión: LATAM invierte solo 12% de lo que invierte EEUU
- Fuga de talento: 35% de ingenieros de ML emigran en los primeros 5 años
- Acceso desigual: la adopción se concentra en grandes ciudades y empresas

### Marco Regulatorio
- Brasil: primera Ley de IA integral de la región (2024)
- México y Colombia: proyectos de ley en proceso para 2025
- Chile: Política Nacional de IA con enfoque ético
- Argentina: sandbox regulatorio para IA en finanzas

## Conclusiones y Recomendaciones

1. **Retención de talento**: Es crucial crear incentivos para retener investigadores y desarrolladores de IA en la región
2. **Inclusión**: Las políticas de IA deben enfocarse en reducir la brecha digital entre zonas urbanas y rurales
3. **Cooperación regional**: Los marcos regulatorios deben armonizarse entre países para facilitar la innovación
4. **Inversión en educación**: El incremento del 180% en programas académicos es positivo pero insuficiente
5. **Agricultura como motor**: El sector agrícola representa una oportunidad única para LATAM por su importancia económica y el impacto demostrable de la IA

## Fuentes

- Informe IA LATAM 2025 — Estado de la Inteligencia Artificial
- BID — Inversión en IA en América Latina
- Marco Regulatorio de IA en LATAM — Panorama 2025
- Talento en IA — Formación y Capacidades en LATAM
- IA en Agricultura de Precisión — Caso LATAM
- IA en Salud — Diagnóstico Asistido en América Latina
"""


def seed():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    cur = conn.cursor()

    # Check if already seeded
    cur.execute("SELECT id FROM ra_research_projects WHERE topic = %s",
                ("Inteligencia Artificial en Latinoamérica 2025",))
    if cur.fetchone():
        print("⚠️  Seed data already exists, skipping.")
        cur.close()
        conn.close()
        return

    # Create project
    cur.execute(
        "INSERT INTO ra_research_projects (topic, status, report, sources_count, completed_at) "
        "VALUES (%s, 'completed', %s, 0, NOW()) RETURNING id",
        ("Inteligencia Artificial en Latinoamérica 2025", REPORT)
    )
    project_id = cur.fetchone()[0]
    print(f"✅ Created project #{project_id}")

    # Insert sources with embeddings
    source_id_map = {}  # offset → list of IDs
    total_chunks = 0
    for i, source in enumerate(SOURCES):
        source_id_map[i] = []
        for chunk in source["chunks"]:
            print(f"   📄 Embedding: {source['title'][:40]}... chunk {total_chunks+1}")
            emb = get_embedding(chunk)
            cur.execute(
                "INSERT INTO ra_sources (project_id, url, title, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (project_id, source["url"], source["title"], chunk, emb)
            )
            sid = cur.fetchone()[0]
            source_id_map[i].append(sid)
            total_chunks += 1

    # Update sources count
    cur.execute(
        "UPDATE ra_research_projects SET sources_count = %s WHERE id = %s",
        (total_chunks, project_id)
    )
    print(f"   📚 {total_chunks} chunks stored")

    # Insert findings
    for f in FINDINGS:
        # Map offset source IDs to actual IDs
        real_ids = []
        for offset in f["source_ids_offset"]:
            real_ids.extend(source_id_map.get(offset, []))

        cur.execute(
            "INSERT INTO ra_findings (project_id, category, finding, confidence, source_ids) "
            "VALUES (%s, %s, %s, %s, %s)",
            (project_id, f["category"], f["finding"], f["confidence"],
             real_ids if real_ids else None)
        )
    print(f"   💡 {len(FINDINGS)} findings stored")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n🎉 Seed complete! Project: 'Inteligencia Artificial en Latinoamérica 2025'")


if __name__ == "__main__":
    seed()
