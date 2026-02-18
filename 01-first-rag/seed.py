"""
Seed — Carga documentos de ejemplo en el sistema RAG.
=====================================================
Ejecutar: python seed.py
"""

import os
import sys
import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://macdenix@localhost/rag_portfolio")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")

# ── Documentos de ejemplo ──────────────────────────────────────────────────

DOCUMENTS = [
    {
        "title": "El Imperio Inca",
        "content": (
            "El Imperio Inca, conocido como Tawantinsuyu, fue el mayor imperio de la América "
            "precolombina. Su territorio abarcaba desde el sur de Colombia hasta el centro de "
            "Chile, incluyendo partes de Argentina, Bolivia, Ecuador y Perú. La capital era "
            "Cusco, considerada el ombligo del mundo. Los incas desarrollaron un sofisticado "
            "sistema de caminos llamado Qhapaq Ñan, que conectaba todo el imperio a lo largo "
            "de más de 30,000 kilómetros. Utilizaban los quipus como sistema de registro y "
            "contabilidad. El último emperador inca, Atahualpa, fue capturado por Francisco "
            "Pizarro en Cajamarca en 1532."
        ),
    },
    {
        "title": "Machu Picchu",
        "content": (
            "Machu Picchu es una ciudadela inca del siglo XV ubicada en la cordillera oriental "
            "de los Andes peruanos, a 2,430 metros sobre el nivel del mar. Fue construida "
            "durante el reinado del emperador Pachacútec como residencia real y santuario "
            "religioso. La ciudadela fue abandonada durante la conquista española y permaneció "
            "oculta hasta 1911, cuando el explorador estadounidense Hiram Bingham la dio a "
            "conocer al mundo. En 1983 fue declarada Patrimonio de la Humanidad por la UNESCO "
            "y en 2007 fue elegida como una de las Nuevas Siete Maravillas del Mundo. "
            "Su arquitectura incluye templos, terrazas agrícolas y un sistema hidráulico avanzado."
        ),
    },
    {
        "title": "Inteligencia Artificial",
        "content": (
            "La inteligencia artificial (IA) es una rama de la informática que busca crear "
            "sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. "
            "Esto incluye el aprendizaje automático (machine learning), el procesamiento del "
            "lenguaje natural (NLP), la visión por computadora y la robótica. Los modelos de "
            "lenguaje grande (LLMs) como GPT y Claude representan avances significativos en NLP. "
            "La IA generativa puede crear texto, imágenes, música y código. Los transformers, "
            "introducidos en el paper 'Attention is All You Need' (2017), revolucionaron el campo. "
            "RAG (Retrieval-Augmented Generation) combina búsqueda de información con generación "
            "de texto para producir respuestas más precisas y fundamentadas."
        ),
    },
    {
        "title": "PostgreSQL y pgvector",
        "content": (
            "PostgreSQL es un sistema de gestión de bases de datos relacional de código abierto, "
            "conocido por su robustez y extensibilidad. pgvector es una extensión de PostgreSQL "
            "que añade soporte para vectores y búsqueda por similitud. Permite almacenar embeddings "
            "de alta dimensión y realizar búsquedas usando distancia coseno, producto interno o "
            "distancia euclidiana. pgvector soporta índices HNSW (Hierarchical Navigable Small "
            "World) e IVFFlat para búsquedas eficientes. Es una alternativa a bases de datos "
            "vectoriales dedicadas como Pinecone o Weaviate, con la ventaja de mantener datos "
            "relacionales y vectoriales en el mismo sistema."
        ),
    },
    {
        "title": "La Gastronomía Peruana",
        "content": (
            "La gastronomía peruana es considerada una de las más diversas y ricas del mundo. "
            "Su cocina fusiona tradiciones indígenas, españolas, africanas, chinas y japonesas. "
            "El ceviche, plato emblemático, consiste en pescado crudo marinado en jugo de limón "
            "con cebolla, ají y cilantro. Otros platos icónicos incluyen el lomo saltado "
            "(fusión chino-peruana), el ají de gallina, la causa limeña y el anticucho. "
            "Lima ha sido nombrada Capital Gastronómica de América Latina múltiples veces. "
            "Restaurantes como Central (dirigido por Virgilio Martínez) y Maido (Mitsuharu "
            "Tsumura) figuran entre los mejores del mundo en la lista The World's 50 Best."
        ),
    },
    {
        "title": "Embeddings y Búsqueda Vectorial",
        "content": (
            "Los embeddings son representaciones numéricas de texto en un espacio vectorial "
            "de alta dimensión. Cada palabra, oración o documento se convierte en un vector "
            "de números flotantes (por ejemplo, 1024 dimensiones). Textos con significado "
            "similar tienen vectores cercanos en este espacio. La búsqueda vectorial encuentra "
            "los documentos más similares comparando la distancia entre vectores. Las métricas "
            "comunes son la similitud coseno (mide el ángulo entre vectores), la distancia "
            "euclidiana (distancia recta) y el producto interno. Los modelos de embedding "
            "como E5, BGE y Ada convierten texto a vectores semánticos de manera eficiente."
        ),
    },
    {
        "title": "El Sistema Solar",
        "content": (
            "El Sistema Solar está formado por el Sol y los cuerpos celestes que orbitan a su "
            "alrededor. Tiene ocho planetas: Mercurio, Venus, Tierra, Marte, Júpiter, Saturno, "
            "Urano y Neptuno. La Tierra es el tercer planeta y el único conocido con vida. "
            "Júpiter es el planeta más grande, con una masa 318 veces la de la Tierra. "
            "El cinturón de asteroides se encuentra entre Marte y Júpiter. Plutón fue "
            "reclasificado como planeta enano en 2006 por la Unión Astronómica Internacional. "
            "El Sol contiene el 99.86% de la masa total del sistema y su luz tarda "
            "aproximadamente 8 minutos y 20 segundos en llegar a la Tierra."
        ),
    },
    {
        "title": "Python como Lenguaje de Programación",
        "content": (
            "Python es un lenguaje de programación de alto nivel, interpretado y de propósito "
            "general creado por Guido van Rossum en 1991. Es conocido por su sintaxis clara "
            "y legible, que enfatiza la legibilidad del código. Python es ampliamente usado en "
            "ciencia de datos, inteligencia artificial, desarrollo web, automatización y scripting. "
            "Frameworks populares incluyen Django y Flask para web, FastAPI para APIs, "
            "NumPy y Pandas para datos, y PyTorch y TensorFlow para machine learning. "
            "Python usa tipado dinámico y recolección de basura automática. Su ecosistema de "
            "paquetes, disponible a través de PyPI, cuenta con más de 400,000 proyectos."
        ),
    },
]


def get_embedding(text: str) -> list[float]:
    """Genera un embedding usando NVIDIA NIM API (síncrono)."""
    resp = httpx.post(
        f"{NVIDIA_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        json={
            "model": EMBEDDING_MODEL,
            "input": [text],
            "input_type": "passage",
            "encoding_format": "float",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def main():
    """Carga los documentos de ejemplo en la base de datos."""
    if not NVIDIA_API_KEY:
        print("❌ Error: NVIDIA_API_KEY no configurada en .env")
        sys.exit(1)

    print("🔌 Conectando a PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # Crear tabla si no existe
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector(1024),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Verificar si ya hay documentos
    cur.execute("SELECT COUNT(*) FROM documents")
    count = cur.fetchone()[0]
    if count > 0:
        print(f"⚠️  Ya hay {count} documentos en la base de datos.")
        resp = input("¿Quieres agregar los documentos de ejemplo de todas formas? (s/n): ")
        if resp.lower() != "s":
            print("Cancelado.")
            cur.close()
            conn.close()
            return

    print(f"\n📚 Cargando {len(DOCUMENTS)} documentos de ejemplo...\n")

    for i, doc in enumerate(DOCUMENTS, 1):
        print(f"  [{i}/{len(DOCUMENTS)}] {doc['title']}...", end=" ", flush=True)
        try:
            embedding = get_embedding(doc["content"])
            cur.execute(
                """INSERT INTO documents (title, content, embedding)
                   VALUES (%s, %s, %s::vector)""",
                (doc["title"], doc["content"], str(embedding)),
            )
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")

    cur.execute("SELECT COUNT(*) FROM documents")
    total = cur.fetchone()[0]
    print(f"\n🎉 Listo. Total de documentos en la base de datos: {total}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
