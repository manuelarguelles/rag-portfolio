"""
Seed data: 2 knowledge bases with sample documents.
Run: python seed.py
"""

import os, sys, time
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI

API_KEY = open(os.path.expanduser("~/.config/nvidia/api_key")).read().strip()
BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"

llm = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def get_conn():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn

def get_embedding(text: str) -> list[float]:
    resp = llm.embeddings.create(input=[text], model=EMBED_MODEL, extra_body={"input_type": "query", "truncate": "END"})
    return resp.data[0].embedding

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks

# ── Knowledge Base Data ──────────────────────────────────────────────────

KNOWLEDGE_BASES = [
    {
        "name": "Economía Perú",
        "description": "Datos macroeconómicos del Perú: PIB, inflación, comercio exterior, sectores productivos y política fiscal.",
        "documents": [
            {
                "title": "PIB y Crecimiento Económico del Perú",
                "content": """El Producto Bruto Interno (PIB) del Perú alcanzó aproximadamente 270 mil millones de dólares en 2024, posicionándose como la sexta economía más grande de América Latina. El crecimiento económico fue del 2.7% en 2024, una recuperación significativa respecto al -0.6% registrado en 2023.

El PIB per cápita se sitúa alrededor de 7,800 dólares nominales. La economía peruana se caracteriza por su diversificación creciente, aunque la minería sigue siendo el motor principal representando cerca del 10% del PIB y más del 60% de las exportaciones.

Los sectores que más contribuyeron al crecimiento en 2024 fueron: minería e hidrocarburos (+4.2%), construcción (+3.8%), servicios (+3.1%) y manufactura (+2.5%). La agricultura tuvo un crecimiento modesto del 1.2% afectada por fenómenos climáticos.

El Banco Central de Reserva del Perú (BCRP) proyecta un crecimiento del 3.1% para 2025, impulsado por la mayor producción minera de cobre, la recuperación de la inversión privada y el impulso fiscal. Los principales riesgos son la incertidumbre política interna, la desaceleración de China (principal socio comercial) y posibles disrupciones climáticas.

Históricamente, Perú experimentó un crecimiento promedio del 5.9% entre 2004-2013, conocido como el boom de las commodities. La pobreza monetaria se redujo de 58.7% en 2004 a 20.2% en 2019, aunque aumentó a 27.5% en 2020 por la pandemia. Para 2024 se estima en 23.1%."""
            },
            {
                "title": "Inflación y Política Monetaria",
                "content": """La inflación en Perú cerró 2024 en 2.4%, dentro del rango meta del BCRP de 1% a 3%. Esta cifra representa una notable desaceleración respecto al 8.5% registrado a mediados de 2022, cuando los precios internacionales de alimentos y energía se dispararon por la guerra en Ucrania.

El BCRP maneja una política de metas de inflación desde 2002, con un rango objetivo de 2% +/- 1 punto porcentual. La tasa de referencia se situó en 5.0% a finales de 2024, tras un ciclo de recortes desde el máximo de 7.75% en 2023.

La inflación subyacente (que excluye alimentos y energía) cerró en 2.8%, mientras que la inflación de alimentos fue 1.9% y la de energía 3.2%. Los rubros con mayor inflación fueron transporte (+4.1%), salud (+3.8%) y educación (+3.5%).

El sol peruano se mantuvo relativamente estable frente al dólar, cerrando 2024 en aproximadamente 3.72 soles por dólar. Las reservas internacionales netas alcanzaron 74 mil millones de dólares, equivalentes a 13 meses de importaciones, una de las posiciones más sólidas de la región.

Para 2025, el BCRP proyecta que la inflación se mantendrá dentro del rango meta, y se espera que la tasa de referencia continúe bajando gradualmente hacia 4.0% durante el año."""
            },
            {
                "title": "Comercio Exterior y Exportaciones",
                "content": """Las exportaciones peruanas alcanzaron 66.5 mil millones de dólares en 2024, un récord histórico impulsado por los altos precios de los metales. Las importaciones sumaron 52.3 mil millones, resultando en un superávit comercial de 14.2 mil millones de dólares.

La estructura de exportaciones está dominada por productos mineros: cobre (33% del total), oro (16%), zinc (5%), hierro (3%) y plomo (2%). Las exportaciones no tradicionales sumaron 20.1 mil millones, con agro-exportaciones liderando con 10.8 mil millones (arándanos, uvas, paltas, espárragos, mangos).

China es el principal destino de exportaciones peruanas (30%), seguido por Estados Unidos (15%), Unión Europea (14%), Corea del Sur (5%) y Japón (4%). En importaciones, China también lidera (28%), seguida de Estados Unidos (20%) y Brasil (6%).

Perú tiene 22 acuerdos comerciales vigentes, incluyendo TLCs con Estados Unidos, China, Unión Europea, Japón, Corea del Sur, y es miembro de la Alianza del Pacífico y el CPTPP (Tratado Integral y Progresista de Asociación Transpacífico).

El sector agro-exportador ha tenido un crecimiento explosivo: de 1.3 mil millones en 2005 a 10.8 mil millones en 2024. Perú se ha convertido en el primer exportador mundial de arándanos y quinua, y segundo de espárragos y paltas."""
            },
            {
                "title": "Sector Minero del Perú",
                "content": """El Perú es una potencia minera global: segundo productor mundial de cobre y zinc, sexto de oro, tercero de plata y segundo de molibdeno. El sector minero genera aproximadamente el 10% del PIB, más del 60% de las exportaciones y emplea directamente a 220 mil personas.

La producción de cobre alcanzó 2.8 millones de toneladas métricas en 2024, con las minas Cerro Verde, Antamina, Las Bambas y Southern Copper como principales productoras. El precio promedio del cobre fue 4.15 dólares por libra en 2024.

La inversión minera totalizo 5.2 mil millones de dólares en 2024. Los principales proyectos en pipeline incluyen: Tía María (Southern Copper, 1.4 mil millones), Zafranal (Teck, 1.3 mil millones), y la expansión de Toromocho (Chinalco, 1.3 mil millones).

El canon minero distribuido a los gobiernos regionales y locales sumó 6.8 mil millones de soles en 2024, siendo Áncash, Arequipa y Cusco los mayores receptores.

Los desafíos del sector incluyen: conflictos sociales (32% de los conflictos sociales en Perú son de origen minero), permisología compleja (obtener permisos puede tomar 8-10 años), y la necesidad de avanzar hacia una minería más sostenible con menor huella de carbono."""
            },
            {
                "title": "Política Fiscal y Deuda Pública",
                "content": """El déficit fiscal del Perú fue 2.8% del PIB en 2024, por encima del límite de la regla fiscal de 2.4%. El Gobierno apunta a reducirlo a 2.2% en 2025 mediante mayor recaudación y control del gasto.

La presión tributaria fue de 16.8% del PIB en 2024, aún baja comparada con el promedio de la OCDE (34%) y de América Latina (21%). La SUNAT (administración tributaria) implementó reformas digitales que aumentaron la recaudación en 8% real.

La deuda pública bruta alcanzó 33.5% del PIB, una de las más bajas de la región. La composición es 55% en moneda extranjera y 45% en soles. La calificación crediticia de Perú es BBB por Fitch y S&P, y Baa1 por Moody's, grado de inversión desde 2008.

El Fondo de Estabilización Fiscal tiene un saldo de 8.2 mil millones de soles. El Perú también cuenta con una Línea de Crédito Flexible del FMI por 5.4 mil millones de dólares, que no ha necesitado utilizar.

Los principales desafíos fiscales incluyen: baja recaudación tributaria, alta informalidad (72% de la PEA), necesidad de inversión en infraestructura (brecha estimada en 110 mil millones de dólares), y la reforma del sistema de pensiones."""
            }
        ]
    },
    {
        "name": "Tecnología 2025",
        "description": "Tendencias tecnológicas, empresas líderes, inversiones y transformación digital en 2025.",
        "documents": [
            {
                "title": "Inteligencia Artificial en 2025",
                "content": """La inteligencia artificial generativa alcanzó un mercado global de 180 mil millones de dólares en 2025, triplicándose desde los 60 mil millones de 2023. Las principales tendencias incluyen: modelos multimodales que procesan texto, imagen, audio y video simultáneamente; agentes autónomos capaces de ejecutar tareas complejas; y la democratización del acceso a través de modelos open-source cada vez más potentes.

OpenAI lanzó GPT-5, un modelo que superó a expertos humanos en múltiples benchmarks científicos. Su capacidad de razonamiento mejoró dramáticamente, logrando resolver problemas de matemáticas de nivel olímpico y generar código funcional para aplicaciones completas. La suscripción a ChatGPT superó los 200 millones de usuarios de pago.

Google DeepMind presentó Gemini Ultra 2.0, que integra capacidades de razonamiento con acceso a la búsqueda de Google en tiempo real. Meta liberó Llama 4, un modelo open-source de 400 mil millones de parámetros que iguala el rendimiento de modelos propietarios. Anthropic lanzó Claude 4, destacándose por sus capacidades de razonamiento largo y seguridad.

El mercado de chips para IA creció a 120 mil millones de dólares. NVIDIA mantiene su dominio con las GPUs H200 y Blackwell, pero enfrenta competencia de AMD (MI350X), Intel (Gaudi 3), y startups como Cerebras y Groq que prometen inferencia ultra-rápida. La escasez de chips GPU continúa, con tiempos de espera de 6-12 meses.

Las aplicaciones empresariales de IA más adoptadas son: asistentes de código (GitHub Copilot alcanzó 5 millones de suscriptores), automatización de servicio al cliente, análisis de documentos legales y médicos, generación de contenido de marketing, y optimización de cadenas de suministro."""
            },
            {
                "title": "Computación en la Nube y Edge Computing",
                "content": """El mercado global de computación en la nube alcanzó 820 mil millones de dólares en 2025, con un crecimiento anual del 19%. Amazon Web Services (AWS) mantiene el liderazgo con 31% del mercado, seguido de Microsoft Azure (25%) y Google Cloud (12%). Los tres hyperscalers invirtieron colectivamente 180 mil millones de dólares en infraestructura durante 2025.

La principal tendencia es la nube soberana: gobiernos y reguladores exigen que los datos se almacenen y procesen dentro de sus fronteras. AWS, Azure y Google han lanzado regiones soberanas en Europa, Asia y América Latina. En Perú, AWS anunció una región local para 2026 con inversión de 500 millones de dólares.

Edge computing creció un 35% alcanzando 61 mil millones de dólares. La proliferación de dispositivos IoT (se estiman 30 mil millones de dispositivos conectados) y la necesidad de procesamiento en tiempo real para vehículos autónomos, manufactura inteligente y gaming en la nube impulsan esta tendencia.

Kubernetes se consolidó como el estándar de facto para orquestar contenedores, con el 85% de las organizaciones usándolo en producción. Las arquitecturas serverless continúan ganando tracción, con AWS Lambda procesando 10 billones de invocaciones al mes.

Multi-cloud es la estrategia dominante: 89% de las empresas usan dos o más proveedores de nube. Herramientas como Terraform, Pulumi y Crossplane facilitan la gestión de infraestructura multi-nube. El gasto en seguridad cloud alcanzó 37 mil millones de dólares, impulsado por regulaciones como el EU AI Act y DORA."""
            },
            {
                "title": "Blockchain y Web3 en 2025",
                "content": """El mercado de criptomonedas alcanzó una capitalización de 4.2 billones de dólares en 2025, con Bitcoin superando los 100,000 dólares por primera vez. La aprobación de ETFs de Bitcoin y Ethereum en Estados Unidos atrajo más de 60 mil millones de dólares en inversión institucional.

Ethereum completó su transición a Proof of Stake y las soluciones de Layer 2 (Arbitrum, Optimism, Base) redujeron las tarifas de transacción a centavos. El TVL (Total Value Locked) en DeFi alcanzó 200 mil millones de dólares, con Aave, Lido, MakerDAO y Uniswap como protocolos líderes.

Las monedas digitales de bancos centrales (CBDCs) avanzaron significativamente: China expandió el yuan digital a nivel nacional, la Unión Europea inició el piloto del euro digital, y Brasil lanzó el Drex. El BIS (Bank for International Settlements) reportó que 130 países están explorando CBDCs.

Tokenización de activos del mundo real (RWA) emergió como la tendencia más disruptiva: bonos del tesoro tokenizados superaron 5 mil millones de dólares, inmuebles tokenizados alcanzaron 3 mil millones, y commodities tokenizados 2 mil millones. BlackRock y Goldman Sachs lanzaron fondos tokenizados en Ethereum.

Los NFTs evolucionaron más allá del arte digital: se usan para identidad digital, credenciales educativas, boletos de eventos, y trazabilidad de cadena de suministro. El mercado de gaming Web3 superó 25 mil millones de dólares con juegos como Illuvium, Star Atlas y Gods Unchained."""
            },
            {
                "title": "Ciberseguridad y Privacidad Digital",
                "content": """El mercado global de ciberseguridad alcanzó 248 mil millones de dólares en 2025, con un crecimiento del 12% anual. El costo global del cibercrimen se estima en 10.5 billones de dólares anuales, superando al PIB de todos los países excepto Estados Unidos y China.

Las principales amenazas en 2025 incluyen: ataques de ransomware impulsados por IA (que se volvieron más sofisticados y dirigidos), deepfakes para fraude corporativo (pérdidas estimadas en 25 mil millones de dólares), ataques a la cadena de suministro de software, y vulnerabilidades en sistemas de IoT.

La arquitectura Zero Trust se convirtió en el estándar de seguridad corporativa, con el 70% de las organizaciones implementándola. Gartner reportó que las empresas con Zero Trust maduro redujeron los costos de brechas de seguridad en 50%.

La IA se usa tanto para ataque como defensa: los atacantes usan IA generativa para crear phishing más convincente, malware polimórfico y evasión de detección. Los defensores usan IA para detección de amenazas en tiempo real, respuesta automatizada a incidentes y análisis de vulnerabilidades.

Regulaciones como el EU AI Act, la actualización de GDPR, y nuevas leyes en Estados Unidos y Asia impulsan la inversión en compliance y privacidad. La criptografía post-cuántica avanzó con NIST publicando estándares finales (ML-KEM, ML-DSA) que las organizaciones empiezan a implementar para prepararse contra computadoras cuánticas."""
            },
            {
                "title": "Startups y Venture Capital Tech 2025",
                "content": """La inversión de venture capital global en tecnología alcanzó 345 mil millones de dólares en 2025, una recuperación del 25% respecto a 2024. La IA acaparó el 35% del total con 121 mil millones, seguida por fintech (15%), healthtech (12%), climate tech (10%) y cybersecurity (8%).

Las rondas más grandes de 2025 incluyen: xAI (Elon Musk) levantó 12 mil millones, Anthropic cerró 8 mil millones (valoración: 80 mil millones), CoreWeave levantó 7.5 mil millones para infraestructura de GPU, y Databricks cerró 5 mil millones (valoración: 62 mil millones).

En América Latina, la inversión VC fue de 8.5 mil millones de dólares. Los principales mercados fueron Brasil (4.2B), México (1.8B), Colombia (800M) y Perú (250M). Los sectores más activos fueron fintech, logtech, y edtech. Destacaron las rondas de Nubank (expansion internacional), Rappi (profundización de superapp) y Kavak (consolidación regional).

El ecosistema de startups en Perú creció con 45 rondas de inversión en 2025. Destacaron: Yape (expansión como superapp financiera), Crehana (Serie C de 80M para educación corporativa), y varias startups de agtech aplicando IA para optimizar la agroexportación.

Los unicornios tecnológicos globales superaron los 1,500, con 250 nuevos en 2025. China produjo 60 nuevos unicornios a pesar de tensiones geopolíticas, India 45, y América Latina contribuyó con 12 nuevos unicornios."""
            }
        ]
    }
]


def seed():
    conn = get_conn()
    cur = conn.cursor()

    # Check if already seeded
    cur.execute("SELECT COUNT(*) FROM ag_knowledge_bases")
    if cur.fetchone()[0] > 0:
        print("⚠️  Ya hay datos en ag_knowledge_bases. Limpiando...")
        cur.execute("DELETE FROM ag_chunks")
        cur.execute("DELETE FROM ag_documents")
        cur.execute("DELETE FROM ag_knowledge_bases")
        conn.commit()

    total_chunks = 0
    for kb_data in KNOWLEDGE_BASES:
        print(f"\n📚 Creando KB: {kb_data['name']}")
        cur.execute(
            "INSERT INTO ag_knowledge_bases (name, description) VALUES (%s, %s) RETURNING id",
            (kb_data["name"], kb_data["description"])
        )
        kb_id = cur.fetchone()[0]

        for doc_data in kb_data["documents"]:
            print(f"  📄 Documento: {doc_data['title']}")
            cur.execute(
                "INSERT INTO ag_documents (kb_id, title, content) VALUES (%s, %s, %s) RETURNING id",
                (kb_id, doc_data["title"], doc_data["content"])
            )
            doc_id = cur.fetchone()[0]

            chunks = chunk_text(doc_data["content"])
            for i, chunk in enumerate(chunks):
                print(f"    🧩 Chunk {i+1}/{len(chunks)} — embedding...", end=" ", flush=True)
                embedding = get_embedding(chunk)
                cur.execute(
                    "INSERT INTO ag_chunks (document_id, content, embedding) VALUES (%s, %s, %s::vector)",
                    (doc_id, chunk, embedding)
                )
                total_chunks += 1
                print("✅")
                time.sleep(0.3)  # Rate limit

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n🎉 Seed completado: {len(KNOWLEDGE_BASES)} KBs, {sum(len(kb['documents']) for kb in KNOWLEDGE_BASES)} documentos, {total_chunks} chunks")


if __name__ == "__main__":
    seed()
