"""
Seed data for LangChain RAG Agent.
Estadísticas y datos numéricos para probar calculator + search combo.
"""

import requests
import time

BASE_URL = "http://localhost:5006"

DOCUMENTS = [
    {
        "title": "Estadísticas Población Mundial 2024",
        "content": """La población mundial alcanzó aproximadamente 8,100 millones de personas en 2024.

China tiene una población de 1,425 millones de habitantes, mientras que India la superó con 1,442 millones.
Estados Unidos ocupa el tercer lugar con 340 millones de habitantes.
Indonesia tiene 277 millones y Pakistán 240 millones.

La tasa de crecimiento poblacional global es de aproximadamente 0.88% anual.
En 1950, la población mundial era de solo 2,500 millones.
Se proyecta que alcanzará 9,700 millones para 2050 y 10,400 millones para 2100.

La densidad poblacional promedio global es de 60 personas por kilómetro cuadrado.
Mónaco es el país más densamente poblado con 26,337 personas por km².
Mongolia es el menos denso con solo 2 personas por km².""",
        "metadata": {"category": "demographics", "year": 2024},
    },
    {
        "title": "PIB Mundial y Economías Principales",
        "content": """El Producto Interno Bruto (PIB) mundial en 2024 fue de aproximadamente 105 billones de dólares (USD).

Las 5 economías más grandes por PIB nominal:
1. Estados Unidos: $28.78 billones (trillion USD)
2. China: $18.53 billones
3. Alemania: $4.59 billones
4. Japón: $4.11 billones
5. India: $3.94 billones

El PIB per cápita promedio mundial es de aproximadamente $13,000 USD.
Luxemburgo tiene el PIB per cápita más alto con $131,384 USD.
Burundi tiene el más bajo con $230 USD.

La tasa de crecimiento del PIB mundial fue de 3.2% en 2024.
La inflación promedio global fue del 5.8%.""",
        "metadata": {"category": "economics", "year": 2024},
    },
    {
        "title": "Energía y Emisiones de CO2",
        "content": """El consumo energético mundial en 2024 fue de aproximadamente 14,400 millones de toneladas equivalentes de petróleo (Mtoe).

Distribución por fuente de energía:
- Petróleo: 30% (4,320 Mtoe)
- Carbón: 26% (3,744 Mtoe)
- Gas natural: 23% (3,312 Mtoe)
- Energías renovables: 14% (2,016 Mtoe)
- Nuclear: 7% (1,008 Mtoe)

Las emisiones globales de CO2 alcanzaron 37,400 millones de toneladas en 2024.
China emite 11,900 millones de toneladas (31.8% del total).
Estados Unidos emite 4,900 millones (13.1%).
India emite 2,800 millones (7.5%).
La Unión Europea emite 2,600 millones (7.0%).

La temperatura global promedio ha aumentado 1.2°C desde la era preindustrial.
Para limitar el calentamiento a 1.5°C, las emisiones deben reducirse un 43% para 2030.""",
        "metadata": {"category": "environment", "year": 2024},
    },
    {
        "title": "Tecnología e Internet Global",
        "content": """En 2024, hay aproximadamente 5,350 millones de usuarios de internet en el mundo, lo que representa el 66% de la población global.

Usuarios de redes sociales: 5,040 millones (62.3% de la población).
Usuarios de smartphones: 6,800 millones.
Dispositivos IoT conectados: 18,800 millones.

Velocidad promedio de internet:
- Banda ancha fija global: 92.43 Mbps
- Internet móvil global: 55.79 Mbps
- País más rápido (fija): Singapur con 300 Mbps
- País más rápido (móvil): UAE con 413 Mbps

El mercado global de computación en la nube alcanzó los $679 mil millones en 2024.
El mercado de inteligencia artificial alcanzó los $214 mil millones.
Se estima que la IA generará $4.4 billones en productividad adicional para 2030.

El tráfico global de internet es de aproximadamente 4.8 zettabytes por año.""",
        "metadata": {"category": "technology", "year": 2024},
    },
    {
        "title": "Salud Global y Esperanza de Vida",
        "content": """La esperanza de vida global promedio en 2024 es de 73.4 años.

Esperanza de vida por región:
- Japón: 84.8 años (más alta del mundo)
- Suiza: 83.8 años
- Australia: 83.5 años
- España: 83.3 años
- Estados Unidos: 77.5 años
- China: 78.2 años
- India: 70.8 años
- Nigeria: 54.7 años
- Chad: 52.5 años (más baja del mundo)

El gasto sanitario mundial es de aproximadamente $9.8 billones de dólares anuales.
Estados Unidos gasta $4.5 billones en salud (el más alto del mundo).
El gasto per cápita en salud en EEUU es de $13,493.
El promedio global de gasto en salud per cápita es de $1,200.

Hay aproximadamente 15 millones de médicos en el mundo.
La ratio global es de 1.75 médicos por cada 1,000 habitantes.
Cuba tiene la ratio más alta con 8.4 médicos por 1,000 habitantes.""",
        "metadata": {"category": "health", "year": 2024},
    },
    {
        "title": "Área y Geografía de Continentes",
        "content": """La superficie total de la Tierra es de 510.1 millones de km².
Superficie terrestre: 148.9 millones de km² (29.2%).
Superficie oceánica: 361.2 millones de km² (70.8%).

Área de los continentes:
- Asia: 44.58 millones de km²
- África: 30.37 millones de km²
- América del Norte: 24.71 millones de km²
- América del Sur: 17.84 millones de km²
- Antártida: 14.2 millones de km²
- Europa: 10.18 millones de km²
- Oceanía: 8.53 millones de km²

Los 5 países más grandes por área:
1. Rusia: 17.1 millones de km²
2. Canadá: 10.0 millones de km²
3. Estados Unidos: 9.83 millones de km²
4. China: 9.60 millones de km²
5. Brasil: 8.52 millones de km²

El punto más alto es el Monte Everest con 8,849 metros.
El punto más profundo es la Fosa de las Marianas con 10,994 metros bajo el nivel del mar.""",
        "metadata": {"category": "geography", "year": 2024},
    },
]


def seed():
    print("🌱 Seeding LangChain Agent knowledge base...\n")

    for doc in DOCUMENTS:
        print(f"📄 Ingesting: {doc['title']}")
        try:
            resp = requests.post(f"{BASE_URL}/ingest", json=doc, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                print(f"   ✅ {data['chunks']} chunks created\n")
            else:
                print(f"   ❌ Error: {resp.text}\n")
        except Exception as e:
            print(f"   ❌ Connection error: {e}\n")
        time.sleep(1)

    print("✅ Seeding complete!")

    # Verify
    try:
        resp = requests.get(f"{BASE_URL}/documents")
        docs = resp.json()
        total_chunks = sum(d["chunk_count"] for d in docs)
        print(f"\n📊 Total: {len(docs)} documents, {total_chunks} chunks")
    except Exception as e:
        print(f"Could not verify: {e}")


if __name__ == "__main__":
    seed()
