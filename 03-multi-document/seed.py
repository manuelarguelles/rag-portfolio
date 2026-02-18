"""
Seed script para Multi-Document RAG.
Crea 3 colecciones con documentos de ejemplo.
"""

import asyncio
import httpx
import sys

BASE = "http://localhost:8003"

SEED_DATA = {
    "Historia de Perú": {
        "description": "Documentos sobre la historia del Perú desde los Incas hasta la República",
        "documents": [
            {
                "title": "El Imperio Inca",
                "source": "Historia General del Perú",
                "content": """El Imperio Inca, también conocido como Tahuantinsuyo, fue el mayor imperio de la América precolombina. Su territorio se extendía desde el sur de Colombia hasta el centro de Chile, abarcando gran parte de los actuales territorios de Perú, Bolivia, Ecuador y Argentina.

El Tahuantinsuyo se dividía en cuatro suyos o regiones: Chinchaysuyo (norte), Antisuyo (este), Collasuyo (sur) y Contisuyo (oeste), todos convergiendo en el Cusco, la capital imperial.

La sociedad inca se organizaba en ayllus, que eran comunidades familiares que compartían tierras y trabajo. El sistema de reciprocidad y redistribución era fundamental: los miembros del ayllu trabajaban colectivamente en la minka, y el Estado redistribuía recursos a través de los tambos.

Los incas desarrollaron un sofisticado sistema de caminos llamado Qhapaq Ñan, que se extendía por más de 30,000 kilómetros conectando todo el imperio. Este sistema de caminos incluía puentes colgantes, tambos (posadas) y chasquis (mensajeros) que podían transmitir mensajes a gran velocidad.

La agricultura inca fue altamente avanzada. Desarrollaron sistemas de terrazas (andenes) que permitían cultivar en las laderas de las montañas. Domesticaron más de 70 especies de plantas, incluyendo la papa, el maíz, la quinua y el algodón. Los quipus eran su sistema de registro basado en cuerdas anudadas.

Pachacútec, el noveno Inca, es considerado el gran transformador del imperio. Bajo su gobierno (1438-1471), el Cusco fue reconstruido y el imperio se expandió enormemente. Mandó construir Machu Picchu, la ciudadela que hoy es Patrimonio de la Humanidad y una de las Nuevas Siete Maravillas del Mundo.""",
            },
            {
                "title": "La Conquista Española",
                "source": "Crónicas de la Conquista",
                "content": """La conquista del Perú por los españoles fue uno de los eventos más transformadores de la historia americana. Francisco Pizarro, junto con Diego de Almagro y Hernando de Luque, organizaron las expediciones que llevarían a la caída del Imperio Inca.

En 1532, Pizarro llegó a Cajamarca con aproximadamente 168 hombres. El Inca Atahualpa, quien acababa de ganar una guerra civil contra su hermano Huáscar, aceptó reunirse con los españoles. En la emboscada de Cajamarca, los españoles capturaron a Atahualpa en un evento que cambiaría la historia del continente.

Atahualpa ofreció llenar una habitación de oro y dos de plata como rescate. A pesar de cumplir su promesa, fue ejecutado el 26 de julio de 1533 en Cajamarca. Los españoles fundaron ciudades como Lima (1535) y establecieron el Virreinato del Perú en 1542.

La conquista trajo devastación demográfica. Las enfermedades europeas como la viruela, el sarampión y la gripe diezmaron a la población indígena, que se redujo de aproximadamente 9 millones a menos de 1 millón en pocas décadas.

El sistema colonial impuso la encomienda, donde los indígenas eran obligados a trabajar para encomenderos españoles. La mita colonial, una versión distorsionada del sistema inca, obligaba a los indígenas a trabajar en las minas de Potosí y Huancavelica en condiciones inhumanas.

Hubo importantes resistencias indígenas. Manco Inca estableció el Estado Neoinca en Vilcabamba (1537-1572), y la resistencia continuó hasta la captura y ejecución de Túpac Amaru I en 1572 por orden del virrey Francisco de Toledo.""",
            },
            {
                "title": "La Independencia del Perú",
                "source": "Historia de la Independencia",
                "content": """La independencia del Perú fue un proceso largo y complejo que involucró tanto corrientes libertadoras externas como movimientos internos de emancipación.

Las rebeliones indígenas del siglo XVIII prepararon el terreno. La más importante fue la de Túpac Amaru II (José Gabriel Condorcanqui) en 1780, que movilizó a miles de indígenas contra el dominio español. Aunque fue derrotada y Túpac Amaru II ejecutado brutalmente en 1781, su rebelión inspiró futuros movimientos independentistas.

La Corriente Libertadora del Sur, liderada por el general argentino José de San Martín, llegó al Perú en 1820. San Martín desembarcó en Paracas con su Expedición Libertadora y avanzó hacia Lima. El 28 de julio de 1821, San Martín proclamó la independencia del Perú en Lima con las célebres palabras: "El Perú es desde este momento libre e independiente por la voluntad general de los pueblos."

Sin embargo, la independencia no estaba asegurada militarmente. San Martín se reunió con Simón Bolívar en Guayaquil en 1822 y posteriormente se retiró del Perú. Bolívar asumió el liderazgo y, junto con el mariscal Antonio José de Sucre, dirigió las campañas finales.

Las batallas decisivas fueron Junín (6 de agosto de 1824) y Ayacucho (9 de diciembre de 1824). En Ayacucho, las fuerzas patriotas bajo el mando de Sucre derrotaron definitivamente al ejército realista del virrey La Serna, sellando la independencia no solo del Perú sino de toda Sudamérica.

La Capitulación de Ayacucho marcó el fin del dominio español en América del Sur. El Perú iniciaba así su vida como república independiente, enfrentando los desafíos de construir una nación desde las cenizas del colonialismo.""",
            },
            {
                "title": "El Perú Republicano",
                "source": "Historia Contemporánea del Perú",
                "content": """Los primeros años de la República peruana estuvieron marcados por la inestabilidad política y los caudillismos militares. Entre 1821 y 1845, el Perú tuvo más de 15 presidentes, la mayoría militares que llegaron al poder mediante golpes de estado.

La era del guano (1845-1866) trajo una efímera prosperidad económica. El guano de las islas del litoral peruano se convirtió en un fertilizante muy demandado en Europa. Ramón Castilla, presidente en dos períodos, utilizó estos ingresos para abolir la esclavitud (1854) y el tributo indígena, además de modernizar la infraestructura del país.

La Guerra del Pacífico (1879-1883) fue el conflicto más devastador de la historia peruana. Chile enfrentó a Perú y Bolivia por el control de los ricos depósitos de salitre en el desierto de Atacama. Tras batallas como Angamos, donde murió el héroe Miguel Grau, y la campaña terrestre que incluyó la ocupación de Lima, el Perú perdió los territorios de Tarapacá y Arica (temporalmente Tacna).

La Reconstrucción Nacional fue liderada por figuras como Andrés Avelino Cáceres, héroe de la resistencia en la sierra durante la guerra. El siglo XX trajo modernización pero también conflictos sociales. El gobierno de Augusto B. Leguía (1919-1930) modernizó Lima pero fue autoritario.

El APRA, fundado por Víctor Raúl Haya de la Torre en 1924, se convirtió en el partido político más importante del siglo XX. El siglo también vio reformas como la del gobierno militar de Juan Velasco Alvarado (1968-1975), que realizó una radical reforma agraria.

El conflicto armado interno (1980-2000) causado por Sendero Luminoso y el MRTA dejó cerca de 70,000 víctimas. La transición democrática del siglo XXI ha buscado consolidar instituciones y enfrentar desafíos de desigualdad y corrupción.""",
            },
        ],
    },
    "Tecnología": {
        "description": "Documentos sobre tecnologías emergentes y transformadoras",
        "documents": [
            {
                "title": "Inteligencia Artificial",
                "source": "Tech Review 2024",
                "content": """La Inteligencia Artificial (IA) es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Desde sus orígenes en la década de 1950 con Alan Turing y su famoso test, la IA ha evolucionado dramáticamente.

El Machine Learning (ML) es un subconjunto de la IA que permite a las máquinas aprender de datos sin ser programadas explícitamente. Los algoritmos de ML incluyen regresión, árboles de decisión, support vector machines y redes neuronales. El aprendizaje supervisado, no supervisado y por refuerzo son los tres paradigmas principales.

El Deep Learning, basado en redes neuronales profundas, revolucionó la IA a partir de 2012. Las Convolutional Neural Networks (CNNs) dominan la visión por computadora, mientras que las Recurrent Neural Networks (RNNs) y LSTMs se usan para datos secuenciales.

Los Transformers, introducidos en el paper "Attention Is All You Need" (2017), transformaron el procesamiento de lenguaje natural. GPT (Generative Pre-trained Transformer) de OpenAI y BERT de Google son arquitecturas basadas en transformers que lograron resultados sin precedentes.

Los Large Language Models (LLMs) como GPT-4, Claude de Anthropic y Llama de Meta representan la frontera actual. Estos modelos, entrenados con billones de tokens, pueden generar texto, código, analizar imágenes y razonar sobre problemas complejos. La técnica de RAG (Retrieval-Augmented Generation) mejora las respuestas conectando LLMs con bases de conocimiento externas.

La IA generativa está transformando industrias: desde la creación de contenido con DALL-E y Midjourney hasta la programación asistida con GitHub Copilot. Sin embargo, plantea desafíos éticos importantes como sesgos, deepfakes, desplazamiento laboral y la necesidad de regulación.""",
            },
            {
                "title": "Blockchain y Criptomonedas",
                "source": "Crypto Economics Review",
                "content": """Blockchain es una tecnología de registro distribuido que permite mantener un ledger inmutable y transparente sin necesidad de intermediarios centralizados. Fue conceptualizada por Satoshi Nakamoto en 2008 con la creación de Bitcoin.

Una blockchain es esencialmente una cadena de bloques donde cada bloque contiene un conjunto de transacciones, un hash del bloque anterior y un nonce. El mecanismo de consenso asegura que todos los nodos de la red acuerden sobre el estado del ledger. Proof of Work (PoW) y Proof of Stake (PoS) son los mecanismos más comunes.

Bitcoin, la primera criptomoneda, fue diseñada como un sistema de efectivo electrónico peer-to-peer. Su suministro está limitado a 21 millones de monedas, con halvings cada 4 años que reducen la recompensa de minería. Bitcoin ha sido llamado "oro digital" por su escasez programada.

Ethereum, lanzado en 2015 por Vitalik Buterin, introdujo los smart contracts: programas que se ejecutan automáticamente en la blockchain. Esto habilitó las aplicaciones descentralizadas (dApps), DeFi (finanzas descentralizadas) y los NFTs (tokens no fungibles). En 2022, Ethereum migró de PoW a PoS con "The Merge".

DeFi permite préstamos, intercambios y yield farming sin intermediarios bancarios. Protocolos como Uniswap, Aave y MakerDAO manejan miles de millones en valor. Los NFTs revolucionaron la propiedad digital en arte, música y gaming.

Los desafíos incluyen escalabilidad (las soluciones Layer 2 como Lightning Network y rollups buscan resolverlo), regulación gubernamental, consumo energético y volatilidad. Las CBDCs (monedas digitales de bancos centrales) representan la respuesta institucional a las criptomonedas.""",
            },
            {
                "title": "Cloud Computing",
                "source": "Cloud Architecture Magazine",
                "content": """Cloud Computing es el modelo de entrega de servicios de computación a través de internet, permitiendo acceso on-demand a recursos como servidores, almacenamiento, bases de datos y software sin gestión directa de infraestructura física.

Los tres modelos de servicio principales son: IaaS (Infrastructure as a Service) que provee máquinas virtuales y almacenamiento; PaaS (Platform as a Service) que ofrece plataformas de desarrollo; y SaaS (Software as a Service) que entrega aplicaciones completas. AWS, Azure y Google Cloud son los principales proveedores.

La arquitectura de microservicios reemplazó a los monolitos en la nube. Las aplicaciones se dividen en servicios pequeños e independientes que se comunican via APIs. Kubernetes se convirtió en el estándar para orquestar contenedores Docker, permitiendo escalado automático y alta disponibilidad.

Serverless computing, con servicios como AWS Lambda y Google Cloud Functions, permite ejecutar código sin gestionar servidores. El modelo de pago por uso reduce costos y simplifica las operaciones. Event-driven architectures aprovechan serverless para construir sistemas reactivos.

DevOps y CI/CD (Continuous Integration/Continuous Deployment) son prácticas esenciales en la nube. Herramientas como Jenkins, GitHub Actions, Terraform y Ansible automatizan el ciclo de vida del software. Infrastructure as Code (IaC) permite definir infraestructura en archivos de configuración versionados.

Edge computing complementa la nube llevando computación cerca de los usuarios finales. CDNs, IoT gateways y servicios como AWS Wavelength reducen la latencia. La arquitectura multi-cloud y hybrid-cloud permite a las empresas distribuir cargas de trabajo entre múltiples proveedores y data centers privados.""",
            },
            {
                "title": "Computación Cuántica",
                "source": "Quantum Computing Today",
                "content": """La computación cuántica utiliza principios de la mecánica cuántica para procesar información de maneras fundamentalmente diferentes a las computadoras clásicas. Mientras los bits clásicos son 0 o 1, los qubits pueden estar en superposición de ambos estados simultáneamente.

El entrelazamiento cuántico permite que dos qubits estén correlacionados de tal manera que el estado de uno determina instantáneamente el estado del otro, sin importar la distancia. Esta propiedad, junto con la superposición, permite a las computadoras cuánticas explorar múltiples soluciones en paralelo.

IBM, Google y startups como IonQ y Rigetti lideran el desarrollo de hardware cuántico. Google afirmó haber logrado la "supremacía cuántica" en 2019 con su procesador Sycamore de 53 qubits, resolviendo en 200 segundos un problema que tardaría 10,000 años en una supercomputadora clásica.

Los algoritmos cuánticos más importantes incluyen el algoritmo de Shor para factorización de números grandes (amenaza la criptografía RSA), el algoritmo de Grover para búsqueda en bases de datos no ordenadas (aceleración cuadrática), y VQE (Variational Quantum Eigensolver) para simulación molecular.

Las aplicaciones potenciales son enormes: descubrimiento de fármacos mediante simulación molecular, optimización de cadenas logísticas, modelado financiero, criptografía cuántica (QKD) y machine learning cuántico. Sin embargo, los desafíos son significativos: la decoherencia, las tasas de error y la necesidad de operar cerca del cero absoluto.

La criptografía post-cuántica está siendo desarrollada para proteger datos contra futuros ataques cuánticos. NIST estandarizó algoritmos como CRYSTALS-Kyber y CRYSTALS-Dilithium en 2024. La computación cuántica tolerante a fallos, que requeriría millones de qubits, sigue siendo un objetivo a largo plazo.""",
            },
        ],
    },
    "Ciencia": {
        "description": "Documentos sobre disciplinas científicas fundamentales",
        "documents": [
            {
                "title": "Física Moderna",
                "source": "Fundamentos de Física",
                "content": """La física moderna se desarrolló a principios del siglo XX con dos revoluciones: la teoría de la relatividad de Albert Einstein y la mecánica cuántica. Estas teorías transformaron nuestra comprensión del universo a escalas tanto cósmicas como subatómicas.

La Relatividad Especial (1905) estableció que la velocidad de la luz es constante para todos los observadores y que el espacio y el tiempo son relativos. La famosa ecuación E=mc² demostró la equivalencia entre masa y energía. La Relatividad General (1915) describió la gravedad como la curvatura del espacio-tiempo causada por la masa.

La mecánica cuántica, desarrollada por Planck, Bohr, Heisenberg, Schrödinger y Dirac, describe el comportamiento de las partículas subatómicas. El principio de incertidumbre de Heisenberg establece que no se pueden conocer simultáneamente la posición y el momento de una partícula con precisión arbitraria.

El Modelo Estándar de física de partículas clasifica todas las partículas elementales conocidas: quarks (up, down, charm, strange, top, bottom), leptones (electrón, muón, tau y sus neutrinos), bosones de gauge (fotón, gluones, W±, Z) y el bosón de Higgs, descubierto en 2012 en el CERN.

La gravedad cuántica sigue siendo uno de los mayores problemas no resueltos. La teoría de cuerdas propone que las partículas fundamentales son en realidad cuerdas vibrantes en dimensiones adicionales. La gravedad cuántica de lazos (Loop Quantum Gravity) ofrece un enfoque alternativo.

La materia oscura (27% del universo) y la energía oscura (68% del universo) son misterios fundamentales. Solo el 5% del universo es materia ordinaria. Las ondas gravitacionales, predichas por Einstein y detectadas por LIGO en 2015, abrieron una nueva ventana para observar el cosmos.""",
            },
            {
                "title": "Biología Molecular y Genética",
                "source": "Biología Contemporánea",
                "content": """La biología molecular estudia los procesos fundamentales de la vida a nivel molecular. El descubrimiento de la estructura del ADN por Watson y Crick en 1953, basado en los datos de difracción de rayos X de Rosalind Franklin, fue uno de los hitos más importantes de la ciencia.

El ADN (ácido desoxirribonucleico) es una doble hélice formada por nucleótidos con cuatro bases: adenina (A), timina (T), citosina (C) y guanina (G). El dogma central de la biología molecular describe el flujo de información: ADN → ARN → Proteínas, mediante transcripción y traducción.

El Proyecto Genoma Humano, completado en 2003, secuenció los aproximadamente 3 mil millones de pares de bases del genoma humano, identificando unos 20,000-25,000 genes codificantes de proteínas. Este logro abrió la era de la genómica y la medicina personalizada.

CRISPR-Cas9, descubierto como sistema inmune bacteriano y adaptado como herramienta de edición genética por Jennifer Doudna y Emmanuelle Charpentier (Premio Nobel 2020), revolucionó la biología. Permite cortar y editar ADN con precisión sin precedentes, con aplicaciones en terapia génica, agricultura y investigación básica.

La epigenética estudia cambios heredables en la expresión génica sin alterar la secuencia de ADN. Modificaciones como la metilación del ADN y la acetilación de histonas regulan qué genes se expresan en cada célula. El microbioma, los billones de microorganismos que habitan nuestro cuerpo, influye en la salud, la inmunidad y hasta el comportamiento.

La biología sintética busca diseñar y construir sistemas biológicos nuevos. Organismos modificados producen medicamentos, biocombustibles y materiales. La terapia con células CAR-T, que modifica las propias células inmunes del paciente para combatir el cáncer, es uno de los avances más prometedores de la medicina moderna.""",
            },
            {
                "title": "Astronomía y Cosmología",
                "source": "El Universo: Una Guía Moderna",
                "content": """La astronomía moderna ha transformado nuestra comprensión del universo. El telescopio espacial Hubble, lanzado en 1990, revolucionó la observación astronómica, y el James Webb Space Telescope (JWST), lanzado en 2021, nos permite ver el universo en sus primeras etapas con detalle sin precedentes.

El Big Bang es el modelo cosmológico estándar que describe el origen del universo hace aproximadamente 13.8 mil millones de años. La radiación cósmica de fondo (CMB), descubierta en 1964, es la evidencia más directa del Big Bang: una radiación de microondas que llena todo el universo.

Las galaxias son los bloques constructivos del universo a gran escala. La Vía Láctea, nuestra galaxia, contiene entre 100 y 400 mil millones de estrellas y un agujero negro supermasivo en su centro llamado Sagitario A*, cuya primera imagen fue capturada en 2022 por el Event Horizon Telescope.

Los exoplanetas son planetas que orbitan otras estrellas. Desde el primer descubrimiento confirmado en 1995, se han identificado más de 5,000 exoplanetas. El telescopio Kepler descubrió miles de ellos. Algunos se encuentran en la "zona habitable" donde podría existir agua líquida, como los del sistema TRAPPIST-1.

Los agujeros negros son regiones donde la gravedad es tan intensa que nada, ni siquiera la luz, puede escapar. Existen agujeros negros estelares (formados por el colapso de estrellas masivas), intermedios y supermasivos (en centros galácticos). La primera imagen de un agujero negro fue capturada en 2019 en la galaxia M87.

La expansión acelerada del universo, descubierta en 1998 (Premio Nobel 2011), implica que el universo se expande cada vez más rápido, impulsado por la energía oscura. El destino final del universo podría ser un "Gran Frío" (Big Freeze) donde toda la energía se disipa. Las ondas gravitacionales y la astronomía multi-mensajero están abriendo nuevas fronteras en nuestra exploración del cosmos.""",
            },
        ],
    },
}


async def seed():
    async with httpx.AsyncClient(base_url=BASE, timeout=120) as client:
        for coll_name, coll_data in SEED_DATA.items():
            print(f"\n{'='*60}")
            print(f"📁 Creando colección: {coll_name}")

            # Create collection
            resp = await client.post("/collections", json={
                "name": coll_name,
                "description": coll_data["description"],
            })
            if resp.status_code == 409:
                print(f"   ⚠️  Ya existe, saltando...")
                # Get collection id
                colls = (await client.get("/collections")).json()
                coll_id = next(c["id"] for c in colls if c["name"] == coll_name)
            elif resp.status_code == 200:
                coll_id = resp.json()["id"]
                print(f"   ✅ Creada (id={coll_id})")
            else:
                print(f"   ❌ Error: {resp.text}")
                continue

            # Check existing docs
            docs_resp = await client.get(f"/collections/{coll_id}/documents")
            existing_docs = docs_resp.json()
            existing_titles = {d["title"] for d in existing_docs}

            # Add documents
            for doc in coll_data["documents"]:
                if doc["title"] in existing_titles:
                    print(f"   📄 '{doc['title']}' ya existe, saltando...")
                    continue

                print(f"   📄 Agregando: {doc['title']}...", end=" ", flush=True)
                resp = await client.post(f"/collections/{coll_id}/documents", json={
                    "title": doc["title"],
                    "content": doc["content"],
                    "source": doc["source"],
                })
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"✅ ({data['chunk_count']} chunks)")
                else:
                    print(f"❌ {resp.text}")

    print(f"\n{'='*60}")
    print("🎉 Seed completado!")
    print("   Abre http://localhost:8003 para ver la interfaz")


if __name__ == "__main__":
    asyncio.run(seed())
