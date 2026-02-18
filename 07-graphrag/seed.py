"""
Seed data for GraphRAG Pipeline.
Ingests 3 rich historical texts with many entities and relationships.
"""

import requests
import time
import sys

BASE = "http://localhost:5007"

DOCUMENTS = [
    {
        "title": "La Conquista del Imperio Inca",
        "content": """En 1532, Francisco Pizarro lideró una expedición española desde Panamá hacia el Perú con el objetivo de conquistar el vasto Imperio Inca. Pizarro, nacido en Trujillo, España, era un conquistador experimentado que ya había explorado las costas de América del Sur. Su expedición contaba con apenas 168 hombres, incluyendo a sus hermanos Hernando Pizarro, Juan Pizarro y Gonzalo Pizarro.

El Imperio Inca, conocido como Tawantinsuyu, se extendía desde el sur de Colombia hasta el centro de Chile, abarcando territorios de Ecuador, Perú y Bolivia. Su capital era Cusco, una ciudad sagrada considerada el ombligo del mundo. El imperio estaba gobernado por el Sapa Inca Atahualpa, quien acababa de ganar una guerra civil contra su hermano Huáscar por el control del trono.

La captura de Atahualpa ocurrió en Cajamarca el 16 de noviembre de 1532. Pizarro tendió una emboscada durante un encuentro supuestamente pacífico. El fraile dominico Vicente de Valverde presentó un breviario a Atahualpa, quien lo arrojó al suelo, lo que sirvió como pretexto para el ataque. Miles de guerreros incas fueron masacrados en la plaza.

Atahualpa ofreció llenar una habitación con oro y dos con plata a cambio de su libertad — el famoso Rescate de Atahualpa. Pese a cumplir su promesa, fue juzgado y ejecutado en julio de 1533. Tras su muerte, Pizarro marchó hacia Cusco y la tomó con la ayuda de grupos indígenas aliados como los Cañaris y Huancas, que resentían el dominio inca.

Pizarro fundó la ciudad de Lima el 18 de enero de 1535 como la Ciudad de los Reyes, estableciéndola como capital del Virreinato del Perú. Diego de Almagro, socio original de Pizarro, disputó el control de Cusco, lo que llevó a las Guerras Civiles entre los conquistadores. Almagro fue ejecutado en 1538, y Pizarro fue asesinado por seguidores de Almagro en Lima en 1541.""",
    },
    {
        "title": "La Revolución Industrial",
        "content": """La Revolución Industrial comenzó en Gran Bretaña a mediados del siglo XVIII y transformó radicalmente la sociedad, la economía y la tecnología mundial. Este proceso de industrialización se extendió luego a Europa continental, Estados Unidos y Japón durante el siglo XIX.

James Watt perfeccionó la máquina de vapor en 1769, mejorando el diseño original de Thomas Newcomen. La máquina de vapor de Watt se convirtió en el motor de la revolución, impulsando fábricas, minas y eventualmente el transporte. Watt trabajó en la Universidad de Glasgow y se asoció con Matthew Boulton para producir sus máquinas en la fábrica Soho Manufactory en Birmingham.

La industria textil fue la primera en mecanizarse. Richard Arkwright inventó la water frame en 1769, mientras que James Hargreaves creó la spinning jenny en 1764 y Edmund Cartwright desarrolló el telar mecánico en 1785. Estas innovaciones transformaron ciudades como Manchester y Liverpool en centros industriales.

George Stephenson construyó la primera línea ferroviaria pública, el Ferrocarril de Stockton y Darlington, inaugurada en 1825. Su locomotora Locomotion No. 1 fue un hito en el transporte. En 1830, el Ferrocarril de Liverpool y Manchester conectó dos de las ciudades más importantes de Inglaterra, con la famosa locomotora Rocket.

La Revolución Industrial trajo consigo profundos cambios sociales. El movimiento ludita, liderado por trabajadores textiles, destruyó maquinaria entre 1811 y 1816 en protesta contra la mecanización. Karl Marx y Friedrich Engels, residentes en Manchester, escribieron El Manifiesto Comunista en 1848, analizando las tensiones de clase producidas por el capitalismo industrial. Robert Owen, empresario galés, fundó New Lanark en Escocia como una comunidad modelo que demostraba que el bienestar de los trabajadores era compatible con la productividad.

La producción de acero se revolucionó con el proceso Bessemer, inventado por Henry Bessemer en 1856, y posteriormente con el proceso Siemens-Martin. Estas innovaciones permitieron la construcción de puentes, edificios y vías férreas a escala masiva. Michael Faraday descubrió la inducción electromagnética en 1831 en la Royal Institution de Londres, sentando las bases para la generación eléctrica que caracterizaría la Segunda Revolución Industrial.""",
    },
    {
        "title": "El Sistema Solar",
        "content": """El Sistema Solar se formó hace aproximadamente 4,600 millones de años a partir de una nube de gas y polvo llamada nebulosa solar. En su centro se encuentra el Sol, una estrella de tipo espectral G2V que contiene el 99.86% de toda la masa del sistema. El Sol está compuesto principalmente de hidrógeno y helio, y genera energía mediante fusión nuclear en su núcleo.

Los cuatro planetas interiores — Mercurio, Venus, Tierra y Marte — son planetas rocosos o terrestres. Mercurio, el más cercano al Sol, tiene una temperatura superficial que varía entre -180°C y 430°C. Venus, a menudo llamado el gemelo de la Tierra, tiene una atmósfera densa de dióxido de carbono que crea un efecto invernadero extremo con temperaturas de 465°C. La Tierra es el único planeta conocido con vida, posee un satélite natural — la Luna — y está ubicada en la zona habitable del Sol. Marte, el planeta rojo, tiene el volcán más grande del sistema solar, el Olympus Mons, y el cañón más profundo, Valles Marineris. Marte posee dos pequeños satélites: Fobos y Deimos.

El Cinturón de Asteroides separa los planetas interiores de los exteriores y contiene millones de objetos rocosos. Ceres, el objeto más grande del cinturón, fue reclasificado como planeta enano por la Unión Astronómica Internacional en 2006.

Los planetas exteriores — Júpiter, Saturno, Urano y Neptuno — son gigantes gaseosos (o gigantes de hielo en el caso de Urano y Neptuno). Júpiter es el planeta más grande, con una masa 318 veces la de la Tierra. Su Gran Mancha Roja es una tormenta que ha durado al menos 400 años. Júpiter tiene 95 lunas conocidas, incluyendo las cuatro lunas galileanas descubiertas por Galileo Galilei en 1610: Ío, Europa, Ganímedes y Calisto.

Saturno es famoso por su sistema de anillos, compuestos de partículas de hielo y roca. Su luna más grande, Titán, es la única luna del sistema solar con una atmósfera densa. La sonda Cassini-Huygens, una misión conjunta de NASA, ESA y ASI (agencia espacial italiana), estudió Saturno entre 2004 y 2017.

Más allá de Neptuno se encuentra el Cinturón de Kuiper, hogar de Plutón, que fue reclasificado como planeta enano en 2006. La sonda New Horizons de la NASA sobrevoló Plutón en julio de 2015, revelando montañas de hielo de agua y una región en forma de corazón llamada Tombaugh Regio, en honor a Clyde Tombaugh, quien descubrió Plutón en 1930 desde el Observatorio Lowell en Arizona.

La Nube de Oort, una esfera hipotética de objetos helados, marca el límite exterior del Sistema Solar y se cree que es el origen de los cometas de periodo largo. La sonda Voyager 1, lanzada por la NASA en 1977, es el objeto humano más distante, habiendo cruzado la heliopausa en 2012 para entrar en el espacio interestelar.""",
    },
]

def main():
    print("🕸️  GraphRAG Seed — Ingesting documents...\n")

    for i, doc in enumerate(DOCUMENTS, 1):
        print(f"[{i}/{len(DOCUMENTS)}] Ingesting: {doc['title']}")
        print(f"    Content: {len(doc['content'])} chars")

        try:
            resp = requests.post(f"{BASE}/ingest", json=doc, timeout=120)
            data = resp.json()
            if "error" in data:
                print(f"    ❌ Error: {data['error']}")
            else:
                print(f"    ✓ {data['entities']} entities, {data['relationships']} relationships, {data['chunks']} chunks")
        except requests.exceptions.ConnectionError:
            print(f"    ❌ Cannot connect to {BASE}. Is the server running?")
            sys.exit(1)
        except Exception as e:
            print(f"    ❌ Error: {e}")

        if i < len(DOCUMENTS):
            print("    Waiting 3s before next...")
            time.sleep(3)

    print("\n✅ Seed complete! Visit http://localhost:5007 to see the graph.")

    # Show stats
    try:
        stats = requests.get(f"{BASE}/stats").json()
        print(f"\n📊 Stats:")
        print(f"   Documents: {stats['documents']}")
        print(f"   Entities: {stats['total_entities']}")
        print(f"   Relationships: {stats['relationships']}")
        print(f"   Chunks: {stats['chunks']}")
        if stats['entity_types']:
            types_str = ', '.join(f"{t['entity_type']}({t['count']})" for t in stats['entity_types'])
            print(f"   Types: {types_str}")
    except:
        pass

if __name__ == "__main__":
    main()
