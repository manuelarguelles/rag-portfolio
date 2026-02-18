"""
Seed script for Multimodal RAG — creates sample text items and generated images.
Run: python seed.py
"""

import json
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample-images"
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

NVIDIA_API_KEY = Path("~/.config/nvidia/api_key").expanduser().read_text().strip()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
DB_URL = "postgresql://macdenix@localhost/rag_portfolio"

nv = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


def embed_text(text: str) -> list[float]:
    resp = nv.embeddings.create(
        input=[text],
        model=EMBED_MODEL,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Sample text items
# ---------------------------------------------------------------------------
TEXT_ITEMS = [
    {
        "title": "The Art of Impressionism",
        "content": (
            "Impressionism originated in France in the 1860s. Artists like Claude Monet, "
            "Pierre-Auguste Renoir, and Edgar Degas sought to capture the fleeting effects "
            "of light and color in their paintings. They often worked outdoors (en plein air) "
            "and used visible brushstrokes and vibrant colors. The movement revolutionized "
            "Western art and paved the way for modern art movements."
        ),
    },
    {
        "title": "Tokyo: A City of Contrasts",
        "content": (
            "Tokyo is a fascinating blend of ultra-modern and traditional. Towering skyscrapers "
            "and neon-lit streets coexist with ancient temples and serene gardens. The city is "
            "famous for its efficient public transit, incredible food scene (from Michelin-starred "
            "restaurants to ramen stands), and cutting-edge technology. Districts like Shibuya, "
            "Akihabara, and Asakusa each offer unique experiences."
        ),
    },
    {
        "title": "The Amazon Rainforest",
        "content": (
            "The Amazon Rainforest spans nine countries in South America and covers about "
            "5.5 million square kilometers. It is the most biodiverse place on Earth, home to "
            "roughly 10% of all known species. The Amazon River, flowing through it, is the "
            "largest river by volume. The forest plays a critical role in regulating the global "
            "climate by absorbing CO2 and producing oxygen."
        ),
    },
    {
        "title": "Introduction to Machine Learning",
        "content": (
            "Machine learning is a subset of artificial intelligence that enables systems to "
            "learn from data without being explicitly programmed. Key paradigms include "
            "supervised learning (classification, regression), unsupervised learning (clustering, "
            "dimensionality reduction), and reinforcement learning. Popular frameworks include "
            "scikit-learn, TensorFlow, and PyTorch. Deep learning, using neural networks with "
            "many layers, has driven breakthroughs in vision, NLP, and more."
        ),
    },
    {
        "title": "Peruvian Cuisine: A Culinary Journey",
        "content": (
            "Peru's cuisine is one of the most diverse in the world, shaped by indigenous, "
            "Spanish, African, Chinese, and Japanese influences. Iconic dishes include ceviche "
            "(fresh fish cured in citrus), lomo saltado (stir-fried beef), and aji de gallina "
            "(creamy chicken). Lima has been named the culinary capital of the Americas. "
            "Peruvian superfoods like quinoa, maca, and lucuma are gaining worldwide popularity."
        ),
    },
]

# ---------------------------------------------------------------------------
# Sample images (generated with Pillow)
# ---------------------------------------------------------------------------
IMAGE_ITEMS = [
    {
        "title": "Map of Peru",
        "description": (
            "A simplified map of Peru showing its outline in green on a dark background. "
            "Peru is located in western South America, bordered by Ecuador, Colombia, Brazil, "
            "Bolivia, and Chile. The capital Lima is marked on the coast. Major geographic "
            "features include the Andes mountains, the Amazon rainforest, and the Pacific coast."
        ),
        "colors": {"bg": "#1a2332", "shape": "#34d399", "text": "#ffffff"},
        "draw_fn": "draw_peru",
    },
    {
        "title": "Python Logo Simplified",
        "description": (
            "A simplified representation of the Python programming language logo featuring "
            "two interlocking shapes in blue and yellow on a dark background. Python is a "
            "high-level, general-purpose programming language known for its readability and "
            "versatility. It is widely used in web development, data science, AI, and automation."
        ),
        "colors": {"bg": "#1a1d27", "blue": "#3776ab", "yellow": "#ffd43b", "text": "#ffffff"},
        "draw_fn": "draw_python",
    },
    {
        "title": "Neural Network Diagram",
        "description": (
            "A diagram of a simple neural network with an input layer, two hidden layers, "
            "and an output layer. Each layer contains multiple nodes (neurons) connected "
            "by lines representing weights. Neural networks are the foundation of deep learning "
            "and are used for tasks like image recognition, natural language processing, "
            "and game playing."
        ),
        "colors": {"bg": "#0f1117", "node": "#7c6aef", "line": "#3d4058", "text": "#e4e6f0"},
        "draw_fn": "draw_neural",
    },
    {
        "title": "Color Palette: Sunset",
        "description": (
            "A horizontal gradient color palette showing warm sunset colors from deep purple "
            "through magenta, orange, and gold. Sunsets create these colors as sunlight "
            "passes through more of the atmosphere, scattering shorter wavelengths and "
            "allowing warm reds, oranges, and yellows to dominate. This palette is often "
            "used in design and photography."
        ),
        "colors": {"c1": "#2d1b69", "c2": "#8b2252", "c3": "#e84545", "c4": "#ff8c42", "c5": "#ffd700"},
        "draw_fn": "draw_sunset",
    },
    {
        "title": "Database Schema Diagram",
        "description": (
            "A simple entity-relationship diagram showing three database tables: Users, "
            "Posts, and Comments. Users have fields for id, name, and email. Posts have id, "
            "title, content, and user_id. Comments have id, text, post_id, and user_id. "
            "Lines show relationships between the tables representing foreign keys."
        ),
        "colors": {"bg": "#0f1117", "table": "#222531", "border": "#7c6aef", "text": "#e4e6f0", "line": "#34d399"},
        "draw_fn": "draw_schema",
    },
]


def get_font(size=20):
    """Try to load a TTF font; fall back to default."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


# ---- Drawing functions ---------------------------------------------------

def draw_peru(img: Image.Image, draw: ImageDraw.Draw, colors: dict):
    W, H = img.size
    font = get_font(28)
    sfont = get_font(16)

    # Simplified Peru shape (polygon)
    pts = [
        (W*0.35, H*0.12), (W*0.55, H*0.08), (W*0.7, H*0.18),
        (W*0.75, H*0.35), (W*0.8, H*0.55), (W*0.65, H*0.75),
        (W*0.5, H*0.88), (W*0.35, H*0.82), (W*0.28, H*0.6),
        (W*0.25, H*0.4),
    ]
    draw.polygon(pts, fill=colors["shape"], outline="#ffffff")

    # Lima dot
    lx, ly = W*0.32, H*0.55
    draw.ellipse([lx-5, ly-5, lx+5, ly+5], fill="#ff4444")
    draw.text((lx+10, ly-8), "Lima", fill=colors["text"], font=sfont)

    # Title
    draw.text((W*0.05, H*0.92), "PERÚ", fill=colors["text"], font=font)


def draw_python(img: Image.Image, draw: ImageDraw.Draw, colors: dict):
    W, H = img.size
    font = get_font(24)
    cx, cy = W//2, H//2
    r = min(W, H) // 5

    # Blue circle (top-left)
    draw.ellipse([cx-r-20, cy-r-20, cx+r-20, cy+r-20], fill=colors["blue"])
    # Yellow circle (bottom-right)
    draw.ellipse([cx-r+20, cy-r+20, cx+r+20, cy+r+20], fill=colors["yellow"])
    # Overlap blend
    draw.ellipse([cx-r//2, cy-r//2, cx+r//2, cy+r//2], fill="#5a9a5a", outline=colors["blue"])

    draw.text((W*0.3, H*0.85), "Python", fill=colors["text"], font=font)


def draw_neural(img: Image.Image, draw: ImageDraw.Draw, colors: dict):
    W, H = img.size
    font = get_font(14)

    layers = [3, 5, 5, 2]  # nodes per layer
    layer_x = [W*0.15, W*0.4, W*0.6, W*0.85]
    positions = []

    for li, (n, x) in enumerate(zip(layers, layer_x)):
        layer_pos = []
        for ni in range(n):
            y = H * (0.15 + 0.7 * ni / max(n-1, 1))
            layer_pos.append((int(x), int(y)))
        positions.append(layer_pos)

    # Draw connections
    for li in range(len(positions)-1):
        for p1 in positions[li]:
            for p2 in positions[li+1]:
                draw.line([p1, p2], fill=colors["line"], width=1)

    # Draw nodes
    r = 12
    for layer in positions:
        for (x, y) in layer:
            draw.ellipse([x-r, y-r, x+r, y+r], fill=colors["node"], outline="#ffffff")

    # Labels
    labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
    for label, x in zip(labels, layer_x):
        draw.text((int(x)-20, H-25), label, fill=colors["text"], font=font)


def draw_sunset(img: Image.Image, draw: ImageDraw.Draw, colors: dict):
    W, H = img.size
    font = get_font(20)
    palette = [colors["c1"], colors["c2"], colors["c3"], colors["c4"], colors["c5"]]
    stripe_w = W // len(palette)

    for i, c in enumerate(palette):
        x0 = i * stripe_w
        x1 = (i+1) * stripe_w if i < len(palette)-1 else W
        draw.rectangle([x0, 0, x1, H*0.8], fill=c)

    # Label
    draw.rectangle([0, H*0.8, W, H], fill="#111111")
    draw.text((W*0.25, H*0.85), "Sunset Palette", fill="#ffffff", font=font)


def draw_schema(img: Image.Image, draw: ImageDraw.Draw, colors: dict):
    W, H = img.size
    font = get_font(14)
    sfont = get_font(11)

    tables = [
        {"name": "Users", "fields": ["id PK", "name", "email"], "pos": (30, 30)},
        {"name": "Posts", "fields": ["id PK", "title", "content", "user_id FK"], "pos": (W//2-60, 30)},
        {"name": "Comments", "fields": ["id PK", "text", "post_id FK", "user_id FK"], "pos": (W-200, 30)},
    ]

    centers = []
    for t in tables:
        x, y = t["pos"]
        tw, th = 160, 22 + len(t["fields"])*20 + 10
        # Table box
        draw.rectangle([x, y, x+tw, y+th], fill=colors["table"], outline=colors["border"], width=2)
        # Header
        draw.rectangle([x, y, x+tw, y+22], fill=colors["border"])
        draw.text((x+8, y+4), t["name"], fill="#ffffff", font=font)
        # Fields
        for i, f in enumerate(t["fields"]):
            fy = y + 26 + i*20
            draw.text((x+12, fy), f, fill=colors["text"], font=sfont)
        centers.append((x + tw//2, y + th//2))

    # Relations
    # Users -> Posts
    draw.line([centers[0][0]+80, centers[0][1], centers[1][0]-80, centers[1][1]],
              fill=colors["line"], width=2)
    # Posts -> Comments
    draw.line([centers[1][0]+80, centers[1][1], centers[2][0]-80, centers[2][1]],
              fill=colors["line"], width=2)
    # Users -> Comments
    draw.line([centers[0][0]+40, centers[0][1]+60, centers[2][0]-40, centers[2][1]+60],
              fill=colors["line"], width=2)

    draw.text((W*0.3, H-30), "Entity-Relationship Diagram", fill=colors["text"], font=font)


DRAW_FUNCTIONS = {
    "draw_peru": draw_peru,
    "draw_python": draw_python,
    "draw_neural": draw_neural,
    "draw_sunset": draw_sunset,
    "draw_schema": draw_schema,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_sample_images():
    """Generate sample images with Pillow."""
    print("🎨 Creating sample images...")
    for item in IMAGE_ITEMS:
        fname = item["title"].lower().replace(" ", "_").replace(":", "") + ".png"
        fpath = SAMPLE_DIR / fname
        item["filename"] = fname

        img = Image.new("RGB", (600, 400), item["colors"].get("bg", "#111111"))
        draw = ImageDraw.Draw(img)
        DRAW_FUNCTIONS[item["draw_fn"]](img, draw, item["colors"])
        img.save(str(fpath))
        print(f"  ✅ {fname}")


def seed_database():
    """Insert sample data into the database."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Check if already seeded
    cur.execute("SELECT COUNT(*) as cnt FROM mm_items")
    if cur.fetchone()["cnt"] > 0:
        print("⚠️  Database already has items. Clearing...")
        cur.execute("DELETE FROM mm_items")
        conn.commit()

    # Seed text items
    print("\n📝 Seeding text items...")
    for item in TEXT_ITEMS:
        emb_text = f"{item['title']} {item['content']}"
        emb = embed_text(emb_text)
        cur.execute(
            """INSERT INTO mm_items (item_type, title, content, embedding)
               VALUES ('text', %s, %s, %s::vector)""",
            (item["title"], item["content"], str(emb)),
        )
        print(f"  ✅ {item['title']}")
    conn.commit()

    # Seed image items
    print("\n🖼️  Seeding image items...")
    import shutil
    for item in IMAGE_ITEMS:
        fname = item["filename"]
        src = SAMPLE_DIR / fname
        thumb_name = f"thumb_{fname}"

        # Copy to uploads
        shutil.copy2(str(src), str(UPLOAD_DIR / fname))

        # Create thumbnail
        img = Image.open(src)
        img.thumbnail((300, 300))
        img.save(str(UPLOAD_DIR / thumb_name))

        emb_text = f"{item['title']} {item['description']}"
        emb = embed_text(emb_text)

        cur.execute(
            """INSERT INTO mm_items
               (item_type, title, description, image_path, thumbnail_path, embedding)
               VALUES ('image', %s, %s, %s, %s, %s::vector)""",
            (item["title"], item["description"], fname, thumb_name, str(emb)),
        )
        print(f"  ✅ {item['title']}")
    conn.commit()

    cur.close()
    conn.close()
    print("\n🎉 Seeding complete!")


if __name__ == "__main__":
    create_sample_images()
    seed_database()
