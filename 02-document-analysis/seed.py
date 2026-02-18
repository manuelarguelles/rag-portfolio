"""Seed script for Document Analysis - uploads sample PDFs via API."""
import httpx
import asyncio
import sys
from pathlib import Path

BASE_URL = "http://localhost:8002"
SAMPLE_DIR = Path(__file__).parent / "sample-pdfs"

async def seed():
    pdfs = list(SAMPLE_DIR.glob("*.pdf"))
    if not pdfs:
        print("No sample PDFs found. Run: python generate_samples.py")
        sys.exit(1)
    
    print(f"Found {len(pdfs)} sample PDFs\n")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check server is running
        try:
            r = await client.get(f"{BASE_URL}/stats")
            print(f"Server stats: {r.json()}\n")
        except httpx.ConnectError:
            print(f"ERROR: Server not running at {BASE_URL}")
            print("Start it with: python app.py")
            sys.exit(1)
        
        for pdf_path in sorted(pdfs):
            print(f"Uploading: {pdf_path.name}...")
            with open(pdf_path, "rb") as f:
                files = {"file": (pdf_path.name, f, "application/pdf")}
                try:
                    r = await client.post(f"{BASE_URL}/upload", files=files)
                    if r.status_code == 200:
                        data = r.json()
                        print(f"  ✅ {data.get('pages', '?')} pages, {data.get('chunks', '?')} chunks")
                    else:
                        print(f"  ❌ Status {r.status_code}: {r.text[:200]}")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        
        # Final stats
        print("\n--- Final Stats ---")
        r = await client.get(f"{BASE_URL}/stats")
        print(r.json())
        
        # List documents
        r = await client.get(f"{BASE_URL}/documents")
        docs = r.json()
        print(f"\nDocuments loaded: {len(docs)}")
        for doc in docs:
            print(f"  - {doc.get('filename', '?')} ({doc.get('total_pages', '?')} pages, {doc.get('total_chunks', '?')} chunks)")

if __name__ == "__main__":
    asyncio.run(seed())
