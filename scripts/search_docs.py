import sys
from pathlib import Path

from src.core.config import get_settings
from src.ingest.embedder_hf import HFEmbedder
from src.index.faiss_store import FaissStore


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_docs.py \"your query\"")
        return

    query = sys.argv[1]
    settings = get_settings()

    store = FaissStore.load(Path(settings.index_dir))
    embedder = HFEmbedder(settings.embedding_model_name)
    qv = embedder.embed_texts([query])

    hits = store.search(qv, k=settings.top_k)

    print(f"QUERY: {query}")
    print("-" * 80)
    for i, h in enumerate(hits, start=1):
        preview = h.record.text[:300].replace("\n", "\\n")
        print(f"{i}) score={h.score:.4f} | {h.record.source_path} | chunk_id={h.record.chunk_id}")
        print(f"   {preview}...")
        print()

if __name__ == "__main__":
    main()
