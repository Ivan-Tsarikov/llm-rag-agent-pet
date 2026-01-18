"""Compare Retriever.search results with a direct FAISS query."""

from __future__ import annotations

import sys
from pathlib import Path

from src.core.config import get_settings
from src.ingest.embedder_hf import HFEmbedder
from src.index.faiss_store import FaissStore
from src.rag.retriever import Retriever


def _print_hits(label: str, hits) -> None:
    """Print a labeled list of hits."""
    print(label)
    for i, h in enumerate(hits, start=1):
        print(f"{i}) score={h.score:.4f} | {h.record.source_path} | chunk_id={h.record.chunk_id}")
    print()


def main() -> None:
    """Run a side-by-side search comparison."""
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.compare_search \"your query\"")
        return

    query = sys.argv[1]
    settings = get_settings()
    top_k = settings.top_k

    retriever = Retriever()
    retriever_hits = retriever.search(query, top_k=top_k)

    store = FaissStore.load(Path(settings.index_dir))
    embedder = HFEmbedder(settings.embedding_model_name)
    qv = embedder.embed_texts([query])
    script_hits = store.search(qv, k=top_k)

    print(f"QUERY: {query}")
    _print_hits("Retriever.search hits:", retriever_hits)
    _print_hits("scripts.search_docs hits:", script_hits)


if __name__ == "__main__":
    main()
