"""Build a FAISS index from local documents.

Example:
    python -m scripts.build_index
"""
from pathlib import Path
import json
from datetime import datetime

from src.core.config import get_settings
from src.ingest.pipeline import build_chunks
from src.ingest.embedder_hf import HFEmbedder
from src.index.faiss_store import ChunkRecord, FaissStore


def main():
    """Create embeddings and persist the FAISS index files."""
    settings = get_settings()
    index_dir = Path(settings.index_dir)

    chunks = build_chunks(settings)
    if not chunks:
        print("No chunks found. Put docs into data/sample_docs first.")
        return

    texts = [c.text for c in chunks]
    records = [
        ChunkRecord(
            source_path=c.source_path,
            chunk_id=c.chunk_id,
            start_char=c.start_char,
            end_char=c.end_char,
            text=c.text,
        )
        for c in chunks
    ]

    print(f"Chunks: {len(records)}")
    print(f"Embedding model: {settings.embedding_model_name}")

    embedder = HFEmbedder(settings.embedding_model_name)
    vectors = embedder.embed_texts(texts)
    print(f"Vectors: {vectors.shape} dtype={vectors.dtype}")

    store = FaissStore.build(vectors=vectors, records=records)
    store.save(index_dir)

    # --- index passport (metadata) -----------
    meta = {
        "created_at": datetime.now().isoformat(),
        "embedding_model_name": settings.embedding_model_name,
        "docs_dir": settings.docs_dir,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "total_chunks": len(records),
        "vector_dim": int(vectors.shape[1]),
    }

    (index_dir / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # -----------------------------------------

    print(f"Saved index to: {index_dir.resolve()}")
    print("Files:")
    print(f" - {index_dir / 'faiss.index'}")
    print(f" - {index_dir / 'chunks.jsonl'}")
    print(f" - {index_dir / 'index_meta.json'}")


if __name__ == "__main__":
    main()
    