from pathlib import Path
from typing import List

from src.core.config import Settings
from src.ingest.loader import iter_documents
from src.ingest.chunker import Chunk, chunk_text
from src.ingest.md_chunker import chunk_markdown


def build_chunks(settings: Settings) -> List[Chunk]:
    docs_dir = Path(settings.docs_dir)

    all_chunks: List[Chunk] = []
    for doc in iter_documents(docs_dir):
        sp = doc.source_path.lower()
        if sp.endswith(".md"):
            chunks = chunk_markdown(
                doc.source_path,
                doc.text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        else:
            chunks = chunk_text(
                doc.source_path,
                doc.text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        all_chunks.extend(chunks)

    return all_chunks
