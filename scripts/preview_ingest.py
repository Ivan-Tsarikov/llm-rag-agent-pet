"""Preview how documents are chunked before building the index."""

from pathlib import Path

from src.ingest.loader import iter_documents
from src.ingest.chunker import chunk_text
from src.ingest.md_chunker import chunk_markdown


def main():
    """Load sample docs and print a short chunk preview."""
    docs_dir = Path("data/sample_docs")
    print(f"Loading docs from: {docs_dir.resolve()}")

    total_docs = 0
    total_chunks = 0

    for doc in iter_documents(docs_dir):
        total_docs += 1
        if doc.source_path.lower().endswith(".md"):
            chunks = chunk_markdown(doc.source_path, doc.text, chunk_size=800, overlap=120)
        else:
            chunks = chunk_text(doc.source_path, doc.text, chunk_size=800, overlap=120)
        total_chunks += len(chunks)

        print("\n" + "=" * 80)
        print(f"FILE: {doc.source_path}")
        print(f"TEXT_LEN: {len(doc.text)} chars | CHUNKS: {len(chunks)}")

        # печатаем первые 2 чанка
        for c in chunks[:2]:
            preview = c.text[:400].replace("\n", "\\n")
            print(f"\n  - chunk_id={c.chunk_id} [{c.start_char}:{c.end_char}] len={len(c.text)}")
            print(f"    preview: {preview}...")

    print("\n" + "-" * 80)
    print(f"TOTAL_DOCS: {total_docs}")
    print(f"TOTAL_CHUNKS: {total_chunks}")


if __name__ == "__main__":
    main()
