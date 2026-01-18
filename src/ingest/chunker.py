"""Text chunking utilities used during index building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """Chunk of source text with character offsets."""
    source_path: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph structure."""
    # минимальная нормализация: убираем лишние пробелы, но не ломаем структуру сильно
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def chunk_text(
    source_path: str,
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
    min_chunk_chars: int = 200,
) -> List[Chunk]:
    """Split text into overlapping chunks by character count.
    Args:
        source_path: Path to the original document.
        text: Raw document text.
        chunk_size: Max characters per chunk.
        overlap: Overlap between adjacent chunks.
        min_chunk_chars: Minimum chunk length to keep.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    i = 0
    chunk_id = 0

    while i < len(text):
        j = min(i + chunk_size, len(text))
        piece = text[i:j].strip()
        if len(piece) >= min_chunk_chars or (j == len(text) and piece):
            chunks.append(
                Chunk(
                    source_path=source_path,
                    chunk_id=chunk_id,
                    text=piece,
                    start_char=i,
                    end_char=j,
                )
            )
            chunk_id += 1
        i += step

    return chunks
