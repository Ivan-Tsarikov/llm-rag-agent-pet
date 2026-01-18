"""Markdown-aware chunking helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .chunker import Chunk, chunk_text


HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def split_markdown_by_headers(md: str) -> List[Tuple[str, str]]:
    """Split Markdown into header-based sections.

    Returns:
        List of (title, body_text) pairs. Title includes the header level.
    """
    md = md.strip()
    if not md:
        return []

    matches = list(HEADER_RE.finditer(md))
    if not matches:
        return [("NO_HEADER", md)]

    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        header = m.group(0).strip()
        body = md[m.end():end].strip()
        title = header
        text = (header + "\n" + body).strip()
        sections.append((title, text))

    return sections


def chunk_markdown(
    source_path: str,
    md_text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Chunk]:
    """Chunk Markdown by headers and then by size."""
    sections = split_markdown_by_headers(md_text)
    chunks: List[Chunk] = []
    global_id = 0

    for _, section_text in sections:
        # дорезаем, если секция слишком длинная
        sub = chunk_text(source_path, section_text, chunk_size=chunk_size, overlap=overlap)
        # перенумеруем chunk_id в глобальной последовательности
        for c in sub:
            chunks.append(
                Chunk(
                    source_path=c.source_path,
                    chunk_id=global_id,
                    text=c.text,
                    start_char=c.start_char,
                    end_char=c.end_char,
                )
            )
            global_id += 1

    return chunks
