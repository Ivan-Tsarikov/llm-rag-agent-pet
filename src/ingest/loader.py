from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import re
import pdfplumber



SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    source_path: str  # relative or absolute path as string
    text: str


def load_text_file(path: Path) -> str:
    # utf-8 with fallback to avoid crashes on weird docs
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def clean_pdf_text(text: str) -> str:
    """
    Минимальная чистка именно PDF-текста:
    - склеиваем переносы вида "паро-\nль" -> "пароль"
    - нормализуем пробелы
    """
    # склейка переносов по дефису на конце строки
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # иногда pdfplumber отдаёт много "рваных" переводов строк
    text = re.sub(r"\n{3,}", "\n\n", text)

    # множественные пробелы -> один
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def load_pdf_file(path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            # use_text_flow часто помогает “собрать” слова/строки более естественно
            txt = page.extract_text(use_text_flow=True) or ""
            txt = txt.strip()
            if txt:
                parts.append(txt)

    text = "\n\n".join(parts).strip()
    return clean_pdf_text(text)


def load_document(path: Path) -> Document:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension: {ext} for {path}")

    if ext in {".txt", ".md"}:
        text = load_text_file(path)
    else:
        text = load_pdf_file(path)

    return Document(source_path=str(path), text=text)


def iter_documents(root_dir: Path) -> Iterable[Document]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {root_dir}")

    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            doc = load_document(path)
            # пропускаем пустые документы
            if doc.text.strip():
                yield doc
