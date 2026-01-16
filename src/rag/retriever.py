from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from src.core.config import get_settings
from src.core.logging import get_logger
from src.ingest.embedder_hf import HFEmbedder
from src.index.faiss_store import FaissStore, SearchHit

log = get_logger(__name__)


class Retriever:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings

        index_dir = Path(settings.index_dir).resolve()
        self.store = FaissStore.load(index_dir)
        self.index_dir = index_dir

        # По умолчанию — модель из текущих настроек
        model_name: str = settings.embedding_model_name

        # Но если есть паспорт индекса — используем модель из него (это и есть "истина")
        meta_path = index_dir / "index_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            built_with: Optional[str] = meta.get("embedding_model_name")

            if built_with and built_with != model_name:
                if os.getenv("RAG_STRICT_INDEX_META", "1").lower() not in {"0", "false", "no"}:
                    raise ValueError(
                        "Embedding model mismatch: index built with "
                        f"'{built_with}', current settings '{model_name}'. "
                        "Rebuild index or update settings."
                    )
                log.warning(
                    "Embedding model mismatch: index built with '%s', current settings '%s'. "
                    "Using '%s' to match index.",
                    built_with,
                    model_name,
                    built_with,
                )
                model_name = built_with

        self.embedding_model_name = model_name
        self.embedder = HFEmbedder(model_name)

    def search(self, query: str, top_k: int) -> List[SearchHit]:
        qv = self.embedder.embed_texts([query])
        return self.store.search(qv, k=top_k)

    def query_vector_norm(self, query: str) -> float:
        qv = self.embedder.embed_texts([query])
        return float((qv**2).sum() ** 0.5)
