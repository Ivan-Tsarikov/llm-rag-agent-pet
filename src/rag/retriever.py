from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.core.config import get_settings
from src.core.logging import get_logger
from src.ingest.embedder_hf import HFEmbedder
from src.index.faiss_store import FaissStore, SearchHit

log = get_logger(__name__)


class Retriever:
    """
    Loads FAISS index + chunk records and provides search(query)->top hits.
    Also exposes embedding_model_name and query_vector_norm() for debugging.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        index_dir = Path(self.settings.index_dir).resolve()

        # 1) Проверим, что индекс на месте
        faiss_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.jsonl"
        if not faiss_path.exists() or not chunks_path.exists():
            raise RuntimeError(
                f"Index files not found in {index_dir}. "
                f"Expected: {faiss_path.name}, {chunks_path.name}. "
                f"Run: python scripts/build_index.py"
            )

        # 2) Загрузим индекс/чанки
        self.store = FaissStore.load(index_dir)

        # 3) Выберем embedding model:
        #    - по умолчанию из settings
        #    - если есть index_meta.json — сверим
        model_name: str = self.settings.embedding_model_name
        self.embedding_model_name: str = model_name

        meta_path = index_dir / "index_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                built_with = meta.get("embedding_model_name")
                if isinstance(built_with, str) and built_with.strip():
                    built_with = built_with.strip()
                    if built_with != model_name:
                        msg = (
                            f"Index built with '{built_with}', but current settings embedding model is '{model_name}'. "
                            f"Rebuild index or align EMBEDDING_MODEL_NAME."
                        )
                        if self.settings.rag_strict_index_meta:
                            raise RuntimeError(msg)
                        log.warning(msg + " (strict meta check disabled)")
                        # Можно принудительно использовать built_with, но это опасно если модели нет локально
                        # Поэтому здесь мы только предупреждаем.
            except Exception as e:
                if self.settings.rag_strict_index_meta:
                    raise RuntimeError(f"Failed to read index_meta.json: {e}") from e
                log.warning("Failed to read index_meta.json: %s", e)

        # 4) Инициализируем эмбеддер
        self.embedder = HFEmbedder(self.embedding_model_name)

    def query_vector_norm(self, query: str) -> float:
        v = self.embedder.embed_texts([query])
        # v: (1, D)
        vv = v[0]
        n = float(np.linalg.norm(vv))
        return n

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        query = query.strip()
        if len(query) < 2:
            return []

        top_k = max(1, min(int(top_k), 50))

        qv = self.embedder.embed_texts([query])  # (1, D)

        # Sanity checks: no NaN/Inf and non-zero vector
        s = float((qv * qv).sum())
        if not math.isfinite(s) or s < 1e-12:
            log.warning("Bad query embedding (nan/inf/zero). query=%r", query[:100])
            return []

        return self.store.search(qv, k=top_k)
