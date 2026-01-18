"""Sentence-transformers embedder wrapper for ingestion and search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class HFEmbedder:
    """Embedder that uses sentence-transformers on CPU."""
    model_name: str

    def __post_init__(self) -> None:
        """Initialize the underlying model."""
        # CPU по умолчанию, чтобы было предсказуемо
        self._model = SentenceTransformer(self.model_name, device="cpu")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return embedding matrix of shape (n, d) as float32."""
        vecs = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)
