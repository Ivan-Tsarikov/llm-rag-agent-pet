from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class HFEmbedder:
    model_name: str

    def __post_init__(self) -> None:
        # CPU по умолчанию, чтобы было предсказуемо
        self._model = SentenceTransformer(self.model_name, device="cpu")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Возвращает матрицу shape (n, d) float32.
        normalize_embeddings=True -> косинусная близость = inner product.
        """
        vecs = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)
