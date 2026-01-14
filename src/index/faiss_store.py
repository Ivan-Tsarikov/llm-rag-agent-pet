from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


@dataclass(frozen=True)
class ChunkRecord:
    source_path: str
    chunk_id: int
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    score: float
    record: ChunkRecord


class FaissStore:
    def __init__(self, index: faiss.Index, records: List[ChunkRecord]) -> None:
        self.index = index
        self.records = records

    @staticmethod
    def build(vectors: np.ndarray, records: List[ChunkRecord]) -> "FaissStore":
        if len(records) != vectors.shape[0]:
            raise ValueError("records count must match vectors rows")

        d = vectors.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product, с normalize_embeddings=True это косинус
        index.add(vectors)
        return FaissStore(index=index, records=records)

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "faiss.index"))

        meta_path = dir_path / "chunks.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    @staticmethod
    def load(dir_path: Path) -> "FaissStore":
        idx_path = dir_path / "faiss.index"
        meta_path = dir_path / "chunks.jsonl"

        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files not found in {dir_path}")

        index = faiss.read_index(str(idx_path))

        records: List[ChunkRecord] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                records.append(ChunkRecord(**obj))

        return FaissStore(index=index, records=records)

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[SearchHit]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        scores, ids = self.index.search(query_vec, k)
        hits: List[SearchHit] = []

        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx == -1:
                continue
            hits.append(SearchHit(score=float(score), record=self.records[idx]))

        return hits
