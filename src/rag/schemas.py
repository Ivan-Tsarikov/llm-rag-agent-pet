from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceItem(BaseModel):
    source_path: str
    chunk_id: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
