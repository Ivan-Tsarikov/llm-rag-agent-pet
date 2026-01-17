from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_from_root(p: str | Path) -> str:
    """If path is relative -> anchor to PROJECT_ROOT. Return as string."""
    pp = Path(p)
    if not pp.is_absolute():
        pp = PROJECT_ROOT / pp
    return str(pp.resolve())


class Settings(BaseSettings):
    """
    Настройки приложения.
    Читаются из переменных окружения и .env в корне проекта.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    app_env: Literal["dev", "prod"] = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    tool_backend: Literal["local", "mcp"] = Field(default="local", alias="TOOL_BACKEND")
    mcp_url: str = Field(default="http://localhost:9001", alias="MCP_URL")

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    docs_dir: str = Field(default=str(PROJECT_ROOT / "data" / "sample_docs"), alias="DOCS_DIR")
    index_dir: str = Field(default=str(PROJECT_ROOT / "data" / "index"), alias="INDEX_DIR")

    # ------------------------------------------------------------------
    # Chunking / Retrieval
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")

    # HF sentence-transformers model (small, multilingual, ok for MVP)
    embedding_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL_NAME",
    )

    # Fail fast if index_meta mismatch (recommended)
    rag_strict_index_meta: bool = Field(default=True, alias="RAG_STRICT_INDEX_META")

    # ------------------------------------------------------------------
    # LLM mode: ollama or openai-compatible (any provider that mimics OpenAI API)
    # ------------------------------------------------------------------
    llm_mode: Literal["ollama", "openai"] = Field(default="ollama", alias="LLM_MODE")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:3b", alias="OLLAMA_MODEL")
    # keep model warm: "10m", "0", "-1" (forever)
    ollama_keep_alive: str = Field(default="10m", alias="OLLAMA_KEEP_ALIVE")
    # generation caps (helps avoid long stalls)
    ollama_num_predict: int = Field(default=512, alias="OLLAMA_NUM_PREDICT")
    ollama_temperature: float = Field(default=0.2, alias="OLLAMA_TEMPERATURE")

    # OpenAI-compatible
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    def normalized(self) -> "Settings":
        """
        Возвращает копию настроек с нормализованными путями.
        """
        s = self.model_copy(deep=True)
        s.docs_dir = _resolve_from_root(s.docs_dir)
        s.index_dir = _resolve_from_root(s.index_dir)

        # sanity
        if s.chunk_overlap < 0:
            s.chunk_overlap = 0
        if s.chunk_overlap >= s.chunk_size:
            s.chunk_overlap = max(0, s.chunk_size // 4)

        if s.top_k < 1:
            s.top_k = 1
        if s.top_k > 20:
            s.top_k = 20

        return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Важно: нормализуем пути всегда одинаково, независимо от cwd
    return Settings().normalized()
