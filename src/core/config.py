from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2] #корневая ппапка проекта

class Settings(BaseSettings):
    # runtime
    app_env: str = "dev"           # dev | prod
    log_level: str = "INFO"        # DEBUG/INFO/WARNING/ERROR

    # api
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # paths
    docs_dir: str = str(PROJECT_ROOT / "data" / "sample_docs")
    index_dir: str = str(PROJECT_ROOT / "data" / "index")

    # chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # embeddings
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    @field_validator("docs_dir", "index_dir", mode="before")
    @classmethod
    def resolve_project_paths(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
