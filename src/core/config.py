from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # runtime
    app_env: str = "dev"           # dev | prod
    log_level: str = "INFO"        # DEBUG/INFO/WARNING/ERROR

    # api
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

        # paths
    docs_dir: str = "data/sample_docs"
    index_dir: str = "data/index"

    # chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # embeddings
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
