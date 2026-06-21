from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = Field(..., validation_alias="ANTHROPIC_API_KEY")
    tavily_api_key: str = Field(..., validation_alias="TAVILY_API_KEY")

    # LLM
    llm_model: str = Field(default="claude-sonnet-4-6", validation_alias="LLM_MODEL")
    eval_model: str = Field(default="claude-haiku-4-5-20251001", validation_alias="EVAL_MODEL")
    llm_temperature: float = Field(default=0.0, validation_alias="LLM_TEMPERATURE")

    # Retrieval
    chroma_persist_dir: str = Field(default="./chroma_db", validation_alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="knowledge_base", validation_alias="COLLECTION_NAME")
    top_k: int = Field(default=5, validation_alias="TOP_K")
    relevance_threshold: float = Field(default=0.6, validation_alias="RELEVANCE_THRESHOLD")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL")

    # API
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
