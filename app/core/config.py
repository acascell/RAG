from pydantic import BaseSettings


class Settings(BaseSettings):
    OLLAMA_URL: str = "http://ollama:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_GENERATION_MODEL: str = "qwen2.5"
    OPENSEARCH_URL: str = "http://opensearch:9200"
    INDEX_NAME: str = "rag-index"


settings = Settings()