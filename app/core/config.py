from pydantic import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENSEARCH_HOST: str = "opensearch"
    OPENSEARCH_PORT: int = 9200
    INDEX_NAME: str = "documents"

settings = Settings()