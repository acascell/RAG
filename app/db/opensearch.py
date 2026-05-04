from opensearchpy import AsyncOpenSearch
from app.core.config import settings

client = AsyncOpenSearch(
    hosts=[settings.OPENSEARCH_URL],
)

