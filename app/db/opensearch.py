from opensearchpy import AsyncOpenSearch
from app.core.config import settings

client = AsyncOpenSearch(
    hosts=[{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
)

