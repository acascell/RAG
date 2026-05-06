"""OpenSearch async client instance.

Provides a shared AsyncOpenSearch client configured with the URL from application settings.
Used across the application for indexing and querying documents.
"""

from opensearchpy import AsyncOpenSearch
from app.core.config import settings

client = AsyncOpenSearch(
    hosts=[settings.OPENSEARCH_URL],
)

