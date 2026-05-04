from opensearchpy import OpenSearch
from app.core.config import settings

client = OpenSearch(
    hosts=[{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
)

