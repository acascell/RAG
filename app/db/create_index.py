from app.db.opensearch import client
from app.core.config import settings

mapping = {
    "settings": {
        "index": {
            "knn": True
        }
    },
    "mappings": {
        "properties": {

            "content": {
                "type": "text"
            },

            "embedding": {
                "type": "knn_vector",
                "dimension": 1536
            },

            "source": {
                "type": "keyword"
            },

            "customer_id": {
                "type": "keyword"
            },

            "document_type": {
                "type": "keyword"
            },

            "timestamp": {
                "type": "date"
            }
        }
    }
}

if not client.indices.exists(settings.INDEX_NAME):

    client.indices.create(
        index=settings.INDEX_NAME,
        body=mapping
    )

    print("Index created")