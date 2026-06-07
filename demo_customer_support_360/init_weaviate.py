"""Utility to (re)create the Weaviate schema for the Customer 360 demo.
Run with: `python init_weaviate.py`
"""

import weaviate
import logging
from shared.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = weaviate.Client(
    url=settings.WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(settings.WEAVIATE_API_KEY) if settings.WEAVIATE_API_KEY else None,
    timeout_config=(5, 15),
)

schema = {
    "class": "CustomerInteraction",
    "properties": [
        {"name": "interaction_id", "dataType": ["string"]},
        {"name": "customer_id", "dataType": ["string"]},
        {"name": "type", "dataType": ["string"]},
        {"name": "summary", "dataType": ["text"]},
        {"name": "date", "dataType": ["date"]},
    ],
    "vectorizer": "text2vec-openai",
}

if client.schema.contains(schema["class"]):
    logger.info("Class already exists – deleting for fresh start")
    client.schema.delete_class(schema["class"])

client.schema.create_class(schema)
logger.info("✅ Weaviate schema created successfully")
