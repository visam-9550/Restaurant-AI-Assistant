import os
from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType,
)
from qdrant_client import QdrantClient

_qdrant_client = None


def get_qdrant_client(path: str = "./qdrant_data"):
    global _qdrant_client
    if _qdrant_client is not None:
        print("Nooooooooooooooooooooooooooo")
        return _qdrant_client

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("...........................................")
    # else:
    #     print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
    #     _qdrant_client = QdrantClient(path=path, force_disable_check_same_thread=True)

    return _qdrant_client


MENU_COLLECTION = "menu_collection"
def setup_qdrant():
    client = get_qdrant_client()

    # 1. Create collection only if it does not exist
    if not client.collection_exists(MENU_COLLECTION):
        client.create_collection(
            collection_name=MENU_COLLECTION,
            vectors_config=VectorParams(
                size=384,  # Match your embedding model vector size
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection created: {MENU_COLLECTION}")

    # 2. Create payload indexes
    indexes = {
        "restaurant_id": PayloadSchemaType.KEYWORD,
        "item_id": PayloadSchemaType.KEYWORD,
        "category": PayloadSchemaType.KEYWORD,
        "diet_type": PayloadSchemaType.KEYWORD,
        "cuisine": PayloadSchemaType.KEYWORD,
        "available": PayloadSchemaType.BOOL,
        "price": PayloadSchemaType.FLOAT,
    }

    for field_name, field_schema in indexes.items():
        try:
            client.create_payload_index(
                collection_name=MENU_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"Index ready: {field_name}")

        except Exception as error:
            # Existing index may throw an error depending on Qdrant version
            print(f"Index already exists / skipped: {field_name} -> {error}")