import os

from qdrant_client import QdrantClient

_qdrant_client = None


def get_qdrant_client(path: str = "./qdrant_data"):
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        _qdrant_client = QdrantClient(path=path, force_disable_check_same_thread=True)

    return _qdrant_client