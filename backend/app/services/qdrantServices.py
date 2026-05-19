from app.db.qdrent import client as qdrant_client
from qdrant_client.models import VectorParams, Distance

class QdrantService:
    def __init__(self, qdrant_client = qdrant_client):
        self.client = qdrant_client

    # def upsert(self, collection_name, points):
    #     self.client.upsert(collection_name=collection_name, points=points)

    def search(self, collection_name, query_vector, top_k=5):
        return self.client.search(collection_name=collection_name, query_vector=query_vector, top_k=top_k)

    def create_collection(self, collection_name, vector_size):
        self.client.create_collection(
            collection_name=collection_name, 
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    # def delete_collection(self, collection_name):
    #     self.client.delete_collection(collection_name=collection_name)