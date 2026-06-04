from app.db.qdrent import get_qdrant_client
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue

class QdrantService:
    def __init__(self, qdrant_client=None):
        self.client = qdrant_client or get_qdrant_client()

    def get_stored_data(self, collection_name):
        records, next_page =  self.client.scroll(
            collection_name=collection_name,
            limit=20
        )
        return records

    def upsert(self, collection_name, points):
        self.client.upsert(collection_name=collection_name, points=points)

    def collection_exists(self,collection_name):
        try:
            collections = (
                self.client.get_collections()
            )
            collection_names = [
                collection.name

                for collection in (
                    collections.collections
                )
            ]
            return (
                collection_name
                in collection_names
            )
        except Exception as e:
            print(e)
            return False

    def search(self, collection_name, query_vector, top_k=50, score_threshold=0.4):
        filter_conditions = None
        if collection_name != "intent_classification":
            filter_conditions = [
                    FieldCondition(
                        key="restaurant_id",
                        match=MatchValue(value="restaurant123")
                    )
                    # Filter(
                    #     must=[
                    #         FieldCondition(
                    #             key="category",
                    #             match=MatchValue(value=category)
                    #         )
                    #     ]
                    # )
                ]
        return self.client.query_points(
            collection_name=collection_name, 
            query=query_vector,
            query_filter=Filter(
                must=filter_conditions
            ),
            limit=top_k,
            score_threshold=score_threshold
        )

    def create_collection(self, collection_name, vector_size):
        self.client.create_collection(
            collection_name=collection_name, 
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def delete_collection(self, collection_name):
        self.client.delete_collection(collection_name=collection_name)