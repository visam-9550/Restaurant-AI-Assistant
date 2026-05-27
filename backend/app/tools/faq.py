from crewai.tools import tool
from app.services.embeddingService import EmbeddingService
from app.services.qdrantServices import QdrantService


embedding_service = EmbeddingService()
qdrant_service = QdrantService()

@tool("FAQ search tool")
def faq_search_tool(query: str) -> str:

    """
    Search restaurant FAQ information
    from Qdrant vector database.
    """

    query_vector = embedding_service.get_embedding(
        query
    )

    print("Query vector:", query_vector)

    search_results = qdrant_service.search(

        collection_name="faq_collection",

        query_vector=query_vector,

        category="faq",

        top_k=5
    )

    print(
        f"Search results for query '{query}':",
        search_results
    )

    # IMPORTANT FIX
    if not search_results.points:

        return "No relevant FAQ information found."

    refined_results = []

    for result in search_results.points:

        text = result.payload.get(
            "text",
            ""
        )

        if text:

            refined_results.append(text)

            print(f"Refined Result: {text}")

    if not refined_results:

        return "No relevant FAQ information found."

    return "\n\n".join(refined_results)