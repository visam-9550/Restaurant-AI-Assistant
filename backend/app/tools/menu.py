from qdrant_client.models import Filter, FieldCondition, MatchValue
from crewai.tools import tool
from app.services.embeddingService import EmbeddingService
from app.services.qdrantServices import QdrantService
from app.db.mongo import users_collection
from bson import ObjectId
from app.services.menuServices import build_preferences_menu_response

embedding_service = EmbeddingService()
qdrant_service = QdrantService()

@tool("User Preferences Search Tool")
def user_preferences_search_tool(user_id: str) -> dict:

    """
    Search user preferences from MongoDB
    based on user ID. Returns user's food preferences.
    """
    print(f"Searching preferences for user ID: {user_id}")
    user_preferences = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "diet_type": 1, "favorite_cuisines": 1, "favorite_foods": 1, "disliked_foods": 1, "spice_level": 1, "preferred_meal_type": 1, "preferred_beverages": 1, "favorite_restaurant_categories": 1, "budget_preference": 1})
    print(f"MongoDB query result for user ID {user_id}: {user_preferences}")
    if not user_preferences:
        return {"message": "No preferences found for this user."}
    print(f"Found preferences for user ID {user_id}: {user_preferences}")
    formated_preferences = build_preferences_menu_response(user_preferences)
    print(f"Formatted preferences for user ID {user_id}: {formated_preferences}")
    return str(formated_preferences)

@tool("Menu search tool")
def menu_search_tool(query: str) -> str:

    """
    Search restaurant menu information
    from Qdrant vector database.
    """
    print(f"Received menu search query: {query}")
    search_vector = embedding_service.get_embedding(query)
    print(f"Search vector: {search_vector}")
    search_results = qdrant_service.search(
        collection_name="menu_collection",
        query_vector=search_vector,
        # category="menu",
        top_k=50
    )

    print("Search results:", search_results)

    if not search_results.points:
        return "No relevant menu information found."

    refined_results = []

    seen_texts = set()

    for result in search_results.points:

        text = result.payload.get("text", "")

        if text and text not in seen_texts:

            refined_results.append(text)

            seen_texts.add(text)

    if not refined_results:
        return "No menu items found."

    return "\n\n".join(refined_results)


# @tool("Extract food items tool")


# @tool("Menu search tool")
# def menu_search_tool(query: str) -> str:
#     """
#     Search restaurant menu information from Qdrant vector database.
#     Returns relevant menu context for customer queries.
#     """

#     # For demonstration, we return a static response.
#     # In a real implementation, this would query the Qdrant database.
#     search_vector = embedding_service.get_embedding(query)
#     print("Search vector for query:", search_vector)
#     search_results = qdrant_service.search(
#         collection_name="menu_collection",
#         query_vector=search_vector,
#         category="menu",
#         top_k=5
#     )
#     print(f"Search results for query '{query}':", search_results)
#     if not search_results:
#         return "No relevant menu information found."
#     refined_results = []

#     for result in search_results:
#         # Handle both tuple and object results
#         if hasattr(result, "score") and hasattr(result, "payload"):
#             score = result.score
#             payload = result.payload
#         elif isinstance(result, (list, tuple)) and len(result) >= 3:
#             # Typical Qdrant tuple: (id, score, payload, ...)
#             score = result[1]
#             payload = result[2]
#         else:
#             continue

#         if score > 0.7:
#             text = payload.get("text", "") if isinstance(payload, dict) else ""
#             if text:
#                 refined_results.append(text)
#                 print(f"Refined Result: {text}")

#     return "\n".join(refined_results)