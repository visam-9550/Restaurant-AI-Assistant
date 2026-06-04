from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["restaurant_ai"]

users_collection = db["users"]
users_collection.create_index("phone_number", unique=True)

chat_history_collection = db["chat_history"]
chat_history_collection.create_index(
    [
        ("session_id", 1),
        ("user_id", 1)
    ],
    unique=True
)

orders_collection = db["orders"]