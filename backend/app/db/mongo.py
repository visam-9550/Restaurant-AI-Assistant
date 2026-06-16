import os
from dotenv import load_dotenv
from pymongo import MongoClient, errors

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "restaurant_ai")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

db = client[MONGO_DB_NAME]

users_collection = db["users"]
chat_history_collection = db["chat_history"]
orders_collection = db["orders"]

# Create indexes lazily and safely to avoid startup failure when MongoDB is unavailable.
def _ensure_indexes():
    try:
        users_collection.create_index("phone_number", unique=True)
        chat_history_collection.create_index(
            [
                ("session_id", 1),
                ("user_id", 1)
            ],
            unique=True
        )
    except errors.ServerSelectionTimeoutError as exc:
        print(f"Warning: MongoDB is not available during module import: {exc}")
    except Exception as exc:
        print(f"Warning: unable to create MongoDB indexes at import time: {exc}")

_ensure_indexes()