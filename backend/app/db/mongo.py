from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["restaurant_ai"]

users_collection = db["users"]
users_collection.create_index("phone_number", unique=True)