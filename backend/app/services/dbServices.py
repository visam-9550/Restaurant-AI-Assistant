from bson import ObjectId

from app.db.mongo import chat_history_collection

from app.db.mongo import orders_collection  

class DBService:
    @staticmethod
    def save_chat_history(chat_history: dict):
        # Use upsert to avoid duplicate key error
        chat_history_collection.update_one(
            {"session_id": chat_history["session_id"], "user_id": chat_history["user_id"]},
            {"$setOnInsert": chat_history},
            upsert=True
        )
    @staticmethod
    def get_chat_history(session_id: str, user_id: str):
        document = chat_history_collection.find_one(
            {
                "session_id": session_id,
                "user_id": user_id
            },
            {
                "_id": 0,
                "message": {"$slice": -10}
            }
        )
        if not document:
            return []

        return document.get("message", [])
    # def get_chat_history(session_id: str, user_id: str):
    #     chat_history = chat_history_collection.find_one({"session_id": session_id, "user_id": user_id})
    #     return chat_history
    @staticmethod
    def update_chat_history(session_id: str, user_id: str, new_message: dict):
        chat_history_collection.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$push": {"message": new_message}}
        )
    @staticmethod
    def delete_chat_history(session_id: str, user_id: str):
        chat_history_collection.delete_one({"session_id": session_id, "user_id": user_id})



class OrderService:
    @staticmethod
    def save_order(order_details: dict):
        # Implement logic to save order to the database
        orders_collection.insert_one(order_details)
    @staticmethod
    def get_order(order_id: str):
        order = orders_collection.find_one({"_id": order_id})
        return order
    @staticmethod
    def update_order(order_id: str, updated_order: dict):
        updated_response = orders_collection.update_one({"_id": ObjectId(order_id)}, {"$set": updated_order})
        new_order_details = orders_collection.find_one({"_id": ObjectId(order_id)})
        return new_order_details
    @staticmethod
    def get_order_details_by_recipt_id(recipt_number: str):
        order = orders_collection.find_one({"recipet_no": recipt_number})
        return order
    @staticmethod
    def delete_order(order_id: str):
        # Implement logic to delete order from the database
        orders_collection.delete_one({"_id": order_id})
        return {"message": "Order deleted successfully. Order ID: " + order_id}
    


class intentService:
    @staticmethod
    def save_intent(intent_details: dict):
        # Implement logic to save intent to the database
        pass
    @staticmethod
    def get_intent(intent_id: str):
        # Implement logic to retrieve intent from the database
        pass
    @staticmethod
    def update_intent(intent_id: str, updated_intent: dict):
        # Implement logic to update intent in the database
        pass
    @staticmethod
    def delete_intent(intent_id: str):
        # Implement logic to delete intent from the database
        pass