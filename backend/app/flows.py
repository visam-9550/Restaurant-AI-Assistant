import uuid
import re
from typing import Any
from qdrant_client.models import PointStruct
from openai import chat
from app.crew import intent_crew, faq_crew, menu_crew, order_crew, menu_parser_crew, greeting_crew
from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel, Field
import json
from app.db.redis import redis_client
from app.db.mongo import users_collection, chat_history_collection, orders_collection
from app.services.dbServices import DBService, OrderService
from app.services.qdrantServices import QdrantService
from app.services.embeddingService import EmbeddingService
from bson import ObjectId
import datetime

qdrant_service = QdrantService()
embedding_service = EmbeddingService()


def serialize_bson(obj):
    """Recursively convert BSON / non-serializable objects to JSON-serializable types."""
    # ObjectId
    if isinstance(obj, ObjectId):
        return str(obj)
    # datetime
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    # dict -> recurse
    if isinstance(obj, dict):
        return {k: serialize_bson(v) for k, v in obj.items()}
    # list/tuple -> recurse
    if isinstance(obj, (list, tuple)):
        return [serialize_bson(v) for v in obj]
    # other types -> return as-is
    return obj

class ordersModel(BaseModel):
    # order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    restaurant_id: str
    items: list[dict]
    total_amount: float
    status: str
    recipet_no: str

class messageModel(BaseModel):
    role: str = ""
    content: str = ""

class RestaurantAssistantFlowModel(BaseModel):
    query: str = ""
    intent_name: str = ""
    response: Any = ""
    user_id: str = ""
    restaurant_id: str = ""
    session_id: str = ""
    message: list[messageModel] = []
    intent_from_db: bool = False
    is_menu_search: bool = False
    is_recommendation: bool = False
    is_menu_price: bool = False
    include_price: bool = False
    include_menu: bool = False
    chat_history: str = ""
    last_menu_response: str = ""

class SampleFlowModel(BaseModel):
    query: str = ""
    response: str = ""

class SampleFlow(Flow):
    print("Initializing SampleFlow")
    @start()
    def start_flow(self):
        print("Starting SampleFlow with query:")
        # self.state.response = "Flow started. Please provide your query."
        return "llllll"

    @listen(start_flow)
    def handle_query(self):
        print("Handling query:")
        # self.state.response = f"Received query: {self.state.query}"
        return "pppppppp"
    
# flow = SampleFlow()
# response = flow.kickoff(inputs={"query": "What is the weather today?"})



class RestaurantAssistantFlow(Flow[RestaurantAssistantFlowModel]):
    @start()
    def classify_intent(self):
        user_query = self.state.query
        
        # Get existing chat history
        history = DBService.get_chat_history(self.state.session_id, self.state.user_id)
        
        # Format chat history for context
        chat_history = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in history
                if msg.get("content")
            ]
        )
        self.state.chat_history = chat_history
        
        # Initialize or update chat history (upsert will handle both cases)
        if history:  # If history exists, update it
            DBService.update_chat_history(self.state.session_id, self.state.user_id, {"role": "user", "content": user_query})
        else:  # If no history exists, create new document
            DBService.save_chat_history({
                "session_id": self.state.session_id,
                "user_id": self.state.user_id,
                "restaurant_id": self.state.restaurant_id,
                "message": [{"role": "user", "content": user_query}]
            })

        # Method 1
        embedding = embedding_service.get_embedding(user_query)
        is_collection_exists = qdrant_service.collection_exists(collection_name="intent_classification")
        if is_collection_exists:
            intent_search_results = qdrant_service.search(
                collection_name="intent_classification",
                query_vector=embedding,
                top_k=1,
                score_threshold=0.94
            )
            self.state.intent_from_db = bool(True) if intent_search_results else False
            if (
                intent_search_results
                and intent_search_results.points
                and intent_search_results.points[0].score >= 0.94
                ):
                top_result = intent_search_results.points[0]

                self.state.intent_from_db = True
                self.state.intent_name = top_result.payload["intent"]
                print(f"Intent found in DB with sufficient score: {self.state.intent_name} (score: {top_result.score})")
                return self.state.intent_name


        # query = user_query.lower().strip()

        # # ==========================================================
        # # KEYWORDS
        # # ==========================================================

        # greeting_keywords = {
        #     "hi",
        #     "hello",
        #     "hey",
        #     "good morning",
        #     "good afternoon",
        #     "good evening",
        #     "greetings"
        # }

        # cancel_order_keywords = [
        #     "cancel order",
        #     "cancel my order",
        #     "cancel the order",
        #     "please cancel",
        #     "please cancel my order",
        #     "i want to cancel my order",
        #     "i need to cancel my order"
        # ]

        # confirm_order_keywords = [
        #     "confirm order",
        #     "confirm my order",
        #     "yes confirm",
        #     "yes please confirm",
        #     "go ahead",
        #     "proceed",
        #     "place it",
        #     "confirm it"
        # ]

        # modify_order_keywords = [
        #     "modify order",
        #     "change order",
        #     "update order",
        #     "edit order",
        #     "add item",
        #     "remove item",
        #     "increase quantity",
        #     "decrease quantity",
        #     "change quantity",
        #     "update quantity",
        #     "add one more",
        #     "one more",
        #     "add to my order",
        #     "remove from my order"
        # ]

        # menu_price_keywords = [
        #     "price of",
        #     "cost of",
        #     "how much is",
        #     "what is the price of",
        #     "what's the price of",
        #     "menu prices",
        #     "food prices",
        #     "show prices",
        #     "show menu with prices",
        #     "price list",
        #     "pricing",
        #     "menu price",
        #     "dish price",
        #     "item price"
        # ]

        # menu_search_keywords = [
        #     "show menu",
        #     "show me menu",
        #     "show me the menu",
        #     "full menu",
        #     "complete menu",
        #     "menu card",
        #     "menu items",
        #     "food menu",
        #     "restaurant menu",
        #     "available items",
        #     "available dishes",
        #     "show all items",
        #     "browse menu",
        #     "see menu",
        #     "list menu",
        #     "display menu",
        #     "what is the menu",
        #     "what's on the menu"
        # ]

        # food_recommendation_keywords = [
        #     "recommend",
        #     "suggest",
        #     "best",
        #     "popular",
        #     "most ordered",
        #     "best seller",
        #     "top selling",
        #     "customer favorite",
        #     "signature dish",
        #     "what should i eat",
        #     "help me choose",
        #     "choose for me",
        #     "pick for me",
        #     "trending",
        #     "special dish",
        #     "spicy food",
        #     "healthy food",
        #     "crispy food",
        #     "sweet dish"
        # ]

        # faq_keywords = [
        #     "parking",
        #     "parking facility",
        #     "parking availability",
        #     "opening hours",
        #     "closing hours",
        #     "restaurant timings",
        #     "location",
        #     "address",
        #     "contact number",
        #     "phone number",
        #     "reservation",
        #     "table booking",
        #     "refund policy",
        #     "cancellation policy",
        #     "payment methods",
        #     "complaint"
        # ]

        # order_food_keywords = [
        #     "place order",
        #     "order",
        #     "give me",
        #     "get me",
        #     "bring me",
        #     "send me",
        #     "can i get",
        #     "may i get",
        #     "i would like",
        #     "i'd like",
        #     "i'll have",
        #     "i will have",
        #     "serve me"
        # ]


        # # ==========================================================
        # # HELPER
        # # ==========================================================

        # def contains_any(text, keywords):
        #     return any(keyword in text for keyword in keywords)


        # # ==========================================================
        # # FLAGS
        # # ==========================================================

        # self.state.include_price = False
        # self.state.include_menu = False


        # # ==========================================================
        # # 1. GREETING
        # # ==========================================================

        # if query in greeting_keywords:

        #     self.state.intent_name = "greeting"


        # # ==========================================================
        # # 2. CANCEL ORDER
        # # ==========================================================

        # elif contains_any(query, cancel_order_keywords):

        #     self.state.intent_name = "cancel_order"


        # # ==========================================================
        # # 3. ORDER MODIFICATION
        # # ==========================================================

        # elif contains_any(query, modify_order_keywords):

        #     self.state.intent_name = "order_modification"


        # # ==========================================================
        # # 4. ORDER CONFIRMATION
        # # ==========================================================

        # elif contains_any(query, confirm_order_keywords):

        #     self.state.intent_name = "order_confirmation"


        # # ==========================================================
        # # 5. MENU RELATED
        # # ==========================================================

        # else:

        #     is_menu_search = contains_any(
        #         query,
        #         menu_search_keywords
        #     )

        #     is_menu_price = contains_any(
        #         query,
        #         menu_price_keywords
        #     )

        #     is_recommendation = contains_any(
        #         query,
        #         food_recommendation_keywords
        #     )

        #     is_faq = contains_any(
        #         query,
        #         faq_keywords
        #     )

        #     # ------------------------------------------------------
        #     # Recommendation
        #     # ------------------------------------------------------

        #     if is_recommendation:

        #         self.state.intent_name = "menu_recommendation"

        #         self.state.include_price = is_menu_price
        #         self.state.include_menu = is_menu_search

        #     # ------------------------------------------------------
        #     # Menu Price
        #     # ------------------------------------------------------

        #     elif is_menu_price:

        #         self.state.intent_name = "menu_price_query"

        #         self.state.include_menu = is_menu_search

        #     # ------------------------------------------------------
        #     # Menu Search
        #     # ------------------------------------------------------

        #     elif is_menu_search:

        #         self.state.intent_name = "menu_search"

        #     # ------------------------------------------------------
        #     # FAQ
        #     # ------------------------------------------------------

        #     elif is_faq:

        #         self.state.intent_name = "faq"

        #     # ------------------------------------------------------
        #     # Order Food
        #     # ------------------------------------------------------

        #     else:

        #         quantity_pattern = re.search(
        #             r"\b\d+\s+\w+\b",
        #             query
        #         )

        #         if (
        #             quantity_pattern
        #             or contains_any(query, order_food_keywords)
        #         ):

        #             self.state.intent_name = "order_food"

        #         else:

        #             self.state.intent_name = "unknown"


        # print("Intent:", self.state.intent_name)
        # print("Include Price:", self.state.include_price)
        # print("Include Menu:", self.state.include_menu)


       
        # if(self.state.intent_name and self.state.intent_name != "unknown"):
        #     print(f"Intent classified based on keyword matching: {self.state.intent_name}")
        #     if not is_collection_exists:
        #         qdrant_service.create_collection(
        #             collection_name="intent_classification",
        #             vector_size=384
        #         )
        #     qdrant_service.upsert(
        #         collection_name="intent_classification",
        #         points=[
        #             PointStruct(
        #                 id=str(uuid.uuid4()),
        #                 vector=embedding,
        #                 payload={
        #                     "query": user_query,
        #                     "intent": self.state.intent_name
        #                 }
        #             )
        #         ]
        #     )
        #     return self.state.intent_name

        # # Method 3

        response = intent_crew.kickoff(inputs={"input": user_query})
        result_text = str(response.raw) if hasattr(response, 'raw') else str(response)
        data = json.loads(result_text)
        intent = data.get("intent", result_text)
        self.state.intent_name = intent
        if not is_collection_exists:
            qdrant_service.create_collection(
                collection_name="intent_classification",
                vector_size=384
            )
        qdrant_service.upsert(
            collection_name="intent_classification",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "query": user_query,
                        "intent": intent
                    }
                )
            ]
        )
        return self.state.intent_name

    @router(classify_intent)
    def route_content(self, intent: str):
        if (intent == "faq" or intent == "restaurant_policy" or intent == "restaurant_timings" or intent == "payment_issue" or intent == "delivery_enquiry" or intent == "complaint" or intent == "parking_enquiry"):
            return "faq_crew"
        elif intent in ["menu_search", "menu_price_query", "menu_availability_query", "menu_recommendation", "food_recommendation"]:
            return "menu_crew"
        elif intent in ["order_food", "order_status", "order_modification", "order_cancellation", "cancel_order", "modify_cart"]:
            return "order_crew"
        elif intent in ["order_modification", "order_cancellation"]:
            # return "order_modification/cancellation" 
            return "order_crew" 
        elif intent in ["confirm_order", "order_confirmation"]:
            return "place_order"
        # elif intent == "menu_parser":
        #     return menu_parser_crew
        elif intent in ["greeting", "gratitude", "acknowledgement"]:
            return "greeting_crew"
        else:
            return "default_crew"

    @listen("greeting_crew")
    def handle_greeting_intent(self):
        greeting_response = greeting_crew.kickoff(inputs={"input": self.state.query, "chat_history": self.state.chat_history})
        greeting_result_text = str(greeting_response.raw) if hasattr(greeting_response, 'raw') else str(greeting_response)
        print(f"Greeting Crew response: {greeting_result_text}")
        self.state.response = greeting_result_text
        print(f"Greeying response: {self.state.response}")
        return f"Greeting intent {greeting_result_text} has been processed"

    @listen("faq_crew")
    def handle_intent(self):
        faq_response = faq_crew.kickoff({"input": self.state.query})
        faq_result_text = str(faq_response.raw) if hasattr(faq_response, 'raw') else str(faq_response)
        self.state.response = faq_result_text
        # print(f"FAQ crew response: {faq_result_text}")
        DBService.update_chat_history(self.state.session_id, self.state.user_id, {"role": "assistant", "content": faq_result_text})
        return f"Intent '{faq_result_text}' has been processed."
    
    @listen("menu_crew")
    def handle_menu_intent(self):

        menu_response = menu_crew.kickoff(
        inputs={
            "message": self.state.query,
            "user_id": self.state.user_id,
            "chat_history": self.state.chat_history
            }
        )
        menu_result_text = (
            str(menu_response.raw)
            if hasattr(menu_response, "raw")
            else str(menu_response)
        )
        DBService.update_chat_history(self.state.session_id, self.state.user_id, {"role": "assistant", "content": menu_result_text})
        self.state.response = menu_result_text
        self.state.last_menu_response = menu_result_text
        return f"Menu intent '{menu_result_text}' has been processed."
    
    @listen("order_crew")
    def handle_order_intent(self):
        print(f"chat history in order crew: {self.state.chat_history}")
        intent = self.state.intent_name
        cache_key = f"order:{self.state.restaurant_id}_{self.state.user_id}"
        if intent == "cancel_order":
            cached_order = redis_client.get(cache_key)
            if not cached_order:
                self.state.response = "No order details found in cache. Unable to process cancellation."
                return "No order details found in cache. Unable to process cancellation."
            
            order_details = json.loads(cached_order)
            receipt_number = str(order_details.get('recipet_no'))
            if receipt_number:
                last_order_details = OrderService.get_order_details_by_recipt_id(receipt_number)
                OrderService.update_order(order_id=str(last_order_details['_id']), updated_order={"status": "cancelled"})
                self.state.response = f"Order with receipt number {receipt_number} has been cancelled."
                redis_client.delete(cache_key)
                return "Order cancellation has been processed."
            self.state.response = f"No receipt number found for the order. Unable to process cancellation."
            return "No receipt number found for the order. Unable to process cancellation."
        
        order_response = order_crew.kickoff(inputs= {"input": self.state.query, "restaurant_id": self.state.restaurant_id, "user_id": self.state.user_id, "chat_history": self.state.chat_history, "last_menu_response": self.state.last_menu_response})
        order_result_text = str(order_response.raw) if hasattr(order_response, 'raw') else str(order_response)
        print(f"Order crew response: {order_result_text}")
        
        response_data = json.loads(order_result_text)
        print(f"Extracted response from order crew: {response_data}")
        
        status = response_data.get("status")
        
        # ========== SUCCESS CASE ==========
        if status == "success":
            items = response_data.get("items", [])
            print(f"Extracted items from order crew response: {items}")
            response = {
                "items": [],
                "total_amount": 0,
                "status": "success"
            }

            for item in items:
                item_total = int(item["quantity"]) * int(item.get("price", 0))
                response["items"].append({
                    **item,
                    "item_total": item_total
                })
                response["total_amount"] += item_total
            
            print(f"Final order details with total amount calculated: {response}")
            self.state.response = response
            
        # ========== PARTIAL SUCCESS CASE ==========
        elif status == "partial_success":
            confirmed_items = response_data.get("items", [])
            unavailable_items = response_data.get("unavailable_items", [])
            
            response = {
                "items": [],
                "total_amount": 0,
                "status": "partial_success",
                "unavailable_items": []
            }
            
            # Process confirmed items
            for item in confirmed_items:
                item_total = int(item["quantity"]) * int(item.get("price", 0))
                response["items"].append({
                    **item,
                    "item_total": item_total
                })
                response["total_amount"] += item_total
            
            # Add unavailable items info
            for item in unavailable_items:
                response["unavailable_items"].append({
                    "food_item_name": item.get("food_item_name"),
                    "quantity": item.get("quantity"),
                    "reason": item.get("reason", "unavailable")
                })
            
            print(f"Partial success order with {len(confirmed_items)} confirmed and {len(unavailable_items)} unavailable items")
            self.state.response = response
            
        # ========== CLARIFICATION REQUIRED CASE ==========
        elif status == "clarification_required":
            response = {
                "status": "clarification_required",
                "requires_clarification": response_data.get("requires_clarification", True),
                "clarification_type": response_data.get("clarification_type"),
                "clarification_question": response_data.get("clarification_question"),
                "clarification_items": response_data.get("clarification_items", []),
                "available_options": response_data.get("available_options", [])
            }
            
            print(f"Clarification required: {response['clarification_question']}")
            self.state.response = response
            return response
            
        # ========== ITEM NOT FOUND CASE ==========
        elif status == "item_not_found":
            unavailable_items = response_data.get("unavailable_items", [])
            response = {
                "status": "item_not_found",
                "message": f"The following items were not found in our menu",
                "unavailable_items": unavailable_items,
                "suggestion": "Please check the menu or try ordering different items."
            }
            
            print(f"Items not found: {[item.get('food_item_name') for item in unavailable_items]}")
            self.state.response = response
            return response
            
        # ========== UNKNOWN STATUS CASE ==========
        else:
            self.state.response = {
                "status": status,
                "message": f"Unable to process order with status: {status}",
                "details": response_data
            }
            print(f"Order crew returned status: {status}")
            return self.state.response

        # ========== ORDER MODIFICATION LOGIC (Only for successful orders) ==========
        if intent in ["order_modification", "modify_cart"] and status == "success":
            order_details_json = redis_client.get(cache_key)
            if not order_details_json:
                self.state.response = "No previous order found. Unable to process order modification."
                return "No previous order found. Unable to process order modification."
                
            order_details_json = json.loads(order_details_json)
            print(f"Order details from cache for modification: {order_details_json}")
            order_status = str(order_details_json.get('order_status'))
            is_pending_order = (order_status == "pending")
            
            if not is_pending_order:
                receipt_number = str(order_details_json.get('recipet_no'))
                if not receipt_number:
                    self.state.response = "No receipt number found in cache. Unable to process order modification."
                    return "No receipt number found in cache. Unable to process order modification."
                last_order_details = OrderService.get_order_details_by_recipt_id(receipt_number)
                print(f"Last order details retrieved from DB using receipt number {receipt_number}: {last_order_details}")
            else:
                print("Order is still pending.")
                last_order_details = order_details_json
                print(f"Last order details retrieved from cache for pending order: {last_order_details}")
            
            previous_items = (
                last_order_details.get("items", [])
                if last_order_details
                else []
            )
            print(f"Previous items from last order details: {previous_items}")
            if not last_order_details:
                self.state.response = "No order found with the provided receipt number. Unable to process order modification."
                return "No order found with the provided receipt number. Unable to process order modification."
            
            new_items = response["items"]
            merged_items = {
                item["food_item_name"]: item.copy()
                for item in previous_items
            }
            for new_item in new_items:
                food_name = new_item["food_item_name"]
                if food_name in merged_items:
                    merged_items[food_name]["quantity"] += new_item["quantity"]
                    merged_items[food_name]["item_total"] += new_item["item_total"]
                else:
                    merged_items[food_name] = new_item.copy()
            updated_items = list(merged_items.values())
            updated_total_amount = sum(
                item["item_total"]
                for item in updated_items
            )
            print(f"Updated items after merging: {updated_items} and updated total amount: {updated_total_amount}")
            
            # Handle pending orders by updating cache, confirmed orders by updating DB
            if is_pending_order:
                # Update cache for pending orders
                response["items"] = updated_items
                response["total_amount"] = updated_total_amount
                response["order_status"] = "pending"
                redis_client.set(cache_key, json.dumps(response), ex=3600)
                self.state.response = response
                return "Order modification has been processed."
            else:
                # Update MongoDB for confirmed orders
                updated_order = OrderService.update_order(order_id=str(last_order_details['_id']), updated_order={"status": "modified", "items": updated_items, "total_amount": updated_total_amount})
                print(f"Updated order details after modification: {updated_order}")
                self.state.response = serialize_bson(updated_order)
                return "Order modification has been processed."
        
        # ========== CACHE SUCCESSFUL ORDER FOR LATER CONFIRMATION ==========
        if status == "success":
            redis_client.delete(cache_key)
            response["order_status"] = "pending"
            redis_client.set(cache_key, json.dumps(response), ex=3600)  # Cache order details for 1 hour
            print(f"Order cached successfully for user {self.state.user_id}")
            return "Order details prepared. Ready for confirmation."
        
        # For non-success cases, return the response as-is without caching
        return f"Order processing completed with status: {status}"
    
    @listen("order_modification/cancellation")
    def handle_order_modification_cancellation(self):
        intent = self.state.intent_name
        if intent == "order_modification":
            OrderService.update_order(order_id=self.state.restaurant_id, updated_order={"status": "modified"})
            return "Order modification has been processed."
        elif intent == "cancel_order":
            OrderService.update_order(order_id=self.state.restaurant_id, updated_order={"status": "cancelled"})
            return "Order cancellation has been processed."


    @listen("place_order")
    def handle_place_order(self):
        cache_key = f"order:{self.state.restaurant_id}_{self.state.user_id}"
        order_details_json = redis_client.get(cache_key)
        print(f"Order details from cache for placing order: {order_details_json}")
        if not order_details_json:
            return "No order details found in cache. Please place your order again."
        
        order_details = json.loads(order_details_json)
        recipet_no = str(uuid.uuid4())
        new_order = ordersModel(
            user_id=self.state.user_id,
            restaurant_id=self.state.restaurant_id,
            items=order_details["items"],
            total_amount=order_details["total_amount"],
            status="confirmed",
            recipet_no=recipet_no
        )
        orders_collection.insert_one(new_order.dict())
        cached_data = redis_client.get(cache_key)

        if cached_data:
            response = json.loads(cached_data)
            response["order_status"] = "confirmed"
            response["recipet_no"] = recipet_no
            redis_client.set(
                cache_key,
                json.dumps(response),
                ex=3600
            )
            cached_data = redis_client.get(cache_key)
            print(cached_data)
        self.state.response = f"Order has been placed successfully with details: {new_order.dict()}"
        return f"Order has been placed successfully with details: {new_order.dict()}"

    @listen("default_crew")
    def handle_default_intent(self):
        self.state.response = "Sorry, I couldn't understand your request. Could you please rephrase it?"
        return self.state.response
