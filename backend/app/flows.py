from urllib import response
import uuid
import re
from qdrant_client.models import PointStruct
from openai import chat
from app.crew import intent_crew, faq_crew, menu_crew, order_crew, menu_parser_crew
from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel, Field
import json
from app.db.redis import redis_client
from app.db.mongo import users_collection, chat_history_collection, orders_collection
from app.services.dbServices import DBService, OrderService
from app.services.qdrantServices import QdrantService
from app.services.embeddingService import EmbeddingService

qdrant_service = QdrantService()
embedding_service = EmbeddingService()

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
    response: str = ""
    user_id: str = ""
    restaurant_id: str = ""
    session_id: str = ""
    message: list[messageModel] = []
    intent_from_db: bool = False
    is_menu_search: bool = False
    is_recommendation: bool = False
    is_menu_price: bool = False

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
        chat_message_is_exist = DBService.get_chat_history(self.state.session_id, self.state.user_id)
        if chat_message_is_exist:
            DBService.update_chat_history(self.state.session_id, self.state.user_id, {"role": "user", "content": user_query})
        else:
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
        
        
        #Method 2
        food_recommendations_key_words = [
                "recommend",
                "suggest",
                "best",
                "popular",
                "special",
                "famous",
                "top",
                "signature",
                "customer favorite",
                "most ordered",
                "craving",
                "feel like eating",
                "want something",
                "in mood for",
                "feel like having",
                "spicy",
                "sweet",
                "crispy",
                "crunchy",
                "cheesy",
                "healthy",
                "light",
                "heavy",
                "hot",
                "cold",
                "most ordered",
                "best seller",
                "top selling",
                "popular",
                "trending",
                "customer favorite"
        ]
        
        menu_search_key_words = [
            "menu",
            "food menu",
            "restaurant menu",
            "full menu",
            "complete menu",
            "menu card",
            "food items",
            "dishes",
            "cuisine",
            "available food",
            "what do you have",
            "what are the options",
            "show me the menu",
            "what can I order",
            "what's on the menu",
            "foods",
            "items",
            "varieties",
            "show menu",
            "see menu",
            "browse menu",
            "menu options",
            "menu list",
            "menu details",
            "menu information",
            "menu offerings",
            "menu choices",
            "menu selection",
            "menu availability",
            "menu price",

        ]
        
        menu_price_query_keywords = [
            "rate",
            "pricing",
            "charges",
            "amount",
            "fee",
            "what is the price",
            "what's the price",
            "what is the cost",
            "what's the cost",
            "how much is",
            "how much does it cost",
            "price list",
            "menu prices",
            "food prices",
            "show prices",
            "show menu with prices",
            "total cost",
            "bill amount",
            "price",
            "cost",
            "how much",
            "what's the price",
            "price of",
            "cost of",
            "how much is",
            "pricing",
            "menu price",
            "food price",
            "dish price",
            "cuisine price",
            "item price",
            "price details",
            "price information",
            "price list",
        ]
        
        order_food_keywords = [
            "place order",
            "order",
            "buy",
            "purchase",
            "want",
            "need",
            "would like",
            "i would like",
            "i'd like",
            "give me",
            "get me",
            "bring me",
            "send me",
            "can i get",
            "may i get",
            "let me have",
            "i'll have",
            "i will have",
            "serve me",
            "prepare",

        ]

        modify_order_keywords =[
            "modify order",
            "change order",
            "update order",
            "edit order",
            "adjust order",
            "alter order",
            "revise order",
            "remove item",
            "add item",
            "change item quantity",
            "update item quantity",
            "increase item quantity",
            "decrease item quantity",
            "i want to change my order",
            "i want to modify my order",
            "i want to update my order",
            "i want to edit my order",
            "i want to adjust my order",
            "i want to alter my order",
            "i want to revise my order",
        ]

        cancel_order_keywords = [
            "cancel order",
            "i want to cancel my order",
            "i need to cancel my order",
            "i'd like to cancel my order",
            "please cancel my order",
            "can i cancel my order",
            "may i cancel my order",
            "let me cancel my order",
            "i will cancel my order",
            "i'll cancel my order",
            "cancel my order",
        ]

        faq_keywords = [
            "what are your timings",
            "when do you open",
            "when do you close",
            "what is your return policy",
            "what is your cancellation policy",
            "do you have parking",
            "is there parking available",
            "what are the parking options",
            "how can I contact you",
            "what is your contact information",
            "where are you located",
            "what is your location",
            "do you have outdoor seating",
            "do you have indoor seating",
            "are reservations required",
            "do you take reservations",
            "what is your reservation policy",
            "parking information",
            "contact information",
            "parking"
        ]
        
        greeting_keywords = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "greetings",
            "what's up",
            "how are you",
            "how's it going",
            "nice to meet you",
            "pleased to meet you",
            "thank you",
            "thanks",
            "thank you very much",
            "thanks a lot",
            "much appreciated",
            "thank you so much",
        ]

        query = user_query.lower().strip()

        scores = {
            "greeting": 0,
            "faq": 0,
            "cancel_order": 0,
            "order_food": 0,
            "order_modification": 0,
            "menu_search": 0,
            "menu_price_query": 0,
            "menu_recommendation": 0
        }


        def calculate_score(query: str, keywords: list):
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1

            return score

        scores["greeting"] = calculate_score(
            query,
            greeting_keywords
        )

        scores["faq"] = calculate_score(
            query,
            faq_keywords
        )

        scores["cancel_order"] = calculate_score(
            query,
            cancel_order_keywords
        )

        scores["order_modification"] = calculate_score(
            query,
            modify_order_keywords
        )

        scores["order_food"] = calculate_score(
            query,
            order_food_keywords
        )

        recommendation_score = calculate_score(
            query,
            food_recommendations_key_words
        )

        menu_search_score = calculate_score(
            query,
            menu_search_key_words
        )

        menu_price_score = calculate_score(
            query,
            menu_price_query_keywords
        )

        if recommendation_score > 0:
            scores["menu_recommendation"] += (
                recommendation_score * 2
            )

        if menu_search_score > 0:
            scores["menu_search"] += (
                menu_search_score
            )

        if menu_price_score > 0:
            scores["menu_price_query"] += (
                menu_price_score * 2
            )
        quantity_pattern = re.search(
            r"\b\d+\s+[a-zA-Z]+\b",
            query
        )

        if quantity_pattern:
            scores["order_food"] += 5

        strong_order_phrases = [
            "i want",
            "give me",
            "get me",
            "can i get",
            "i would like",
            "i'd like",
            "i'll have",
            "order"
        ]

        for phrase in strong_order_phrases:
            if phrase in query:
                scores["order_food"] += 5

        priority_weights = {
            "order_food": 5,
            "order_modification": 4,
            "cancel_order": 4,
            "menu_price_query": 3,
            "menu_recommendation": 2,
            "menu_search": 1,
            "faq": 1,
            "greeting": 0
        }

        for intent, weight in priority_weights.items():
            if scores[intent] > 0:
                scores[intent] += weight

        max_score = max(scores.values())
        if max_score == 0:
            self.state.intent_name = "unknown"
        else:
            self.state.intent_name = max(
                scores,
                key=scores.get
            )

        print("Intent Scores:", scores)
        print("Detected Intent:", self.state.intent_name)
        
        if(self.state.intent_name):
            print(f"Intent classified based on keyword matching: {self.state.intent_name}")
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

        # Method 3

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
        elif intent == "confirm_order":
            return "place_order"
        # elif intent == "menu_parser":
        #     return menu_parser_crew
        else:
            return "default_crew"

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
            "user_id": self.state.user_id
            }
        )
        menu_result_text = (
            str(menu_response.raw)
            if hasattr(menu_response, "raw")
            else str(menu_response)
        )
        DBService.update_chat_history(self.state.session_id, self.state.user_id, {"role": "assistant", "content": menu_result_text})
        self.state.response = menu_result_text
        return f"Menu intent '{menu_result_text}' has been processed."
    
    @listen("order_crew")
    def handle_order_intent(self):
        intent = self.state.intent_name
        cache_key = f"order:{self.state.restaurant_id}_{self.state.user_id}"
        if intent == "cancel_order":
            order_details = json.loads(redis_client.get(cache_key))
            receipt_number = str(order_details.get('recipet_no'))
            if not order_details:
                self.state.response = "No order details found in cache. Unable to process cancellation."
                return "No order details found in cache. Unable to process cancellation."
            if(receipt_number):
                last_order_details = OrderService.get_order_details_by_recipt_id(receipt_number)
                OrderService.update_order(order_id=str(last_order_details['_id']), updated_order={"status": "cancelled"})
                self.state.response = f"Order with receipt number {receipt_number} has been cancelled."
                redis_client.delete(cache_key)
                return "Order cancellation has been processed."
            self.state.response = f"No receipt number found for the order. Unable to process cancellation."
            return "No receipt number found for the order. Unable to process cancellation."
        
        order_response = order_crew.kickoff({"input": self.state.query})
        order_result_text = str(order_response.raw) if hasattr(order_response, 'raw') else str(order_response)
        self.state.response = order_result_text
        items = json.loads(self.state.response)

        response = {
            "items": [],
            "total_amount": 0
        }

        for item in items:
            item_total = int(item["quantity"]) * int(item["price"])
            response["items"].append({
                **item,
                "item_total": item_total
            })
            response["total_amount"] += item_total

        self.state.response = response

        if intent in ["order_modification", "modify_cart"]:
            order_details_json = json.loads(redis_client.get(cache_key))
            receipt_number = str(order_details_json.get('recipet_no'))
            if not receipt_number:
                self.state.response = "No receipt number found in cache. Unable to process order modification."
                return "No receipt number found in cache. Unable to process order modification."
            last_order_details = OrderService.get_order_details_by_recipt_id(receipt_number)
            previous_items = last_order_details['items']
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

            OrderService.update_order(order_id=str(last_order_details['_id']), updated_order={"status": "modified", "items": updated_items, "total_amount": updated_total_amount})
            self.response = f"Order modification has been processed. Your updated order details are: {response}"
            return f"Order modification has been processed., Response: {response}"
        
        redis_client.delete(cache_key)
        response["order_status"] = "pending"
        redis_client.set(cache_key, json.dumps(response), ex=3600)  # Cache order details for 1 hour
        self.state.response = f"Order intent '{order_result_text}' has been processed with order details: {response}"
        return f"Order intent has been processed with order details: {response}"
    
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
