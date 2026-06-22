import asyncio
import shutil

from fastapi import APIRouter, File, UploadFile
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import os
import json
from app.crews.chatCrew import chat_crew
from app.crew import intent_crew, faq_crew, menu_crew, order_crew
import fitz # PyMuPDF for PDF processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.qdrantServices import QdrantService
from app.services.embeddingService import EmbeddingService
from qdrant_client.models import PointStruct
import uuid
from app.db.mongo import users_collection
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from enum import Enum
from app.crew import menu_parser_crew
import re
from app.flows import RestaurantAssistantFlow, SampleFlow
from fastapi.concurrency import run_in_threadpool


embedding_service = EmbeddingService()
qdrant_service = QdrantService()

router = APIRouter()
load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

@router.get("/health")
async def health_check():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    user_id: str
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    print(f"Received message from user {request.user_id}: {request.message}")
    response = intent_crew.kickoff({"input": request.message})
    result_text = str(response.raw) if hasattr(response, 'raw') else str(response)
    print("Response from intent crew:", result_text)
    try:
        data = json.loads(result_text)
        print("Parsed response:", data)
        return {"message": f"Intent classified: {data.get('intent', 'unknown')}", "intent": data}
    except json.JSONDecodeError:
        return {"message": f"Received message from user {request.user_id}: {request.message}", "raw_response": result_text}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), type_of_data: str = "menu", restaurant_id: str = "restaurant123"):
    print(f"Received file upload request: {file.filename}, {file}")
    text =""
    os.makedirs(
            "uploads",
            exist_ok=True
        )
    saved_file_path = os.path.join(
        "uploads",
        file.filename
    )
    with open(
            saved_file_path,
            "wb"
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )
    pdf = fitz.open(saved_file_path)
    for page in pdf:
        text += page.get_text()
    pdf.close()
    collection_name = f"{type_of_data}_collection"
    isExist = qdrant_service.collection_exists(collection_name)
    if not isExist:
        qdrant_service.create_collection(collection_name, vector_size=384)
    points = []   
    if type_of_data != "menu":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        embeddings = embedding_service.get_embeddings(chunks)
        for index, embedding in enumerate(embeddings):
            point = PointStruct(
                    id= str(uuid.uuid4()),  # Unique ID for each point 
                    vector=embedding,  # The embedding vector for the chunk
                    payload={
                        "user_id": "user123",  # Example user ID, replace with actual user ID if available
                        "text": chunks[index],  # The original text chunk as payload
                        "category": type_of_data,  # The category of the data
                        "restaurant_id": restaurant_id  # Example restaurant ID, replace with actual restaurant ID if available
                    }
                )
            
            points.append(point)

    if(type_of_data == "menu"):
        menu_parser_response = menu_parser_crew.kickoff({"menu_text": text})
        if not menu_parser_response:
            return {"message": "No menu items found in the uploaded document."}

        response = (
        menu_parser_response.raw
        if hasattr(menu_parser_response, "raw")
        else str(menu_parser_response)
        )

        cleaned_response = re.sub(
        r"```json|```",
        "",
        response
        ).strip()

        # Convert to Python list
        menu_items = json.loads(cleaned_response)
        for item in menu_items:

            text = f"""
            {item.get('food_item_name')} is a
            {item.get('food_type')} food item.
            Quantity: {item.get('quantity')}
            Price: ₹{item.get('price')}
            Category: {item.get('category')}
            """
            embedding = embedding_service.get_embedding(text)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "food_item_name": item.get("food_item_name"),
                    "quantity": item.get("quantity"),
                    "price": item.get("price"),
                    "category": item.get("category"),
                    "food_type": item.get("food_type"),
                    "restaurant_id": restaurant_id
                }
            )

            points.append(point)

    qdrant_service.upsert(collection_name, points)
    return {"message": "File upload endpoint - to be implemented", "type_of_data": type_of_data, "text": text}

    # print(f"Extracted text from PDF: {text}...")  # Print the first 500 characters for verification
    
    # print("Response from Menu Parser Crew:", menu_parser_response)
    #Step--2 Split text into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    # langchain's split_text expects a single string, not a list.
    # Pass `text` directly so both langchain and the fallback work.
    # chunks = text_splitter.split_text(text)

    #Step--3 Create vector embeddings for chunks using OpenAI API
    # embeddings = embedding_service.get_embeddings(chunks)
    # print(f"Generated embeddings for {len(chunks)} chunks.", type_of_data)

    

    # Remove markdown if exists
    


    # print(f"Generated {len(points)} points for Qdrant upload.", type_of_data)
    # Step--4 Create a collection in Qdrant oif not exist
    # Store the chunks in the Qdrant vector database

    # for index, embedding in enumerate(embeddings):
    #     points = [
    #         PointStruct(
    #             id= str(uuid.uuid4()),  # Unique ID for each point 
    #             vector=embedding,  # The embedding vector for the chunk
    #             payload={
    #                 "user_id": "user123",  # Example user ID, replace with actual user ID if available
    #                 "text": chunks[index],  # The original text chunk as payload
    #                 "category": type_of_data,  # The category of the data
    #                 "restaurant_id": restaurant_id  # Example restaurant ID, replace with actual restaurant ID if available
    #             }
    #         )
    #     ]

    # stored_data = qdrant_service.get_stored_data(collection_name)
    # print(f"Retrieved {len(stored_data)} records from Qdrant collection '{collection_name}' for verification.")

    # print(f"Extracted text from PDF: {text[:500]}...")  # Print the first 500 characters for verification
    

@router.get("/search")
async def search(collection_name: str, query: str):
    query_vector = embedding_service.get_embedding(query)  # Get embedding for the query
    top_k = 5  # Number of top results to retrieve
    results = qdrant_service.search(collection_name, query_vector, top_k)
    return {"results": results}

@router.delete("/collection/{collection_name}")
async def delete_collection(collection_name: str):
    qdrant_service.delete_collection(collection_name)
    return {"message": f"Collection '{collection_name}' deleted."}

@router.get("/collections")
async def list_collections():
    collections = qdrant_service.list_collections()
    return {"collections": collections}

@router.get("/stored_data/{collection_name}")
async def get_stored_data(collection_name: str):
    stored_data = qdrant_service.get_stored_data(collection_name)
    return {"stored_data": stored_data}

class UserMessage(BaseModel):
    user_id: str
    message: str
    session_id: str
    restaurant_id: str

def sample_execute_flow(query: str):
    print("Before kickoff")
    sample_flow = SampleFlow()
    print("Middle kickoff")
    # response = await run_in_threadpool(
    #     sample_flow.kickoff,
    #     inputs={"query": "What is the weather today?"}
    # )
    response =  sample_flow.kickoff(
        inputs={
            "query": query
        }
    )
    print("After kickoff")
    print("Sample Flow response:", response)
    return {"message": response}


def restaurant_execute_flow(message: str, user_id:str, session_id: int, restaurant_id: str):
    try:
        flow = RestaurantAssistantFlow()
        inputs={
                "query": message,
                "user_id": user_id,
                "session_id": session_id,
                "restaurant_id": restaurant_id
        }

        response = flow.kickoff(inputs=inputs)
        print(f"Flow response: {response}")
        intent = getattr(flow.state, 'intent_name', None)
        message = getattr(flow.state, 'response', None)
        print(f"Intent: {intent}, Message: {message}")
        if hasattr(message, "raw"):
            message = message.raw
        if not isinstance(message, (str, dict, list, int, float, bool, type(None))):
            try:
                message = jsonable_encoder(message)
            except Exception:
                message = str(message)
        return {

            "intent": intent,
            "message": message
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("FLOW ERROR:", str(e))
        print("TRACEBACK:\n", error_details)
        return {
            "intent": None,
            "message": f"Error processing flow: {str(e)}",
            "error": str(e),
            "traceback": error_details
        }


@router.post("/user/message")
async def handle_user_message(request: UserMessage):
    try:
        print(
            f"Received message from user "
            f"{request.user_id}: {request.message}"
        )
        response = await run_in_threadpool(
            restaurant_execute_flow,
            request.message,
            request.user_id,
            request.session_id,
            request.restaurant_id
        )

        return {
            "status_code": 200,
            "status": "success",
            "message": "Message processed successfully",
            "query": request.message,
            "data": {
                "intent": response["intent"],
                "message": response['message']
            }
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("FLOW ERROR:", str(e))
        print("TRACEBACK:\n", error_details)
        return {
            "status_code": 500,
            "status": "error",
            "message": str(e),
            "data": {
                "error": str(e),
                "traceback": error_details
            }
        }

class dietType(str, Enum):
    veg = "veg"
    non_veg = "non_veg"
    vegan = "vegan"
    eggitarian = "eggitarian"
class spiceLevel(str, Enum):
    mild = "mild"
    medium = "medium"
    hot = "hot"
class Preferences(BaseModel):
    diet_type: dietType = Field(description="Diet type of the user")
    favorite_cuisines: Optional[list] = Field(description="List of favorite cuisines")
    favorite_foods: list = Field(description="List of favorite foods")
    disliked_foods: list = Field(description="List of disliked foods")
    spice_level: spiceLevel = Field(description="Preferred spice level")
    preferred_meal_type: Optional[str] = Field(description="Preferred meal type", max_length=20, min_length=3)
    preferred_beverages: Optional[list] = Field(description="List of preferred beverages")
    favorite_restaurant_categories: Optional[list] = Field(description="List of favorite restaurant categories")
    budget_preference: Optional[dict] = Field(description="Budget preference")
class UserPreferences(BaseModel):
    name: str = Field(description="Name of the user", max_length=18, min_length=3)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(description="Age of the user", ge=12, le=120)
    phone_number: int = Field(description="Phone number of the user")
    preferences: Preferences = Field(description="User's food preferences")

@router.post("/user/preferences")
async def handle_user_preferences(request: UserPreferences):
    print(f"Received user preferences from user {request.name}: {request.preferences}")
    # Here you would process the user preferences and update the user's profile in your database
    print(f"Storing user preferences in database for user {request.preferences}")
    user = users_collection.insert_one({
        "name": request.name or "Unknown",
        "email": request.email,
        "age": request.age,
        "phone_number": request.phone_number,
        "diet_type": request.preferences.diet_type.value if request.preferences.diet_type else "", # e.g., # veg | non_veg | vegan | eggitarian
        "favorite_cuisines": request.preferences.favorite_cuisines, # e.g., ["Italian", "Chinese", "Indian", "south_indian"]
        "favorite_foods": request.preferences.favorite_foods, # e.g., ["Pizza", "Sushi", "Pasta", "Biryani"]
        "disliked_foods": request.preferences.disliked_foods, # e.g., ["Broccoli", "Mushrooms"]
        "spice_level": request.preferences.spice_level.value if request.preferences.spice_level else "", # e.g., "mild", "medium", "hot"
        "preferred_meal_type": request.preferences.preferred_meal_type, # e.g., "breakfast", "lunch", "dinner", "snacks"
        "preferred_beverages": request.preferences.preferred_beverages, # e.g., ["Coke", "Pepsi", "Juice"]
        "favorite_restaurant_categories": request.preferences.favorite_restaurant_categories, # e.g., ["fast_food", "fine_dining", "cafe", "buffet"]
        "budget_preference": request.preferences.budget_preference # e.g., {"min": 100, "max": 500}
    })
    return {"message": "User preferences updated successfully.", "user_id": str(user.inserted_id)}