import shutil

from fastapi import APIRouter, File, UploadFile
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
from app.crews.chatCrew import chat_crew
from app.crew import intent_crew
import fitz # PyMuPDF for PDF processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.qdrent import client as qdrant_client
from app.services.qdrantServices import QdrantService
from app.services.embeddingService import EmbeddingService

embedding_service = EmbeddingService()
qdrant_service = QdrantService(qdrant_client)

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
async def upload_file(file: UploadFile = File(...), type_of_data: str = "menu"):
    print(f"Received file upload request: {file.filename}, {file}")

    #Step--1 Extract text from PDF
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

    #Step--2 Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    # langchain's split_text expects a single string, not a list.
    # Pass `text` directly so both langchain and the fallback work.
    chunks = text_splitter.split_text(text)

    #Step--3 Create vector embeddings for chunks using OpenAI API
    embeddings = embedding_service.get_embeddings(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks.", type_of_data)


    # Store the chunks in the Qdrant vector database


    print(f"Extracted text from PDF: {text[:500]}...")  # Print the first 500 characters for verification
    return {"message": "File upload endpoint - to be implemented", "type_of_data": type_of_data, "text": text, "chunks": chunks, "embeddings": embeddings}