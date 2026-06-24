from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from app.db.qdrent import setup_qdrant
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Restaurant AI Assistant...")
    setup_qdrant()  # Creates collection + indexes at startup

    yield

    print("Stopping Restaurant AI Assistant...")


app = FastAPI(
    title="Restaurant AI Assistant",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://restaurant-ai-assisstant.onrender.com",
        "https://restaurant-ai-assisstant.onrenders.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

#check endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chat API!"}
