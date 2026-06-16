from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
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
