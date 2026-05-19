from fastapi import FastAPI
from app.routes import router
app = FastAPI()


app.include_router(router, prefix="/api")

#check endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chat API!"}
