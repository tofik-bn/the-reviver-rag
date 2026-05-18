# src/api.py
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.query import ask_question

app = FastAPI(title="The Reviver - Al-Ghazali RAG API")

# Allow your frontend to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. In production, restrict to your frontend domain.
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "The Reviver API is running. Send POST /ask with {'question': '...'}"}

@app.post("/ask")
def ask(req: QuestionRequest):
    answer = ask_question(req.question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)