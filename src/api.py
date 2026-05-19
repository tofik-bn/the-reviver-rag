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

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        from src.query import ask_question
        answer = ask_question(req.question)
        return {"answer": answer}
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return {"error": str(e), "trace": error_detail}, 500
        @app.get("/debug")
def debug():
    import os
    return {
        "has_groq_key": "GROQ_API_KEY" in os.environ,
        "key_preview": os.environ.get("GROQ_API_KEY", "")[:10] if "GROQ_API_KEY" in os.environ else None
    }