# src/api.py
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your RAG query function (will be called inside /ask)
from src.query import ask_question

app = FastAPI(title="The Reviver - Al-Ghazali RAG API")

# Enable CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "The Reviver API is running. Send POST /ask with {'question': '...'}"}

@app.get("/debug")
def debug():
    """Check if the GROQ_API_KEY environment variable is set."""
    has_key = "GROQ_API_KEY" in os.environ
    preview = os.environ.get("GROQ_API_KEY", "")[:10] if has_key else None
    return {"has_groq_key": has_key, "key_preview": preview}

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        answer = ask_question(req.question)
        return {"answer": answer}
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return {"error": str(e), "trace": error_detail}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)