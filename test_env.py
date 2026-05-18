# test_env.py - Test that we can read the API key from .env
from dotenv import load_dotenv
import os

load_dotenv()  # This reads the .env file

api_key = os.getenv("GROQ_API_KEY")
print(f"API key loaded. First 10 characters: {api_key[:10]}...")
print(f"Length of key: {len(api_key)} characters")