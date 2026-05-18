FROM python:3.12-slim

# Install system dependency for FAISS (libomp)
RUN apt-get update && apt-get install -y libomp-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]