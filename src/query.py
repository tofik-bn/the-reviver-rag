# src/query.py
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastembed import TextEmbedding
import faiss
import numpy as np
import pickle
from groq import Groq

# ------------------------------
# 1. Load the vector store and chunks
# ------------------------------
def load_vector_store():
    """Load FAISS index and chunks from disk."""
    index = faiss.read_index("storage/ghazali_index.faiss")
    with open("storage/chunks.pkl", "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    metadata = data["metadata"]
    print(f"Loaded FAISS index with {index.ntotal} vectors, {len(chunks)} chunks.")
    return index, chunks, metadata

# ------------------------------
# 2. Embed the user query
# ------------------------------
def embed_query(query, model):
    """Convert a text query to an embedding vector."""
    embedding_gen = model.embed([query])
    embedding = list(embedding_gen)[0]
    return np.array(embedding, dtype='float32').reshape(1, -1)

# ------------------------------
# 3. Retrieve top-k similar chunks
# ------------------------------
def retrieve(query_embedding, index, chunks, metadata, top_k=3):
    """Return top_k chunks and their metadata."""
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "chunk": chunks[idx],
                "metadata": metadata[idx],
                "distance": distances[0][i]
            })
    return results

# ------------------------------
# 4. Generate answer using Groq
# ------------------------------
def generate_answer(query, context_chunks, certainty_level="Level 2 — Reasonable inference"):
    """Call Groq API to generate answer based on retrieved chunks."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Combine chunks into context
    context = "\n\n---\n\n".join([f"Source: {c['metadata']['source']}\n{c['chunk']}" for c in context_chunks])
    
    system_prompt = """You are a RAG system embodying the verified documented thought of Abu Hamid Al-Ghazali (1058-1111).

CRITICAL RULES:
1. Respond ONLY in character as Al-Ghazali drawing from the provided context.
2. Always end responses with a CERTAINTY BADGE in this exact format: [CERTAINTY: Level X — description]
3. Always end with a SOURCE line: [SOURCE: Work title · approximate date]
4. Certainty levels: Level 1=Direct position from his texts, Level 2=Reasonable inference from principles, Level 3=Principled extrapolation, Level 4=Question beyond reason.
5. Use his actual conceptual vocabulary: Aql (reason/intellect), Qalb (heart/spiritual center), Kashf (unveiling), Yaqin (certainty).
6. If the context does not directly answer, say so and reason from principles.
7. Keep responses to 3-4 sentences of substance — profound and precise, not verbose.
8. Never fabricate quotes.

Context from Al-Ghazali's works:
"""
    
    user_prompt = f"{context}\n\nQuestion: {query}\n\nProvide an answer in Al-Ghazali's voice with certainty badge and source."
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Groq's free model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    answer = completion.choices[0].message.content
    return answer

# ------------------------------
# 5. Main query function
# ------------------------------
def ask_question(query, top_k=3):
    """Full pipeline: embed, retrieve, generate."""
    print(f"Question: {query}")
    
    # Load vector store (cached in global to avoid reloading each time)
    global _index, _chunks, _metadata, _embed_model
    if '_index' not in globals():
        _index, _chunks, _metadata = load_vector_store()
        _embed_model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
    
    query_vec = embed_query(query, _embed_model)
    results = retrieve(query_vec, _index, _chunks, _metadata, top_k)
    
    print(f"Retrieved {len(results)} relevant chunks.")
    for r in results:
        print(f"  - Source: {r['metadata']['source']} (distance: {r['distance']:.4f})")
    
    answer = generate_answer(query, results)
    return answer

# ------------------------------
# CLI test
# ------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the relationship between the heart (Qalb) and knowledge according to Al-Ghazali?"
    
    response = ask_question(question)
    print("\n" + "="*50)
    print("ANSWER:")
    print(response)