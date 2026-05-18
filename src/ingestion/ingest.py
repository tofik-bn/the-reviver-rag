# src/ingestion/ingest.py
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastembed import TextEmbedding
import faiss
import numpy as np
import pickle

def extract_text_from_txt(txt_path):
    """Read entire text file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2 (for text-based PDFs)."""
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            full_text += f"\n--- Page {page_num} ---\n{text}"
    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_vector_store(data_folder="data/raw", max_files=1):
    print("Loading embedding model (fastembed)...")
    model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
    
    all_chunks = []
    all_metadata = []
    
    # Get all .txt and .pdf files
    files = []
    for ext in ['.txt', '.pdf']:
        files.extend([f for f in os.listdir(data_folder) if f.endswith(ext)])
    
    print(f"Found {len(files)} files: {files[:max_files]}")
    
    for filename in files[:max_files]:
        filepath = os.path.join(data_folder, filename)
        print(f"Processing: {filename}")
        
        # Choose extraction method based on extension
        if filename.endswith('.txt'):
            full_text = extract_text_from_txt(filepath)
        else:
            full_text = extract_text_from_pdf(filepath)
        
        if not full_text or len(full_text.strip()) < 100:
            print(f"Warning: {filename} has very little text. Skipping.")
            continue
        
        chunks = chunk_text(full_text, chunk_size=500, overlap=50)
        print(f"  Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": filename,
                "chunk_index": i,
                "text_preview": chunk[:100]
            })
    
    print(f"Total chunks created: {len(all_chunks)}")
    if len(all_chunks) == 0:
        print("No chunks created. Check your files.")
        return None, None, None
    
    print("Creating embeddings (this may take a few minutes)...")
    embeddings_generator = model.embed(all_chunks)
    embeddings = list(embeddings_generator)
    embeddings = np.array(embeddings, dtype='float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs("storage", exist_ok=True)
    faiss.write_index(index, "storage/ghazali_index.faiss")
    with open("storage/chunks.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)
    
    print(f"Vector store saved to storage/ folder. Index size: {index.ntotal} vectors.")
    return index, all_chunks, all_metadata

if __name__ == "__main__":
    build_vector_store(max_files=None)