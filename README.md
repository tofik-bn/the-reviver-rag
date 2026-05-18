# The Reviver — AI that thinks like Al‑Ghazali

> *“Doubt is not removed except by certainty.”*  
> — Abu Hamid Al‑Ghazali (1058–1111)

**The Reviver** is a sophisticated RAG (Retrieval‑Augmented Generation) system that embodies the documented thought of one of history’s most complex minds: the theologian, philosopher, jurist, and Sufi mystic Al‑Ghazali.

Unlike a simple chatbot, this system respects the full arc of his intellectual journey — from the confident academic who wrote *Tahafut al‑Falasifa* (The Incoherence of the Philosophers) to the transformed wanderer who authored *Ihya Ulum al‑Din* (The Revival of Religious Sciences). It navigates his three distinct voices (philosophical, spiritual, legal), maps modern questions to original Arabic concepts (`Aql`, `Qalb`, `Kashf`, `Yaqin`), and delivers every answer with a **certainty gradient** (Level 1–4) and a verifiable source tag.

**The six hard problems we solved:**

1. **Conceptual translation** – No collapsing of Arabic terms into vague English equivalents.  
2. **Three voices** – The system knows when to speak as a philosopher, a mystic, or a jurist.  
3. **Historical context** – Every argument surfaces the intellectual opponent (Ibn Sina, Al‑Farabi, etc.).  
4. **Modern bridging** – Questions about AI consciousness, scientific limits, or ethical uncertainty are mapped to his documented framework (clearly labelled as inference).  
5. **Certainty gradient** – Direct position → reasoned inference → principled extrapolation → beyond reason.  
6. **Transformation arc** – The early and late Ghazali differ meaningfully; the system flags and explains the change.

**Technical stack (100% free tools):**

- Python 3.12 + FastAPI backend  
- FAISS vector database (6,978 semantic chunks from primary texts)  
- Groq API (free LLM for generation)  
- Streamlit‑like custom frontend (HTML/CSS/JS) with gold Arabic design  
- Local embeddings with `fastembed` (BAAI/bge‑small‑en‑v1.5)

**Primary sources ingested:**

- *Ihya Ulum al‑Din* (Revival of Religious Sciences)  
- *Kimiya yi Sa’adat* (The Alchemy of Happiness)  
- *Tahafut al‑Falasifa* (The Incoherence of the Philosophers)  

**Live demo & code:**  
[Link to your deployed frontend] · [GitHub repository]

Built as a portfolio project to demonstrate end‑to‑end AI engineering: ETL, embeddings, vector search, LLM integration, API development, and a complete UI.
