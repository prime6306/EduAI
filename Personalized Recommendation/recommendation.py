# ============================================================
# features.py  (Final Production-Ready Main File)
# ============================================================

import json
import numpy as np
import faiss
from datetime import datetime
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 1. MONGO DATABASE CONNECTION
# ------------------------------------------------------------
mongo_uri = "mongodb+srv://studentUser:1234@cluster0.zodzonx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
mongo_db = client["studentdb"]

# New table for Q/A storage
qa_collection = mongo_db["qa_history"]


# ------------------------------------------------------------
# 2. LOAD FAISS, CHUNKS, EMBEDDING MODEL, LLM
# ------------------------------------------------------------
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print("📦 Loading FAISS index, chunks, and AI models...")

INDEX_PATH = "book_faiss_index.index"
CHUNKS_PATH = "book_chunks.npy"

index = faiss.read_index(INDEX_PATH)
chunks = np.load(CHUNKS_PATH, allow_pickle=True)

# Embedding model
embed_model = SentenceTransformer("BAAI/bge-m3")

# LLM model (TinyLlama)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
generator = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

print(f"✅ Loaded FAISS with {index.ntotal} vectors")
print(f"✅ Loaded {len(chunks)} text chunks")
print("✅ Models loaded successfully!")


# ------------------------------------------------------------
# 3. FAISS PIPELINE FUNCTIONS
# ------------------------------------------------------------
def encode_query(query):
    instruction = "Represent the question for retrieving relevant text passages:"
    return embed_model.encode([[instruction, query]])


def search_faiss(query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    results = []

    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "text": chunks[idx]
        })

    return results


# ------------------------------------------------------------
# 4. LLM ANSWER GENERATION
# ------------------------------------------------------------
def generate_answer(question, context):
    prompt = f"""
You are an experienced university professor explaining technical topics clearly and accurately.

Your goal is to answer the question **only using information from the given book context**.
If the answer is not in the context, respond exactly with:
"The book does not discuss this topic."

---
### Context:
{context}

---
### Question:
{question}

---
### Instructions:
- Base your answer strictly on the provided context.
- Explain concepts in a clear, simple, and educational tone.
- Include relevant technical details where appropriate.
- Use an example or analogy only if it helps clarify the concept.
- Keep the explanation concise and focused.
- Do **not** add any external or fabricated information.

---
### Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.2,
        top_k=50,
        top_p=0.9,
    )[0]["generated_text"]

    return result.split("### Answer:")[-1].strip()


# ------------------------------------------------------------
# 5. SAVE RESULTS TO MONGO DB
# ------------------------------------------------------------
def save_to_mongo(question, answer, chunks_used):
    record = {
        "question": question,
        "answer": answer,
        "chunks_used": chunks_used,
        "timestamp": datetime.utcnow()
    }
    qa_collection.insert_one(record)
    print("💾 Saved to MongoDB!")


# ------------------------------------------------------------
# 6. MAIN FUNCTION — Called by Flask
# ------------------------------------------------------------
def run_model(user_question):
    """
    Full pipeline:
    1. Encode → 2. Retrieve chunks → 3. Generate answer → 4. Save to DB
    """

    # Step 1 — Encode
    query_embedding = encode_query(user_question)

    # Step 2 — Retrieve chunks
    retrieved_chunks = search_faiss(query_embedding, k=5)
    context = "\n\n".join([c["text"] for c in retrieved_chunks])

    # Step 3 — Generate answer using TinyLlama
    final_answer = generate_answer(user_question, context)

    # Step 4 — Save in MongoDB
    save_to_mongo(
        question=user_question,
        answer=final_answer,
        chunks_used=len(retrieved_chunks)
    )

    return {
        "question": user_question,
        "answer": final_answer,
        "chunks_used": len(retrieved_chunks)
    }
