# retriever.py
import numpy as np
from datetime import datetime
from bson import Binary
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------
# MongoDB Setup
# -------------------------
mongo_uri = "mongodb+srv://studentUser:1234@cluster0.zodzonx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
mongo_db = client["studentdb"]

pdfs_collection = mongo_db["pdfs"]
pdf_chunks_collection = mongo_db["pdf_chunks"]
qa_collection = mongo_db["qa_history"]

# -------------------------
# Embedding Model (BGE-M3)
# -------------------------
print("📦 Loading retrieval embedding model: BAAI/bge-m3 ...")
embed_model = SentenceTransformer("BAAI/bge-m3")
print("✅ Embedding model loaded.")

# -------------------------
# TinyLlama LLM
# -------------------------
print("📦 Loading TinyLlama LLM ...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
generator = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
print("✅ LLM loaded.")

# -------------------------
# Helper: Deserialize Embedding
# -------------------------
def deserialize_embedding(bin_data: Binary, shape, dtype="float32") -> np.ndarray:
    arr = np.frombuffer(bin_data, dtype=np.float32)
    return arr.reshape(shape)

# -------------------------
# Cosine Similarity
# -------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm  # shape (N,)

# -------------------------
# LLM Answer Generation
# -------------------------
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

# -------------------------
# Main Search Function
# -------------------------
def search_query(query: str, pdf_id: str = None, top_k: int = 5):

    # 1️⃣ Encode query
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0].astype(np.float32)

    # 2️⃣ Fetch embeddings from Mongo
    query_filter = {}
    if pdf_id:
        query_filter["pdf_id"] = pdf_id

    cursor = pdf_chunks_collection.find(
        query_filter,
        {"_id": 0, "pdf_id": 1, "chunk_id": 1, "text": 1, "embedding": 1, "embedding_shape": 1}
    )

    docs = list(cursor)
    if len(docs) == 0:
        return {
            "question": query,
            "answer": "No chunks found for this PDF." if pdf_id else "No chunks in database.",
            "chunks_used": [],
            "pdf_id_searched": pdf_id
        }

    # 3️⃣ Rebuild embedding matrix
    embeddings = []
    texts = []
    chunk_ids = []
    pdf_ids_found = []

    for d in docs:
        vec = deserialize_embedding(d["embedding"], d["embedding_shape"])
        embeddings.append(vec)
        texts.append(d["text"])
        chunk_ids.append(d["chunk_id"])
        pdf_ids_found.append(d["pdf_id"])

    embeddings = np.vstack(embeddings).astype(np.float32)

    # 4️⃣ Compute cosine similarity
    sims = cosine_sim(q_emb, embeddings)

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    # 5️⃣ Threshold logic → Reject irrelevant questions
    SIMILARITY_THRESHOLD = 0.42

    if best_score < SIMILARITY_THRESHOLD:
        qa_doc = {
            "question": query,
            "answer": "The book does not discuss this topic.",
            "chunks_used": [],        # no chunks returned
            "pdf_id_searched": pdf_id,
            "best_score": best_score,
            "timestamp": datetime.utcnow()
        }
        qa_collection.insert_one(qa_doc)

        return {
            "question": query,
            "answer": "The book does not discuss this topic.",
            "chunks_used": [],
            "best_score": best_score,
            "pdf_id_searched": pdf_id
        }

    # 6️⃣ Retrieve top-k only when similarity ≥ threshold
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        results.append({
            "rank": rank + 1,
            "chunk_id": int(chunk_ids[idx]),
            "score": float(sims[idx]),
            "pdf_id": pdf_ids_found[idx],
            "text": texts[idx]
        })

    # 7️⃣ Build context → LLM answer
    context = "\n\n".join([r["text"] for r in results])
    answer = generate_answer(query, context)

    # 8️⃣ Save everything to history
    qa_doc = {
        "question": query,
        "answer": answer,
        "chunks_used": results,
        "pdf_id_searched": pdf_id,
        "timestamp": datetime.utcnow()
    }
    qa_collection.insert_one(qa_doc)

    # 9️⃣ Final response
    return {
        "question": query,
        "answer": answer,
        "chunks_used": results,
        "pdf_id_searched": pdf_id
    }
