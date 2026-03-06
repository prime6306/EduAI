# EduAI — AI-Powered Academic Intelligence Platform

EduAI is an **AI-driven academic intelligence platform** that integrates **Deep Learning, LLMs, and Embedding-based Retrieval** to support personalized learning, automated evaluation, and academic analytics.

The platform combines **computer vision, NLP, and generative AI** to assist students and educators with **content generation, attendance verification, plagiarism detection, and intelligent Q&A systems**.

---

# 🚀 Key AI Features

## AI Learning Assistant
EduAI generates structured learning content using large language models.

Capabilities include:

- Subtopic generation  
- Concept explanations  
- MCQ generation  
- Content summarization  
- Question answering  
- Learning resource recommendations  

Models used:

- `TinyLlama-1.1B-Chat`
- `Llama-3.1-8B-Instruct`
- `Mistral-7B-Instruct`

These models power dynamic **academic content generation pipelines**.

---

## Embedding-Based Knowledge Retrieval

EduAI uses **semantic embeddings** to retrieve relevant context before generating responses.

Embedding model:

`BAAI/bge-m3 (SentenceTransformer)`

This enables:

- Semantic search
- Context-aware Q&A
- Knowledge retrieval
- Question recommendation

---

## AI Attendance with Anti-Spoofing

EduAI includes a **computer vision attendance system** designed for group verification.

Features:

- Face recognition
- Deep learning anti-spoofing
- Protection against photo/video proxy attempts

Model used:

`ResNet CNN`

Performance:

**99% spoof detection accuracy**

This prevents fraudulent attendance marking.

---

## Assignment Similarity & Plagiarism Detection

EduAI detects copied assignments using:

- **N-gram similarity**
- **Embedding similarity**
- Semantic comparison

This allows instructors to quickly detect duplicate or copied submissions.

---

## AI Recommendation System

EduAI recommends:

- Related questions
- Learning topics
- Study resources

Recommendations are generated using **embedding similarity and past interactions**.

---

# 🧠 AI Architecture

| Component | Technology |
|----------|------------|
| NLP Embeddings | BAAI/bge-m3 |
| Generative AI | TinyLlama, Llama-3.1-8B, Mistral-7B |
| Computer Vision | ResNet |
| Retrieval | Embedding similarity |
| Backend | Python + Flask |
| Database | MongoDB |
| Frontend | Lovable |

---

# ⚙️ Core AI Pipeline

EduAI includes a reusable AI pipeline module:

`features.py`

Responsibilities:

- Subtopic generation
- Concept explanations
- MCQ generation
- Article retrieval
- YouTube resource search
- Summarization
- Plagiarism detection
- MongoDB storage

Example storage function:

```python
def save_to_mongo(question, answer, chunks_used):
    record = {
        "question": question,
        "answer": answer,
        "chunks_used": chunks_used,
        "timestamp": datetime.utcnow()
    }
    qa_collection.insert_one(record)
