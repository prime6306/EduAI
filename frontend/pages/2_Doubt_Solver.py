"""
frontend/pages/2_Doubt_Solver.py – AI-powered doubt solver chat.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth
from groq import Groq
from backend.config import get_settings

st.set_page_config(page_title="Doubt Solver – EduAI", page_icon="💬", layout="wide")
user = require_auth()
st.title("💬 AI Doubt Solver")
st.caption("Powered by Llama 3 70B • Ask anything from your syllabus")

# ── Session chat history ───────────────────────────────────
if "doubt_messages" not in st.session_state:
    st.session_state.doubt_messages = []

# ── Sidebar settings ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    subject = st.text_input("Subject context", placeholder="e.g., Digital Electronics")
    level = st.selectbox("Explanation level", ["Simple", "Intermediate", "Advanced"])
    if st.button("🗑️ Clear Chat"):
        st.session_state.doubt_messages = []
        st.rerun()

SYSTEM_PROMPT = f"""You are EduAI, an expert academic tutor for {user.get('branch', 'Engineering')} students
({user.get('year', '')}). Subject context: {subject or 'General Engineering'}.
Explanation level requested: {level}.
- Give clear, structured answers with examples
- Use numbered steps for problem-solving
- Reference real-world applications
- Keep answers appropriately detailed for the level
- For math/formulas, explain each term
"""

# ── Display chat history ───────────────────────────────────
for msg in st.session_state.doubt_messages:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "🧑‍🎓"):
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────
prompt = st.chat_input("Ask your doubt here...")
if prompt:
    st.session_state.doubt_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            try:
                client = Groq(api_key=get_settings().groq_api_key)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages += st.session_state.doubt_messages[-10:]
                resp = client.chat.completions.create(
                    model=get_settings().groq_model,
                    messages=messages,
                    max_tokens=1000,
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                answer = f"❌ Error: {e}"
        st.markdown(answer)

    st.session_state.doubt_messages.append({"role": "assistant", "content": answer})

# ── Suggested questions ───────────────────────────────────
if not st.session_state.doubt_messages:
    st.divider()
    st.subheader("💡 Suggested questions to get started:")
    suggestions = [
        "Explain the difference between RAM and ROM with examples",
        "What is Fourier Transform and where is it used?",
        "How does a microprocessor work?",
        "Explain MOSFET working principle",
        "What is the difference between TCP and UDP?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions[:4]):
        with cols[i % 2]:
            if st.button(q, key=f"sug_{i}"):
                st.session_state.doubt_messages.append({"role": "user", "content": q})
                st.rerun()
