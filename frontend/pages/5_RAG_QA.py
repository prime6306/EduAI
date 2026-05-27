"""
frontend/pages/5_RAG_QA.py – PDF upload + RAG-powered Q&A.
"""
import streamlit as st, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth, post, get, delete

st.set_page_config(page_title="RAG Q&A – EduAI", page_icon="📄", layout="wide")
user = require_auth()
st.title("📄 PDF Q&A — Ask Your Study Material")
st.caption("Upload textbooks, notes, or any PDF → ask questions → get Llama 3 powered answers with hallucination scoring")

# ── Sidebar: My PDFs ───────────────────────────────────────
with st.sidebar:
    st.header("📁 My Uploaded PDFs")
    pdfs_result = get("/rag/my-pdfs")
    pdfs = pdfs_result.get("data", []) if pdfs_result["ok"] else []

    selected_pdf_id = None
    if pdfs:
        pdf_options = {"All my PDFs": None}
        for p in pdfs:
            label = f"📄 {p['filename']} ({p['chunk_count']} chunks)"
            pdf_options[label] = p["pdf_id"]
        choice = st.selectbox("Filter by PDF", list(pdf_options.keys()))
        selected_pdf_id = pdf_options[choice]

        if selected_pdf_id:
            if st.button("🗑️ Delete this PDF", type="secondary"):
                result = delete(f"/rag/pdf/{selected_pdf_id}")
                if result["ok"]:
                    st.success("Deleted!")
                    st.rerun()
    else:
        st.info("No PDFs uploaded yet.")

# ── Upload new PDF ─────────────────────────────────────────
st.subheader("⬆️ Upload Document")
uploaded = st.file_uploader(
    "Upload PDF, DOCX, or TXT file",
    type=["pdf", "docx", "txt", "md"],
    help="File will be chunked, embedded, and stored permanently for Q&A"
)

if uploaded:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"📄 **{uploaded.name}** ({uploaded.size // 1024} KB)")
    with col2:
        if st.button("📤 Upload & Index", type="primary"):
            with st.spinner("Chunking and embedding document... (may take 30–60s)"):
                result = post(
                    "/rag/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
            if result["ok"]:
                data = result["data"]
                if data.get("cached"):
                    st.info(f"📌 Already indexed — {data['chunks']} chunks available.")
                else:
                    st.success(f"✅ Indexed {data['chunks']} chunks from **{data['filename']}**")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")

st.divider()

# ── Q&A Interface ──────────────────────────────────────────
st.subheader("🤖 Ask a Question")

if not pdfs:
    st.info("👆 Upload a document above to get started.")
else:
    # Chat history
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "🧑‍🎓"):
            st.markdown(msg["content"])
            if msg.get("hallucination"):
                h = msg["hallucination"]
                score = h.get("grounded_score", 50)
                verdict = h.get("verdict", "unknown")
                color = "#22c55e" if score > 70 else "#f59e0b" if score > 40 else "#ef4444"
                st.markdown(
                    f'<div style="font-size:0.75rem;color:{color};margin-top:4px">'
                    f'🔍 Grounding: {score}/100 ({verdict})</div>',
                    unsafe_allow_html=True,
                )
            if msg.get("recommendations"):
                with st.expander("💡 Related questions you might ask"):
                    for rec in msg["recommendations"]:
                        st.markdown(f"• {rec['question']} *(similarity: {rec['similarity']})*")

    question = st.chat_input("Ask a question about your uploaded material...")
    if question:
        st.session_state.rag_messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Searching your documents and generating answer..."):
                result = post("/rag/ask", {"question": question, "pdf_id": selected_pdf_id, "top_k": 5})

            if result["ok"]:
                data = result["data"]
                answer = data["answer"]
                st.markdown(answer)
                h = data.get("hallucination_score", {})
                score = h.get("grounded_score", 50)
                verdict = h.get("verdict", "unknown")
                color = "#22c55e" if score > 70 else "#f59e0b" if score > 40 else "#ef4444"
                st.markdown(
                    f'<div style="font-size:0.75rem;color:{color}">'
                    f'🔍 Grounding score: {score}/100 ({verdict}) — {h.get("reason","")}</div>',
                    unsafe_allow_html=True,
                )
                if data.get("recommendations"):
                    with st.expander("💡 Related questions"):
                        for rec in data["recommendations"]:
                            st.markdown(f"• {rec['question']}")
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "hallucination": h,
                    "recommendations": data.get("recommendations", []),
                })
            else:
                err = f"❌ {result['error']}"
                st.error(err)
                st.session_state.rag_messages.append({"role": "assistant", "content": err})

    if st.button("🗑️ Clear Chat History"):
        st.session_state.rag_messages = []
        st.rerun()
