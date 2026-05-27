"""
frontend/Home.py – EduAI Landing + Login/Register page.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.utils.api_client import post

st.set_page_config(
    page_title="EduAI – AI Academic Platform",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  .main-title { font-size: 2.8rem; font-weight: 800; color: #4F46E5; text-align: center; }
  .subtitle   { font-size: 1.1rem; color: #6B7280; text-align: center; margin-bottom: 2rem; }
  .feature-card { background: #F9FAFB; border-radius: 12px; padding: 1rem;
                  border-left: 4px solid #4F46E5; margin-bottom: 0.5rem; }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 EduAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your AI-Powered Academic Companion</div>', unsafe_allow_html=True)

# ── Already logged in ──────────────────────────────────────
if st.session_state.get("token"):
    user = st.session_state.get("user", {})
    st.success(f"✅ Welcome back, **{user.get('name', 'Student')}**!")
    st.info("Use the sidebar to navigate to any feature.")
    if st.button("🚪 Logout", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.stop()

# ── Feature highlights ─────────────────────────────────────
with st.expander("✨ What can EduAI do for you?", expanded=True):
    features = [
        ("🧠", "AI Doubt Solver", "Instant answers powered by Llama 3 70B"),
        ("📚", "RAG Q&A", "Ask questions from your uploaded PDFs"),
        ("❓", "Smart Quiz", "MCQs generated from any topic"),
        ("📝", "Study Material", "Full subtopics, explanations & summaries"),
        ("📅", "Study Planner", "Personalised AI-generated weekly schedule"),
        ("📷", "Face Attendance", "Anti-spoof protected attendance marking"),
        ("⚠️", "Dropout Risk", "Early warning classifier for at-risk students"),
        ("💙", "Wellness Chat", "Empathetic AI wellness companion"),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(
                f'<div class="feature-card"><b>{icon} {title}</b><br>'
                f'<small style="color:#6B7280">{desc}</small></div>',
                unsafe_allow_html=True,
            )

st.divider()

# ── Auth tabs ──────────────────────────────────────────────
tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

with tab_login:
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@university.edu")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

    if submitted:
        if not email or not password:
            st.error("Please fill in all fields")
        else:
            with st.spinner("Logging in..."):
                result = post("/auth/login", {"email": email, "password": password})
            if result["ok"]:
                data = result["data"]
                st.session_state["token"] = data["access_token"]
                st.session_state["user"] = data["user"]
                st.success(f"✅ Welcome, {data['user']['name']}!")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")

with tab_register:
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            email_r = st.text_input("Email", key="reg_email")
            password_r = st.text_input("Password", type="password", key="reg_pw")
        with col2:
            role = st.selectbox("Role", ["student", "teacher"])
            branch = st.selectbox("Branch", ["ECE", "CS", "ME", "CE", "EE", "IT", "Other"])
            year = st.selectbox("Year", ["1st year", "2nd year", "3rd year", "4th year"])
        student_id = st.text_input("Student/Employee ID (optional)")
        submitted_r = st.form_submit_button("Create Account", type="primary", use_container_width=True)

    if submitted_r:
        if not all([name, email_r, password_r]):
            st.error("Name, email and password are required")
        else:
            with st.spinner("Creating account..."):
                result = post("/auth/register", {
                    "name": name, "email": email_r, "password": password_r,
                    "role": role, "branch": branch, "year": year, "student_id": student_id,
                })
            if result["ok"]:
                data = result["data"]
                st.session_state["token"] = data["access_token"]
                st.session_state["user"] = data["user"]
                st.success("✅ Account created! Redirecting...")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
