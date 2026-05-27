"""
frontend/pages/1_Dashboard.py – Student/Teacher dashboard.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth, get

st.set_page_config(page_title="Dashboard – EduAI", page_icon="📊", layout="wide")

user = require_auth()
st.title(f"📊 Dashboard — Welcome, {user.get('name', 'Student')}!")
st.caption(f"{user.get('branch', '')} • {user.get('year', '')} • {user.get('role', 'student').title()}")
st.divider()

# ── Quick stats ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎓 Role", user.get("role", "student").title())
col2.metric("🏫 Branch", user.get("branch", "—"))
col3.metric("📅 Year", user.get("year", "—"))
col4.metric("🆔 Student ID", user.get("student_id", "—") or "—")

st.divider()

# ── Attendance ─────────────────────────────────────────────
st.subheader("📷 Recent Attendance Logs")
att = get("/attendance/logs", {"limit": 20})
if att["ok"] and att["data"]:
    logs = att["data"]
    st.dataframe(
        [{"Name": l.get("name"), "Student ID": l.get("student_id"),
          "Total Attendance": l.get("total_attendance"), "Time": l.get("timestamp", "")[:19]}
         for l in logs],
        use_container_width=True,
    )
else:
    st.info("No attendance logs yet.")

# ── Teacher extras ─────────────────────────────────────────
if user.get("role") == "teacher":
    st.divider()
    st.subheader("👩‍🏫 Class Overview")
    students = get("/attendance/students")
    if students["ok"] and students["data"]:
        df_data = students["data"]
        att_counts = [s.get("total_attendance", 0) for s in df_data]
        names = [s.get("name", s.get("student_id")) for s in df_data]
        fig = px.bar(x=names, y=att_counts, labels={"x": "Student", "y": "Total Attendance"},
                     color=att_counts, color_continuous_scale="Blues", title="Class Attendance")
        st.plotly_chart(fig, use_container_width=True)

# ── Navigation cards ───────────────────────────────────────
st.divider()
st.subheader("🚀 Quick Navigation")
nav_items = [
    ("💬", "Doubt Solver", "Ask any academic question", "2_Doubt_Solver"),
    ("📚", "Study Material", "Generate notes + summaries", "3_Study_Material"),
    ("❓", "Quiz", "Practice with AI-generated MCQs", "4_Quiz"),
    ("📄", "RAG Q&A", "Ask questions from your PDFs", "5_RAG_QA"),
    ("📷", "Attendance", "Mark attendance with face recognition", "6_Attendance"),
    ("⚠️", "Dropout Risk", "Check your academic risk level", "7_Dropout_Risk"),
    ("💙", "Wellness", "Talk to your wellness companion", "8_Wellness"),
]
cols = st.columns(4)
for i, (icon, title, desc, page) in enumerate(nav_items):
    with cols[i % 4]:
        st.markdown(f"**{icon} {title}**")
        st.caption(desc)
        st.page_link(f"pages/{page}.py", label=f"Open {title} →")
