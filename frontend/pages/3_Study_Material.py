"""
frontend/pages/3_Study_Material.py – Generate subtopics, explanations, summaries, study plan.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth, post

st.set_page_config(page_title="Study Material – EduAI", page_icon="📚", layout="wide")
user = require_auth()
st.title("📚 AI Study Material Generator")
st.caption("Auto-generates subtopics, detailed explanations, summaries, YouTube links & articles")

tab1, tab2 = st.tabs(["📖 Generate Notes", "📅 Study Planner"])

# ── TAB 1: Generate Notes ──────────────────────────────────
with tab1:
    with st.form("notes_form"):
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Topic *", placeholder="e.g., Fourier Transform")
            subject = st.text_input("Subject *", placeholder="e.g., Signals & Systems")
        with col2:
            branch = st.selectbox("Branch", ["ECE", "CS", "ME", "CE", "EE", "IT"],
                                  index=["ECE","CS","ME","CE","EE","IT"].index(user.get("branch","ECE"))
                                  if user.get("branch","ECE") in ["ECE","CS","ME","CE","EE","IT"] else 0)
            year = st.selectbox("Year", ["1st year","2nd year","3rd year","4th year"],
                                index=["1st year","2nd year","3rd year","4th year"].index(user.get("year","3rd year"))
                                if user.get("year","3rd year") in ["1st year","2nd year","3rd year","4th year"] else 2)
        submitted = st.form_submit_button("🚀 Generate Study Material", type="primary", use_container_width=True)

    if submitted:
        if not topic or not subject:
            st.error("Topic and Subject are required")
        else:
            with st.spinner("🤖 AI is generating your study material (30–60 sec)..."):
                result = post("/nlp/full-pipeline", {
                    "topic": topic, "subject": subject, "branch": branch, "year": year
                })

            if result["ok"]:
                data = result["data"]
                st.success(f"✅ Generated in {data.get('processing_time_sec', '—')}s")

                # Summary
                st.subheader("📋 Summary")
                st.info(data.get("summary", ""))

                # Subtopics + Explanations
                st.subheader("📖 Subtopics & Explanations")
                subtopics = data.get("subtopics", {})
                explanations = data.get("explanations", {})
                for st_name, points in subtopics.items():
                    with st.expander(f"📌 {st_name}"):
                        if points:
                            st.markdown("**Key Points:**")
                            for p in points:
                                st.markdown(f"• {p}")
                        if st_name in explanations:
                            st.divider()
                            st.markdown("**Detailed Explanation:**")
                            st.write(explanations[st_name])

                # YouTube
                youtube = data.get("youtube", {})
                if any(youtube.values()):
                    st.subheader("▶️ YouTube Resources")
                    cols = st.columns(3)
                    idx = 0
                    for st_name, videos in youtube.items():
                        for v in videos:
                            with cols[idx % 3]:
                                st.markdown(f"**{v['title'][:60]}**")
                                st.markdown(f"[Watch ▶]({v['url']})")
                            idx += 1

                # Articles
                articles = data.get("articles", {})
                if any(a.get("link") for a in articles.values()):
                    st.subheader("🔗 Reference Articles")
                    for st_name, art in articles.items():
                        if art.get("link"):
                            st.markdown(f"**{st_name}** → [{art.get('title', art['link'])}]({art['link']})")
                            if art.get("snippet"):
                                st.caption(art["snippet"])
            else:
                st.error(f"❌ {result['error']}")

# ── TAB 2: Study Planner ───────────────────────────────────
with tab2:
    st.subheader("📅 AI Study Planner")
    with st.form("planner_form"):
        subjects_input = st.text_area("Subjects (one per line)", height=100,
                                       placeholder="Digital Electronics\nSignals & Systems\nControl Systems")
        col1, col2 = st.columns(2)
        with col1:
            exam_date = st.date_input("Exam Date")
        with col2:
            hours = st.slider("Study hours per day", 1, 12, 4)
        plan_branch = st.selectbox("Branch", ["ECE","CS","ME","CE","EE","IT"], key="plan_branch")
        plan_year = st.selectbox("Year", ["1st year","2nd year","3rd year","4th year"], key="plan_year", index=2)
        plan_submit = st.form_submit_button("📅 Generate Plan", type="primary", use_container_width=True)

    if plan_submit:
        subjects = [s.strip() for s in subjects_input.strip().split("\n") if s.strip()]
        if not subjects:
            st.error("Enter at least one subject")
        else:
            with st.spinner("Generating your study plan..."):
                result = post("/nlp/study-plan", {
                    "subjects": subjects,
                    "exam_date": str(exam_date),
                    "hours_per_day": hours,
                    "branch": plan_branch,
                    "year": plan_year,
                })
            if result["ok"]:
                plan = result["data"]
                for day in plan.get("plan", []):
                    with st.expander(f"📆 {day['day']} — {day.get('total_hours', 0)}h total"):
                        for session in day.get("sessions", []):
                            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                                session.get("priority", "").lower(), "⚪")
                            st.markdown(
                                f"{priority_color} **{session['subject']}** — {session['topic']} "
                                f"({session.get('duration_hours', 0)}h)"
                            )
                if plan.get("tips"):
                    st.subheader("💡 Study Tips")
                    for tip in plan["tips"]:
                        st.markdown(f"• {tip}")
            else:
                st.error(f"❌ {result['error']}")
