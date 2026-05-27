"""
frontend/pages/9_Teacher_Dashboard.py – Teacher-only analytics + plagiarism + class overview.
"""
import streamlit as st, sys, os, plotly.express as px, plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth, post, get

st.set_page_config(page_title="Teacher Dashboard – EduAI", page_icon="👩‍🏫", layout="wide")
user = require_auth()

if user.get("role") not in ("teacher", "admin"):
    st.error("🚫 This page is for teachers only.")
    st.stop()

st.title("👩‍🏫 Teacher Dashboard")
st.caption("Class analytics • Plagiarism detection • Attendance heatmap • Dropout monitoring")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "🔍 Plagiarism", "📷 Attendance", "⚠️ Dropout Batch"])

# ── TAB 1: Analytics ──────────────────────────────────────
with tab1:
    st.subheader("📊 Class Performance Analytics")
    from backend.db.mongodb import get_sync_db
    try:
        db = get_sync_db()

        col1, col2, col3, col4 = st.columns(4)
        total_students = db.students.count_documents({})
        total_pipelines = db.topic_pipelines.count_documents({})
        total_qa = db.qa_history.count_documents({})
        total_quiz = db.quiz_results.count_documents({})
        col1.metric("👥 Students", total_students)
        col2.metric("📚 Study Sessions", total_pipelines)
        col3.metric("🤖 Q&A Queries", total_qa)
        col4.metric("❓ Quizzes Taken", total_quiz)

        st.divider()

        # Quiz score distribution
        quiz_data = list(db.quiz_results.find({}, {"score": 1, "topic": 1, "email": 1, "grade": 1}).limit(100))
        if quiz_data:
            col1, col2 = st.columns(2)
            with col1:
                scores = [q.get("score", 0) for q in quiz_data]
                fig = px.histogram(x=scores, nbins=10, title="Quiz Score Distribution",
                                   labels={"x": "Score %", "y": "Count"},
                                   color_discrete_sequence=["#4F46E5"])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                grades = [q.get("grade", "F") for q in quiz_data]
                grade_counts = {g: grades.count(g) for g in set(grades)}
                fig2 = px.pie(values=list(grade_counts.values()), names=list(grade_counts.keys()),
                              title="Grade Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig2, use_container_width=True)

        # Dropout predictions
        dropout_data = list(db.dropout_predictions.find({}, {"risk_level":1,"dropout_probability":1}).limit(200))
        if dropout_data:
            st.subheader("⚠️ Dropout Risk Overview")
            risk_counts = {"High": 0, "Medium": 0, "Low": 0}
            for d in dropout_data:
                risk_counts[d.get("risk_level", "Low")] = risk_counts.get(d.get("risk_level","Low"), 0) + 1
            fig3 = go.Figure(go.Bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                marker_color=["#ef4444", "#f59e0b", "#22c55e"]
            ))
            fig3.update_layout(title="Dropout Risk Predictions Distribution")
            st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load analytics: {e}")

# ── TAB 2: Plagiarism ─────────────────────────────────────
with tab2:
    st.subheader("🔍 Plagiarism Detector")
    st.info("Upload multiple student submissions (PDF/DOCX/TXT) or paste text directly.")

    method = st.radio("Input method", ["📁 Upload Files", "📝 Paste Text"], horizontal=True)

    if method == "📁 Upload Files":
        files = st.file_uploader("Upload student assignments", type=["pdf","docx","txt"],
                                  accept_multiple_files=True)
        threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.72)
        if files and st.button("🔍 Check Plagiarism", type="primary"):
            with st.spinner("Analysing submissions..."):
                file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
                result = post("/nlp/plagiarism/upload", files=file_tuples)
            if result["ok"]:
                data = result["data"]
                pairs = data.get("suspicious_pairs", [])
                st.metric("Submissions analysed", data.get("total_submissions", 0))
                st.metric("Suspicious pairs found", len(pairs))
                if pairs:
                    st.error(f"⚠️ {len(pairs)} suspicious pair(s) detected!")
                    for p in pairs:
                        st.markdown(
                            f"🔴 **{p['student1']}** ↔ **{p['student2']}** — "
                            f"Combined: `{p['combined_score']*100:.1f}%` "
                            f"(N-gram: {p['ngram_score']*100:.1f}% | Embedding: {p['embedding_score']*100:.1f}%)"
                        )
                else:
                    st.success("✅ No plagiarism detected above threshold.")
            else:
                st.error(result["error"])

    else:
        st.caption("Add student name and paste their submission text (one per row).")
        num_students = st.number_input("Number of students", 2, 10, 3)
        submissions = {}
        for i in range(int(num_students)):
            c1, c2 = st.columns([1, 3])
            with c1:
                name = st.text_input(f"Student {i+1} name", key=f"plag_name_{i}", value=f"Student_{i+1}")
            with c2:
                text = st.text_area(f"Submission {i+1}", key=f"plag_text_{i}", height=80)
            if name and text:
                submissions[name] = text
        threshold_t = st.slider("Threshold", 0.5, 1.0, 0.72, key="plag_thresh_text")
        if st.button("🔍 Analyse", type="primary") and submissions:
            with st.spinner("Checking..."):
                result = post("/nlp/plagiarism", {"submissions": submissions, "threshold": threshold_t})
            if result["ok"]:
                pairs = result["data"].get("suspicious_pairs", [])
                if pairs:
                    st.error(f"⚠️ {len(pairs)} suspicious pair(s)!")
                    for p in pairs:
                        st.markdown(f"🔴 **{p['student1']}** ↔ **{p['student2']}** — {p['combined_score']*100:.1f}%")
                else:
                    st.success("✅ No plagiarism detected.")
            else:
                st.error(result["error"])

# ── TAB 3: Attendance ─────────────────────────────────────
with tab3:
    st.subheader("📷 Class Attendance Overview")
    students_r = get("/attendance/students")
    if students_r["ok"] and students_r["data"]:
        data = students_r["data"]
        st.dataframe(data, use_container_width=True)
        if data:
            names = [s.get("name", s.get("student_id")) for s in data]
            counts = [s.get("total_attendance", 0) for s in data]
            fig = px.bar(x=names, y=counts, color=counts, color_continuous_scale="Blues",
                         title="Total Attendance per Student",
                         labels={"x": "Student", "y": "Sessions"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No student data yet.")

# ── TAB 4: Dropout Batch ──────────────────────────────────
with tab4:
    st.subheader("⚠️ Batch Dropout Risk Screening")
    st.info("Coming soon: Upload a CSV of student grades and get batch risk predictions with visualisations.")
    st.markdown("""
    **Expected CSV columns:**
    `age, studytime, failures, absences, G1, G2, sex, address, schoolsup, famsup, freetime, goout, health, famrel`
    """)
    uploaded_csv = st.file_uploader("Upload student data CSV", type=["csv"])
    if uploaded_csv:
        import pandas as pd
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head(), use_container_width=True)
        if st.button("Run Batch Prediction", type="primary"):
            results = []
            with st.spinner(f"Processing {len(df)} students..."):
                for _, row in df.iterrows():
                    r = post("/dropout/predict", row.to_dict())
                    if r["ok"]:
                        results.append({**row.to_dict(), **r["data"]})
            if results:
                res_df = pd.DataFrame(results)
                st.dataframe(res_df[["age","G1","G2","risk_level","dropout_probability"]], use_container_width=True)
                fig = px.pie(res_df, names="risk_level", title="Batch Risk Distribution",
                             color="risk_level",
                             color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"})
                st.plotly_chart(fig, use_container_width=True)
