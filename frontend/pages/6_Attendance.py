"""
frontend/pages/6_Attendance.py – Face recognition + anti-spoof attendance.
"""
import streamlit as st, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth, post, get
import plotly.express as px

st.set_page_config(page_title="Attendance – EduAI", page_icon="📷", layout="wide")
user = require_auth()
st.title("📷 Smart Face Attendance")
st.caption("Anti-spoofing protected • MobileNetV2 + ResNet18 ensemble • 99% accuracy")

tab1, tab2 = st.tabs(["📸 Mark Attendance", "📊 Attendance Report"])

# ── TAB 1: Mark Attendance ─────────────────────────────────
with tab1:
    st.subheader("Upload Class Photo")
    st.info("📌 Upload a group or individual photo. Anti-spoofing checks will run automatically on each detected face.")

    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

    if uploaded:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(uploaded, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.metric("File", uploaded.name)
            st.metric("Size", f"{uploaded.size // 1024} KB")
            if st.button("🚀 Process Attendance", type="primary", use_container_width=True):
                with st.spinner("Running face detection + anti-spoofing..."):
                    result = post(
                        "/attendance/mark",
                        files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                    )

                if result["ok"]:
                    data = result["data"]
                    st.success(f"✅ Processed — {data.get('faces_detected', 0)} faces detected")

                    students = data.get("students", [])
                    timings = data.get("timings", {})

                    # Results table
                    st.subheader("Results")
                    for s in students:
                        status = s.get("status", "unknown")
                        if status == "marked":
                            st.success(f"✅ **{s.get('name', s.get('student_id'))}** — Attendance marked (Total: {s.get('total_attendance')})")
                        elif status == "spoof":
                            st.error(f"🚫 **{s.get('student_id')}** — SPOOF DETECTED. Attendance NOT marked.")
                        elif status == "duplicate":
                            st.warning(f"⏱️ **{s.get('name', s.get('student_id'))}** — Already marked recently.")
                        elif status == "no_match":
                            st.warning("❓ Unknown face detected")
                        else:
                            st.info(f"ℹ️ {s.get('student_id')} — {status}")

                    # Timings
                    if timings:
                        st.subheader("⏱️ Processing Timings")
                        st.metric("Total Time", f"{timings.get('total_time', 0):.3f}s")
                        per_face = timings.get("per_face", [])
                        if per_face:
                            st.dataframe(
                                [{"Student": t.get("student_id", "Unknown"),
                                  "Recognition (s)": t.get("recognition_time", 0),
                                  "Anti-spoof (s)": t.get("antispoof_time", 0),
                                  "Total (s)": t.get("face_time", 0)} for t in per_face],
                                use_container_width=True,
                            )
                else:
                    st.error(f"❌ {result['error']}")

# ── TAB 2: Report ──────────────────────────────────────────
with tab2:
    st.subheader("📊 Attendance Report")
    logs_result = get("/attendance/logs", {"limit": 100})

    if logs_result["ok"] and logs_result["data"]:
        logs = logs_result["data"]

        # Summary metrics
        unique_students = len(set(l.get("student_id") for l in logs))
        st.metric("Total Sessions Logged", len(logs))

        col1, col2 = st.columns(2)
        with col1:
            # Bar chart
            student_totals = {}
            for l in logs:
                sid = l.get("name", l.get("student_id", "Unknown"))
                student_totals[sid] = max(student_totals.get(sid, 0), l.get("total_attendance", 0))
            if student_totals:
                fig = px.bar(
                    x=list(student_totals.keys()),
                    y=list(student_totals.values()),
                    labels={"x": "Student", "y": "Attendance Count"},
                    title="Total Attendance by Student",
                    color=list(student_totals.values()),
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Recent logs table
            st.subheader("Recent Logs")
            st.dataframe(
                [{"Name": l.get("name", l.get("student_id")),
                  "Total": l.get("total_attendance"),
                  "Time": l.get("timestamp", "")[:16]} for l in logs[:15]],
                use_container_width=True,
            )
    else:
        st.info("No attendance logs yet. Upload a photo to get started.")

    if user.get("role") == "teacher":
        st.divider()
        st.subheader("👩‍🏫 All Students")
        students_result = get("/attendance/students")
        if students_result["ok"] and students_result["data"]:
            st.dataframe(students_result["data"], use_container_width=True)
