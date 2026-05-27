"""
frontend/pages/7_Dropout_Risk.py – Student dropout risk classifier.
"""
import streamlit as st, sys, os, plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth, post

st.set_page_config(page_title="Dropout Risk – EduAI", page_icon="⚠️", layout="wide")
user = require_auth()
st.title("⚠️ Student Dropout Risk Predictor")
st.caption("Random Forest + Logistic Regression ensemble • MLflow tracked • Early warning system")

st.info(
    "📌 Fill in the academic details below to get a dropout risk assessment. "
    "This is a **supportive tool** — high risk means extra attention is needed, not a judgment."
)

col_form, col_result = st.columns([1, 1])

with col_form:
    st.subheader("📋 Student Information")
    with st.form("dropout_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 15, 25, 19)
            sex = st.selectbox("Gender", ["M", "F"])
            address = st.selectbox("Address", ["U (Urban)", "R (Rural)"])
            studytime = st.slider("Weekly Study Time (hrs)", 1, 10, 3)
            failures = st.number_input("Past Academic Failures", 0, 5, 0)
        with c2:
            absences = st.number_input("Number of Absences", 0, 50, 5)
            G1 = st.slider("Grade Period 1 (0-20)", 0, 20, 12)
            G2 = st.slider("Grade Period 2 (0-20)", 0, 20, 11)
            schoolsup = st.selectbox("School Support", ["yes", "no"])
            famsup = st.selectbox("Family Support", ["yes", "no"])
        c3, c4 = st.columns(2)
        with c3:
            freetime = st.slider("Free Time (1-5)", 1, 5, 3)
            goout = st.slider("Goes Out with Friends (1-5)", 1, 5, 3)
        with c4:
            health = st.slider("Health Status (1-5)", 1, 5, 3)
            famrel = st.slider("Family Relationship (1-5)", 1, 5, 3)
        model_type = st.selectbox("Model", ["rf (Random Forest)", "lr (Logistic Regression)"])
        submit = st.form_submit_button("🔍 Predict Risk", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Risk Assessment")
    if submit:
        with st.spinner("Running prediction..."):
            result = post("/dropout/predict", {
                "age": age, "sex": sex,
                "address": address[0],
                "studytime": studytime, "failures": failures,
                "absences": absences, "G1": G1, "G2": G2,
                "schoolsup": schoolsup, "famsup": famsup,
                "freetime": freetime, "goout": goout,
                "health": health, "famrel": famrel,
                "model_type": model_type.split()[0],
            })

        if result["ok"]:
            data = result["data"]
            risk = data["risk_level"]
            prob = data["dropout_probability"]
            factors = data.get("risk_factors", [])

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Dropout Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if risk=="High" else "#f59e0b" if risk=="Medium" else "#22c55e"},
                    "steps": [
                        {"range": [0, 35], "color": "#dcfce7"},
                        {"range": [35, 60], "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 60},
                },
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Risk badge
            badge_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}[risk]
            st.markdown(
                f'<div style="text-align:center;background:{badge_color}22;border:2px solid {badge_color};'
                f'border-radius:12px;padding:1rem;margin:0.5rem 0">'
                f'<b style="font-size:1.5rem;color:{badge_color}">{risk} Risk</b><br>'
                f'Dropout probability: <b>{prob*100:.1f}%</b></div>',
                unsafe_allow_html=True,
            )

            # Risk factors
            if factors:
                st.subheader("🔍 Key Risk Factors")
                for f in factors:
                    st.markdown(f"• **{f.replace('_', ' ').title()}**")

            # Recommendations
            st.subheader("💡 Recommendations")
            if risk == "High":
                recs = [
                    "📚 Increase study time to at least 2 hours daily",
                    "🏫 Attend extra support classes offered by the college",
                    "👨‍🏫 Meet with your academic advisor this week",
                    "💙 Consider speaking to the student wellness team",
                    "📝 Create a structured study schedule using our Study Planner",
                ]
            elif risk == "Medium":
                recs = [
                    "📖 Review topics where grades dipped in Period 1 or 2",
                    "⏰ Try to reduce absences — each missed class increases risk",
                    "🤝 Form a study group with peers",
                    "📅 Use the EduAI Study Planner to stay organised",
                ]
            else:
                recs = [
                    "🎉 You're on track — keep up the good work!",
                    "📈 Try practicing with more quizzes to maintain performance",
                    "🧑‍🤝‍🧑 Consider helping peers who might be struggling",
                ]
            for r in recs:
                st.markdown(r)
        else:
            st.error(f"❌ {result['error']}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#9CA3AF">
        <div style="font-size:3rem">📊</div>
        <p>Fill in the form and click <b>Predict Risk</b> to see your assessment</p>
        </div>""", unsafe_allow_html=True)

# ── Teacher retrain ────────────────────────────────────────
if user.get("role") == "teacher":
    st.divider()
    st.subheader("🔄 Model Management (Teacher)")
    if st.button("Retrain Dropout Models", type="secondary"):
        with st.spinner("Retraining..."):
            result = post("/dropout/retrain", {})
        if result["ok"]:
            d = result["data"]
            st.success(
                f"✅ Retrained! LR accuracy: {d.get('lr_accuracy',0)*100:.1f}% | "
                f"RF accuracy: {d.get('rf_accuracy',0)*100:.1f}%"
            )
        else:
            st.error(result["error"])
