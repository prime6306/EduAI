"""
frontend/pages/8_Wellness.py – Mental wellness assessment + AI companion chat.
"""
import streamlit as st, sys, os, uuid, plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontend.utils.api_client import require_auth, post, get

st.set_page_config(page_title="Wellness – EduAI", page_icon="💙", layout="wide")
user = require_auth()
st.title("💙 Student Wellness Companion")
st.caption("Confidential • AI-powered empathetic support • Crisis helplines always available")

st.markdown("""
<div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:1rem;border-radius:8px;margin-bottom:1rem">
<b>📌 Important:</b> This tool is for support and reflection. It is <b>not</b> a replacement for
professional mental health care. If you're in crisis, please call <b>iCall: 9152987821</b> (India).
</div>""", unsafe_allow_html=True)

# ── State ──────────────────────────────────────────────────
for k, v in [
    ("wellness_step", "intro"),        # intro → assess → chat
    ("wellness_answers", []),
    ("wellness_assessment", {}),
    ("wellness_session_id", None),
    ("wellness_messages", []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── STEP: Intro ────────────────────────────────────────────
if st.session_state.wellness_step == "intro":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("How are you feeling today?")
        st.write(
            "This quick wellness check takes about **2 minutes**. "
            "Your responses are private and help personalise the support you receive."
        )
        st.markdown("""
        **What you'll get:**
        - 📊 A personalised wellness assessment
        - 💬 A private chat with an empathetic AI companion
        - 🧘 Coping strategies and study tips
        - 📞 Helpline resources if needed
        """)
        if st.button("💙 Start Wellness Check", type="primary", use_container_width=True):
            st.session_state.wellness_step = "assess"
            st.rerun()
    with col2:
        st.markdown("""
        <div style="background:#f0fdf4;border-radius:12px;padding:1.5rem;text-align:center">
        <div style="font-size:3rem">🌱</div>
        <b>You are not alone.</b><br><br>
        <div style="font-size:0.85rem;color:#6B7280">
        Thousands of students face academic stress.<br>
        Seeking support is a sign of strength.
        </div>
        </div>""", unsafe_allow_html=True)

# ── STEP: Assessment ───────────────────────────────────────
elif st.session_state.wellness_step == "assess":
    q_result = get("/wellness/questions")
    if not q_result["ok"]:
        st.error("Could not load questions")
        st.stop()

    questions = q_result["data"]["questions"]
    st.subheader("📋 Wellness Questionnaire")
    st.caption("Answer based on how you've been feeling this past week.")

    answers = []
    valid = True
    for i, q in enumerate(questions):
        st.markdown(f"**{i+1}. {q['text']}**")
        choice = st.radio("", q["options"], key=f"wq_{i}", index=None, label_visibility="collapsed")
        if choice is None:
            valid = False
        else:
            idx = q["options"].index(choice)
            answers.append(q["scores"][idx])
        st.divider()

    if st.button("📊 Get My Assessment", type="primary", use_container_width=True):
        if not valid or len(answers) < len(questions):
            st.error("Please answer all questions before continuing.")
        else:
            with st.spinner("Analysing your responses..."):
                result = post("/wellness/assess", {"answers": answers})
            if result["ok"]:
                data = result["data"]
                st.session_state.wellness_assessment = data["assessment"]
                st.session_state.wellness_session_id = data["session_id"]
                st.session_state.wellness_answers = answers
                st.session_state.wellness_step = "results"
                st.rerun()
            else:
                st.error(result["error"])

# ── STEP: Results + Chat ───────────────────────────────────
elif st.session_state.wellness_step in ("results", "chat"):
    assessment = st.session_state.wellness_assessment
    level = assessment.get("level", "mild")
    score = assessment.get("total_score", 0)
    message = assessment.get("message", "")
    color_map = {"minimal": "#22c55e", "mild": "#84cc16", "moderate": "#f59e0b", "severe": "#ef4444"}
    color = color_map.get(level, "#6b7280")

    # Score display
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Wellness Score"},
            gauge={
                "axis": {"range": [0, 21]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 4], "color": "#dcfce7"},
                    {"range": [4, 9], "color": "#fef9c3"},
                    {"range": [9, 14], "color": "#fed7aa"},
                    {"range": [14, 21], "color": "#fee2e2"},
                ],
            },
        ))
        fig.update_layout(height=240, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            f'<div style="background:{color}22;border:2px solid {color};border-radius:12px;padding:1.5rem">'
            f'<b style="font-size:1.3rem;color:{color}">{level.title()} — {score}/21</b><br><br>'
            f'{message}</div>',
            unsafe_allow_html=True,
        )
        if level == "severe":
            st.error("📞 **Please reach out:** iCall: **9152987821** • NIMHANS: **080-46110007**")

    st.divider()

    # ── Chat ──────────────────────────────────────────────
    st.subheader("💬 Talk to Your Wellness Companion")
    st.caption("Your companion has been personalised based on your assessment. Everything you share stays private.")

    for msg in st.session_state.wellness_messages:
        role = msg["role"]
        avatar = "💙" if role == "assistant" else "🧑‍🎓"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sentiment"):
                s = msg["sentiment"]
                compound = s.get("compound", 0)
                emoji = "😊" if compound > 0.05 else "😔" if compound < -0.05 else "😐"
                st.caption(f"{emoji} Sentiment: {s.get('label','neutral')} ({compound:+.2f})")

    # Greeting on first open
    if not st.session_state.wellness_messages and st.session_state.wellness_step == "results":
        greeting_map = {
            "minimal": "Hi! Great to hear you're doing well. I'm here if you want to talk about anything — studies, goals, or just have a chat. 😊",
            "mild": "Hi there! I can see you're managing okay but have some things on your mind. I'm here to listen and help however I can. What's been on your mind lately?",
            "moderate": "Hey, I'm really glad you're here. It sounds like things have been a bit tough lately. You don't have to face this alone — I'm here to listen and support you. 💙",
            "severe": "Hi, I'm really glad you reached out. I want you to know that how you're feeling right now is valid, and there are people who care about you. I'm here with you. 💙",
        }
        greeting = greeting_map.get(level, "Hi! I'm your EduAI wellness companion. How are you feeling?")
        st.session_state.wellness_messages.append({"role": "assistant", "content": greeting})
        st.session_state.wellness_step = "chat"
        st.rerun()

    user_input = st.chat_input("Share what's on your mind...")
    if user_input:
        st.session_state.wellness_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="💙"):
            with st.spinner("..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.wellness_messages[:-1]
                ]
                result = post("/wellness/chat", {
                    "message": user_input,
                    "session_id": st.session_state.wellness_session_id,
                    "conversation_history": history,
                    "assessment": assessment,
                })
            if result["ok"]:
                data = result["data"]
                response = data["response"]
                if data.get("crisis_detected"):
                    st.error(response)
                else:
                    st.markdown(response)
                st.session_state.wellness_messages.append({
                    "role": "assistant",
                    "content": response,
                    "sentiment": data.get("sentiment"),
                })
            else:
                st.error(result.get("error"))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Assessment"):
            for k in ["wellness_step","wellness_answers","wellness_assessment","wellness_session_id","wellness_messages"]:
                st.session_state.pop(k, None)
            st.rerun()
    with col2:
        st.markdown("📞 **Crisis helpline:** iCall **9152987821**")
