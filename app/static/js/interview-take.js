/**
 * interview-take.js — drives the distraction-free mock interview page.
 * Reads window.__INTERVIEW__ (session id, urls, optional resumeTurn) set
 * by interview/take.html, talks to /api/interview/<sid>/start + /answer
 * via the CSRF-aware apiFetch() from app.js, and hands each question off
 * to VoiceIO for speech in/out.
 */
(function () {
  const cfg = window.__INTERVIEW__;

  const interviewerAvatar = document.getElementById("interviewerAvatar");
  const interviewerName = document.getElementById("interviewerName");
  const interviewerTitle = document.getElementById("interviewerTitle");
  const levelPill = document.getElementById("levelPill");
  const qProgress = document.getElementById("qProgress");
  const questionText = document.getElementById("questionText");
  const micRing = document.getElementById("micRing");
  const voiceStatus = document.getElementById("voiceStatus");
  const transcriptBox = document.getElementById("transcriptBox");
  const submitBtn = document.getElementById("submitBtn");
  const replayBtn = document.getElementById("replayBtn");
  const progressDots = document.getElementById("progressDots");
  const turnLog = document.getElementById("turnLog");

  let state = { turnId: null, question: "", voiceOpts: { rate: 1, pitch: 1 }, isListening: false };

  if (!window.VoiceIO || !VoiceIO.supported) {
    voiceStatus.textContent = "Voice not supported in this browser — type your answer below instead.";
    micRing.classList.add("disabled");
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s || "";
    return div.innerHTML;
  }

  function setInterviewer(interviewer) {
    interviewerAvatar.textContent = interviewer.initials;
    interviewerAvatar.className = "interviewer-avatar" + (interviewer.key === "recruiter_b" ? " arjun" : "");
    interviewerName.textContent = interviewer.name;
    interviewerTitle.textContent = interviewer.title;
    state.voiceOpts = { rate: interviewer.voice_rate || 1, pitch: interviewer.voice_pitch || 1 };
  }

  function renderDots(asked, target) {
    progressDots.innerHTML = "";
    for (let i = 1; i <= target; i++) {
      const d = document.createElement("div");
      d.className = "dot" + (i < asked ? " done" : i === asked ? " current" : "");
      progressDots.appendChild(d);
    }
  }

  function setQuestion(payload) {
    state.turnId = payload.turn_id;
    state.question = payload.question;

    setInterviewer(payload.interviewer);
    levelPill.textContent = payload.level_name;
    qProgress.textContent = `Question ${payload.questions_this_level} of ${payload.questions_target_this_level} this round`;
    questionText.textContent = payload.question;
    transcriptBox.textContent = "";
    renderDots(payload.questions_this_level, payload.questions_target_this_level);

    if (window.VoiceIO && VoiceIO.supported) {
      VoiceIO.speak(payload.question, null, state.voiceOpts);
    }
  }

  function logTurn(interviewerName_, question, answer, evaluation) {
    const entry = document.createElement("div");
    entry.className = "entry";
    entry.innerHTML =
      '<div class="interviewer-tag">' + escapeHtml(interviewerName_) + '</div>' +
      '<div class="q">' + escapeHtml(question) + '</div>' +
      '<div class="a">' + escapeHtml(answer) + '</div>' +
      '<div class="score">Score: ' + (evaluation ? evaluation.quality_score : "-") + '/10' +
      (evaluation && evaluation.what_could_be_better ? ' — ' + escapeHtml(evaluation.what_could_be_better) : '') +
      '</div>';
    turnLog.prepend(entry);
  }

  micRing.addEventListener("click", function () {
    if (!window.VoiceIO || !VoiceIO.supported) return;
    if (state.isListening) {
      VoiceIO.stopListening();
      return;
    }
    VoiceIO.startListening({
      onStart: () => {
        state.isListening = true;
        micRing.classList.add("listening");
        voiceStatus.textContent = "Listening… click the mic again to stop.";
      },
      onInterim: (text) => { transcriptBox.textContent = text; },
      onFinal: (text) => { transcriptBox.textContent = text; },
      onStop: () => {
        state.isListening = false;
        micRing.classList.remove("listening");
        voiceStatus.textContent = "Got it — edit the text if needed, then submit.";
      },
      onError: (err) => {
        state.isListening = false;
        micRing.classList.remove("listening");
        voiceStatus.textContent = "Voice error: " + err.message + " — you can type instead.";
      },
    });
  });

  replayBtn.addEventListener("click", function () {
    if (window.VoiceIO && VoiceIO.supported) VoiceIO.speak(state.question, null, state.voiceOpts);
  });

  submitBtn.addEventListener("click", async function () {
    const answer = transcriptBox.textContent.trim();
    if (!answer) {
      voiceStatus.textContent = "Please record or type an answer first.";
      return;
    }
    if (state.isListening) VoiceIO.stopListening();
    submitBtn.disabled = true;
    submitBtn.textContent = "Evaluating…";

    try {
      const res = await apiFetch(cfg.answerUrl, {
        method: "POST",
        body: JSON.stringify({ turn_id: state.turnId, answer: answer }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed.");

      logTurn(interviewerName.textContent, state.question, answer, data.evaluation);

      if (data.is_final) {
        questionText.textContent = "That's a wrap — building your report…";
        voiceStatus.textContent = "";
        micRing.style.display = "none";
        document.querySelector(".answer-actions").style.display = "none";
        window.location.href = cfg.reportUrl;
        return;
      }

      setQuestion(data);
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit Answer →";
      voiceStatus.textContent = window.VoiceIO && VoiceIO.supported
        ? "Click the mic and speak, or type your answer below."
        : "Type your answer below.";
    } catch (err) {
      voiceStatus.textContent = "Something went wrong: " + err.message;
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit Answer →";
    }
  });

  // ── Kick off: resume an in-progress turn, or start a fresh interview ──
  if (cfg.resumeTurn) {
    setQuestion(cfg.resumeTurn);
  } else {
    apiFetch(cfg.startUrl, { method: "POST" })
      .then((r) => r.json().then((data) => ({ ok: r.ok, data })))
      .then(({ ok, data }) => {
        if (!ok) throw new Error(data.error || "Could not start the interview.");
        setQuestion(data);
      })
      .catch((err) => {
        questionText.textContent = "Couldn't start the interview: " + err.message;
      });
  }
})();
