/**
 * test-take.js — student test-taking interface. Handles mcq/tf (radio),
 * short/long (textarea) answer types, autosaves every 30s, and supports
 * an optional overall timer.
 */
(function () {
  "use strict";

  const questions = JSON.parse(document.getElementById("questionsData").textContent);
  const savedAnswers = JSON.parse(document.getElementById("savedAnswersData").textContent);
  const attemptId = window.ATTEMPT_ID;
  const timeLimitMinutes = window.TIME_LIMIT_MINUTES;

  const answers = Object.assign({}, savedAnswers);
  const flagged = new Set();
  let currentIndex = 0;
  const startTime = Date.now();
  let remainingSeconds = timeLimitMinutes ? timeLimitMinutes * 60 : null;
  let submitting = false;

  const questionText = document.getElementById("questionText");
  const questionMeta = document.getElementById("questionMeta");
  const optionsContainer = document.getElementById("optionsContainer");
  const questionProgress = document.getElementById("questionProgress");
  const answeredProgress = document.getElementById("answeredProgress");
  const paletteGrid = document.getElementById("paletteGrid");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const flagBtn = document.getElementById("flagBtn");
  const submitBtn = document.getElementById("submitBtn");
  const timerEl = document.getElementById("quizTimer");
  const timerText = document.getElementById("timerText");
  const autosaveNote = document.getElementById("autosaveNote");

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function renderQuestion() {
    const q = questions[currentIndex];
    questionText.textContent = `Q${currentIndex + 1}. ${q.text}`;
    questionMeta.innerHTML = `<span class="badge badge-accent">${q.marks} marks</span> <span class="badge badge-neutral">${q.type.toUpperCase()}</span>`;
    questionProgress.textContent = `Question ${currentIndex + 1} of ${questions.length}`;

    optionsContainer.innerHTML = "";
    if (q.type === "mcq" || q.type === "tf") {
      q.options.forEach((optText) => {
        const div = document.createElement("div");
        div.className = "quiz-option" + (answers[q.id] === optText ? " selected" : "");
        div.innerHTML = `<span class="radio-dot"></span><span class="option-text">${escapeHtml(optText)}</span>`;
        div.addEventListener("click", () => { answers[q.id] = optText; renderQuestion(); });
        optionsContainer.appendChild(div);
      });
    } else {
      const textarea = document.createElement("textarea");
      textarea.className = "form-control";
      textarea.rows = q.type === "long" ? 10 : 4;
      textarea.placeholder = q.type === "long" ? "Write your detailed answer..." : "Write your answer...";
      textarea.value = answers[q.id] || "";
      textarea.addEventListener("input", () => { answers[q.id] = textarea.value; renderPalette(); renderProgressBar(); });
      optionsContainer.appendChild(textarea);
    }

    flagBtn.textContent = flagged.has(currentIndex) ? "Unflag this question" : "Flag this question";
    prevBtn.disabled = currentIndex === 0;
    nextBtn.textContent = currentIndex === questions.length - 1 ? "Finish" : "Next";

    renderPalette();
    renderProgressBar();
  }

  function renderPalette() {
    paletteGrid.innerHTML = "";
    questions.forEach((q, i) => {
      const cell = document.createElement("div");
      let cls = "palette-cell";
      if (i === currentIndex) cls += " current";
      if (answers[q.id] && String(answers[q.id]).trim()) cls += " answered";
      if (flagged.has(i)) cls += " flagged";
      cell.className = cls;
      cell.textContent = i + 1;
      cell.addEventListener("click", () => { currentIndex = i; renderQuestion(); });
      paletteGrid.appendChild(cell);
    });
  }

  function renderProgressBar() {
    const answeredCount = questions.filter((q) => answers[q.id] && String(answers[q.id]).trim()).length;
    answeredProgress.style.width = Math.round((answeredCount / questions.length) * 100) + "%";
  }

  prevBtn.addEventListener("click", () => { if (currentIndex > 0) { currentIndex--; renderQuestion(); } });
  nextBtn.addEventListener("click", () => {
    if (currentIndex < questions.length - 1) { currentIndex++; renderQuestion(); }
    else { submitTest(); }
  });
  flagBtn.addEventListener("click", () => {
    if (flagged.has(currentIndex)) flagged.delete(currentIndex); else flagged.add(currentIndex);
    renderQuestion();
  });
  submitBtn.addEventListener("click", () => submitTest());

  async function doAutosave() {
    try {
      await apiFetch("/api/tests/autosave", {
        method: "POST",
        body: JSON.stringify({ attempt_id: attemptId, answers }),
      });
      autosaveNote.textContent = `Saved ${new Date().toLocaleTimeString()}`;
    } catch (e) { /* non-critical */ }
  }
  setInterval(doAutosave, 30000);

  function formatTime(sec) {
    const m = Math.floor(sec / 60).toString().padStart(2, "0");
    const s = Math.floor(sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }
  if (timeLimitMinutes && timerEl) {
    timerText.textContent = formatTime(remainingSeconds);
    setInterval(() => {
      remainingSeconds--;
      timerText.textContent = formatTime(Math.max(remainingSeconds, 0));
      timerEl.classList.toggle("amber", remainingSeconds <= 300 && remainingSeconds > 60);
      timerEl.classList.toggle("danger", remainingSeconds <= 60);
      if (remainingSeconds <= 0) submitTest();
    }, 1000);
  }

  async function submitTest() {
    if (submitting) return;
    const unanswered = questions.filter((q) => !answers[q.id] || !String(answers[q.id]).trim()).length;
    if (unanswered > 0 && (!timeLimitMinutes || remainingSeconds > 0)) {
      if (!confirm(`You have ${unanswered} unanswered question(s). Submit anyway?`)) return;
    }
    submitting = true;
    submitBtn.disabled = true;
    submitBtn.classList.add("loading");

    const timeTakenSec = Math.round((Date.now() - startTime) / 1000);
    try {
      const resp = await apiFetch("/api/tests/submit", {
        method: "POST",
        body: JSON.stringify({ attempt_id: attemptId, answers, time_taken_sec: timeTakenSec }),
      });
      const data = await resp.json();
      if (data.attempt_id) {
        window.location.href = `/tests/results/${data.attempt_id}`;
      } else {
        alert(data.error || "Could not submit. Please try again.");
        submitting = false;
        submitBtn.disabled = false;
        submitBtn.classList.remove("loading");
      }
    } catch (e) {
      alert("Connection error submitting the test. Please try again.");
      submitting = false;
      submitBtn.disabled = false;
      submitBtn.classList.remove("loading");
    }
  }

  renderQuestion();
})();
