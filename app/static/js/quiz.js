/**
 * quiz.js — quiz-taking interface. Standalone page (no app.js sidebar/menu
 * deps beyond apiFetch, which app.js also provides here).
 */
(function () {
  "use strict";

  const questions = JSON.parse(document.getElementById("questionsData").textContent);
  const quizId = window.QUIZ_ID;
  const timed = window.QUIZ_TIMED;
  const SECONDS_PER_QUESTION = 120;

  const answers = {}; // index -> selected option index
  const flagged = new Set();
  let currentIndex = 0;
  const startTime = Date.now();
  let remainingSeconds = questions.length * SECONDS_PER_QUESTION;
  let timerInterval = null;
  let submitting = false;

  const questionText = document.getElementById("questionText");
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

  function renderQuestion() {
    const q = questions[currentIndex];
    questionText.textContent = `Q${currentIndex + 1}. ${q.question}`;
    questionProgress.textContent = `Question ${currentIndex + 1} of ${questions.length}`;

    optionsContainer.innerHTML = "";
    q.options.forEach((optText, i) => {
      const div = document.createElement("div");
      div.className = "quiz-option" + (answers[currentIndex] === i ? " selected" : "");
      div.innerHTML = `<span class="radio-dot"></span><span class="option-text">${escapeHtml(optText)}</span>`;
      div.addEventListener("click", () => selectOption(i));
      optionsContainer.appendChild(div);
    });

    flagBtn.textContent = flagged.has(currentIndex) ? "Unflag this question" : "Flag this question";
    prevBtn.disabled = currentIndex === 0;
    nextBtn.textContent = currentIndex === questions.length - 1 ? "Finish" : "Next";

    renderPalette();
    renderProgressBar();
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function selectOption(optIndex) {
    answers[currentIndex] = optIndex;
    renderQuestion();
  }

  function renderPalette() {
    paletteGrid.innerHTML = "";
    questions.forEach((_, i) => {
      const cell = document.createElement("div");
      let cls = "palette-cell";
      if (i === currentIndex) cls += " current";
      if (answers[i] !== undefined) cls += " answered";
      if (flagged.has(i)) cls += " flagged";
      cell.className = cls;
      cell.textContent = i + 1;
      cell.addEventListener("click", () => { currentIndex = i; renderQuestion(); });
      paletteGrid.appendChild(cell);
    });
  }

  function renderProgressBar() {
    const answeredCount = Object.keys(answers).length;
    const pct = Math.round((answeredCount / questions.length) * 100);
    answeredProgress.style.width = pct + "%";
  }

  prevBtn.addEventListener("click", () => {
    if (currentIndex > 0) { currentIndex--; renderQuestion(); }
  });
  nextBtn.addEventListener("click", () => {
    if (currentIndex < questions.length - 1) { currentIndex++; renderQuestion(); }
    else { submitQuiz(); }
  });
  flagBtn.addEventListener("click", () => {
    if (flagged.has(currentIndex)) flagged.delete(currentIndex);
    else flagged.add(currentIndex);
    renderQuestion();
  });
  submitBtn.addEventListener("click", () => submitQuiz());

  // ── Timer ─────────────────────────────────────────────────────
  function formatTime(sec) {
    const m = Math.floor(sec / 60).toString().padStart(2, "0");
    const s = Math.floor(sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }

  if (timed && timerEl) {
    timerText.textContent = formatTime(remainingSeconds);
    timerInterval = setInterval(() => {
      remainingSeconds--;
      timerText.textContent = formatTime(Math.max(remainingSeconds, 0));
      timerEl.classList.toggle("amber", remainingSeconds <= 300 && remainingSeconds > 60);
      timerEl.classList.toggle("danger", remainingSeconds <= 60);
      if (remainingSeconds <= 0) {
        clearInterval(timerInterval);
        submitQuiz();
      }
    }, 1000);
  }

  // ── Submit ────────────────────────────────────────────────────
  async function submitQuiz() {
    if (submitting) return;

    const unanswered = questions.length - Object.keys(answers).length;
    if (unanswered > 0 && remainingSeconds > 0) {
      const proceed = confirm(`You have ${unanswered} unanswered question(s). Submit anyway?`);
      if (!proceed) return;
    }

    submitting = true;
    submitBtn.disabled = true;
    submitBtn.classList.add("loading");
    if (timerInterval) clearInterval(timerInterval);

    const timeTakenSec = Math.round((Date.now() - startTime) / 1000);

    try {
      const resp = await apiFetch("/api/quiz/submit", {
        method: "POST",
        body: JSON.stringify({ quiz_id: quizId, answers, time_taken_sec: timeTakenSec }),
      });
      const data = await resp.json();
      if (data.result_id) {
        window.location.href = `/quiz/results/${data.result_id}`;
      } else {
        alert(data.error || "Could not submit the quiz. Please try again.");
        submitting = false;
        submitBtn.disabled = false;
        submitBtn.classList.remove("loading");
      }
    } catch (e) {
      alert("Connection error submitting the quiz. Please try again.");
      submitting = false;
      submitBtn.disabled = false;
      submitBtn.classList.remove("loading");
    }
  }

  renderQuestion();
})();
