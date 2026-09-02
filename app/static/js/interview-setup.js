/**
 * interview-setup.js — shows the same "this takes a little while" loading
 * overlay used by Study Material's form (see study/form.html) while the
 * start form kicks off a real multi-call, dual-LLM analysis pipeline.
 */
(function () {
  const form = document.getElementById("startInterviewForm");
  if (!form) return;

  const submitBtn = document.getElementById("startInterviewBtn");
  const overlay = document.getElementById("interviewLoadingOverlay");

  form.addEventListener("submit", function () {
    const jdText = form.querySelector("[name=jd_text]").value.trim();
    const resumeText = form.querySelector("[name=resume_text]").value.trim();
    const jdFile = form.querySelector("[name=jd_file]").files.length > 0;
    const resumeFile = form.querySelector("[name=resume_file]").files.length > 0;

    if ((!jdText && !jdFile) || (!resumeText && !resumeFile)) {
      return; // let the server-side validation message explain it
    }

    submitBtn.disabled = true;
    if (overlay) overlay.classList.add("open");
  });
})();
