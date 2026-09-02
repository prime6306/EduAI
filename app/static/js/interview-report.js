/**
 * interview-report.js — lets a teacher save a personalised comment on a
 * student's interview report (AJAX, CSRF-aware via apiFetch from app.js).
 */
(function () {
  const form = document.getElementById("teacherFeedbackForm");
  if (!form) return;

  const textarea = document.getElementById("teacherFeedbackInput");
  const saveBtn = document.getElementById("teacherFeedbackSave");
  const statusEl = document.getElementById("teacherFeedbackStatus");
  const displayBox = document.getElementById("teacherFeedbackDisplay");

  saveBtn.addEventListener("click", async function () {
    const comment = textarea.value.trim();
    if (!comment) {
      statusEl.textContent = "Write a comment first.";
      return;
    }
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving…";
    statusEl.textContent = "";

    try {
      const res = await apiFetch(form.dataset.feedbackUrl, {
        method: "POST",
        body: JSON.stringify({ comment: comment }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not save the comment.");

      if (displayBox) {
        displayBox.querySelector(".tf-comment").textContent = data.comment;
        displayBox.style.display = "block";
      }
      statusEl.textContent = "Saved.";
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = "Save Comment";
    }
  });
})();
