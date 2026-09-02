/**
 * test-builder.js — the three question-add methods, drag-to-reorder,
 * and publish for the Custom Test Creator builder page.
 */
(function () {
  "use strict";

  const testId = window.TEST_ID;

  document.querySelectorAll(".builder-method-tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".builder-method-tabs button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".builder-method-panel").forEach((p) => p.classList.remove("active"));
      document.getElementById(`method-${btn.getAttribute("data-method")}`).classList.add("active");
    });
  });

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function questionListItemHtml(q, index) {
    return `<div class="test-question-item" draggable="true" data-id="${q.id}">
      <svg class="icon icon-16 drag-handle" aria-hidden="true"><use href="/static/icons/lucide.svg#menu"></use></svg>
      <div class="tq-body">
        <div class="tq-text">${index}. ${escapeHtml(q.text)}</div>
        <span class="badge badge-neutral">${q.type.toUpperCase()}</span>
        <span class="badge badge-accent">${q.marks} marks</span>
      </div>
      <div class="tq-actions">
        <button class="btn-icon" data-delete-q="${q.id}"><svg class="icon icon-14" aria-hidden="true"><use href="/static/icons/lucide.svg#trash-2"></use></svg></button>
      </div>
    </div>`;
  }

  function appendQuestionToList(q) {
    const list = document.getElementById("questionList");
    const div = document.createElement("div");
    div.innerHTML = questionListItemHtml(q, list.children.length + 1);
    const el = div.firstElementChild;
    list.appendChild(el);
    attachDragHandlers(el);
    attachDeleteHandler(el.querySelector("[data-delete-q]"));
  }

  const aiGenerateBtn = document.getElementById("aiGenerateBtn");
  const aiCandidates = document.getElementById("aiCandidates");

  aiGenerateBtn.addEventListener("click", async () => {
    aiGenerateBtn.disabled = true;
    aiGenerateBtn.classList.add("loading");
    aiCandidates.innerHTML = "";

    const topic = document.getElementById("aiTopic").value;
    const q_type = document.getElementById("aiType").value;
    const n = parseInt(document.getElementById("aiCount").value) || 5;

    try {
      const resp = await apiFetch(`/api/tests/${testId}/questions/generate`, {
        method: "POST",
        body: JSON.stringify({ topic, q_type, n }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        aiCandidates.innerHTML = `<div class="alert alert-danger">${escapeHtml(data.error || "Generation failed.")}</div>`;
        return;
      }
      if (!data.candidates.length) {
        aiCandidates.innerHTML = `<div class="alert alert-info">No questions generated. Try a different topic.</div>`;
        return;
      }
      data.candidates.forEach((c) => {
        const card = document.createElement("div");
        card.className = "candidate-card";
        card.innerHTML = `
          <div class="c-text">${escapeHtml(c.text)} <span class="badge badge-accent">${c.marks} marks</span></div>
          <div class="c-actions">
            <button class="btn-primary btn-sm" data-accept>Accept</button>
            <button class="btn-ghost btn-sm" data-reject>Reject</button>
          </div>
        `;
        card.querySelector("[data-accept]").addEventListener("click", async () => {
          const resp2 = await apiFetch(`/api/tests/${testId}/questions/add`, {
            method: "POST",
            body: JSON.stringify({ question: c }),
          });
          const data2 = await resp2.json();
          if (resp2.ok) {
            appendQuestionToList(data2.question);
            card.remove();
          }
        });
        card.querySelector("[data-reject]").addEventListener("click", () => card.remove());
        aiCandidates.appendChild(card);
      });
    } catch (e) {
      aiCandidates.innerHTML = `<div class="alert alert-danger">Connection error.</div>`;
    } finally {
      aiGenerateBtn.disabled = false;
      aiGenerateBtn.classList.remove("loading");
    }
  });

  const bankSearchBtn = document.getElementById("bankSearchBtn");
  const bankResults = document.getElementById("bankResults");
  const addFromBankBtn = document.getElementById("addFromBankBtn");

  async function searchBank() {
    const subject = document.getElementById("bankSearchInput").value;
    const resp = await apiFetch(`/api/question-bank/search?subject=${encodeURIComponent(subject)}`);
    const data = await resp.json();
    bankResults.innerHTML = "";
    if (!data.length) {
      bankResults.innerHTML = `<p class="text-sm" style="color:var(--text-muted);">No matching questions found.</p>`;
      addFromBankBtn.disabled = true;
      return;
    }
    data.forEach((q) => {
      const row = document.createElement("label");
      row.className = "bank-picker-row";
      row.innerHTML = `<input type="checkbox" value="${q.id}"> ${escapeHtml(q.text)} <span class="badge badge-neutral">${q.type.toUpperCase()}</span>`;
      bankResults.appendChild(row);
    });
    addFromBankBtn.disabled = false;
  }
  bankSearchBtn.addEventListener("click", searchBank);

  addFromBankBtn.addEventListener("click", async () => {
    const ids = Array.from(bankResults.querySelectorAll("input:checked")).map((c) => c.value);
    if (!ids.length) return;
    addFromBankBtn.disabled = true;
    const resp = await apiFetch(`/api/tests/${testId}/questions/from-bank`, {
      method: "POST",
      body: JSON.stringify({ question_ids: ids }),
    });
    const data = await resp.json();
    (data.added || []).forEach((q) => appendQuestionToList(q));
    addFromBankBtn.disabled = false;
  });

  let manualType = "mcq";
  document.querySelectorAll(".manual-type-select button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".manual-type-select button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      manualType = btn.getAttribute("data-mtype");
      const isMcqLike = manualType === "mcq";
      document.getElementById("manualOptionsWrap").style.display = isMcqLike ? "block" : "none";
      document.getElementById("manualAnswerWrap").style.display = isMcqLike ? "none" : "block";
    });
  });

  document.getElementById("addManualBtn").addEventListener("click", async () => {
    const text = document.getElementById("manualText").value.trim();
    if (!text) { alert("Question text is required."); return; }

    const question = {
      text, type: manualType,
      marks: parseInt(document.getElementById("manualMarks").value) || 1,
      explanation: document.getElementById("manualExplanation").value,
    };

    if (manualType === "mcq") {
      const options = [0, 1, 2, 3].map((i) => document.getElementById(`manualOpt${i}`).value.trim());
      if (options.some((o) => !o)) { alert("All 4 options are required."); return; }
      question.options = options;
      question.correct_answer = options[parseInt(document.getElementById("manualCorrectIdx").value)];
    } else if (manualType === "tf") {
      question.options = ["True", "False"];
      question.correct_answer = document.getElementById("manualAnswer").value.trim() || "True";
    } else {
      question.correct_answer = document.getElementById("manualAnswer").value.trim();
    }

    const resp = await apiFetch(`/api/tests/${testId}/questions/add`, {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    const data = await resp.json();
    if (resp.ok) {
      appendQuestionToList(data.question);
      document.getElementById("manualText").value = "";
      [0, 1, 2, 3].forEach((i) => { const el = document.getElementById(`manualOpt${i}`); if (el) el.value = ""; });
      const ansEl = document.getElementById("manualAnswer");
      if (ansEl) ansEl.value = "";
    } else {
      alert(data.error || "Could not add question.");
    }
  });

  function attachDeleteHandler(btn) {
    btn.addEventListener("click", async () => {
      const qid = btn.getAttribute("data-delete-q");
      const resp = await apiFetch(`/api/tests/${testId}/questions/delete`, {
        method: "POST",
        body: JSON.stringify({ question_id: qid }),
      });
      if (resp.ok) btn.closest(".test-question-item").remove();
    });
  }
  document.querySelectorAll("[data-delete-q]").forEach(attachDeleteHandler);

  let draggedEl = null;
  function attachDragHandlers(el) {
    el.addEventListener("dragstart", () => { draggedEl = el; el.classList.add("dragging"); });
    el.addEventListener("dragend", () => { el.classList.remove("dragging"); saveOrder(); });
    el.addEventListener("dragover", (e) => {
      e.preventDefault();
      const list = document.getElementById("questionList");
      const after = getDragAfterElement(list, e.clientY);
      if (!draggedEl) return;
      if (after == null) list.appendChild(draggedEl);
      else list.insertBefore(draggedEl, after);
    });
  }
  function getDragAfterElement(container, y) {
    const els = [...container.querySelectorAll(".test-question-item:not(.dragging)")];
    return els.reduce((closest, child) => {
      const box = child.getBoundingClientRect();
      const offset = y - box.top - box.height / 2;
      if (offset < 0 && offset > closest.offset) return { offset, element: child };
      return closest;
    }, { offset: -Infinity }).element;
  }
  document.querySelectorAll(".test-question-item").forEach(attachDragHandlers);

  async function saveOrder() {
    const ids = Array.from(document.querySelectorAll(".test-question-item")).map((el) => el.getAttribute("data-id"));
    await apiFetch(`/api/tests/${testId}/questions/reorder`, {
      method: "POST",
      body: JSON.stringify({ ordered_ids: ids }),
    });
  }

  document.getElementById("publishBtn").addEventListener("click", async () => {
    const assignedTo = Array.from(document.getElementById("assignedTo").selectedOptions).map((o) => o.value);
    const payload = {
      available_from: document.getElementById("availableFrom").value || null,
      available_until: document.getElementById("availableUntil").value || null,
      assigned_to: assignedTo,
      shuffle_questions: document.getElementById("shuffleQuestions").checked,
      shuffle_options: document.getElementById("shuffleOptions").checked,
    };
    const resp = await apiFetch(`/api/tests/${testId}/publish`, { method: "POST", body: JSON.stringify(payload) });
    const data = await resp.json();
    if (resp.ok) {
      alert("Test published!");
      window.location.href = `/tests/${testId}/results`;
    } else {
      alert(data.error || "Could not publish.");
    }
  });
})();
