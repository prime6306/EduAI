/**
 * rag-chat.js — PDF Q&A interface: document upload, selection, and
 * grounded question/answer flow with hallucination-score badges.
 */
(function () {
  "use strict";

  const uploadBtn = document.getElementById("uploadBtn");
  const fileInput = document.getElementById("fileInput");
  const docList = document.getElementById("docList");
  const selectedDocTitle = document.getElementById("selectedDocTitle");
  const qaArea = document.getElementById("qaArea");
  const emptyState = document.getElementById("ragEmptyState");
  const inputBar = document.getElementById("ragInputBar");
  const questionInput = document.getElementById("ragQuestionInput");
  const askBtn = document.getElementById("ragAskBtn");

  let selectedPdfId = null;
  let selectedPdfName = null;
  let asking = false;

  function selectDocument(pdfId, name, el) {
    selectedPdfId = pdfId;
    selectedPdfName = name;
    document.querySelectorAll(".rag-doc-item").forEach((d) => d.classList.remove("selected"));
    if (el) el.classList.add("selected");
    selectedDocTitle.textContent = name;
    qaArea.querySelectorAll(".rag-qa-pair").forEach((el2) => el2.remove());
    emptyState.style.display = "none";
    inputBar.style.display = "block";
    questionInput.placeholder = `Ask a question about ${name}...`;
    questionInput.focus();
  }

  document.querySelectorAll(".rag-doc-item[data-id]").forEach((el) => {
    el.addEventListener("click", () => selectDocument(el.getAttribute("data-id"), el.getAttribute("data-name"), el));
  });

  // ── Upload ────────────────────────────────────────────────────
  uploadBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadBtn.classList.add("loading");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await apiFetch("/rag/api/rag/upload", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) {
        alert(data.error || "Upload failed.");
        return;
      }
      const item = document.createElement("div");
      item.className = "rag-doc-item";
      item.setAttribute("data-id", data.pdf_id);
      item.setAttribute("data-name", data.filename);
      item.innerHTML = `<div class="doc-name">${escapeHtml(data.filename)}</div><div class="doc-meta">${data.chunk_count} chunks · just now</div>`;
      item.addEventListener("click", () => selectDocument(data.pdf_id, data.filename, item));

      const emptyMsg = docList.querySelector(".rag-doc-empty");
      if (emptyMsg) emptyMsg.remove();
      docList.prepend(item);
      selectDocument(data.pdf_id, data.filename, item);
    } catch (e) {
      alert("Upload failed. Please try again.");
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.classList.remove("loading");
      fileInput.value = "";
    }
  });

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Ask ───────────────────────────────────────────────────────
  async function askQuestion(question) {
    if (!question.trim() || !selectedPdfId || asking) return;
    asking = true;
    askBtn.disabled = true;
    askBtn.classList.add("loading");
    questionInput.value = "";

    const pair = document.createElement("div");
    pair.className = "rag-qa-pair";
    pair.innerHTML = `
      <div class="rag-question">${escapeHtml(question)}</div>
      <div class="rag-answer"><div class="msg-typing"><span></span><span></span><span></span></div></div>
    `;
    qaArea.appendChild(pair);
    qaArea.scrollTop = qaArea.scrollHeight;

    try {
      const resp = await apiFetch("/rag/api/rag/ask", {
        method: "POST",
        body: JSON.stringify({ pdf_id: selectedPdfId, question }),
      });
      const data = await resp.json();
      const answerEl = pair.querySelector(".rag-answer");

      if (!resp.ok) {
        answerEl.textContent = data.error || "Something went wrong.";
        return;
      }

      answerEl.innerHTML = typeof marked !== "undefined" ? marked.parse(data.answer) : escapeHtml(data.answer);

      const groundingRow = document.createElement("div");
      groundingRow.className = "grounding-row";
      groundingRow.innerHTML = `<span class="grounding-badge ${data.verdict}">Grounding: ${data.grounding_score}/100 — ${verdictLabel(data.verdict)}</span>`;
      answerEl.appendChild(groundingRow);

      if (data.related_questions && data.related_questions.length) {
        const rq = document.createElement("div");
        rq.className = "related-questions";
        rq.innerHTML = "Related: " + data.related_questions.map((q) =>
          `<span class="rq-item" data-q="${escapeHtml(q)}">${escapeHtml(q)}</span>`
        ).join(" ");
        rq.querySelectorAll(".rq-item").forEach((item) => {
          item.addEventListener("click", () => askQuestion(item.getAttribute("data-q")));
        });
        answerEl.appendChild(rq);
      }
    } catch (e) {
      pair.querySelector(".rag-answer").textContent = "Connection error. Please try again.";
    } finally {
      asking = false;
      askBtn.disabled = false;
      askBtn.classList.remove("loading");
      qaArea.scrollTop = qaArea.scrollHeight;
    }
  }

  function verdictLabel(v) {
    return { grounded: "Grounded ✓", partial: "Partial", hallucinated: "Hallucinated", off_topic: "Off-topic" }[v] || v;
  }

  askBtn.addEventListener("click", () => askQuestion(questionInput.value));
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") askQuestion(questionInput.value);
  });
})();
