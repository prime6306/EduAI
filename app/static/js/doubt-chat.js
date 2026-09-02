/**
 * doubt-chat.js — Doubt Solver chat interface.
 * Streams /api/doubt/chat via fetch + ReadableStream (SSE-formatted body,
 * not a native EventSource, since EventSource can't POST). Renders
 * markdown, code highlighting, and KaTeX math on completed messages.
 */
(function () {
  "use strict";

  const messagesEl = document.getElementById("chatMessages");
  const emptyState = document.getElementById("emptyState");
  const input = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const subjectInput = document.getElementById("subjectInput");
  const levelSelect = document.getElementById("levelSelect");
  const newConversationBtn = document.getElementById("newConversationBtn");
  const recentList = document.getElementById("recentList");

  let conversationId = null;
  let sending = false;

  function renderMarkdown(el, text) {
    if (typeof marked !== "undefined") {
      el.innerHTML = marked.parse(text);
    } else {
      el.textContent = text;
    }
    if (typeof hljs !== "undefined") {
      el.querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
    }
    if (typeof renderMathInElement !== "undefined") {
      renderMathInElement(el, {
        delimiters: [
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$$", right: "$$", display: true },
        ],
        throwOnError: false,
      });
    }
  }

  function addMessage(role, text) {
    emptyState.style.display = "none";
    const row = document.createElement("div");
    row.className = `msg-row ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    if (role === "user") {
      bubble.textContent = text;
    } else {
      renderMarkdown(bubble, text);
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function addTypingBubble() {
    const row = document.createElement("div");
    row.className = "msg-row assistant";
    row.innerHTML = '<div class="msg-bubble"><div class="msg-typing"><span></span><span></span><span></span></div></div>';
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return row.querySelector(".msg-bubble");
  }

  async function sendMessage(text) {
    if (!text.trim() || sending) return;
    sending = true;
    sendBtn.disabled = true;

    addMessage("user", text);
    input.value = "";
    autosize();

    const bubble = addTypingBubble();
    let accumulated = "";
    let firstChunk = true;

    try {
      const resp = await apiFetch("/api/doubt/chat", {
        method: "POST",
        body: JSON.stringify({
          message: text,
          conversation_id: conversationId,
          subject: subjectInput.value.trim(),
          level: levelSelect.value,
        }),
      });

      if (!resp.ok || !resp.body) {
        bubble.textContent = "Something went wrong reaching the AI tutor. Please try again.";
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
          const line = part.replace(/^data:\s*/, "").trim();
          if (!line) continue;
          let payload;
          try {
            payload = JSON.parse(line);
          } catch (e) {
            continue;
          }

          if (payload.error) {
            bubble.innerHTML = "";
            bubble.textContent = payload.error;
            return;
          }
          if (payload.delta) {
            if (firstChunk) {
              bubble.innerHTML = "";
              firstChunk = false;
            }
            accumulated += payload.delta;
            bubble.textContent = accumulated;
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }
          if (payload.done) {
            conversationId = payload.conversation_id;
            renderMarkdown(bubble, accumulated);
            refreshRecentList();
          }
        }
      }
    } catch (err) {
      bubble.textContent = "Connection lost. Please try again.";
    } finally {
      sending = false;
      sendBtn.disabled = false;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  }
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });
  sendBtn.addEventListener("click", () => sendMessage(input.value));

  document.querySelectorAll(".starter-pill").forEach((pill) => {
    pill.addEventListener("click", () => sendMessage(pill.getAttribute("data-q")));
  });

  newConversationBtn.addEventListener("click", () => {
    conversationId = null;
    messagesEl.querySelectorAll(".msg-row").forEach((el) => el.remove());
    emptyState.style.display = "flex";
    input.focus();
  });

  async function refreshRecentList() {
    try {
      const resp = await apiFetch("/api/doubt/history");
      if (!resp.ok) return;
      const items = await resp.json();
      recentList.innerHTML = "";
      if (!items.length) {
        recentList.innerHTML = '<div class="chat-recent-empty">No conversations yet.</div>';
        return;
      }
      items.slice(0, 5).forEach((c) => {
        const div = document.createElement("div");
        div.className = "chat-recent-item";
        div.setAttribute("data-id", c.id);
        div.textContent = c.title;
        div.addEventListener("click", () => loadConversation(c.id));
        recentList.appendChild(div);
      });
    } catch (e) { /* non-critical */ }
  }

  async function loadConversation(id) {
    try {
      const resp = await apiFetch(`/api/doubt/conversation/${id}`);
      if (!resp.ok) return;
      const data = await resp.json();
      conversationId = data.id;
      subjectInput.value = data.subject || "";
      levelSelect.value = data.level || "Intermediate";
      messagesEl.querySelectorAll(".msg-row").forEach((el) => el.remove());
      if (data.messages.length) {
        emptyState.style.display = "none";
        data.messages.forEach((m) => addMessage(m.role, m.content));
      } else {
        emptyState.style.display = "flex";
      }
    } catch (e) { /* non-critical */ }
  }

  document.querySelectorAll(".chat-recent-item[data-id]").forEach((el) => {
    el.addEventListener("click", () => loadConversation(el.getAttribute("data-id")));
  });
})();
