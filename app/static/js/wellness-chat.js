/**
 * wellness-chat.js — non-streaming chat (unlike Doubt Solver): a single
 * JSON round trip per message, since responses need to carry sentiment
 * and a crisis flag before being rendered.
 */
(function () {
  "use strict";

  const messagesEl = document.getElementById("chatMessages");
  const emptyState = document.getElementById("emptyState");
  const input = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");

  let sessionId = null;
  let sending = false;

  function renderMarkdown(text) {
    return typeof marked !== "undefined" ? marked.parse(text) : text;
  }

  function addMessage(role, text, sentiment) {
    emptyState.style.display = "none";
    const row = document.createElement("div");
    row.className = `msg-row ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    if (role === "user") {
      bubble.textContent = text;
      if (sentiment) {
        const tag = document.createElement("div");
        tag.className = `sentiment-tag ${sentiment.label}`;
        tag.textContent = `${sentiment.label} - ${sentiment.compound.toFixed(2)}`;
        bubble.appendChild(document.createElement("br"));
        bubble.appendChild(tag);
      }
    } else {
      bubble.innerHTML = renderMarkdown(text);
    }

    row.appendChild(bubble);
    messagesEl.appendChild(row);
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
    input.value = "";
    autosize();

    addMessage("user", text, null);
    const typingBubble = addTypingBubble();

    try {
      const resp = await apiFetch("/api/wellness/chat", {
        method: "POST",
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      const data = await resp.json();

      typingBubble.closest(".msg-row").remove();

      if (!resp.ok) {
        addMessage("assistant", data.error || "Something went wrong. Please try again.", null);
        return;
      }

      sessionId = data.session_id;
      const bubble = addMessage("assistant", data.reply, null);
      if (data.crisis) {
        bubble.style.borderColor = "var(--danger-border)";
        bubble.style.background = "var(--danger-bg)";
      }
    } catch (err) {
      typingBubble.closest(".msg-row").remove();
      addMessage("assistant", "Connection error. Please try again.", null);
    } finally {
      sending = false;
      sendBtn.disabled = false;
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
})();
