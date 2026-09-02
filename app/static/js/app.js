/**
 * app.js — global behaviour shared by every authenticated page:
 * mobile sidebar toggle, flash message auto-dismiss, user menu dropdown,
 * and a small fetch() wrapper that attaches the CSRF token automatically.
 */
(function () {
  "use strict";

  // ── CSRF-aware fetch ─────────────────────────────────────────
  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  window.apiFetch = function apiFetch(url, options = {}) {
    const opts = Object.assign({}, options);
    opts.headers = Object.assign(
      { "X-CSRFToken": getCsrfToken() },
      options.headers || {}
    );
    if (opts.body && !(opts.body instanceof FormData) && !opts.headers["Content-Type"]) {
      opts.headers["Content-Type"] = "application/json";
    }
    return fetch(url, opts);
  };

  // ── Mobile sidebar toggle ────────────────────────────────────
  const menuToggle = document.querySelector(".menu-toggle");
  const sidebar = document.querySelector(".sidebar");
  if (menuToggle && sidebar) {
    menuToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
    document.addEventListener("click", (e) => {
      if (
        sidebar.classList.contains("open") &&
        !sidebar.contains(e.target) &&
        !menuToggle.contains(e.target)
      ) {
        sidebar.classList.remove("open");
      }
    });
  }

  // ── User menu dropdown ───────────────────────────────────────
  const trigger = document.querySelector(".user-menu-trigger");
  const dropdown = document.querySelector(".user-menu-dropdown");
  if (trigger && dropdown) {
    trigger.addEventListener("click", (e) => {
      e.stopPropagation();
      dropdown.classList.toggle("open");
    });
    document.addEventListener("click", () => dropdown.classList.remove("open"));
  }

  // ── Flash message auto-dismiss ───────────────────────────────
  function dismissAlert(el) {
    el.classList.add("dismissing");
    setTimeout(() => el.remove(), 200);
  }

  document.querySelectorAll(".alert[data-flash]").forEach((el) => {
    const timer = setTimeout(() => dismissAlert(el), 5000);
    const btn = el.querySelector(".alert-dismiss");
    if (btn) {
      btn.addEventListener("click", () => {
        clearTimeout(timer);
        dismissAlert(el);
      });
    }
  });

  // ── Generic modal open/close helpers ─────────────────────────
  document.querySelectorAll("[data-modal-open]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const modal = document.getElementById(btn.getAttribute("data-modal-open"));
      if (modal) modal.classList.add("open");
    });
  });
  document.querySelectorAll("[data-modal-close]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const overlay = btn.closest(".modal-overlay");
      if (overlay) overlay.classList.remove("open");
    });
  });
  document.querySelectorAll(".modal-overlay").forEach((overlay) => {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) overlay.classList.remove("open");
    });
  });

  // ── Generic tab bar (data-tab buttons + #tab-<name> panels) ─────
  document.querySelectorAll(".tab-btn[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const group = btn.closest(".tab-bar");
      const name = btn.getAttribute("data-tab");
      group.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      const panel = document.getElementById(`tab-${name}`);
      if (panel) panel.classList.add("active");
    });
  });
})();
