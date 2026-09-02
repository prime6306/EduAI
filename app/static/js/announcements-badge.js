/**
 * announcements-badge.js — polls /api/announcements/unread-count every 60s
 * and updates the sidebar nav badge. Student-only (included conditionally
 * by base.html).
 */
(function () {
  "use strict";

  const badge = document.getElementById("announcementUnreadBadge");
  if (!badge) return;

  async function refresh() {
    try {
      const resp = await apiFetch("/api/announcements/unread-count");
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.count > 0) {
        badge.textContent = data.count;
        badge.style.display = "inline-flex";
      } else {
        badge.style.display = "none";
      }
    } catch (e) { /* non-critical */ }
  }

  refresh();
  setInterval(refresh, 60000);
})();
