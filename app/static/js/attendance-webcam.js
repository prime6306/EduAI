/**
 * attendance-webcam.js — Group Photo dropzone/upload and Live Webcam
 * capture flows for the teacher Attendance page.
 */
(function () {
  "use strict";

  const STATUS_ICON = {
    marked: "check-circle-2",
    spoof: "x-circle",
    duplicate: "clock",
    unknown: "help-circle",
    antispoof_unavailable: "alert-triangle",
  };

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function resultLine(r) {
    const icon = STATUS_ICON[r.status] || "info";
    let label;
    if (r.status === "marked") label = `${escapeHtml(r.name)} — Marked (Total: ${r.total_attendance})`;
    else if (r.status === "spoof") label = `${escapeHtml(r.name || "Unknown")} — Spoof detected`;
    else if (r.status === "duplicate") label = `${escapeHtml(r.name)} — Already marked ${r.seconds_elapsed}s ago`;
    else if (r.status === "unknown") label = "Unknown face — not enrolled";
    else if (r.status === "antispoof_unavailable") label = `${escapeHtml(r.name || "")} — Anti-spoof model unavailable`;
    else label = r.label || r.status;

    const timingBits = [];
    if (r.timings) {
      if (r.timings.recognition_ms !== undefined) timingBits.push(`recog ${r.timings.recognition_ms}ms`);
      if (r.timings.antispoof_ms !== undefined) timingBits.push(`spoof-check ${r.timings.antispoof_ms}ms`);
      if (r.timings.db_ms !== undefined) timingBits.push(`db ${r.timings.db_ms}ms`);
    }

    return `<div class="result-row ${r.status}">
      <svg class="icon icon-16" aria-hidden="true"><use href="/static/icons/lucide.svg#${icon}"></use></svg>
      <span>${label}</span>
      ${timingBits.length ? `<span class="r-timings">${timingBits.join(" · ")}</span>` : ""}
    </div>`;
  }

  const dropzone = document.getElementById("groupDropzone");
  const fileInput = document.getElementById("groupFileInput");
  const previewArea = document.getElementById("groupPreviewArea");
  const previewImg = document.getElementById("groupPreviewImg");
  const fileMeta = document.getElementById("groupFileMeta");
  const processBtn = document.getElementById("processPhotoBtn");
  const resultsEl = document.getElementById("groupResults");
  let selectedFile = null;

  if (dropzone) {
    dropzone.addEventListener("click", () => fileInput.click());
    ["dragover", "dragenter"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
    );
    dropzone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });
  }

  function handleFile(file) {
    if (file.size > 10 * 1024 * 1024) {
      alert("File is larger than 10MB. Please choose a smaller photo.");
      return;
    }
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    fileMeta.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`;
    previewArea.style.display = "block";
    resultsEl.innerHTML = "";
  }

  if (processBtn) {
    processBtn.addEventListener("click", async () => {
      if (!selectedFile) return;
      processBtn.disabled = true;
      processBtn.classList.add("loading");
      resultsEl.innerHTML = "";

      const formData = new FormData();
      formData.append("photo", selectedFile);

      try {
        const resp = await apiFetch("/api/attendance/mark-photo", { method: "POST", body: formData });
        const data = await resp.json();
        if (!resp.ok) {
          resultsEl.innerHTML = `<div class="alert alert-danger">${escapeHtml(data.error || "Something went wrong.")}</div>`;
          return;
        }
        if (!data.results || !data.results.length) {
          resultsEl.innerHTML = `<div class="alert alert-info">${escapeHtml(data.message || "No faces detected.")}</div>`;
          return;
        }
        resultsEl.innerHTML = data.results.map(resultLine).join("");
      } catch (e) {
        resultsEl.innerHTML = `<div class="alert alert-danger">Connection error. Please try again.</div>`;
      } finally {
        processBtn.disabled = false;
        processBtn.classList.remove("loading");
      }
    });
  }

  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("webcamCanvas");
  const captureBtn = document.getElementById("captureBtn");
  const retakeBtn = document.getElementById("retakeBtn");
  const verifyBtn = document.getElementById("verifyBtn");
  const webcamResult = document.getElementById("webcamResult");
  const cooldownNote = document.getElementById("cooldownNote");

  let stream = null;

  async function startWebcam() {
    if (!video) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = stream;
    } catch (e) {
      cooldownNote.textContent = "Couldn't access the camera. Check browser permissions.";
    }
  }

  document.querySelectorAll('.tab-btn[data-tab="webcam"]').forEach((btn) => {
    btn.addEventListener("click", () => { if (!stream) startWebcam(); }, { once: true });
  });

  if (captureBtn) {
    captureBtn.addEventListener("click", () => {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      video.style.display = "none";
      canvas.style.display = "block";
      captureBtn.style.display = "none";
      retakeBtn.style.display = "inline-flex";
      verifyBtn.style.display = "inline-flex";
      webcamResult.innerHTML = "";
    });
  }

  if (retakeBtn) {
    retakeBtn.addEventListener("click", () => {
      video.style.display = "block";
      canvas.style.display = "none";
      captureBtn.style.display = "inline-flex";
      retakeBtn.style.display = "none";
      verifyBtn.style.display = "none";
      webcamResult.innerHTML = "";
    });
  }

  if (verifyBtn) {
    verifyBtn.addEventListener("click", () => {
      verifyBtn.disabled = true;
      verifyBtn.classList.add("loading");
      canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");
        try {
          const resp = await apiFetch("/api/attendance/mark-webcam", { method: "POST", body: formData });
          const data = await resp.json();
          webcamResult.innerHTML = resultLine(data);
          if (data.status === "duplicate") {
            startCooldownCountdown(15 - (data.seconds_elapsed || 0));
          }
        } catch (e) {
          webcamResult.innerHTML = `<div class="alert alert-danger">Connection error. Please try again.</div>`;
        } finally {
          verifyBtn.disabled = false;
          verifyBtn.classList.remove("loading");
        }
      }, "image/jpeg", 0.9);
    });
  }

  function startCooldownCountdown(seconds) {
    let remaining = Math.max(seconds, 0);
    cooldownNote.textContent = `Re-capture available in ${remaining}s`;
    const interval = setInterval(() => {
      remaining--;
      if (remaining <= 0) {
        cooldownNote.textContent = "";
        clearInterval(interval);
      } else {
        cooldownNote.textContent = `Re-capture available in ${remaining}s`;
      }
    }, 1000);
  }
})();
