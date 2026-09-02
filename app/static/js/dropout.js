/**
 * dropout.js — prediction form (gauge chart result) + teacher batch CSV mode.
 */
(function () {
  "use strict";

  document.querySelectorAll('.slider-row input[type="range"]').forEach((input) => {
    const label = document.getElementById(`v_${input.name}`);
    if (label) input.addEventListener("input", () => { label.textContent = input.value; });
  });

  const RISK_COLORS = { Low: "#16A34A", Medium: "#D97706", High: "#DC2626" };
  let gaugeChart = null;

  function renderGauge(probability, riskLevel) {
    const ctx = document.getElementById("gaugeChart");
    const color = RISK_COLORS[riskLevel] || "#6B7280";
    if (gaugeChart) gaugeChart.destroy();
    gaugeChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        datasets: [{
          data: [probability, 100 - probability],
          backgroundColor: [color, "#E9EAEC"],
          borderWidth: 0,
        }],
      },
      options: {
        cutout: "75%",
        circumference: 270,
        rotation: 225,
        plugins: { tooltip: { enabled: false } },
        animation: { animateRotate: true, duration: 700 },
      },
    });
  }

  const predictForm = document.getElementById("predictForm");
  if (predictForm) {
    predictForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("predictBtn");
      btn.disabled = true;
      btn.classList.add("loading");

      const payload = {};
      predictForm.querySelectorAll("[name]").forEach((el) => { payload[el.name] = el.value; });
      payload.model_choice = document.getElementById("modelChoice").value;

      try {
        const resp = await apiFetch("/api/dropout/predict", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          alert(data.error || `Request failed (${resp.status}). Please try again.`);
          return;
        }
        const data = await resp.json();

        document.getElementById("emptyResultCard").style.display = "none";
        document.getElementById("resultCard").style.display = "block";
        document.getElementById("gaugePct").textContent = `${data.probability}%`;
        renderGauge(data.probability, data.risk_level);

        const badge = document.getElementById("riskBadge");
        badge.textContent = `${data.risk_level} Risk`;
        badge.className = "badge risk-badge-lg badge-" +
          (data.risk_level === "High" ? "danger" : data.risk_level === "Medium" ? "warning" : "success");

        const factorsSection = document.getElementById("factorsSection");
        const factorChips = document.getElementById("factorChips");
        if (data.top_factors && data.top_factors.length) {
          factorsSection.style.display = "block";
          factorChips.innerHTML = data.top_factors.map((f) => `<span class="factor-chip">${f}</span>`).join("");
        } else {
          factorsSection.style.display = "none";
        }

        const recList = document.getElementById("recommendationList");
        recList.innerHTML = data.recommendations.map((r) => `
          <div class="recommendation-item">
            <svg class="icon icon-16" aria-hidden="true"><use href="/static/icons/lucide.svg#check-circle-2"></use></svg>
            <span>${r}</span>
          </div>
        `).join("");
      } catch (err) {
        alert("Network error. Please check your connection and try again.");
      } finally {
        btn.disabled = false;
        btn.classList.remove("loading");
      }
    });
  }

  const batchDropzone = document.getElementById("batchDropzone");
  const batchFileInput = document.getElementById("batchFileInput");
  const batchFileName = document.getElementById("batchFileName");
  const batchSubmitBtn = document.getElementById("batchSubmitBtn");
  let batchFile = null;
  let lastBatchCsv = "";

  if (batchDropzone) {
    batchDropzone.addEventListener("click", () => batchFileInput.click());
    batchFileInput.addEventListener("change", () => {
      if (batchFileInput.files[0]) {
        batchFile = batchFileInput.files[0];
        batchFileName.textContent = batchFile.name;
        batchSubmitBtn.disabled = false;
      }
    });

    batchSubmitBtn.addEventListener("click", async () => {
      if (!batchFile) return;
      batchSubmitBtn.disabled = true;
      batchSubmitBtn.classList.add("loading");

      const formData = new FormData();
      formData.append("csv_file", batchFile);

      try {
        const resp = await apiFetch("/api/dropout/batch", { method: "POST", body: formData });
        const data = await resp.json();
        if (!resp.ok) {
          alert(data.error || "Batch prediction failed.");
          return;
        }
        lastBatchCsv = data.csv;
        renderBatchResults(data);
      } catch (err) {
        alert("Connection error. Please try again.");
      } finally {
        batchSubmitBtn.disabled = false;
        batchSubmitBtn.classList.remove("loading");
      }
    });
  }

  function renderBatchResults(data) {
    document.getElementById("batchResultCard").style.display = "block";
    const tbody = document.getElementById("batchResultsBody");
    tbody.innerHTML = data.results.map((r) => {
      const gc = r.risk_level === "High" ? "danger" : r.risk_level === "Medium" ? "warning" : "success";
      return `<tr><td>${r.student_id || "—"}</td><td>${r.probability}%</td><td><span class="badge badge-${gc}">${r.risk_level}</span></td></tr>`;
    }).join("");

    new Chart(document.getElementById("batchPieChart"), {
      type: "doughnut",
      data: {
        labels: ["Low", "Medium", "High"],
        datasets: [{
          data: [data.risk_counts.Low, data.risk_counts.Medium, data.risk_counts.High],
          backgroundColor: [RISK_COLORS.Low, RISK_COLORS.Medium, RISK_COLORS.High],
        }],
      },
      options: { plugins: { legend: { display: true, position: "bottom" } } },
    });
  }

  const downloadBtn = document.getElementById("downloadBatchCsvBtn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
      if (!lastBatchCsv) return;
      const blob = new Blob([lastBatchCsv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "dropout_batch_results.csv";
      a.click();
      URL.revokeObjectURL(url);
    });
  }
})();
