/**
 * chart-init.js — Chart.js global defaults. Loaded on any page that draws
 * a chart, after the Chart.js library itself and before page-specific
 * chart-building code.
 */
(function () {
  "use strict";
  if (typeof Chart === "undefined") return;

  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 13;
  Chart.defaults.color = "#6B7280";
  Chart.defaults.borderColor = "#E5E7EB";
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = "#111827";
  Chart.defaults.plugins.tooltip.titleColor = "#F9FAFB";
  Chart.defaults.plugins.tooltip.bodyColor = "#D1D5DB";
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.displayColors = false;

  window.EduAIChartColors = {
    accent: "#1D4ED8",
    accentSoft: "rgba(29, 78, 216, 0.8)",
    success: "#16A34A",
    warning: "#D97706",
    danger: "#DC2626",
    border: "#E5E7EB",
    surface3: "#E9EAEC",
  };
})();
