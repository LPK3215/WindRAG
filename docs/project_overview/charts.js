/* WindRAG project overview - Chart.js config.
   Numeric data is generated into overview-data.js from the real reports. */
(function () {
  "use strict";

  var WD_DATA = window.WD_OVERVIEW_DATA;
  if (!WD_DATA) throw new Error("overview-data.js must load before charts.js");

  function css(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function pct(x, d) { return (x == null) ? "n/a" : (x * 100).toFixed(d == null ? 1 : d) + "%"; }

  function fillTables() {
    var lt = document.querySelector("#tblLayers tbody");
    if (lt && !lt.childElementCount) {
      lt.innerHTML = WD_DATA.layers.map(function (r) {
        return "<tr><td>" + r.name + "</td><td>" + r.count + "</td><td>" + pct(r.recall, 2) +
          "</td><td>" + pct(r.mrr, 2) + "</td><td>" + pct(r.refusal, 1) + "</td></tr>";
      }).join("");
    }
    var at = document.querySelector("#tblAblation tbody");
    if (at && !at.childElementCount) {
      at.innerHTML = WD_DATA.ablation.map(function (r) {
        return "<tr><td>" + r[0] + "</td><td>" + pct(r[1], 2) + "</td><td>" + pct(r[2], 2) +
          "</td><td>" + pct(r[3], 2) + "</td><td>" + pct(r[4], 2) + "</td></tr>";
      }).join("");
    }
    var pt = document.querySelector("#tblAlpha tbody");
    if (pt && !pt.childElementCount) {
      pt.innerHTML = WD_DATA.alpha.map(function (r) {
        return "<tr><td>" + r[0] + "</td><td>" + pct(r[1], 2) + "</td><td>" + pct(r[2], 2) +
          "</td><td>" + pct(r[3], 2) + "</td></tr>";
      }).join("");
    }
  }

  var charts = {};
  function gridOpts() {
    return {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: css("--text", "#16283b") } } },
      scales: {
        x: { ticks: { color: css("--text-soft", "#4a6178") }, grid: { color: css("--card-border", "rgba(0,0,0,.08)") } },
        y: { min: 0, max: 1, ticks: { color: css("--text-soft", "#4a6178"), callback: function (v) { return (v * 100) + "%"; } }, grid: { color: css("--card-border", "rgba(0,0,0,.08)") } }
      }
    };
  }

  function buildLayers() {
    if (charts.layers || !window.Chart) return;
    var withRecall = WD_DATA.layers.filter(function (r) { return r.recall != null; });
    var ctx = document.getElementById("chartLayers");
    if (!ctx) return;
    charts.layers = new Chart(ctx, {
      type: "bar",
      data: {
        labels: withRecall.map(function (r) { return r.name; }),
        datasets: [
          { label: "Recall@3", data: withRecall.map(function (r) { return r.recall; }), backgroundColor: "rgba(34,211,238,0.55)", borderColor: "#22d3ee", borderWidth: 1, borderRadius: 6 },
          { label: "MRR", data: withRecall.map(function (r) { return r.mrr; }), backgroundColor: "rgba(217,119,6,0.45)", borderColor: "#d97706", borderWidth: 1, borderRadius: 6 }
        ]
      },
      options: gridOpts()
    });
  }

  function buildAblation() {
    if (charts.ablation || !window.Chart) return;
    var ctx = document.getElementById("chartAblation");
    if (!ctx) return;
    charts.ablation = new Chart(ctx, {
      type: "bar",
      data: {
        labels: WD_DATA.ablation.map(function (r) { return r[0]; }),
        datasets: [
          { label: "Recall@k", data: WD_DATA.ablation.map(function (r) { return r[1]; }), backgroundColor: "rgba(34,211,238,0.55)", borderColor: "#22d3ee", borderWidth: 1, borderRadius: 6 },
          { label: "MRR", data: WD_DATA.ablation.map(function (r) { return r[2]; }), backgroundColor: "rgba(52,211,153,0.45)", borderColor: "#34d399", borderWidth: 1, borderRadius: 6 }
        ]
      },
      options: gridOpts()
    });
  }

  function buildAlpha() {
    if (charts.alpha || !window.Chart) return;
    var ctx = document.getElementById("chartAlpha");
    if (!ctx) return;
    charts.alpha = new Chart(ctx, {
      type: "line",
      data: {
        labels: WD_DATA.alpha.map(function (r) { return "alpha=" + r[0]; }),
        datasets: [
          { label: "Recall@k", data: WD_DATA.alpha.map(function (r) { return r[1]; }), borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,0.15)", fill: true, tension: 0.3 },
          { label: "MRR", data: WD_DATA.alpha.map(function (r) { return r[2]; }), borderColor: "#d97706", backgroundColor: "rgba(217,119,6,0.12)", fill: true, tension: 0.3 }
        ]
      },
      options: gridOpts()
    });
  }

  var builders = { layers: buildLayers, ablation: buildAblation, alpha: buildAlpha };

  window.WD_CHART = {
    data: WD_DATA,
    build: function (kind) {
      fillTables();
      if (builders[kind]) builders[kind]();
    },
    restyle: function () {
      Object.keys(charts).forEach(function (k) {
        if (charts[k]) { charts[k].options = gridOpts(); charts[k].update(); }
      });
    }
  };
})();
