/* WindRAG project overview — Chart.js config + data.
   All numbers are snapshots read from data/pure_index/report.json and ablation.json
   at generation time. This is the single source for the Results section (charts and
   tables share WD_DATA). Regenerate from the real reports if the corpus changes. */
(function () {
  "use strict";

  var WD_DATA = {
    layers: [
      { name: "QA verbatim",  count: 209, recall: 0.985645933014354, mrr: 0.965709728867624, refusal: null },
      { name: "QA rewrite",   count: 190, recall: 0.9473684210526315, mrr: 0.9149122807017545, refusal: null },
      { name: "Fault case",   count: 581, recall: 0.9655765920826161, mrr: 0.8668961560527826, refusal: null },
      { name: "Doc title",    count: 406, recall: 1.0,               mrr: 0.9950738916256158, refusal: null },
      { name: "Near-miss",    count: 18,  recall: null,              mrr: null,               refusal: 1.0 },
      { name: "Out-of-domain",count: 10,  recall: null,              mrr: null,               refusal: 1.0 },
      { name: "Missing-material", count: 6, recall: null,            mrr: null,               refusal: 0.5 },
      { name: "Colloquial (human)", count: 53, recall: 0.58,         mrr: 0.4066666666666666, refusal: 0.6666666666666666 }
    ],
    // name, recall@k, mrr, rewrite_recall, case_recall, avg_latency_ms
    ablation: [
      ["BM25 only",        0.9805194805194806, 0.9565897065897067, 0.9263157894736842, 0.9776247848537005, 1.2665748596191406],
      ["TF-IDF only",      0.9567099567099567, 0.88997113997114,   0.8473684210526315, 0.963855421686747,  1.1098861694335938],
      ["Hybrid a=0.7",     0.9682539682539683, 0.9285714285714286, 0.868421052631579,  0.9707401032702238, 1.4566898345947266],
      ["+ adaptive a",     0.9682539682539683, 0.9313371813371812, 0.8736842105263158, 0.9707401032702238, 1.5224456787109375],
      ["+ synonyms",       0.9747474747474747, 0.924963924963925,  0.9473684210526315, 0.9621342512908778, 1.6445000966389975],
      ["+ query expand",   0.9552669552669553, 0.8894901394901392, 0.968421052631579,  0.9087779690189329, 1.7851670583089194],
      ["+ MMR rerank",     0.9668109668109668, 0.8938191438191436, 0.9894736842105263, 0.9294320137693631, 2.0653565724690757],
      ["+ knowledge graph",0.9668109668109668, 0.8938191438191436, 0.9894736842105263, 0.9294320137693631, 1.4960289001464844],
      ["full - new-words", 0.9718614718614719, 0.9111351611351609, 1.0,                0.9345955249569707, 3.1904061635335283]
    ],
    // alpha, recall@k, mrr, case_recall, avg_latency_ms
    alpha: [
      ["0.3",      0.9812409812409812, 0.9582732082732083, 0.9724612736660929, 1.4632542928059895],
      ["0.4",      0.9805194805194806, 0.9505772005772006, 0.9707401032702238, 1.4664808909098308],
      ["0.5",      0.9790764790764791, 0.9436026936026937, 0.9707401032702238, 1.7452875773111978],
      ["0.6",      0.9776334776334776, 0.9341029341029342, 0.9690189328743546, 1.2115478515625],
      ["0.7",      0.9747474747474747, 0.924001924001924,  0.9621342512908778, 1.3833045959472656],
      ["0.8",      0.9689754689754689, 0.9093314093314094, 0.9483648881239243, 1.2972036997477214],
      ["adaptive", 0.9747474747474747, 0.924963924963925,  0.9621342512908778, 1.2666543324788413]
    ]
  };

  function css(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function pct(x, d) { return (x == null) ? "—" : (x * 100).toFixed(d == null ? 1 : d) + "%"; }

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
          "</td><td>" + pct(r[3], 2) + "</td><td>" + pct(r[4], 2) + "</td><td>" + r[5].toFixed(2) + "</td></tr>";
      }).join("");
    }
    var pt = document.querySelector("#tblAlpha tbody");
    if (pt && !pt.childElementCount) {
      pt.innerHTML = WD_DATA.alpha.map(function (r) {
        return "<tr><td>" + r[0] + "</td><td>" + pct(r[1], 2) + "</td><td>" + pct(r[2], 2) +
          "</td><td>" + pct(r[3], 2) + "</td><td>" + r[4].toFixed(2) + "</td></tr>";
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
        labels: WD_DATA.alpha.map(function (r) { return "a=" + r[0]; }),
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
