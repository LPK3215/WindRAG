/* WindRAG project overview — all interaction logic. */
(function () {
  "use strict";

  /* ---------- theme (system default + manual toggle, persisted) ---------- */
  var THEME_KEY = "wd-theme";
  var root = document.documentElement;

  function preferredTheme() {
    var stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function applyTheme(t) {
    root.setAttribute("data-theme", t);
    var btn = document.getElementById("themeToggle");
    if (btn) btn.textContent = t === "dark" ? "☀" : "☾";
    if (window.WD_CHART && window.WD_CHART.restyle) window.WD_CHART.restyle();
  }
  applyTheme(preferredTheme());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
    if (!localStorage.getItem(THEME_KEY)) applyTheme(e.matches ? "dark" : "light");
  });
  var themeBtn = document.getElementById("themeToggle");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });

  /* ---------- scroll progress bar ---------- */
  var progress = document.getElementById("progressBar");
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var pct = max > 0 ? (h.scrollTop || document.body.scrollTop) / max * 100 : 0;
    if (progress) progress.style.width = pct + "%";
    var bt = document.getElementById("backTop");
    if (bt) bt.classList.toggle("show", (h.scrollTop || window.scrollY) > window.innerHeight * 0.9);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- mobile hamburger ---------- */
  var menuBtn = document.getElementById("menuBtn");
  var navLinks = document.getElementById("navLinks");
  if (menuBtn && navLinks) menuBtn.addEventListener("click", function () {
    navLinks.classList.toggle("open");
  });
  if (navLinks) navLinks.addEventListener("click", function (e) {
    if (e.target.tagName === "A") navLinks.classList.remove("open");
  });

  /* ---------- active nav highlight ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
  var byId = {};
  links.forEach(function (a) {
    var id = a.getAttribute("href").replace("#", "");
    var el = document.getElementById(id);
    if (el) byId[id] = a;
  });
  var sectionObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) {
        links.forEach(function (a) { a.classList.remove("active"); });
        var a = byId[en.target.id];
        if (a) a.classList.add("active");
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });
  Object.keys(byId).forEach(function (id) {
    var el = document.getElementById(id);
    if (el) sectionObserver.observe(el);
  });

  /* ---------- generated overview facts ---------- */
  var overviewData = window.WD_OVERVIEW_DATA;
  if (!overviewData) throw new Error("overview-data.js must load before script.js");

  function applyOverviewData() {
    document.querySelectorAll("[data-overview]").forEach(function (el) {
      var key = el.getAttribute("data-overview");
      if (!Object.prototype.hasOwnProperty.call(overviewData, key)) {
        throw new Error("Missing overview value: " + key);
      }
      var value = overviewData[key];
      var format = el.getAttribute("data-overview-format") || "text";
      var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
      if (format === "version") {
        el.textContent = "v" + value;
      } else if (format === "percent") {
        var percent = Number(value) * 100;
        if (el.classList.contains("num")) {
          el.setAttribute("data-count", String(percent));
          el.setAttribute("data-suffix", "%");
        } else {
          el.textContent = percent.toFixed(decimals) + "%";
        }
      } else if (el.classList.contains("num")) {
        el.setAttribute("data-count", String(value));
      } else {
        el.textContent = String(value);
      }
    });
  }
  applyOverviewData();

  /* ---------- number count-up animation ---------- */
  function animateNum(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
    var suffix = el.getAttribute("data-suffix") || "";
    var dur = 1300, start = null;
    function frame(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      var val = target * eased;
      el.textContent = val.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ",") + suffix;
      if (p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  var numObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting && !en.target.dataset.done) {
        en.target.dataset.done = "1";
        animateNum(en.target);
        numObserver.unobserve(en.target);
      }
    });
  }, { threshold: 0.4 });
  document.querySelectorAll(".num[data-count]").forEach(function (el) { numObserver.observe(el); });

  /* ---------- copy buttons ---------- */
  document.querySelectorAll(".copy-btn[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var block = btn.closest(".code-block");
      var code = block && block.querySelector("code");
      if (!code) return;
      var text = code.innerText;
      navigator.clipboard.writeText(text).then(function () {
        var old = btn.textContent; btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = old; }, 1400);
      });
    });
  });

  /* ---------- tabs (lazy-build charts) ---------- */
  var tabs = document.querySelectorAll(".tab");
  var panels = document.querySelectorAll(".tab-panel");
  function showTab(kind) {
    tabs.forEach(function (t) { t.classList.toggle("active", t.dataset.tab === kind); });
    panels.forEach(function (p) { p.classList.toggle("active", p.dataset.panel === kind); });
    if (window.WD_CHART) window.WD_CHART.build(kind);
  }
  tabs.forEach(function (t) { t.addEventListener("click", function () { showTab(t.dataset.tab); }); });
  if (window.WD_CHART) window.WD_CHART.build("layers");

  /* ---------- architecture minimap tooltip ---------- */
  var tip = document.getElementById("mmTip");
  var minimap = document.querySelector(".minimap");
  document.querySelectorAll(".mm-node").forEach(function (node) {
    node.addEventListener("mouseenter", function (e) {
      if (!tip || !minimap) return;
      tip.textContent = node.getAttribute("data-tip");
      tip.hidden = false;
      var r = minimap.getBoundingClientRect();
      tip.style.left = (e.clientX - r.left) + "px";
      tip.style.top = (e.clientY - r.top - 10) + "px";
    });
    node.addEventListener("mousemove", function (e) {
      if (!tip || !minimap) return;
      var r = minimap.getBoundingClientRect();
      tip.style.left = (e.clientX - r.left) + "px";
      tip.style.top = (e.clientY - r.top - 10) + "px";
    });
    node.addEventListener("mouseleave", function () { if (tip) tip.hidden = true; });
  });

  /* ---------- back to top ---------- */
  var backTop = document.getElementById("backTop");
  if (backTop) backTop.addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  /* ---------- footer stamp ---------- */
  var yr = new Date().getFullYear();
  var fs = document.querySelector(".footer span");
  if (fs) fs.textContent = fs.textContent.replace("{{year}}", yr);
})();
