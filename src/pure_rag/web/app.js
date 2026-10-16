// WindRAG 对话界面逻辑（原生 JS，无框架、无外部依赖）
//
// 安全约定：用户输入与答案文本一律先 esc() 再插入；来源正文与要点摘录
// （content_html / excerpt_html）由服务端转义后生成，可直接插入。

const EXAMPLES = [
  "齿轮箱温度过高应该怎么处理？",
  "ISO VG 320齿轮油更换周期",
  "偏航系统的工作原理是什么？",
  "如何提高风电机组的发电效率？",
  "变压器过热怎么排查？",
];

const STORAGE_KEY = "windrag.session.v1";
// 会话记录本地保留的轮数上限（localStorage 有容量限制，只留最近几轮）
const MAX_STORED_TURNS = 10;
// 要点摘录在原文里前后各扩展多少字符，只回显选中句的话信息量太低
const EXCERPT_CONTEXT = 80;

const $ = (id) => document.getElementById(id);
const transcript = $("transcript");
const welcome = $("welcome");
const input = $("input");
const sendBtn = $("send");

let thread = null;
let turns = 0;
let busy = false;
let sessionId = newSessionId();
let history = [];              // [{question, data}]，用于刷新后恢复对话

/* ---------------- 工具 ---------------- */

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (ch) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[ch]));

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function newSessionId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "s-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      sessionId,
      turns,
      messages: history.slice(-MAX_STORED_TURNS),
    }));
  } catch (err) {
    // 超出配额时退化成只记会话号，不影响当前使用
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessionId, turns, messages: [] })); }
    catch (ignored) { /* 隐私模式下完全不落盘 */ }
  }
}

function restore() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch (err) { saved = null; }
  if (!saved || !saved.sessionId) return;
  sessionId = saved.sessionId;
  // 旧版本可能只写了 turns 没写 messages（或两者不符），一律以实际能恢复的气泡数为准
  const messages = Array.isArray(saved.messages) ? saved.messages : [];
  messages.forEach((item) => {
    renderAnswer(item.question, item.data);
    history.push(item);
  });
  turns = history.length;
  updateTurns();
  if (history.length) toBottom();   // 恢复完应停在最新一条，而不是顶部
}

/* ---------------- 会话容器 ---------------- */

function ensureThread() {
  if (thread) return thread;
  welcome.style.display = "none";
  thread = el("div", "thread");
  transcript.appendChild(thread);
  return thread;
}

function resetThread() {
  if (thread) { thread.remove(); thread = null; }
  turns = 0;
  history = [];
  updateTurns();
  welcome.style.display = "";
}

function updateTurns() { $("turns").textContent = "共 " + turns + " 轮"; }

function toBottom() { transcript.scrollTop = transcript.scrollHeight; }

/* ---------------- 顶栏状态 ---------------- */

function paintLlmChip(state, detail) {
  const chip = $("chip-llm");
  const suffix = detail ? "（" + esc(String(detail).slice(0, 60)) + "）" : "";
  const map = {
    on: ["模型 <b>已启用</b>", false],
    cooling: ["模型 <b>不可用</b>" + suffix, true],
    off: ["模型 <b>未启用（抽取式）</b>", true],
  };
  const [html, dim] = map[state] || map.off;
  chip.innerHTML = html;
  chip.className = "chip" + (dim ? " off" : "");
}

async function loadStatus() {
  try {
    const resp = await fetch("/api/stats");
    const s = await resp.json();
    paintLlmChip(s.llm_configured ? (s.llm_enabled ? "on" : "cooling") : "off",
                 s.llm_enabled ? "" : s.llm_last_error);
    $("chip-corpus").innerHTML = "语料 <b>" + esc(s.source_documents) + " 篇</b> / "
      + esc(s.chunks) + " 块 · 词表 " + esc(s.vocabulary);
  } catch (err) {
    $("chip-llm").textContent = "状态获取失败";
  }
}

// 拒答原因必须按引擎给出的 reason 分支写，否则会出现“覆盖率 100% 低于阈值 35%”这种矛盾文案
function refusalNotice(conf) {
  const coverage = Math.round((conf.coverage || 0) * 100);
  const threshold = Math.round((conf.threshold || 0) * 100);
  const byReason = {
    invalid_combination: "部件与故障的搭配在知识图谱里不存在（具体归属见下方答案）。",
    no_content_term: "问题关键词没有落在知识库用词范围内（靠单字片段凑到 " + coverage + "%，没有实词命中）。",
    low_coverage: "查询词覆盖率 " + coverage + "% 低于阈值 " + threshold + "%。",
    empty_question: "问题为空。",
  };
  return "已拒答：" + (byReason[conf.reason] || ("查询词覆盖率 " + coverage + "%，阈值 " + threshold + "%。"))
    + " 可补充部件名称、故障现象或标准编号后重试。";
}

/* ---------------- 渲染组件 ---------------- */

function metaLine(data) {
  const hybrid = !data.method || data.method === "hybrid";
  const refused = data.mode === "refused";
  const items = [
    ["生成方式", data.mode],
    // alpha 只在混合检索下起作用；拒答轮没用到融合结果排序，列出来只会多余
    ...(hybrid && !refused ? [["alpha", Number(data.alpha).toFixed(3)]] : []),
    ["检索", data.method || "hybrid"],
    [(refused ? "检索耗时" : "耗时"),
     Math.round(data.timings.total * 1000) + " ms" + (data.cached ? "（缓存）" : "")],
    ["候选", data.candidates],
    ["图谱命中", (data.kg_entities || []).map((e) => e.name).join("、") || "无"],
  ];
  // 图谱在提示词里占一个编号，但模型不一定引用；只在真的被引用时才标出来
  if (data.kg_citation_index && new RegExp("\\[" + data.kg_citation_index + "\\]").test(data.answer || "")) {
    items.push(["图谱编号", "[" + data.kg_citation_index + "]"]);
  }
  const box = el("div", "meta-line" + (data.mode === "llm" ? " mode-llm" : ""));
  items.forEach(([k, v]) => box.appendChild(el("span", null, esc(k) + " <b>" + esc(v) + "</b>")));
  return box;
}

// 把答案里的 [1] [2, 3] [2-5] 变成可点击引用；data-cite 保存全部编号，
// 指向哪个条目由本轮映射表决定
function answerText(text) {
  const escaped = esc(text).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return escaped.replace(/\[(\d+(?:\s*(?:[,，]\s*\d+|[-–~至]\s*\d+))*)\]/g, (whole, inner) => {
    const nums = (inner.match(/\d+/g) || []).map(Number);
    if (!nums.length) return whole;
    return '<span class="ref" data-cite="' + nums.join(",") + '" title="定位到来源">' + whole + "</span>";
  });
}

function detailsBlock(title, bodyNode, open) {
  const box = el("details");
  if (open) box.open = true;
  box.appendChild(el("summary", null, esc(title)));
  const body = el("div", "details-body");
  body.appendChild(bodyNode);
  box.appendChild(body);
  return box;
}

// 引用编号与定位锚点都放在本轮映射表里，不用 DOM id：
// 同一篇文档可能在多个回合重复出现，id 会冲突，getElementById 只能命中第一个
function sourcesBlock(docs, kgIndex, kgContext) {
  const wrap = el("div");
  const byDocId = {}, byCite = {};
  if (!docs.length) wrap.appendChild(el("p", "empty-src", "本轮没有检索到文档。"));
  docs.forEach((d) => {
    const m = d.metadata || {};
    const row = el("div", "doc");
    const cite = d.citation_index ? "[" + d.citation_index + "]" : "未引用";
    row.appendChild(el("div", "doc-head",
      '<span class="cite">' + esc(cite) + " " + esc(m.source || "-") + " / " + esc(m.title || "-")
      + " / chunk " + esc(m.chunk_id ?? "-") + "</span><span>score " + esc(Number(d.score).toFixed(4)) + "</span>"));
    const body = el("div", "doc-body");
    body.innerHTML = d.content_html || esc(d.content || "");
    row.appendChild(body);
    wrap.appendChild(row);
    byDocId[d.doc_id] = row;
    if (d.citation_index) byCite[d.citation_index] = row;
  });
  if (kgIndex && kgContext) {
    const row = el("div", "doc");
    row.appendChild(el("div", "doc-head",
      '<span class="cite">[' + esc(kgIndex) + "] 知识图谱</span><span>上下文</span>"));
    row.appendChild(el("div", "doc-body kgline", esc(kgContext)));
    wrap.appendChild(row);
    byCite[kgIndex] = row;
  }
  return { node: wrap, byDocId, byCite };
}

function focusDoc(scope, row) {
  if (!row) return;
  scope.querySelectorAll(".doc.active").forEach((n) => n.classList.remove("active"));
  const holder = row.closest("details");
  if (holder) holder.open = true;
  row.classList.add("active");
  row.scrollIntoView({ behavior: "smooth", block: "center" });
}

// 摘录带上下文：只显示选中句的话，读者看不到它前后在讲什么
function excerptWindow(doc, start, end) {
  if (!doc || typeof doc.content !== "string") return null;
  const from = Math.max(0, start - EXCERPT_CONTEXT);
  const to = Math.min(doc.content.length, end + EXCERPT_CONTEXT);
  const head = esc(doc.content.slice(from, start));
  const mid = esc(doc.content.slice(start, end));
  const tail = esc(doc.content.slice(end, to));
  return (from > 0 ? "… " : "") + head + "<mark>" + mid + "</mark>" + tail
    + (to < doc.content.length ? " …" : "");
}

function claimsBlock(claims, sources, docsById, scope) {
  const wrap = el("div");
  const excerpt = el("div", "excerpt");
  excerpt.style.display = "none";
  claims.forEach((c) => {
    const item = el("div", "claim",
      "<b>" + esc(c.index) + "</b>" + (c.excerpt_html || esc(c.text))
      + '<span class="where">doc' + esc(c.doc_id) + " · 字符 " + esc(c.start) + "-" + esc(c.end) + "</span>");
    item.addEventListener("click", () => {
      focusDoc(scope, sources.byDocId[c.doc_id]);
      const window = excerptWindow(docsById[c.doc_id], c.start, c.end);
      excerpt.style.display = "";
      excerpt.innerHTML = '<div class="cap">doc' + esc(c.doc_id) + " · " + esc(c.source) + "/" + esc(c.title || "")
        + " · 字符 " + esc(c.start) + "-" + esc(c.end) + "</div><div>"
        + (window || c.excerpt_html || esc(c.text)) + "</div>";
    });
    wrap.appendChild(item);
  });
  wrap.appendChild(excerpt);
  return wrap;
}

function cardBlock(card) {
  const table = el("table", "card");
  table.appendChild(el("caption", null,
    "故障卡片：" + esc(card.title)
    + (card.severity ? ' <span class="sev">严重程度：' + esc(card.severity) + "</span>" : "")));
  (card.fields || []).forEach((f) => {
    const tr = document.createElement("tr");
    tr.appendChild(el("th", null, esc(f.label)));
    tr.appendChild(el("td", null, esc(f.value)));
    table.appendChild(tr);
  });
  return table;
}

function renderAnswer(question, data, turn) {
  if (!turn) {
    turn = el("div", "turn");
    turn.appendChild(el("div", "msg-user", esc(question)));
    ensureThread().appendChild(turn);
  }
  const box = el("div", "msg-answer");
  const docs = data.retrieved_docs || [];
  const sources = sourcesBlock(docs, data.kg_citation_index, data.kg_context);
  const docsById = {};
  docs.forEach((d) => { docsById[d.doc_id] = d; });

  box.appendChild(el("div", "answer-text", answerText(data.answer)));

  if (data.llm_error) {
    // 不用“调用失败”开头：冷却期根本没发起调用，与“耗时 1 ms”并列会矛盾
    box.appendChild(el("div", "notice notice-warn", "本条答案退回抽取式：" + esc(data.llm_error)));
  }
  if (data.mode === "refused") {
    box.appendChild(el("div", "notice notice-refused", refusalNotice(data.confidence || {})));
  }
  if (data.resolved_question && data.resolved_question !== data.question) {
    box.appendChild(el("div", "notice notice-coref", "已按上一轮补全为：" + esc(data.resolved_question)));
  }

  box.appendChild(metaLine(data));

  const claims = data.claims || [];
  if (data.card) box.appendChild(detailsBlock("故障卡片", cardBlock(data.card), true));
  if (claims.length) {
    box.appendChild(detailsBlock("要点定位（" + claims.length + " 条，点击跳原文）",
      claimsBlock(claims, sources, docsById, box)));
  }
  const extra = data.kg_citation_index ? " + 知识图谱" : "";
  // 拒答轮没生成引用，来源条目全部标“未引用”，标题再写“引用来源”就与条目互相矛盾
  const refused = data.mode === "refused";
  const srcTitle = (refused ? "检索候选（" : "引用来源（") + docs.length + " 篇" + extra
    + (refused ? "，未作为回答依据）" : "）");
  box.appendChild(detailsBlock(srcTitle, sources.node, false));

  turn.appendChild(box);

  // 答案里的 [N] 点击后展开本轮来源区：多编号引用（[1, 2, 3]）全部描边，滚到第一个
  box.querySelectorAll(".ref").forEach((chip) => {
    chip.addEventListener("click", () => {
      const rows = chip.dataset.cite.split(",")
        .map((n) => sources.byCite[n]).filter(Boolean);
      if (!rows.length) return;
      box.querySelectorAll(".doc.active").forEach((n) => n.classList.remove("active"));
      const holder = rows[0].closest("details");
      if (holder) holder.open = true;
      rows.forEach((row) => row.classList.add("active"));
      rows[0].scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
  return turn;
}

/* ---------------- 发送 ---------------- */

function pendingRow(question) {
  const turn = el("div", "turn");
  turn.appendChild(el("div", "msg-user", esc(question)));
  const box = el("div", "msg-answer");
  const line = el("div", "pending", '<span class="dots"><i></i><i></i><i></i></span><span>检索中 0s</span>');
  box.appendChild(line);
  turn.appendChild(box);
  ensureThread().appendChild(turn);
  const started = Date.now();
  const timer = setInterval(() => {
    line.lastChild.textContent = "检索中 " + Math.round((Date.now() - started) / 1000) + "s";
  }, 500);
  return { turn, box, stop: () => clearInterval(timer) };
}

function setBusy(flag) {
  busy = flag;
  sendBtn.classList.toggle("busy", flag);
  sendBtn.disabled = flag || !input.value.trim();
}

async function ask() {
  const question = input.value.trim();
  if (!question || busy) return;
  setBusy(true);
  input.value = "";
  autosize();
  const holder = pendingRow(question);
  toBottom();

  const alphaRaw = $("alpha").value;
  const payload = {
    question,
    top_k: parseInt($("topk").value || "3", 10),
    method: $("method").value,
    alpha: alphaRaw === "" ? null : parseFloat(alphaRaw),
    session_id: sessionId,
  };

  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    holder.stop();
    holder.box.remove();
    renderAnswer(question, data, holder.turn);
    turns = data.turns || (turns + 1);
    updateTurns();
    history.push({ question, data });
    persist();
    if (data.llm_error) paintLlmChip("cooling", data.llm_error);
    loadStatus();
  } catch (err) {
    holder.stop();
    holder.box.innerHTML = '<div class="notice notice-warn">请求失败：' + esc(err.message) + "</div>";
  } finally {
    setBusy(false);
    toBottom();
    input.focus();
  }
}

/* ---------------- 输入框行为 ---------------- */

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

function bind() {
  const examples = $("examples");
  EXAMPLES.forEach((text) => {
    const chip = el("button", "example", esc(text));
    chip.type = "button";
    chip.addEventListener("click", () => { input.value = text; autosize(); ask(); });
    examples.appendChild(chip);
  });

  $("composer").addEventListener("submit", (e) => { e.preventDefault(); ask(); });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
  input.addEventListener("input", () => {
    autosize();
    sendBtn.disabled = busy || !input.value.trim();
  });

  $("new-chat").addEventListener("click", () => {
    const old = sessionId;
    // 先清界面再发请求，否则点完一秒内仍显示旧轮次与旧对话
    sessionId = newSessionId();
    resetThread();
    persist();
    loadStatus();
    fetch("/api/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: old }),
    }).catch(() => { /* 服务端不可达时以本地为准 */ });
    input.focus();
  });
}

restore();
bind();
loadStatus();
setBusy(false);
