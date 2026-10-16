"""
零依赖 Web 界面（标准库 http.server 实现）

替代 gradio 版界面，无需安装任何第三方包：
    GET  /           单页应用
    POST /api/query  问答接口，入参 {"question","top_k","method","alpha"}
    GET  /api/stats  索引统计
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pure_rag.highlight import clip_spans, to_html

# 检索结果预览长度（超过则截断）
PREVIEW_CHARS = 400

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WindRAG · 纯 Python 检索内核</title>
<style>
  :root { --bg:#0f1720; --card:#18232e; --line:#26384a; --fg:#e8eef5; --dim:#8ba0b4; --accent:#3ddc97; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
  header { padding:22px 28px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:20px; letter-spacing:.5px; }
  header p { margin:6px 0 0; color:var(--dim); font-size:13px; }
  main { max-width:960px; margin:0 auto; padding:22px 20px 60px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }
  textarea { width:100%; min-height:74px; resize:vertical; background:#0d151d; color:var(--fg);
             border:1px solid var(--line); border-radius:8px; padding:10px; font-size:14px; }
  .row { display:flex; gap:10px; align-items:center; margin-top:10px; flex-wrap:wrap; }
  select, input[type=number] { background:#0d151d; color:var(--fg); border:1px solid var(--line);
             border-radius:8px; padding:7px 9px; font-size:13px; }
  button { background:var(--accent); color:#06251a; border:0; border-radius:8px; padding:9px 18px;
           font-size:14px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.55; cursor:progress; }
  .label { color:var(--dim); font-size:12px; }
  .badges { display:flex; gap:8px; flex-wrap:wrap; margin:14px 0 0; }
  .badge { background:#0d151d; border:1px solid var(--line); border-radius:999px;
           padding:4px 11px; font-size:12px; color:var(--dim); }
  .badge b { color:var(--accent); font-weight:600; }
  #answer { white-space:pre-wrap; line-height:1.75; font-size:14.5px; margin:14px 0 0; }
  .doc { border-top:1px solid var(--line); padding:12px 0; }
  .doc:first-child { border-top:0; }
  .doc-head { display:flex; justify-content:space-between; gap:12px; font-size:12.5px; color:var(--dim); }
  .doc-body { margin-top:6px; font-size:13.5px; line-height:1.65; color:#c8d6e2; }
  .doc-body mark { background:var(--accent); color:#06251a; border-radius:3px; padding:0 2px; }
  .doc.active { outline:1px solid var(--accent); outline-offset:5px; border-radius:8px; }
  .doc.active .doc-head span:first-child { color:var(--accent); }
  .claim { cursor:pointer; padding:8px 11px; margin:6px 0; background:#0d151d; font-size:13.5px;
           color:#c8d6e2; border-left:3px solid var(--line); border-radius:0 8px 8px 0; line-height:1.6; }
  .claim:hover { border-left-color:var(--accent); color:var(--fg); }
  .claim b { color:var(--accent); margin-right:6px; }
  .claim mark { background:var(--accent); color:#06251a; border-radius:3px; padding:0 2px; }
  .claim .meta { color:var(--dim); font-size:11.5px; margin-left:8px; white-space:nowrap; }
  .card-head { font-size:14.5px; font-weight:600; margin-bottom:8px; }
  .card-head .sev { color:#ffb454; font-size:12px; font-weight:400; margin-left:8px; }
  #card table { width:100%; border-collapse:collapse; font-size:13.5px; }
  #card th { text-align:left; color:var(--dim); font-weight:500; width:78px; vertical-align:top;
             padding:5px 8px 5px 0; white-space:nowrap; }
  #card td { padding:5px 0; color:#c8d6e2; line-height:1.65; }
  .locate-title { color:var(--dim); font-size:12px; margin-bottom:6px; }
  .locate-body { font-size:13.5px; line-height:1.7; color:var(--fg); }
  .locate-body mark { background:var(--accent); color:#06251a; border-radius:3px; padding:0 2px; }
  .bar { height:3px; background:#0d151d; border-radius:3px; margin-top:8px; }
  .bar span { display:block; height:100%; background:var(--accent); border-radius:3px; }
  h2 { font-size:14px; color:var(--dim); margin:26px 0 10px; font-weight:600; }
  .err { color:#ff8080; font-size:13px; margin-top:8px; }
  .examples span { display:inline-block; margin:4px 8px 0 0; font-size:12.5px; color:var(--accent);
                   cursor:pointer; border-bottom:1px dashed #2f6b52; }
</style>
</head>
<body>
<header>
  <h1>WindRAG · 纯 Python 检索内核</h1>
  <p>零第三方依赖 · BM25 + TF-IDF 混合检索 · 自适应 α 融合 · 知识图谱增强 · MMR 重排</p>
</header>
<main>
  <div class="panel">
    <textarea id="q" placeholder="例如：齿轮箱温度过高应该怎么处理？"></textarea>
    <div class="row">
      <button id="go">检索并回答</button>
      <span class="label">方法</span>
      <select id="method">
        <option value="hybrid">hybrid</option>
        <option value="vector">vector</option>
        <option value="bm25">bm25</option>
      </select>
      <span class="label">top-k</span>
      <input id="topk" type="number" value="3" min="1" max="10" style="width:64px">
      <span class="label">α（留空自适应）</span>
      <input id="alpha" type="number" step="0.05" min="0" max="1" style="width:76px">
    </div>
    <div class="examples" id="examples"></div>
    <div class="err" id="err"></div>
    <div class="badges" id="badges"></div>
    <div id="answer"></div>
  </div>
  <div id="card"></div>
  <div id="claims"></div>
  <h2>参考来源</h2>
  <div class="panel" id="docs"><span class="label">尚无检索结果</span></div>
  <h2>定位摘录</h2>
  <div class="panel" id="locate"><span class="label">点击上方要点可定位到原文</span></div>
</main>
<script>
const esc = s => String(s).replace(/[&<>"']/g,
  ch => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[ch]));

// LLM 常返回 Markdown 片段，而答案区是纯文本渲染；这里只做一层转义后的加粗转换，
// 不引入 Markdown 库（先 esc 再替换，注入面已封闭）
const mdLite = s => esc(s).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

const examples = ["齿轮箱温度过高应该怎么处理？", "变速箱异响怎么排查？",
                  "偏航系统的工作原理是什么？", "ISO VG 320齿轮油更换周期",
                  "如何提高风电机组的发电效率？"];
const ex = document.getElementById("examples");
examples.forEach(t => {
  const s = document.createElement("span");
  s.textContent = t;
  s.onclick = () => { document.getElementById("q").value = t; run(); };
  ex.appendChild(s);
});

async function run() {
  const question = document.getElementById("q").value.trim();
  if (!question) return;
  const btn = document.getElementById("go");
  btn.disabled = true;
  document.getElementById("err").textContent = "";
  document.getElementById("answer").textContent = "检索中…";
  ["card", "claims"].forEach(id => { document.getElementById(id).innerHTML = ""; });
  document.getElementById("locate").innerHTML = "<span class='label'>点击上方要点可定位到原文</span>";

  const alphaRaw = document.getElementById("alpha").value;
  const payload = {
    question,
    top_k: parseInt(document.getElementById("topk").value || "3", 10),
    method: document.getElementById("method").value,
    alpha: alphaRaw === "" ? null : parseFloat(alphaRaw)
  };

  try {
    const resp = await fetch("/api/query", {method:"POST", headers:{"Content-Type":"application/json"},
                                            body: JSON.stringify(payload)});
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    document.getElementById("answer").innerHTML = mdLite(data.answer);
    if (data.llm_error) {
      document.getElementById("err").textContent = "LLM 调用失败，本条答案退回抽取式：" + data.llm_error;
    }
    const badges = document.getElementById("badges");
    badges.innerHTML = "";
    const badgeItems = [["α", data.alpha.toFixed(3)],
      ["耗时", (data.timings.total*1000).toFixed(0) + " ms" + (data.cached ? "（缓存）" : "")],
      ["生成方式", data.mode], ["候选", data.candidates],
      ["图谱命中", data.kg_entities.map(e => e.name).join("、") || "无"]];
    // 知识图谱在提示词里占最后一个资料编号，不标出来读者找不到 [N] 对应什么
    if (data.kg_citation_index) badgeItems.push(["图谱引用", "[" + data.kg_citation_index + "]"]);
    badgeItems.forEach(([k, v]) => {
        const b = document.createElement("div");
        b.className = "badge";
        b.innerHTML = k + " <b>" + v + "</b>";
        badges.appendChild(b);
      });

    renderCard(data.card);
    renderClaims(data.claims);

    const docs = document.getElementById("docs");
    docs.innerHTML = data.retrieved_docs.length ? "" : "<span class='label'>未检索到文档</span>";
    const max = Math.max(...data.retrieved_docs.map(d => d.score), 1e-6);
    data.retrieved_docs.forEach(d => {
      const div = document.createElement("div");
      div.className = "doc";
      div.id = d.anchor || ("doc-" + (d.metadata || {}).chunk_id);
      const m = d.metadata || {};
      // 行首编号与答案里的 [N] 同一口径（由服务端按引用列表回映射），未被引用的标“未引用”
      const cite = d.citation_index ? `[${d.citation_index}]` : "未引用";
      div.innerHTML = `<div class="doc-head"><span>${cite} ${esc(m.source || "-")} / ${esc(m.title || "-")} / chunk ${m.chunk_id ?? "-"}</span>
                       <span>score ${d.score.toFixed(4)}</span></div>
                       <div class="doc-body"></div>
                       <div class="bar"><span style="width:${(d.score/max*100).toFixed(1)}%"></span></div>`;
      div.querySelector(".doc-body").innerHTML = d.content_html;
      docs.appendChild(div);
    });

    // 答案里的 [N] 如果指向知识图谱，在来源区补一行同编号条目，否则这一引用无处可查
    if (data.kg_citation_index && data.kg_context) {
      const kgRow = document.createElement("div");
      kgRow.className = "doc";
      kgRow.id = "doc-kg";
      kgRow.innerHTML = `<div class="doc-head"><span>[${data.kg_citation_index}] 知识图谱</span>
                         <span>上下文</span></div>
                         <div class="doc-body"></div>`;
      kgRow.querySelector(".doc-body").textContent = data.kg_context;
      docs.appendChild(kgRow);
    }
  } catch (e) {
    document.getElementById("err").textContent = "查询失败：" + e.message;
    document.getElementById("answer").textContent = "";
  } finally {
    btn.disabled = false;
  }
}

function renderCard(card) {
  const box = document.getElementById("card");
  if (!card) return;
  const rows = (card.fields || []).map(f =>
    `<tr><th>${esc(f.label)}</th><td>${esc(f.value)}</td></tr>`).join("");
  const sev = card.severity ? `<span class="sev">严重程度：${esc(card.severity)}</span>` : "";
  box.innerHTML = `<div class="panel"><div class="card-head">故障卡片：${esc(card.title)}${sev}</div>
                   <table>${rows}</table></div>`;
}

function renderClaims(claims) {
  const box = document.getElementById("claims");
  if (!claims || !claims.length) return;
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<div class="card-head">要点定位（点击跳转到原文）</div>`;
  claims.forEach(c => {
    const item = document.createElement("div");
    item.className = "claim";
    item.innerHTML = `<b>${c.index}</b>${c.excerpt_html}` +
                     `<span class="meta">doc${c.doc_id} · 字符 ${c.start}–${c.end}</span>`;
    item.onclick = () => locate(c);
    panel.appendChild(item);
  });
  box.appendChild(panel);
}

function locate(claim) {
  document.querySelectorAll(".doc").forEach(d => d.classList.remove("active"));
  const target = document.getElementById(claim.anchor);
  if (target) {
    target.classList.add("active");
    target.scrollIntoView({behavior: "smooth", block: "center"});
  }
  document.getElementById("locate").innerHTML =
    `<div class="locate-title">doc${claim.doc_id} · ${esc(claim.source)}/${esc(claim.title || "")}` +
    ` · 字符 ${claim.start}–${claim.end}</div>` +
    `<div class="locate-body">${claim.excerpt_html}</div>`;
}

document.getElementById("go").onclick = run;
document.getElementById("q").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});
</script>
</body>
</html>
"""


class RAGRequestHandler(BaseHTTPRequestHandler):
    """极简 HTTP 处理器：静态页 + 两个 JSON 接口"""

    engine = None
    lock = threading.Lock()

    def do_GET(self):                                    # noqa: N802 - 标准库命名约定
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/stats":
            self._send_json(self.engine.stats())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):                                   # noqa: N802
        if self.path != "/api/query":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = (payload.get("question") or "").strip()
            if not question:
                raise ValueError("问题不能为空")
            # 引擎内部有缓存字典，加锁保证多线程下安全
            with self.lock:
                result = self.engine.query(
                    question,
                    top_k=int(payload.get("top_k") or 3),
                    method=payload.get("method") or "hybrid",
                    alpha=payload.get("alpha"),
                )
            self._render_highlights(result)
            self._send_json(result)
        except Exception as exc:                         # noqa: BLE001 - 统一返回错误信息
            self._send_json({"error": str(exc)}, status=400)

    @staticmethod
    def _render_highlights(result: dict) -> None:
        """
        在服务端把高亮渲染成 HTML，前端直接插入即可（内容已做转义）

        - 每篇文档：content_html（查询词高亮）+ anchor（供要点跳转定位）
        - 每条要点：excerpt_html（要点句内命中词高亮）+ 所属文档的 anchor
        """
        documents = result.get("retrieved_docs", [])
        spans_by_doc = {document.get("doc_id"): document.get("spans", []) for document in documents}

        # 答案里的 [N] 编号属于生成用的引用列表（已按相关性下限过滤），
        # 而下面列出的是全部检索结果，两者不是同一集合，按 doc_id 回映射才能对上
        citation_by_doc = {citation.get("doc_id"): citation["index"]
                           for citation in result.get("citations", [])}
        if result.get("kg_context") and result.get("mode") == "llm":
            result["kg_citation_index"] = len(result.get("citations", [])) + 1

        for document in documents:
            preview = document["content"][:PREVIEW_CHARS].replace("\n", " ")
            document["content_html"] = to_html(
                preview, clip_spans(document.get("spans", []), PREVIEW_CHARS)
            )
            document["anchor"] = f"doc-{document.get('doc_id')}"
            document["citation_index"] = citation_by_doc.get(document.get("doc_id"))
            document.pop("spans", None)

        for claim in result.get("claims", []):
            start, end = claim.get("start", 0), claim.get("end", 0)
            inner = [
                (span_start - start, span_end - start, term)
                for span_start, span_end, term in spans_by_doc.get(claim.get("doc_id"), [])
                if span_start >= start and span_end <= end
            ]
            claim["excerpt_html"] = to_html(claim.get("text", ""), inner)
            claim["anchor"] = f"doc-{claim.get('doc_id')}"

    def _send_json(self, data, status: int = 200):
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):                    # 静默访问日志
        return


def serve(engine, host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动 HTTP 服务（阻塞）"""
    handler = type("BoundHandler", (RAGRequestHandler,), {"engine": engine})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Web 界面已启动： http://{host}:{port}   （Ctrl+C 退出）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        httpd.server_close()
