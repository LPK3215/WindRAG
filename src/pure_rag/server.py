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
  <h2>参考来源</h2>
  <div class="panel" id="docs"><span class="label">尚无检索结果</span></div>
</main>
<script>
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

    document.getElementById("answer").textContent = data.answer;
    const badges = document.getElementById("badges");
    badges.innerHTML = "";
    [["α", data.alpha.toFixed(3)], ["耗时", (data.timings.total*1000).toFixed(0)+" ms"],
     ["生成方式", data.mode], ["候选", data.candidates],
     ["图谱命中", data.kg_entities.map(e => e.name).join("、") || "无"]]
      .forEach(([k, v]) => {
        const b = document.createElement("div");
        b.className = "badge";
        b.innerHTML = k + " <b>" + v + "</b>";
        badges.appendChild(b);
      });

    const docs = document.getElementById("docs");
    docs.innerHTML = data.retrieved_docs.length ? "" : "<span class='label'>未检索到文档</span>";
    const max = Math.max(...data.retrieved_docs.map(d => d.score), 1e-6);
    data.retrieved_docs.forEach(d => {
      const div = document.createElement("div");
      div.className = "doc";
      const m = d.metadata || {};
      div.innerHTML = `<div class="doc-head"><span>${m.source || "-"} / ${m.title || "-"} / chunk ${m.chunk_id ?? "-"}</span>
                       <span>score ${d.score.toFixed(4)}</span></div>
                       <div class="doc-body"></div>
                       <div class="bar"><span style="width:${(d.score/max*100).toFixed(1)}%"></span></div>`;
      div.querySelector(".doc-body").innerHTML = d.content_html;
      docs.appendChild(div);
    });
  } catch (e) {
    document.getElementById("err").textContent = "查询失败：" + e.message;
    document.getElementById("answer").textContent = "";
  } finally {
    btn.disabled = false;
  }
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
            # 在服务端渲染高亮 HTML，前端直接插入即可（内容已转义）
            for document in result.get("retrieved_docs", []):
                preview = document["content"][:PREVIEW_CHARS].replace("\n", " ")
                document["content_html"] = to_html(
                    preview, clip_spans(document.get("spans", []), PREVIEW_CHARS)
                )
                document.pop("spans", None)
            self._send_json(result)
        except Exception as exc:                         # noqa: BLE001 - 统一返回错误信息
            self._send_json({"error": str(exc)}, status=400)

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
