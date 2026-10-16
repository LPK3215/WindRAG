"""
零依赖 Web 界面（标准库 http.server 实现）

替代 gradio 版界面，不安装任何第三方包，前端也不用框架：页面结构、样式与脚本
是 `src/pure_rag/web/` 下的静态文件，由本模块按白名单读取后原样返回。

路由：
    GET  /                    对话式单页应用
    GET  /static/<file>       style.css / app.js（仅白名单内文件）
    POST /api/query           问答，入参 {"question","top_k","method","alpha","session_id"}
    POST /api/session/reset   清空某个会话的多轮上下文
    GET  /api/stats           索引与模型状态

多轮：服务端按 session_id 保存最近若干轮的问答，传给引擎做指代消解；
前端也带 session_id，因此刷新页面后服务端上下文仍在。
"""

import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pure_rag.highlight import clip_spans, to_html

# 检索结果预览长度（超过则截断）
PREVIEW_CHARS = 400

# 静态资源目录与白名单（不做路径拼接，杜绝目录穿越）
WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

# 每个会话最多保留多少轮问答（供指代消解使用）
MAX_TURNS_PER_SESSION = 8
# 内存里最多同时保留多少个会话
MAX_SESSIONS = 64


class ConversationStore:
    """会话上下文存储：线程安全 + 按最近使用淘汰"""

    def __init__(self, max_sessions: int = MAX_SESSIONS, max_turns: int = MAX_TURNS_PER_SESSION):
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.max_turns = max_turns

    def history(self, session_id: str) -> list:
        """该会话已有的问答（不含当前这轮）"""
        with self._lock:
            return list(self._data.get(session_id, []))

    def append(self, session_id: str, question: str, answer: str) -> int:
        """追加一轮，返回该会话当前轮数"""
        with self._lock:
            turns = self._data.setdefault(session_id, [])
            turns.append({"question": question, "answer": answer})
            del turns[self.max_turns:]
            self._data.move_to_end(session_id)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)
            return len(turns)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


class RAGRequestHandler(BaseHTTPRequestHandler):
    """极简 HTTP 处理器：静态页 + 问答接口 + 状态接口"""

    engine = None
    sessions = ConversationStore()
    lock = threading.Lock()

    def do_GET(self):                                    # noqa: N802 - 标准库命名约定
        if self.path == "/api/stats":
            self._send_json(self.engine.stats())
        elif self.path in STATIC_FILES:
            self._send_static(self.path)
        elif self.path == "/favicon.ico":
            self._send(204, b"", "text/plain; charset=utf-8")
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):                                   # noqa: N802
        if self.path == "/api/session/reset":
            payload = self._read_json_body()
            self.sessions.reset(str(payload.get("session_id") or ""))
            self._send_json({"reset": True})
            return

        if self.path != "/api/query":
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            payload = self._read_json_body()
            question = (payload.get("question") or "").strip()
            if not question:
                raise ValueError("问题不能为空")
            session_id = str(payload.get("session_id") or "")
            # 只有带 session_id 才启用多轮，缺省保持原来的单轮无状态行为
            history = self.sessions.history(session_id) if session_id else None

            # 引擎内部有缓存字典，加锁保证多线程下安全
            with self.lock:
                result = self.engine.query(
                    question,
                    top_k=int(payload.get("top_k") or 3),
                    method=payload.get("method") or "hybrid",
                    alpha=payload.get("alpha"),
                    history=history,
                )
            if session_id:
                result["turns"] = self.sessions.append(session_id, question, result["answer"])
            self._render_highlights(result)
            self._send_json(result)
        except Exception as exc:                         # noqa: BLE001 - 统一返回错误信息
            self._send_json({"error": str(exc)}, status=400)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        return json.loads(raw or "{}")

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

    def _send_static(self, route: str):
        filename, content_type = STATIC_FILES[route]
        path = WEB_DIR / filename
        if not path.is_file():
            self._send_json({"error": f"缺少前端文件 {path}"}, status=500)
            return
        body = path.read_bytes()
        self._send(200, body, content_type, extra={"Cache-Control": "no-store"})

    def _send_json(self, data, status: int = 200):
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str, extra: dict = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
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
