"""
数据加载（仅标准库）

1. 项目自带数据：复用 src/data_processing.py 的解析逻辑（该模块本身零依赖）
2. 自有文档：自动纳入 data/user_docs/ 下的 txt / md / json / csv / docx
   —— docx 本质是 zip + XML，用标准库 zipfile + 正则即可解析，无需 python-docx
"""

import csv
import html
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List

from data_processing import DataProcessor

USER_DIR_NAME = "user_docs"
GENERATED_DIR_NAME = "generated"
SUPPORTED_SUFFIXES = (".txt", ".md", ".markdown", ".json", ".csv", ".docx")
_PARAGRAPH_BREAKS = ("</w:p>", "<w:br/>", "<w:cr/>")

# 纯文本类文件的候选编码：UTF-8（含 BOM）优先，再试中文 Windows 下常见的 GB18030
# （GB18030 向后兼容 GBK/GB2312）。两者都失败时报错，不静默丢字节。
_TEXT_ENCODINGS = ("utf-8-sig", "gb18030")


def read_plain_text(path: Path) -> str:
    """按候选编码读纯文本（标准库实现，不依赖 chardet）"""
    last_error = None
    for encoding in _TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def _read_docx(path: Path) -> str:
    """解析 .docx：读取 word/document.xml，按段落断开后剥离标签"""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", "ignore")
    for token in _PARAGRAPH_BREAKS:
        xml = xml.replace(token, "\n")
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def _flatten_json_item(item) -> str:
    """从 JSON 条目中抽取正文（兼容常见字段名）"""
    if isinstance(item, dict):
        for key in ("content", "answer", "text", "body", "正文", "内容"):
            if key in item:
                head = item.get("title") or item.get("question") or item.get("name") or ""
                return f"{head}\n{item[key]}".strip()
        return json.dumps(item, ensure_ascii=False)
    return str(item)


def _read_json_doc(path: Path) -> str:
    data = json.loads(read_plain_text(path))
    items = data if isinstance(data, list) else [data]
    return "\n\n".join(block for block in (_flatten_json_item(i) for i in items) if block)


def _read_csv_doc(path: Path) -> str:
    """CSV：每行拼成"列名: 值"，保持列语义不丢失"""
    rows = list(csv.DictReader(read_plain_text(path).splitlines()))
    lines = []
    for index, row in enumerate(rows, 1):
        pairs = "; ".join(f"{key}: {value}" for key, value in row.items() if value)
        if pairs:
            lines.append(f"[{index}] {pairs}")
    return "\n".join(lines)


def _read_markdown(path: Path) -> str:
    """Markdown：去掉标题井号、代码围栏与强调符号，只保留可检索的正文"""
    text = read_plain_text(path)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)      # 标题
    text = re.sub(r"^\s*```.*$", "", text, flags=re.M)             # 代码围栏
    return text.replace("**", "").replace("`", "")


def read_text_file(path: Path) -> str:
    """按后缀分派解析器"""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".json":
        return _read_json_doc(path)
    if suffix == ".csv":
        return _read_csv_doc(path)
    if suffix in (".md", ".markdown"):
        return _read_markdown(path)
    return read_plain_text(path)


def load_user_documents(data_dir: str = "data") -> List[Dict]:
    """加载 data/user_docs/ 下的自有文档，返回 [{source, content, metadata}]"""
    user_dir = Path(data_dir) / USER_DIR_NAME
    user_dir.mkdir(parents=True, exist_ok=True)   # 便于用户直接把文件丢进来

    documents: List[Dict] = []
    for path in sorted(user_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            content = read_text_file(path).strip()
        except Exception as exc:                     # noqa: BLE001 - 单个文件失败不影响整体
            print(f"[warn] 跳过 {path.name}：{exc}")
            continue
        if not content:
            continue
        documents.append({
            "source": USER_DIR_NAME,
            "content": content,
            "metadata": {"type": "user_doc", "title": path.stem, "file": path.name},
        })
    return documents


def load_generated_documents(data_dir: str = "data") -> List[Dict]:
    """
    加载 data/generated/ 下由 `src/simulate_corpus.py` 生成的规模化语料

    与 user_docs 的区别：这里每个 JSON 条目就是一篇独立文档，并保留
    component / category / farm 等结构化元数据，而不是把整个文件拼成一篇。
    目录不存在时返回空列表（只有小语料也能正常跑）。
    """
    generated_dir = Path(data_dir) / GENERATED_DIR_NAME
    if not generated_dir.exists():
        return []

    documents = []
    for path in sorted(generated_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:          # noqa: BLE001 - 单文件失败不影响整体
            print(f"[warn] 跳过 {path.name}：{exc}")
            continue
        items = data if isinstance(data, list) else next(
            (value for value in data.values() if isinstance(value, list)), [])
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            source = item.get("source", path.stem)
            metadata = dict(item.get("metadata", {}))
            metadata.setdefault("source", source)
            metadata.setdefault("title", metadata.get("component")
                                or metadata.get("category") or path.stem)
            documents.append({"source": source, "content": item["content"],
                              "metadata": metadata})
    return documents


def load_documents(data_dir: str = "data") -> List[Dict]:
    """加载全部文档：项目数据 + 规模化生成语料 + 自有文档"""
    processor = DataProcessor(str(data_dir))
    documents = processor.load_raw_text() + processor.load_synthetic_data()
    documents += load_generated_documents(data_dir)
    documents += load_user_documents(data_dir)

    for document in documents:
        metadata = document.setdefault("metadata", {})
        # 统一标题字段，便于分块时做字段加权
        if not metadata.get("title"):
            metadata["title"] = (
                metadata.get("component")
                or metadata.get("category")
                or document.get("source", "")
            )
        metadata.setdefault("source", document.get("source", "unknown"))

    return documents


def data_stats(data_dir: str = "data") -> Dict[str, int]:
    """统计各类数据的数量，用于 CLI 的 stats 子命令"""
    stats: Dict[str, int] = {}
    for document in load_documents(data_dir):
        source = document.get("source", "unknown")
        stats[source] = stats.get(source, 0) + 1
    return stats


def ensure_dir(path) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
