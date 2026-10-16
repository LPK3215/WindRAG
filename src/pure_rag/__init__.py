"""
纯 Python RAG 内核（零第三方依赖）

与原研究版（sentence-transformers + rank-bm25 + gradio）并存，特点：
- 只用 Python 标准库，不需要安装任何依赖即可构建索引与问答
- BM25 / TF-IDF / 分词 / 融合重排 / 答案生成全部从底层实现
- 复用 src/knowledge_graph.py 与 src/model_manager.py（二者本就只依赖标准库）

用法：
    from pure_rag import RAGEngine
    engine = RAGEngine()
    engine.build()
    print(engine.query("齿轮箱温度过高怎么处理？")["answer"])
"""

import os
import sys

# 让 pure_rag 能够复用 src/ 目录下已有的标准库实现
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pure_rag.engine import RAGEngine  # noqa: E402

__version__ = "1.0.0"
__all__ = ["RAGEngine", "__version__"]
