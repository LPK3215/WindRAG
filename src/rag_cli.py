"""
纯 Python RAG 命令行入口（零第三方依赖）

用法示例：
    python src/rag_cli.py build
    python src/rag_cli.py ask "齿轮箱温度过高应该怎么处理？" --show-docs
    python src/rag_cli.py chat
    python src/rag_cli.py eval -n 15
    python src/rag_cli.py stats
    python src/rag_cli.py serve --port 8000
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pure_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
