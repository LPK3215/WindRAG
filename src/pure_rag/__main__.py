"""模块形式入口。

包在 src/ 下，仓库根里直接 `python -m pure_rag` 找不到包，需要把 src 放进搜索路径：
    PYTHONPATH=src python -m pure_rag stats        # Linux / macOS
    $env:PYTHONPATH="src"; python -m pure_rag stats  # Windows PowerShell
日常使用推荐直接走仓库根入口：python src/rag_cli.py <子命令>
"""

import sys

from pure_rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
