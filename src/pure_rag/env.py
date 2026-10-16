"""
极简 .env 加载器（替代 python-dotenv，仅标准库）

支持：KEY=VALUE、# 注释、空行、值两侧引号；默认不覆盖已存在的环境变量。
"""

import os
from pathlib import Path
from typing import Dict


def parse_env_file(path) -> Dict[str, str]:
    """解析 .env 文件为字典"""
    values: Dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def load_env(path, override: bool = False) -> Dict[str, str]:
    """把 .env 内容写入 os.environ，返回解析结果"""
    values = parse_env_file(path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values
