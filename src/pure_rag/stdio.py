"""
标准流编码加固（仅标准库）

Windows 中文环境下有两个必现故障点：

1. stdout 被重定向（管道 / 写文件 / CI 日志）时，Python 按系统代码页（cp936/GBK）
   编码输出，遇到代码页外的字符（✓ ✗ − 等）直接抛 UnicodeEncodeError 中断命令；
2. 管道读入的中文若解码失败会得到孤立代理项（\udc80 之类），
   随后写 UTF-8 文件时报 "surrogates not allowed"。

处理策略：
- 非终端（被重定向）：三个流统一切到 UTF-8，日志与管道内容可跨平台、可被工具读取；
- 终端交互：保持原编码（不改用户看到的字符集），只把 errors 降级为
  backslashreplace / replace，保证编码不了或解码不了的字符不会中断命令。
"""

import sys

# (流名, 重定向时的编码, 输出 errors, 输入 errors)
_OUTPUT_ERRORS = "backslashreplace"
_INPUT_ERRORS = "replace"
_REDIRECT_ENCODING = "utf-8"


def configure_streams() -> None:
    """入口调用一次，让命令在任意代码页下都不会因编码问题中断"""
    for name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        is_console = bool(getattr(stream, "isatty", lambda: False)())
        options = {"errors": _OUTPUT_ERRORS if name != "stdin" else _INPUT_ERRORS}
        if not is_console:
            options["encoding"] = _REDIRECT_ENCODING
        try:
            stream.reconfigure(**options)
        except (ValueError, OSError):
            # 流被换成不支持 reconfigure 的对象（例如测试框架的 capture），保持原样
            continue
