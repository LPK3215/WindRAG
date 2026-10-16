"""
检索结果高亮渲染（零第三方依赖）

同一份字符级区间（由 text_utils.find_spans 产生）渲染到三种载体：
- 终端：ANSI 粗体黄色
- Web：`<mark>` 标签，且先做 HTML 转义，避免文档内容里的尖括号注入
- Markdown：`**加粗**`，用于会话记录导出
"""

import html as _html
from typing import Iterable, List, Sequence, Tuple

Span = Tuple[int, int, str]

ANSI_START = "\033[1;33m"
ANSI_END = "\033[0m"


def _iter_segments(text: str, spans: Sequence[Span]):
    """按区间把文本切成 (片段, 是否命中) 依次产出"""
    cursor = 0
    for start, end, _term in spans:
        if start > cursor:
            yield text[cursor:start], False
        yield text[start:end], True
        cursor = end
    if cursor < len(text):
        yield text[cursor:], False


def ansi(text: str, spans: Sequence[Span], color: bool = True) -> str:
    """渲染为带 ANSI 颜色的终端文本"""
    if not spans or not color:
        return text
    return "".join(
        f"{ANSI_START}{segment}{ANSI_END}" if hit else segment
        for segment, hit in _iter_segments(text, spans)
    )


def to_html(text: str, spans: Sequence[Span]) -> str:
    """渲染为 HTML 片段（转义后再插入 <mark>）"""
    if not spans:
        return _html.escape(text)
    return "".join(
        f"<mark>{_html.escape(segment)}</mark>" if hit else _html.escape(segment)
        for segment, hit in _iter_segments(text, spans)
    )


def to_markdown(text: str, spans: Sequence[Span]) -> str:
    """渲染为 Markdown（命中词加粗）"""
    if not spans:
        return text
    return "".join(
        f"**{segment}**" if hit else segment
        for segment, hit in _iter_segments(text, spans)
    )


def clip_spans(spans: Sequence[Span], limit: int) -> List[Span]:
    """把区间裁剪到预览长度内，避免渲染越界"""
    return [(start, min(end, limit), term) for start, end, term in spans if start < limit]


def matched_terms(spans: Iterable[Span]) -> List[str]:
    """返回去重后的命中词列表，保持出现顺序"""
    seen: List[str] = []
    for _start, _end, term in spans:
        if term not in seen:
            seen.append(term)
    return seen
