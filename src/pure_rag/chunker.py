"""
文档分块（从零实现）

相比原版 data_processing.chunk_documents 的改进：
- 长段落先拆句，避免单个 chunk 远超 chunk_size
- 相邻 chunk 之间保留 overlap 字符的重叠，缓解跨块语义断裂
- 超长句子再按固定窗口硬切，保证块长上界
"""

from typing import Dict, List

from pure_rag.text_utils import split_paragraphs, split_sentences


def _split_units(text: str, chunk_size: int) -> List[str]:
    """把文本切成不超过 chunk_size 的语义单元（段落 -> 句子 -> 硬切）"""
    units: List[str] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(
                    sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)
                )
    return units


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """把一段文本切成带重叠的块"""
    units = _split_units(text, chunk_size)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        if current and current_len + len(unit) + 1 > chunk_size:
            chunks.append("\n".join(current))
            # 从尾部回捞若干句作为重叠上下文
            carry: List[str] = []
            carry_len = 0
            for sentence in reversed(current):
                if carry_len + len(sentence) > overlap:
                    break
                carry.insert(0, sentence)
                carry_len += len(sentence) + 1
            current, current_len = carry, carry_len

        current.append(unit)
        current_len += len(unit) + 1

    if current:
        chunks.append("\n".join(current))

    return [c for c in chunks if c.strip()]


def chunk_documents(documents: List[Dict], chunk_size: int = 400,
                    overlap: int = 80) -> List[Dict]:
    """
    对文档列表分块

    输入元素需包含 content / metadata / source 字段（与 data_processing 输出一致），
    输出元素为 {content, metadata}，metadata 中补充 chunk_id / title 等检索辅助字段。
    """
    chunks: List[Dict] = []

    for document in documents:
        metadata = dict(document.get("metadata", {}))
        source = document.get("source", metadata.get("source", "unknown"))
        title = metadata.get("title") or metadata.get("component") or ""

        for chunk_id, content in enumerate(chunk_text(document["content"], chunk_size, overlap)):
            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                "source": source,
                "chunk_id": chunk_id,
                "title": title,
            })
            chunks.append({"content": content, "metadata": chunk_metadata})

    return chunks


def build_index_text(chunk: Dict) -> str:
    """
    构造用于建索引的文本：标题重复一次做字段加权（简易 BM25F 思想）
    正文与元数据分离存储，检索打分时不会因为标题拼入正文而污染展示内容
    """
    metadata = chunk.get("metadata", {})
    title = metadata.get("title") or metadata.get("component") or ""
    keywords = metadata.get("keywords") or ""
    if isinstance(keywords, list):
        keywords = " ".join(keywords)
    boost = " ".join(part for part in (title, title, keywords) if part)
    return f"{boost}\n{chunk['content']}"
