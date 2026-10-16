"""
查询理解：问题类型分类 + 自适应融合权重 α 预测 + 查询扩展

纯规则实现，无需训练；与项目原有的 adaptive_fusion 规则先验保持一致，
区别在于不依赖 numpy，可直接在零依赖环境下运行。
"""

import re
from typing import Dict, List

from pure_rag.text_utils import DOMAIN_WORDS, STOPWORDS, Tokenizer, expand_query, normalize

# 问题类型关键词
QUESTION_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "故障诊断": ["故障", "异常", "报警", "温度高", "振动大", "异响", "泄漏", "过热",
                 "怎么处理", "如何判断", "原因", "跳闸", "停机"],
    "维护操作": ["维护", "检修", "更换", "检查", "保养", "操作", "规程", "周期",
                 "步骤", "如何操作", "加注", "紧固"],
    "系统原理": ["原理", "作用", "功能", "结构", "组成", "工作", "什么是", "为什么",
                 "机制", "怎么工作"],
    "性能优化": ["效率", "优化", "提高", "改善", "提升", "降低", "减少", "功率曲线",
                 "发电量"],
}

# 标准编号 / 数值规格 / 百分比
SPEC_PATTERN = re.compile(r"(iso|gb|iec|din|nb)\s*/?\s*[a-z]*\s*\d+", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\d+\.?\d*")
UNIT_PATTERN = re.compile(r"\d+\.?\d*\s*(mm|cm|m|s|min|h|℃|°c|kw|mw|v|a|hz|rpm|%|kg|nm|mpa)")
RATIO_PATTERN = re.compile(r"\d+\s*%")

# 融合权重先验表（α 越大越偏向语义向量检索，越小越偏向 BM25 精确匹配）
ALPHA_BY_TYPE: Dict[str, float] = {
    "故障诊断": 0.7,
    "系统原理": 0.8,
    "性能优化": 0.75,
    "维护操作": 0.6,
}
ALPHA_DEFAULT = 0.65


def classify_question(query: str) -> Dict[str, float]:
    """问题类型软分类：返回各类型的归一化得分（和可能为 0）"""
    query = normalize(query)
    scores = {
        qtype: float(sum(1 for kw in keywords if kw in query))
        for qtype, keywords in QUESTION_TYPE_KEYWORDS.items()
    }
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def adaptive_alpha(query: str, tokenizer: Tokenizer = None) -> float:
    """
    根据查询特征预测混合检索权重 α

    α → 0：BM25 精确匹配更重要（含标准编号、数值规格）
    α → 1：语义向量检索更重要（概念解释、口语化描述）
    """
    query = normalize(query)
    tokenizer = tokenizer or Tokenizer()
    tokens = tokenizer.cut(query)

    # 规则1：标准编号（ISO/GB/IEC/DIN）
    if SPEC_PATTERN.search(query):
        return 0.3
    # 规则2：数值规格（mm、℃、kW、%）
    if UNIT_PATTERN.search(query) or RATIO_PATTERN.search(query):
        return 0.4
    # 规则3：领域术语密集
    term_count = sum(1 for word in DOMAIN_WORDS if len(word) >= 2 and word in query)
    if term_count >= 3:
        return 0.5
    # 规则4~7：按问题类型
    scores = classify_question(query)
    if scores:
        best_type = max(scores, key=lambda k: scores[k])
        if scores[best_type] > 0:
            return ALPHA_BY_TYPE.get(best_type, ALPHA_DEFAULT)
    # 规则8：短查询偏关键词匹配
    if len(tokens) <= 2:
        return 0.5
    return ALPHA_DEFAULT


def expand_queries(query: str) -> List[str]:
    """查询扩展：同义词替换 + 关键词拼接（返回不含原查询的变体列表）"""
    variants = expand_query(query)
    # 抽取式扩展：把查询中的核心词直接拼成短查询，提升 BM25 命中率
    words = [w for w in Tokenizer().cut(query) if w not in STOPWORDS and len(w) >= 2]
    if len(words) >= 2:
        short = " ".join(words)
        if short not in variants:
            variants.append(short)
    return variants[:3]
