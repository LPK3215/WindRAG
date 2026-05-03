"""
查询特征提取器
从用户查询中提取多维度特征，用于自适应检索权重预测
"""

import re
import numpy as np
from typing import Dict, List


# 风电领域专业术语词典
DOMAIN_TERMS = {
    # 部件名称
    "齿轮箱", "发电机", "叶片", "变桨", "偏航", "液压", "制动",
    "塔筒", "变压器", "轴承", "传感器", "PLC", "变频器", "主轴",
    "联轴器", "机舱", "轮毂", "导流罩", "散热器", "冷却器",
    # 故障术语
    "振动", "温度", "过热", "磨损", "裂纹", "腐蚀", "疲劳",
    "异响", "泄漏", "卡涩", "失速", "过载", "短路", "断路",
    # 运维术语
    "润滑", "螺栓", "力矩", "绝缘", "油位", "油温", "油质",
    "巡检", "大修", "预防性维护", "纠正性维护",
    # 标准编号模式
    "ISO", "GB", "IEC", "DIN",
    # 性能术语
    "功率", "效率", "可利用率", "MTBF", "功率曲线",
    "切入风速", "额定风速", "切出风速",
}

# 问题类型关键词映射
QUESTION_TYPE_KEYWORDS = {
    "故障诊断": ["故障", "异常", "报警", "温度高", "振动大", "异响", "泄漏", "过热", "怎么处理", "如何判断", "原因"],
    "维护操作": ["维护", "检修", "更换", "检查", "保养", "操作", "规程", "周期", "步骤", "如何操作"],
    "系统原理": ["原理", "作用", "功能", "结构", "组成", "工作", "什么是", "为什么", "机制"],
    "性能优化": ["效率", "优化", "提高", "改善", "提升", "降低", "减少", "功率曲线"],
}

# 数值/规格模式
NUMBER_PATTERN = re.compile(r'\d+\.?\d*')
SPEC_PATTERN = re.compile(r'(ISO|GB|IEC|DIN)\s*\d+')
RATIO_PATTERN = re.compile(r'\d+\s*[%％]')


class QueryFeatureExtractor:
    """查询特征提取器"""

    def __init__(self, domain_terms: set = None):
        self.domain_terms = domain_terms or DOMAIN_TERMS

    def extract(self, query: str) -> Dict[str, float]:
        """提取查询的多维度特征，返回特征字典"""
        features = {}

        # === 1. 基础文本特征 ===
        features["query_length"] = len(query)                          # 查询长度
        features["char_count"] = len(query.replace(" ", ""))           # 去空格字符数
        features["word_count"] = len(query.split())                    # 词数（按空格分）

        # === 2. 领域术语特征 ===
        term_hits = [t for t in self.domain_terms if t in query]
        features["domain_term_count"] = len(term_hits)                 # 领域术语数量
        features["domain_term_ratio"] = len(term_hits) / max(len(query), 1)  # 术语密度

        # === 3. 问题类型特征（one-hot风格） ===
        type_scores = {}
        for qtype, keywords in QUESTION_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query)
            type_scores[qtype] = score

        total_type_score = sum(type_scores.values()) or 1
        features["is_fault_diagnosis"] = type_scores.get("故障诊断", 0) / total_type_score
        features["is_maintenance"] = type_scores.get("维护操作", 0) / total_type_score
        features["is_principle"] = type_scores.get("系统原理", 0) / total_type_score
        features["is_optimization"] = type_scores.get("性能优化", 0) / total_type_score

        # === 4. 数值/规格特征 ===
        numbers = NUMBER_PATTERN.findall(query)
        features["has_number"] = 1.0 if numbers else 0.0              # 是否含数值
        features["number_count"] = len(numbers)                        # 数值个数
        features["has_spec"] = 1.0 if SPEC_PATTERN.search(query) else 0.0  # 是否含标准编号
        features["has_ratio"] = 1.0 if RATIO_PATTERN.search(query) else 0.0  # 是否含百分比

        # === 5. 疑问词特征 ===
        question_words = ["怎么", "如何", "为什么", "什么", "哪些", "是否", "多少", "怎样"]
        features["has_question_word"] = 1.0 if any(w in query for w in question_words) else 0.0

        # === 6. 语义特征（查询embedding的统计量，需要外部提供） ===
        # 这些在调用时通过 set_embedding_features 补充

        return features

    def extract_vector(self, query: str, query_embedding: np.ndarray = None) -> np.ndarray:
        """提取特征向量（用于模型输入）"""
        features = self.extract(query)

        # 如果提供了embedding，计算其统计特征
        if query_embedding is not None:
            features["emb_norm"] = float(np.linalg.norm(query_embedding))
            features["emb_max"] = float(np.max(query_embedding))
            features["emb_std"] = float(np.std(query_embedding))
        else:
            features["emb_norm"] = 0.0
            features["emb_max"] = 0.0
            features["emb_std"] = 0.0

        # 按固定顺序组成向量
        feature_order = [
            "query_length", "char_count", "word_count",
            "domain_term_count", "domain_term_ratio",
            "is_fault_diagnosis", "is_maintenance", "is_principle", "is_optimization",
            "has_number", "number_count", "has_spec", "has_ratio",
            "has_question_word",
            "emb_norm", "emb_max", "emb_std",
        ]

        return np.array([features[k] for k in feature_order], dtype=np.float32)

    @staticmethod
    def get_feature_names() -> List[str]:
        """返回特征名列表（固定顺序）"""
        return [
            "query_length", "char_count", "word_count",
            "domain_term_count", "domain_term_ratio",
            "is_fault_diagnosis", "is_maintenance", "is_principle", "is_optimization",
            "has_number", "number_count", "has_spec", "has_ratio",
            "has_question_word",
            "emb_norm", "emb_max", "emb_std",
        ]


if __name__ == "__main__":
    extractor = QueryFeatureExtractor()

    test_queries = [
        "风电机的齿轮箱温度过高应该怎么处理？",
        "ISO VG 320齿轮油更换周期是多久？",
        "偏航系统的工作原理是什么？",
        "如何提高风电机组的发电效率？",
        "齿轮箱振动值超过7mm/s需要怎么维护？",
    ]

    for q in test_queries:
        features = extractor.extract(q)
        print(f"\n查询: {q}")
        for k, v in features.items():
            print(f"  {k}: {v:.3f}")
