"""
TF-IDF 稀疏向量检索（从零实现，仅用标准库）

替代原版的 sentence-transformers 语义向量：
    tf(t,d)   = 1 + ln(count)                 亚线性词频，抑制高频词
    idf(t)    = ln((1 + N) / (1 + df)) + 1    平滑 IDF，且不为 0
    weight    = tf · idf，文档向量做 L2 归一化
    sim(q, d) = q̂ · d̂                        归一化后点积 = 余弦相似度

存储结构：倒排表 term -> {doc_id: 权重}，检索时只累加命中文档；
同时保留正排 doc_terms 以支持文档间相似度（供 MMR 多样性重排使用）。
"""

import json
import math
from typing import Dict, List, Sequence, Tuple


class TfidfIndex:
    """TF-IDF 向量索引（稀疏表示 + 倒排索引）"""

    def __init__(self, sublinear_tf: bool = True):
        self.sublinear_tf = sublinear_tf
        self.doc_count = 0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, Dict[int, float]] = {}   # term -> {doc_id: 归一化权重}
        self.doc_terms: List[Dict[str, float]] = []       # doc_id -> {term: 归一化权重}

    # -- 构建 ---------------------------------------------------------------
    def build(self, token_lists: Sequence[Sequence[str]]) -> "TfidfIndex":
        """一次性构建索引"""
        self.doc_count = len(token_lists)
        self.df = {}
        term_freqs: List[Dict[str, int]] = []

        for tokens in token_lists:
            counter: Dict[str, int] = {}
            for token in tokens:
                counter[token] = counter.get(token, 0) + 1
            term_freqs.append(counter)
            for term in counter:
                self.df[term] = self.df.get(term, 0) + 1

        self.idf = {
            term: math.log((1 + self.doc_count) / (1 + df)) + 1.0
            for term, df in self.df.items()
        }

        self.postings = {}
        self.doc_terms = []
        for doc_id, counter in enumerate(term_freqs):
            weights = {
                term: self._tf(count) * self.idf[term]
                for term, count in counter.items()
            }
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            weights = {term: w / norm for term, w in weights.items()}
            self.doc_terms.append(weights)
            for term, weight in weights.items():
                self.postings.setdefault(term, {})[doc_id] = weight
        return self

    def _tf(self, count: int) -> float:
        return 1.0 + math.log(count) if self.sublinear_tf else float(count)

    # -- 检索 ---------------------------------------------------------------
    def encode(self, tokens: Sequence[str]) -> Dict[str, float]:
        """把查询编码成 L2 归一化的稀疏向量"""
        counter: Dict[str, int] = {}
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1

        weights = {
            term: self._tf(count) * self.idf_of(term)
            for term, count in counter.items()
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {term: w / norm for term, w in weights.items()}

    def idf_of(self, term: str) -> float:
        """查询词的 IDF，未登录词按"只出现一次"的稀有权重估计"""
        if term in self.idf:
            return self.idf[term]
        return math.log((1 + self.doc_count) / 2) + 1.0

    def score(self, tokens: Sequence[str]) -> Dict[int, float]:
        """返回 {doc_id: 余弦相似度}"""
        scores: Dict[int, float] = {}
        for term, query_weight in self.encode(tokens).items():
            posting = self.postings.get(term)
            if not posting:
                continue
            for doc_id, doc_weight in posting.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + query_weight * doc_weight
        return scores

    def search(self, tokens: Sequence[str], top_k: int = None) -> List[Tuple[int, float]]:
        """检索并返回 [(doc_id, 相似度)]，按相似度降序"""
        ranked = sorted(self.score(tokens).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k] if top_k else ranked

    def explain(self, tokens: Sequence[str], doc_id: int) -> Dict[str, float]:
        """单个文档上每个查询词贡献的余弦分量（可解释性展示用）"""
        contributions: Dict[str, float] = {}
        for term, query_weight in self.encode(tokens).items():
            posting = self.postings.get(term)
            if posting and doc_id in posting:
                contributions[term] = query_weight * posting[doc_id]
        return contributions

    def similarity(self, doc_a: int, doc_b: int) -> float:
        """两个文档向量的余弦相似度（两者均已归一化，点积即为余弦）"""
        if doc_a == doc_b:
            return 1.0
        terms_a = self.doc_terms[doc_a]
        terms_b = self.postings  # 复用倒排表按需查表，避免保存完整矩阵
        score = 0.0
        for term, weight in terms_a.items():
            posting = terms_b.get(term)
            if posting and doc_b in posting:
                score += weight * posting[doc_b]
        return score

    # -- 持久化 -------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "sublinear_tf": self.sublinear_tf,
            "doc_count": self.doc_count,
            "idf": self.idf,
            "df": self.df,
            "doc_terms": self.doc_terms,
            "postings": {term: list(posting.items()) for term, posting in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TfidfIndex":
        index = cls(sublinear_tf=data.get("sublinear_tf", True))
        index.doc_count = data.get("doc_count", 0)
        index.idf = {k: float(v) for k, v in data.get("idf", {}).items()}
        index.df = {k: int(v) for k, v in data.get("df", {}).items()}
        index.doc_terms = [
            {term: float(w) for term, w in weights.items()} for weights in data.get("doc_terms", [])
        ]
        for term, posting in data.get("postings", {}).items():
            index.postings[term] = {int(doc_id): float(w) for doc_id, w in posting}
        return index

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "TfidfIndex":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
