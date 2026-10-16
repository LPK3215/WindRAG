"""
Okapi BM25 检索（从零实现，仅用标准库）

打分公式：
    score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
    IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

实现要点：倒排表 term -> {doc_id: tf}，检索时只遍历查询词命中的文档，
避免对全量文档打分（对应原版的 rank-bm25 + BM25Okapi，但无第三方依赖）。
"""

import json
import math
from typing import Dict, List, Sequence, Tuple


class BM25Index:
    """Okapi BM25 索引"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: 词频}
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self._idf: Dict[str, float] = {}

    # -- 构建 ---------------------------------------------------------------
    def add(self, tokens: Sequence[str]) -> int:
        """加入一篇文档，返回其 doc_id"""
        doc_id = len(self.doc_len)
        self.doc_len.append(len(tokens))
        for token in tokens:
            self.postings.setdefault(token, {})
            self.postings[token][doc_id] = self.postings[token].get(doc_id, 0) + 1
        return doc_id

    def build(self) -> "BM25Index":
        """索引收尾：计算平均文档长度与 IDF"""
        n = len(self.doc_len)
        self.avgdl = sum(self.doc_len) / n if n else 0.0
        self._idf = {}
        for term, posting in self.postings.items():
            df = len(posting)
            self._idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0
        return self

    # -- 检索 ---------------------------------------------------------------
    def idf(self, term: str) -> float:
        """查询词的逆文档频率，未登录词取最大 IDF（稀有权重高）"""
        if term in self._idf:
            return self._idf[term]
        n = len(self.doc_len)
        return math.log(1 + (n + 0.5) / 0.5) if n else 0.0

    def _saturation(self, tf: int, doc_id: int) -> float:
        """词频饱和项：tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))"""
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
        return tf * (self.k1 + 1) / (tf + norm)

    def score(self, tokens: Sequence[str]) -> Dict[int, float]:
        """对查询词命中的文档打分，返回 {doc_id: score}"""
        scores: Dict[int, float] = {}
        if not self.doc_len or self.avgdl <= 0:
            return scores

        for term in set(tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for doc_id, tf in posting.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * self._saturation(tf, doc_id)
        return scores

    def explain(self, tokens: Sequence[str], doc_id: int) -> Dict[str, float]:
        """单个文档上每个查询词各自贡献的 BM25 分数（可解释性展示用）"""
        contributions: Dict[str, float] = {}
        if self.avgdl <= 0 or doc_id >= len(self.doc_len):
            return contributions
        for term in set(tokens):
            posting = self.postings.get(term)
            if posting and doc_id in posting:
                contributions[term] = self.idf(term) * self._saturation(posting[doc_id], doc_id)
        return contributions

    def search(self, tokens: Sequence[str], top_k: int = None) -> List[Tuple[int, float]]:
        """检索并返回 [(doc_id, score)]，按分数降序"""
        ranked = sorted(self.score(tokens).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k] if top_k else ranked

    # -- 持久化 -------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_len": self.doc_len,
            "idf": self._idf,
            "postings": {term: list(posting.items()) for term, posting in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BM25Index":
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        index.doc_len = list(data.get("doc_len", []))
        index.avgdl = data.get("avgdl", 0.0)
        index._idf = {k: float(v) for k, v in data.get("idf", {}).items()}
        for term, posting in data.get("postings", {}).items():
            index.postings[term] = {int(doc_id): int(tf) for doc_id, tf in posting}
        return index

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "BM25Index":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
