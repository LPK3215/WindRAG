"""
检索结果融合与重排（从零实现，仅用标准库）

三种策略：
1. 归一化加权融合：S = α·Ŝ_vec + (1-α)·Ŝ_bm25（与原版一致，但无需 numpy）
2. RRF（Reciprocal Rank Fusion）：S = Σ 1/(k + rank)，用于合并多路/多查询结果
3. MMR（最大边际相关）：λ·相关性 - (1-λ)·与已选结果的最大相似度，抑制冗余
"""

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

RankedList = Sequence[Tuple[int, float]]


def min_max(scores: Dict[int, float]) -> Dict[int, float]:
    """把分数线性映射到 [0, 1]，全等时统一置 1.0"""
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return {doc_id: 1.0 for doc_id in scores}
    span = high - low
    return {doc_id: (score - low) / span for doc_id, score in scores.items()}


def scale_by_max(scores: Dict[int, float]) -> Dict[int, float]:
    """按最大值缩放到 (0, 1]，保留相对大小关系（最高分恒为 1.0）"""
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 1e-12:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: score / top for doc_id, score in scores.items()}


def weighted_fusion(vector_scores: Dict[int, float], bm25_scores: Dict[int, float],
                    alpha: float) -> Dict[int, float]:
    """归一化加权融合，缺失的一路记 0 分"""
    vec_norm = min_max(vector_scores)
    bm25_norm = min_max(bm25_scores)
    doc_ids = set(vec_norm) | set(bm25_norm)
    return {
        doc_id: alpha * vec_norm.get(doc_id, 0.0) + (1 - alpha) * bm25_norm.get(doc_id, 0.0)
        for doc_id in doc_ids
    }


def reciprocal_rank_fusion(ranked_lists: Iterable[RankedList], k: int = 60,
                           weights: Sequence[float] = None) -> Dict[int, float]:
    """
    RRF 融合：只看排名不看绝对分数，能自然合并量纲不同的多路结果
    常用于"原查询 + 同义词扩展查询"的多路召回合并
    """
    ranked_lists = list(ranked_lists)
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused: Dict[int, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    return fused


def mmr_select(candidates: RankedList, top_k: int, similarity: Callable[[int, int], float],
               lambda_: float = 0.7) -> List[Tuple[int, float]]:
    """
    MMR 多样性重排

    candidates: [(doc_id, 相关性分数)]，分数建议先归一化到 [0, 1]
    similarity: 两个 doc_id 之间的相似度函数（0~1）
    lambda_:   1.0 表示纯相关性排序，越小越强调多样性
    """
    remaining = list(candidates)
    selected: List[Tuple[int, float]] = []

    while remaining and len(selected) < top_k:
        best_item, best_value = None, float("-inf")
        for doc_id, score in remaining:
            redundancy = max((similarity(doc_id, picked_id) for picked_id, _ in selected), default=0.0)
            value = lambda_ * score - (1 - lambda_) * redundancy
            if value > best_value:
                best_item, best_value = (doc_id, score), value
        selected.append(best_item)
        remaining.remove(best_item)

    return selected


def merge_scores(primary: Dict[int, float], secondary: Dict[int, float],
                 weight: float = 0.5) -> Dict[int, float]:
    """把两路已归一化分数按权重相加（用于把扩展查询的召回结果并回主结果）"""
    merged = dict(primary)
    for doc_id, score in secondary.items():
        merged[doc_id] = merged.get(doc_id, 0.0) + weight * score
    return merged
