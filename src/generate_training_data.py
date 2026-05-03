"""
训练数据生成脚本
通过网格搜索为每个查询找到最优α，生成自适应融合模型的训练数据

核心思路：
1. 对每个查询，遍历不同α值（0.1~0.9）
2. 用该α执行混合检索，计算检索质量指标
3. 选择使指标最优的α作为该查询的ground-truth标签
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import jieba
from query_features import QueryFeatureExtractor


class TrainingDataGenerator:
    """训练数据生成器：通过网格搜索找到每个查询的最优α"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.processed_dir / "index"

        # 加载模型和索引
        print("加载嵌入模型...")
        self.model = SentenceTransformer("shibing624/text2vec-base-chinese")

        print("加载索引...")
        # 向量索引
        embeddings_path = self.index_dir / "embeddings.npy"
        self.embeddings = np.load(embeddings_path)

        # 元数据
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # 文档块
        chunks_path = self.processed_dir / "chunks" / "documents.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # BM25索引
        bm25_path = self.index_dir / "bm25.pkl"
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)

        # 特征提取器
        self.feature_extractor = QueryFeatureExtractor()

    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """向量检索（L2归一化余弦相似度）"""
        query_embedding = self.model.encode([query])

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normalized_embeddings = self.embeddings / norms

        q_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        q_norm = np.where(q_norm == 0, 1e-8, q_norm)
        normalized_query = query_embedding / q_norm

        similarities = np.dot(normalized_embeddings, normalized_query.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        return list(zip(top_indices.tolist(), top_scores.tolist()))

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25检索（jieba分词）"""
        query_tokens = list(jieba.cut_for_search(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return list(zip(top_indices.tolist(), top_scores.tolist()))

    def _hybrid_search(self, query: str, alpha: float, top_k: int = 10) -> List[Dict]:
        """混合检索（指定α）"""
        vec_results = self._vector_search(query, top_k=top_k)
        bm25_results = self._bm25_search(query, top_k=top_k)

        # 归一化
        vec_scores = [s for _, s in vec_results]
        bm25_scores = [s for _, s in bm25_results]

        def normalize(scores):
            arr = np.array(scores)
            smin, smax = arr.min(), arr.max()
            return (arr - smin) / (smax - smin + 1e-8) if smax > smin else np.ones_like(arr)

        vec_norm = normalize(vec_scores)
        bm25_norm = normalize(bm25_scores)

        # 融合
        vec_dict = {idx: s for (idx, _), s in zip(vec_results, vec_norm)}
        bm25_dict = {idx: s for (idx, _), s in zip(bm25_results, bm25_norm)}

        all_indices = set(vec_dict.keys()) | set(bm25_dict.keys())
        fused = []
        for idx in all_indices:
            v = vec_dict.get(idx, 0.0)
            b = bm25_dict.get(idx, 0.0)
            score = alpha * v + (1 - alpha) * b
            fused.append({"index": idx, "score": score})

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    def _compute_retrieval_quality(self, query: str, results: List[Dict],
                                    relevant_indices: set = None) -> float:
        """
        计算检索质量分数（综合语义相似度和关键词匹配率）
        使用加权组合避免纯embedding指标偏向高α
        """
        if not results:
            return 0.0

        if relevant_indices:
            hits = sum(1 for r in results if r["index"] in relevant_indices)
            return hits / len(results)

        # 指标1：位置加权语义相似度
        query_emb = self.model.encode([query])
        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        q_norm = np.where(q_norm == 0, 1e-8, q_norm)
        norm_q = query_emb / q_norm

        semantic_score = 0.0
        for i, r in enumerate(results[:5]):
            doc_emb = self.embeddings[r["index"]].reshape(1, -1)
            d_norm = np.linalg.norm(doc_emb, axis=1, keepdims=True)
            d_norm = np.where(d_norm == 0, 1e-8, d_norm)
            norm_d = doc_emb / d_norm
            sim = float(np.dot(norm_q, norm_d.T)[0][0])
            weight = 1.0 / np.log2(i + 2)  # 位置衰减
            semantic_score += weight * sim

        # 指标2：关键词精确匹配率（从查询中提取术语，检查是否在检索文档中出现）
        from query_features import DOMAIN_TERMS
        query_terms = [t for t in DOMAIN_TERMS if t in query]
        keyword_score = 0.0
        if query_terms:
            for i, r in enumerate(results[:5]):
                doc_content = self.chunks[r["index"]]["content"] if r["index"] < len(self.chunks) else ""
                hits = sum(1 for t in query_terms if t in doc_content)
                weight = 1.0 / np.log2(i + 2)
                keyword_score += weight * (hits / len(query_terms))
        else:
            keyword_score = semantic_score  # 无术语时退化为语义分数

        # 综合质量 = 0.5*语义 + 0.5*关键词匹配
        return 0.5 * semantic_score + 0.5 * keyword_score

    def generate(self, queries: List[str], alpha_range: Tuple[float, float] = (0.1, 0.9),
                 alpha_step: float = 0.05) -> List[Dict]:
        """
        为每个查询网格搜索最优α，生成训练样本

        Returns:
            训练样本列表，每个样本包含 {features, optimal_alpha, query}
        """
        alphas = np.arange(alpha_range[0], alpha_range[1] + alpha_step / 2, alpha_step)
        training_data = []

        print(f"开始网格搜索，α范围: {alphas}")
        print(f"查询数: {len(queries)}")

        for i, query in enumerate(queries):
            if (i + 1) % 10 == 0:
                print(f"  处理查询 {i+1}/{len(queries)}")

            best_alpha = 0.5
            best_quality = -1.0
            quality_scores = {}

            for alpha in alphas:
                results = self._hybrid_search(query, alpha=alpha, top_k=10)
                quality = self._compute_retrieval_quality(query, results)
                quality_scores[round(alpha, 2)] = quality

                if quality > best_quality:
                    best_quality = quality
                    best_alpha = alpha

            # 提取查询特征
            query_embedding = self.model.encode([query])[0]
            feature_vector = self.feature_extractor.extract_vector(query, query_embedding)

            training_data.append({
                "query": query,
                "features": feature_vector.tolist(),
                "optimal_alpha": round(best_alpha, 2),
                "quality_scores": {str(k): round(v, 4) for k, v in quality_scores.items()},
                "best_quality": round(best_quality, 4),
            })

        return training_data

    def save_training_data(self, data: List[Dict], output_path: str):
        """保存训练数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"训练数据已保存到 {output_path}，共 {len(data)} 条")


def load_queries(data_dir: str = "data") -> List[str]:
    """从数据集中加载查询"""
    queries = []

    # 从QA对加载
    qa_path = Path(data_dir) / "synthetic" / "qa_pairs.json"
    if qa_path.exists():
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        qa_list = qa_data.get("qa_pairs", qa_data) if isinstance(qa_data, dict) else qa_data
        for qa in qa_list:
            if isinstance(qa, dict) and "question" in qa:
                queries.append(qa["question"])

    # 从故障案例生成查询
    fault_path = Path(data_dir) / "synthetic" / "fault_cases.json"
    if fault_path.exists():
        with open(fault_path, 'r', encoding='utf-8') as f:
            fault_data = json.load(f)
        fault_list = fault_data.get("fault_cases", fault_data) if isinstance(fault_data, dict) else fault_data
        for fault in fault_list:
            # 生成多种风格的查询
            component = fault.get("component", "")
            fault_type = fault.get("fault_type", "")
            symptoms = fault.get("symptoms", "")

            if component and fault_type:
                queries.append(f"{component}{fault_type}怎么处理？")
            if component and symptoms:
                queries.append(f"风电机{symptoms}是什么原因？")
            if component:
                queries.append(f"如何诊断{component}的故障？")

    # 补充多样化的查询模板
    diverse_queries = [
        "风电机组齿轮箱油温超过85℃应该怎么处理？",
        "ISO VG 320齿轮油多久更换一次？",
        "偏航系统偏航偏差过大怎么维修？",
        "叶片前缘腐蚀严重需要更换吗？",
        "变桨电机过热的原因有哪些？",
        "液压系统压力不足如何排查？",
        "制动器制动力矩如何检测？",
        "塔筒螺栓力矩检查周期是多久？",
        "发电机绝缘电阻低于多少需要停机？",
        "变压器油温异常升高怎么处理？",
        "风电机组可利用率怎么计算？",
        "功率曲线偏离标准值说明什么问题？",
        "切入风速3m/s切出风速25m/s正常吗？",
        "PLC控制器故障怎么排查？",
        "主轴轴承温度异常怎么诊断？",
        "联轴器对中不良会有什么后果？",
        "齿轮箱振动值7mm/s是否需要停机？",
        "叶片雷击损伤如何修复？",
        "偏航制动器磨损怎么更换？",
        "变频器IGBT模块过热保护怎么处理？",
    ]
    queries.extend(diverse_queries)

    # 去重
    queries = list(dict.fromkeys(queries))
    return queries


if __name__ == "__main__":
    print("=" * 60)
    print("自适应融合训练数据生成")
    print("=" * 60)

    # 加载查询
    queries = load_queries()
    print(f"共加载 {len(queries)} 条查询")

    # 生成训练数据
    generator = TrainingDataGenerator()
    training_data = generator.generate(queries, alpha_range=(0.1, 0.9), alpha_step=0.1)

    # 保存
    output_path = "data/adaptive_fusion_training.json"
    generator.save_training_data(training_data, output_path)

    # 统计
    alphas = [d["optimal_alpha"] for d in training_data]
    print(f"\n最优α分布:")
    print(f"  均值: {np.mean(alphas):.3f}")
    print(f"  标准差: {np.std(alphas):.3f}")
    print(f"  范围: [{min(alphas)}, {max(alphas)}]")
    print(f"  中位数: {np.median(alphas):.3f}")

    # 按区间统计
    from collections import Counter
    alpha_counts = Counter(alphas)
    print(f"\nα值频率分布:")
    for a in sorted(alpha_counts.keys()):
        bar = "█" * alpha_counts[a]
        print(f"  α={a:.1f}: {bar} ({alpha_counts[a]})")
