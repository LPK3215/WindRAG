"""
对比实验：固定α vs 自适应α
验证自适应融合方法的有效性

实验设计：
1. Baseline: 固定α=0.7（当前系统默认）
2. Proposed: 自适应α（查询感知预测）
3. 对比指标：检索质量（位置加权相似度）、关键词匹配率
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import time
from run_rag import WindTurbineRAG
from query_features import QueryFeatureExtractor


def load_test_queries(data_dir: str = "data") -> List[Dict]:
    """加载测试查询（带期望关键词）"""
    test_data = []

    # 从QA对加载
    qa_path = Path(data_dir) / "synthetic" / "qa_pairs.json"
    if qa_path.exists():
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        qa_list = qa_data.get("qa_pairs", qa_data) if isinstance(qa_data, dict) else qa_data
        for qa in qa_list:
            if isinstance(qa, dict) and "question" in qa:
                # 从答案中提取关键词
                answer = qa.get("answer", "")
                keywords = extract_keywords(answer)
                test_data.append({
                    "query": qa["question"],
                    "keywords": keywords,
                    "category": qa.get("category", ""),
                })

    # 补充多样化测试查询（增加更多需要低α的精确匹配查询）
    extra_queries = [
        {"query": "齿轮箱振动值超过7mm/s怎么处理？", "keywords": ["振动", "7mm/s", "停机", "振动分析"], "category": "故障诊断"},
        {"query": "ISO VG 320齿轮油更换周期", "keywords": ["ISO VG 320", "齿轮油", "2-3年", "油质分析"], "category": "维护操作"},
        {"query": "偏航系统偏航偏差过大怎么维修？", "keywords": ["偏航", "偏差", "偏航电机", "风向传感器"], "category": "故障诊断"},
        {"query": "叶片前缘腐蚀检测技术有哪些？", "keywords": ["叶片", "前缘腐蚀", "超声波", "红外热成像"], "category": "维护操作"},
        {"query": "变桨系统的工作原理是什么？", "keywords": ["变桨", "桨距角", "功率控制", "变桨电机"], "category": "系统原理"},
        {"query": "如何提高风电机组的发电效率？", "keywords": ["效率", "偏航", "叶片清洁", "MPPT"], "category": "性能优化"},
        {"query": "发电机绝缘电阻测试方法", "keywords": ["发电机", "绝缘", "绝缘电阻", "测试"], "category": "维护操作"},
        {"query": "液压系统压力不足的原因", "keywords": ["液压", "压力", "油泵", "溢流阀"], "category": "故障诊断"},
        {"query": "制动器制动力矩如何检测和调整？", "keywords": ["制动", "力矩", "制动力矩", "调整"], "category": "维护操作"},
        {"query": "风电机组功率曲线异常怎么分析？", "keywords": ["功率曲线", "异常", "故障诊断", "偏差"], "category": "故障诊断"},
        # 新增：含标准编号/数值规格的精确匹配查询（需要低α）
        {"query": "GB/T 19071风力发电机组技术条件", "keywords": ["GB/T 19071", "技术条件", "标准"], "category": "维护操作"},
        {"query": "IEC 61400-4齿轮箱设计要求", "keywords": ["IEC 61400-4", "齿轮箱", "设计要求"], "category": "系统原理"},
        {"query": "轴承温度超过85℃报警阈值", "keywords": ["轴承", "85℃", "温度", "报警"], "category": "故障诊断"},
        {"query": "偏航电机额定功率5.5kW参数", "keywords": ["偏航电机", "5.5kW", "额定功率"], "category": "维护操作"},
        {"query": "齿轮箱油位低于标尺下限怎么处理", "keywords": ["齿轮箱", "油位", "标尺", "下限"], "category": "故障诊断"},
        {"query": "ISO 81400-4风电齿轮箱标准", "keywords": ["ISO 81400-4", "齿轮箱", "标准"], "category": "维护操作"},
        {"query": "螺栓力矩值M20为440Nm", "keywords": ["螺栓", "M20", "440Nm", "力矩"], "category": "维护操作"},
        {"query": "发电机绝缘电阻应大于100MΩ", "keywords": ["发电机", "绝缘电阻", "100MΩ"], "category": "维护操作"},
        {"query": "液压系统工作压力16MPa", "keywords": ["液压", "16MPa", "工作压力"], "category": "维护操作"},
        {"query": "风电机组切入风速3m/s额定风速12m/s", "keywords": ["切入风速", "3m/s", "额定风速", "12m/s"], "category": "系统原理"},
    ]
    test_data.extend(extra_queries)
    return test_data


def extract_keywords(text: str) -> List[str]:
    """从文本中提取关键词（简单方法：取2-4字的领域术语）"""
    from query_features import DOMAIN_TERMS
    found = []
    for term in DOMAIN_TERMS:
        if term in text:
            found.append(term)
    # 去重，保留顺序
    seen = set()
    keywords = []
    for k in found:
        if k not in seen:
            seen.add(k)
            keywords.append(k)
    return keywords[:8]  # 最多8个


def compute_retrieval_quality(rag: WindTurbineRAG, query: str,
                               retrieved_docs: List[Dict]) -> float:
    """计算位置加权检索质量（nDCG风格的代理指标）"""
    if not retrieved_docs:
        return 0.0

    query_emb = rag.model.encode([query])
    q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    q_norm = np.where(q_norm == 0, 1e-8, q_norm)
    norm_q = query_emb / q_norm

    total_score = 0.0
    for i, doc in enumerate(retrieved_docs[:5]):
        doc_emb = rag.model.encode([doc["content"]])
        d_norm = np.linalg.norm(doc_emb, axis=1, keepdims=True)
        d_norm = np.where(d_norm == 0, 1e-8, d_norm)
        norm_d = doc_emb / d_norm
        sim = float(np.dot(norm_q, norm_d.T)[0][0])
        # 位置衰减 1/log2(rank+2)
        weight = 1.0 / np.log2(i + 2)
        total_score += weight * sim

    return total_score


def compute_keyword_coverage(retrieved_docs: List[Dict], keywords: List[str]) -> float:
    """计算关键词覆盖率"""
    if not keywords:
        return 1.0
    all_content = " ".join(d["content"] for d in retrieved_docs).lower()
    covered = sum(1 for kw in keywords if kw.lower() in all_content)
    return covered / len(keywords)


def run_experiment(rag: WindTurbineRAG, test_data: List[Dict],
                   mode: str = "adaptive") -> Dict:
    """
    运行实验
    mode: "fixed" 使用固定α=0.7, "adaptive" 使用自适应α
    """
    results = []
    total_quality = 0.0
    total_coverage = 0.0
    total_time = 0.0

    for item in test_data:
        query = item["query"]
        keywords = item["keywords"]

        start = time.time()

        if mode == "fixed":
            # 固定α=0.7
            docs = rag.retrieve_documents(query, top_k=5, method="hybrid", use_rerank=True, alpha=0.7)
            alpha_used = 0.7
        else:
            # 自适应α：先预测α，再强制传入
            if rag.adaptive_fusion_loaded:
                query_emb = rag.model.encode([query])[0]
                alpha_used = rag.adaptive_fusion.predict_alpha(query, query_emb)
            else:
                alpha_used = 0.7
            docs = rag.retrieve_documents(query, top_k=5, method="hybrid", use_rerank=True, alpha=alpha_used)

        elapsed = time.time() - start

        quality = compute_retrieval_quality(rag, query, docs)
        coverage = compute_keyword_coverage(docs, keywords)

        results.append({
            "query": query,
            "alpha": round(alpha_used, 3),
            "quality": round(quality, 4),
            "keyword_coverage": round(coverage, 4),
            "response_time": round(elapsed, 3),
            "category": item.get("category", ""),
        })

        total_quality += quality
        total_coverage += coverage
        total_time += elapsed

    n = len(test_data)
    summary = {
        "mode": mode,
        "total_queries": n,
        "avg_quality": round(total_quality / n, 4),
        "avg_keyword_coverage": round(total_coverage / n, 4),
        "avg_response_time": round(total_time / n, 3),
        "detailed_results": results,
    }
    return summary


def main():
    print("=" * 60)
    print("对比实验：固定α vs 自适应α")
    print("=" * 60)

    # 初始化RAG系统
    print("\n初始化RAG系统...")
    rag = WindTurbineRAG()

    # 加载测试数据
    print("\n加载测试数据...")
    test_data = load_test_queries()
    print(f"共 {len(test_data)} 条测试查询")

    # 实验1：固定α=0.7（Baseline）
    print("\n" + "-" * 40)
    print("实验1：固定α=0.7 (Baseline)")
    print("-" * 40)
    # 临时禁用自适应
    original_loaded = rag.adaptive_fusion_loaded
    rag.adaptive_fusion_loaded = False
    fixed_results = run_experiment(rag, test_data, mode="fixed")
    rag.adaptive_fusion_loaded = original_loaded

    # 实验2：自适应α（Proposed）
    print("\n" + "-" * 40)
    print("实验2：自适应α (Proposed)")
    print("-" * 40)
    adaptive_results = run_experiment(rag, test_data, mode="adaptive")

    # 对比分析
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    print(f"\n{'指标':<20} {'固定α=0.7':<15} {'自适应α':<15} {'提升':<10}")
    print("-" * 60)

    quality_diff = adaptive_results["avg_quality"] - fixed_results["avg_quality"]
    quality_pct = (quality_diff / fixed_results["avg_quality"] * 100) if fixed_results["avg_quality"] > 0 else 0

    coverage_diff = adaptive_results["avg_keyword_coverage"] - fixed_results["avg_keyword_coverage"]
    coverage_pct = (coverage_diff / fixed_results["avg_keyword_coverage"] * 100) if fixed_results["avg_keyword_coverage"] > 0 else 0

    print(f"{'检索质量(WDS)':<20} {fixed_results['avg_quality']:<15.4f} {adaptive_results['avg_quality']:<15.4f} {quality_pct:+.2f}%")
    print(f"{'关键词覆盖率':<20} {fixed_results['avg_keyword_coverage']:<15.4f} {adaptive_results['avg_keyword_coverage']:<15.4f} {coverage_pct:+.2f}%")
    print(f"{'平均响应时间(s)':<20} {fixed_results['avg_response_time']:<15.3f} {adaptive_results['avg_response_time']:<15.3f} {'---':<10}")

    # 逐查询对比
    print(f"\n逐查询对比（前10条）：")
    print(f"{'查询':<30} {'α_fixed':<10} {'α_adapt':<10} {'Q_fixed':<10} {'Q_adapt':<10} {'ΔQ':<10}")
    print("-" * 100)
    for i in range(min(10, len(fixed_results["detailed_results"]))):
        fr = fixed_results["detailed_results"][i]
        ar = adaptive_results["detailed_results"][i]
        q_diff = ar["quality"] - fr["quality"]
        query_short = fr["query"][:28]
        print(f"{query_short:<30} {0.7:<10.1f} {ar['alpha']:<10.3f} {fr['quality']:<10.4f} {ar['quality']:<10.4f} {q_diff:+.4f}")

    # 保存实验结果
    experiment_report = {
        "experiment": "fixed_alpha_vs_adaptive_alpha",
        "fixed_alpha_results": fixed_results,
        "adaptive_alpha_results": adaptive_results,
        "comparison": {
            "quality_improvement_pct": round(quality_pct, 2),
            "coverage_improvement_pct": round(coverage_pct, 2),
        }
    }

    output_path = "data/ablation_experiment.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_report, f, ensure_ascii=False, indent=2)
    print(f"\n实验报告已保存到 {output_path}")


if __name__ == "__main__":
    main()
