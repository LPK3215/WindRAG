"""
检索与答案质量评估（零第三方依赖）

由于缺少人工标注的相关性数据，这里采用「自召回」评测：
把 QA 对的问题当查询，看能否召回包含该 QA 对的文档块。
这是一个无需额外标注、可复现的检索质量代理指标：
    Recall@k  —— top-k 内命中率
    MRR       —— 命中位置的倒数排名均值
    nDCG@k    —— 二值相关性下的归一化折损累计增益
答案质量用参考答案中领域术语的覆盖率衡量，性能用响应时间分布衡量。
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

from pure_rag.text_utils import DOMAIN_WORDS, SYNONYMS, normalize


class Evaluator:
    """RAG 系统评估器"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    # -- 数据 ---------------------------------------------------------------
    def load_qa_pairs(self, limit: int = None) -> List[Dict]:
        """加载问答对作为评测集"""
        path = self.data_dir / "synthetic" / "qa_pairs.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            pairs = json.load(f).get("qa_pairs", [])
        return pairs[:limit] if limit else pairs

    def build_rewrites(self, pairs: List[Dict] = None, limit: int = None) -> List[Dict]:
        """
        构造「改写查询」评测集，用来衡量语义鲁棒性（而非仅仅是原句匹配）：

        1. 同义词替换：把问题里的标准词换成口语别名（齿轮箱 → 变速箱）
        2. 关键词式提问：去掉"怎么/如何"，退化为关键词查询

        返回 [{"question": 改写后的问题, "target": 原问题文本}]，
        目标文档块仍是包含原问题的那一块。
        """
        pairs = self.load_qa_pairs() if pairs is None else pairs
        samples = []
        for pair in pairs:
            rewritten = self._rewrite(pair["question"])
            if rewritten and rewritten != pair["question"]:
                samples.append({"question": rewritten, "target": pair["question"]})
        return samples[:limit] if limit else samples

    @staticmethod
    def _rewrite(question: str) -> str:
        """按同义词替换优先、否则退化为关键词查询"""
        for alias, standards in SYNONYMS.items():
            for standard in standards:
                if standard in question:
                    return question.replace(standard, alias, 1)

        for phrase in ("怎么", "如何", "什么", "为什么", "哪些", "是否", "怎样"):
            if phrase in question:
                return question.replace(phrase, "").replace("？", "").replace("?", "").strip()
        return ""

    def load_negatives(self, limit: int = None) -> List[Dict]:
        """加载拒答评测集（知识库答不了的问题）"""
        path = self.data_dir / "synthetic" / "negative_queries.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f).get("negative_queries", [])
        return items[:limit] if limit else items

    def evaluate_negatives(self, engine, limit: int = None, verbose: bool = False) -> Dict:
        """
        拒答能力评测：期望系统对这些「答不了的问题」明确说不知道

        按类型分组统计：out_of_domain（域外）与 missing_in_kb（域内缺资料）。
        后者当前机制无法可靠识别，分组统计正是为了把这个边界暴露出来而不是掩盖。
        """
        negatives = self.load_negatives(limit)
        if not negatives:
            return {"count": 0, "refused": 0, "refusal_rate": 0.0,
                    "false_answer_rate": 0.0, "by_type": {}, "details": []}

        engine.ensure_loaded()
        details = []
        for item in negatives:
            result = engine.query(item["query"])
            refused = bool(result["confidence"].get("refused"))
            details.append({
                "query": item["query"],
                "type": item.get("type", "unknown"),
                "refused": refused,
                "coverage": result["confidence"]["coverage"],
                "answer": result["answer"][:60],
            })
            if verbose:
                mark = "✓" if refused else "✗"
                print(f"  {mark} 覆盖率={result['confidence']['coverage']:.2f} {item['query'][:36]}")

        by_type: Dict[str, Dict] = {}
        for item in details:
            bucket = by_type.setdefault(item["type"], {"count": 0, "refused": 0})
            bucket["count"] += 1
            bucket["refused"] += 1 if item["refused"] else 0

        total = len(details)
        refused_count = sum(1 for d in details if d["refused"])
        for bucket in by_type.values():
            bucket["refusal_rate"] = bucket["refused"] / bucket["count"]

        return {
            "count": total,
            "refused": refused_count,
            "refusal_rate": refused_count / total,
            "false_answer_rate": (total - refused_count) / total,
            "by_type": by_type,
            "details": details,
        }

    def calibrate_threshold(self, engine, thresholds: Sequence[float] = None,
                            num_questions: int = 15, top_k: int = 3,
                            method: str = "hybrid") -> List[Dict]:
        """
        拒答阈值标定：给出不同阈值下「正例回答率 / 正例召回 / 反例拒答率」的权衡表

        覆盖率只取决于检索结果、与阈值无关，因此每个问题只算一次覆盖率，再离线扫一遍阈值。
        """
        thresholds = thresholds or (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
        engine.ensure_loaded()

        positive_stats = []
        for pair in self.load_qa_pairs(num_questions):
            question = pair["question"]
            retrieval = engine.retrieve(question, top_k=top_k, method=method)
            positive_stats.append((
                engine.confidence(question, retrieval["documents"]),
                self._rank_of(engine, question, [d["doc_id"] for d in retrieval["documents"]]),
            ))

        negative_coverages = [
            engine.confidence(
                item["query"], engine.retrieve(item["query"], top_k=top_k,
                                                method=method)["documents"]
            )
            for item in self.load_negatives()
        ]
        if not positive_stats or not negative_coverages:
            return []

        rows = []
        for threshold in thresholds:
            answered = [c for c, _ in positive_stats if not engine.should_refuse(c, threshold)]
            answered_correct = sum(
                1 for c, rank in positive_stats
                if rank and not engine.should_refuse(c, threshold)
            )
            refused = sum(1 for c in negative_coverages if engine.should_refuse(c, threshold))
            rows.append({
                "threshold": threshold,
                "answer_rate": len(answered) / len(positive_stats),
                "positive_recall": answered_correct / len(positive_stats),
                "negative_refusal_rate": refused / len(negative_coverages),
            })
        return rows

    def measure(self, engine, items: List[tuple], top_k: int = 3,
                method: str = "hybrid", verbose: bool = False) -> Dict:
        """对 [(查询, 目标文本)] 计算 Recall@k / MRR / nDCG@k"""
        engine.ensure_loaded()
        recall = mrr = ndcg = 0.0
        details = []

        for query, target in items:
            retrieval = engine.retrieve(query, top_k=top_k, method=method)
            rank = self._rank_of(engine, target,
                                 [doc["doc_id"] for doc in retrieval["documents"]])
            recall += 1.0 if rank else 0.0
            mrr += 1.0 / rank if rank else 0.0
            ndcg += 1.0 / math.log2(rank + 1) if rank else 0.0
            details.append({"query": query, "rank": rank})
            if verbose:
                print(f"  {'✓' if rank else '✗'} rank={rank or '-'}  {query[:40]}")

        total = len(items) or 1
        return {
            "count": len(items),
            "recall": recall / total,
            "mrr": mrr / total,
            "ndcg": ndcg / total,
            "details": details,
        }

    # -- 指标 ---------------------------------------------------------------
    @staticmethod
    def _rank_of(engine, target_text: str, ranked_ids: List[int]) -> int:
        """返回包含目标文本的文档块排名（1 起），未命中返回 0"""
        target = normalize(target_text)
        for rank, doc_id in enumerate(ranked_ids, 1):
            if target and target in normalize(engine.chunks[doc_id]["content"]):
                return rank
        return 0

    def evaluate(self, engine, num_questions: int = 15, top_k: int = 3,
                 method: str = "hybrid", verbose: bool = True,
                 robust: bool = False, negative: bool = False) -> Dict:
        """
        跑一轮评测，返回整体指标与逐条明细

        robust=True   额外评测同义词/关键词改写查询
        negative=True 额外评测拒答能力与拒答阈值标定
        """
        pairs = self.load_qa_pairs(num_questions)
        if not pairs:
            raise RuntimeError("未找到 data/synthetic/qa_pairs.json，无法评测")

        engine.ensure_loaded()
        details: List[Dict] = []
        latencies: List[float] = []

        for index, pair in enumerate(pairs, 1):
            question = pair["question"]
            start = time.time()
            retrieval = engine.retrieve(question, top_k=top_k, method=method)
            ranked_ids = [doc["doc_id"] for doc in retrieval["documents"]]
            elapsed = time.time() - start
            latencies.append(elapsed)

            rank = self._rank_of(engine, question, ranked_ids)
            answer = engine.generator.generate(
                question, retrieval["documents"], retrieval["kg_context"], engine.tfidf.idf_of
            )["answer"]

            details.append({
                "question": question,
                "rank": rank,
                "recall": 1.0 if 0 < rank <= top_k else 0.0,
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
                "ndcg": 1.0 / math.log2(rank + 1) if rank else 0.0,
                "keyword_coverage": self._keyword_coverage(pair.get("answer", ""), answer),
                "answer_length": len(answer),
                "latency": round(elapsed, 4),
                "alpha": retrieval["alpha"],
            })

            if verbose:
                mark = "✓" if 0 < rank <= top_k else "✗"
                print(f"  {mark} [{index}/{len(pairs)}] rank={rank or '-'} "
                      f"α={retrieval['alpha']:.2f} {elapsed:.3f}s  {question[:36]}")

        total = len(details)
        hits = sum(1 for d in details if d["recall"] > 0)
        latencies.sort()

        result = {
            "config": {"top_k": top_k, "method": method, "questions": total},
            "metrics": {
                "recall_at_k": hits / total,
                "mrr": sum(d["reciprocal_rank"] for d in details) / total,
                "ndcg_at_k": sum(d["ndcg"] for d in details) / total,
                "avg_keyword_coverage": sum(d["keyword_coverage"] for d in details) / total,
                "avg_answer_length": sum(d["answer_length"] for d in details) / total,
                "avg_latency": sum(latencies) / total,
                "p95_latency": latencies[min(int(total * 0.95), total - 1)],
                "avg_alpha": sum(d["alpha"] for d in details) / total,
            },
            "details": details,
        }

        if robust:
            rewrites = self.build_rewrites(pairs)
            if verbose:
                print(f"\n  改写查询评测（{len(rewrites)} 条）：")
            result["robustness"] = self.measure(
                engine, [(s["question"], s["target"]) for s in rewrites], top_k, method,
                verbose=verbose,
            )

        if negative:
            if verbose:
                print(f"\n  拒答能力评测（{len(self.load_negatives())} 条答不了的问题）：")
            result["negatives"] = self.evaluate_negatives(engine, verbose=verbose)
            result["threshold_calibration"] = self.calibrate_threshold(
                engine, num_questions=num_questions, top_k=top_k, method=method
            )
        return result

    @staticmethod
    def _keyword_coverage(reference: str, answer: str) -> float:
        """参考答案中的领域术语有多大比例出现在生成答案中"""
        if not reference:
            return 0.0
        keywords = [w for w in DOMAIN_WORDS if len(w) >= 2 and w in reference]
        if not keywords:
            return 0.0
        answer_text = normalize(answer)
        return sum(1 for w in keywords if normalize(w) in answer_text) / len(keywords)

    # -- 输出 ---------------------------------------------------------------
    @staticmethod
    def print_summary(result: Dict) -> None:
        metrics = result["metrics"]
        config = result["config"]
        print("\n" + "=" * 62)
        print(f"RAG 检索评估报告（{config['questions']} 条 QA / top_k={config['top_k']}"
              f" / method={config['method']}）")
        print("=" * 62)
        print("【检索质量】")
        print(f"  Recall@{config['top_k']}       : {metrics['recall_at_k']:.2%}")
        print(f"  MRR               : {metrics['mrr']:.4f}")
        print(f"  nDCG@{config['top_k']}         : {metrics['ndcg_at_k']:.4f}")
        print("【答案质量】")
        print(f"  领域术语覆盖率    : {metrics['avg_keyword_coverage']:.2%}")
        print(f"  平均答案长度      : {metrics['avg_answer_length']:.0f} 字符")
        print("【性能与权重】")
        print(f"  平均响应时间      : {metrics['avg_latency'] * 1000:.1f} ms")
        print(f"  P95 响应时间      : {metrics['p95_latency'] * 1000:.1f} ms")
        print(f"  平均自适应 α      : {metrics['avg_alpha']:.3f}")

        if "robustness" in result:
            robust = result["robustness"]
            print(f"【语义鲁棒性】（{robust['count']} 条同义词/关键词改写查询）")
            print(f"  Recall@{config['top_k']}       : {robust['recall']:.2%}")
            print(f"  MRR               : {robust['mrr']:.4f}")
            print(f"  nDCG@{config['top_k']}         : {robust['ndcg']:.4f}")

        if "negatives" in result:
            negatives = result["negatives"]
            print(f"【拒答能力】（{negatives['count']} 条答不了的问题）")
            print(f"  拒答率            : {negatives['refusal_rate']:.2%}")
            for name, bucket in sorted(negatives["by_type"].items()):
                label = {"out_of_domain": "域外问题",
                         "missing_in_kb": "域内缺资料"}.get(name, name)
                print(f"  {label:<14}: {bucket['refusal_rate']:.2%} "
                      f"({bucket['refused']}/{bucket['count']})")

        if result.get("threshold_calibration"):
            print(f"【拒答阈值标定】阈值 → 正例回答率 / 正例召回@{config['top_k']} / 反例拒答率")
            for row in result["threshold_calibration"]:
                print(f"  {row['threshold']:.2f} → {row['answer_rate']:>7.2%}"
                      f" / {row['positive_recall']:>7.2%} / {row['negative_refusal_rate']:>7.2%}")
        print("=" * 62)

    @staticmethod
    def save_report(result: Dict, path: str = "data/pure_index/evaluation.json") -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"评估报告已保存到 {output}")
