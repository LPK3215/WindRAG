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
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

from pure_rag.datasets import rewrite_query
from pure_rag.text_utils import DOMAIN_WORDS, normalize


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

    # -- 分层评测（目标可以是文本签名，也可以是整篇文档） --------------------
    @staticmethod
    def _resolve_target(item) -> tuple:
        """把评测条目归一化成 (查询, 目标文本, 目标文档, 目标部件, 目标故障)"""
        if isinstance(item, dict):
            return (item.get("query", ""), item.get("target_text"), item.get("target_doc"),
                    item.get("target_component"), item.get("target_fault"))
        if isinstance(item, (tuple, list)):
            return (item[0], item[1] if len(item) > 1 else None, None,
                    item[2] if len(item) > 2 else None, item[3] if len(item) > 3 else None)
        return item, None, None, None, None

    @staticmethod
    def _chunk_fields(content: str) -> Dict[str, str]:
        """从工单块内容里取出部件 / 故障类型"""
        out = {}
        for label in ("部件", "故障类型"):
            match = re.search(rf"{label}：(.+)", content)
            if match:
                out[label] = match.group(1).strip()
        return out

    @staticmethod
    def rank_of_item(engine, item, ranked_ids: List[int]) -> int:
        """条目在给定检索结果里的排名（1 起，未命中 0）"""
        _query, target_text, target_doc, target_component, target_fault = Evaluator._resolve_target(item)
        target = normalize(target_text) if target_text else ""
        for rank, doc_id in enumerate(ranked_ids, 1):
            chunk = engine.chunks[doc_id]
            if target and target in normalize(chunk["content"]):
                return rank
            if target_doc:
                metadata = chunk.get("metadata", {})
                if (metadata.get("source") == target_doc.get("source")
                        and metadata.get("title") == target_doc.get("title")):
                    return rank
            # 口语化层：命中「同类工单」即算对（不限定某一条），按部件+故障类型匹配
            if target_component and "故障ID：" in chunk["content"]:
                fields = Evaluator._chunk_fields(chunk["content"])
                if fields.get("部件") == target_component and fields.get("故障类型") == target_fault:
                    return rank
        return 0

    def evaluate_layers(self, engine, dataset: Dict = None, top_k: int = 3,
                        method: str = "hybrid", verbose: bool = False) -> Dict:
        """
        按层评测：回答层看召回（Recall@k / MRR / nDCG@k），拒答层看拒答率

        同时按数据来源（fault_cases / technical_docs / ...）与问题类型（故障诊断 / ...）分组，
        便于定位系统在哪一类问题上更弱。
        """
        from pure_rag.datasets import load_dataset

        dataset = dataset or load_dataset(self.data_dir)
        engine.ensure_loaded()

        layers, by_source, by_qtype = {}, {}, {}
        details = []

        for item in dataset.get("items", []):
            begin = time.time()
            result = engine.query(item["query"], top_k=top_k, method=method)
            latency_ms = (time.time() - begin) * 1000
            ranked = [doc["doc_id"] for doc in result["retrieved_docs"]]
            rank = (self.rank_of_item(engine, item, ranked)
                    if item.get("expect") == "answer" else 0)
            refused = bool(result["confidence"].get("refused"))

            record = {
                "query": item["query"], "layer": item["layer"],
                "source": item.get("source", ""), "qtype": item.get("qtype", ""),
                "rank": rank, "refused": refused, "latency_ms": latency_ms,
                "answer": result["answer"],
            }
            details.append(record)

            for bucket_key, key in ((layers, item["layer"]), (by_source, item.get("source")),
                                    (by_qtype, item.get("qtype"))):
                bucket = bucket_key.setdefault(key or "其他", {
                    "count": 0, "answer_count": 0, "refuse_count": 0,
                    "hits": 0, "mrr_sum": 0.0, "ndcg_sum": 0.0, "refused": 0})
                bucket["count"] += 1
                # 召回与拒答分别只在自己的分母上统计，避免"应拒答"的条目被算成未命中
                if item.get("expect") == "answer":
                    bucket["answer_count"] += 1
                    if rank:
                        bucket["hits"] += 1
                        bucket["mrr_sum"] += 1.0 / rank
                        bucket["ndcg_sum"] += 1.0 / math.log2(rank + 1)
                else:
                    bucket["refuse_count"] += 1
                    if refused:
                        bucket["refused"] += 1
            if verbose:
                mark = "拒答" if refused else (f"rank={rank}" if rank else "未命中")
                print(f"  [{item['layer']}] {mark:<8} {item['query'][:36]}")

        def _finalize(buckets: Dict) -> Dict:
            for bucket in buckets.values():
                answers = bucket["answer_count"] or 1
                refuses = bucket["refuse_count"] or 1
                bucket["recall"] = bucket["hits"] / answers
                bucket["mrr"] = bucket["mrr_sum"] / answers
                bucket["ndcg"] = bucket["ndcg_sum"] / answers
                bucket["refusal_rate"] = bucket["refused"] / refuses
            return buckets

        latencies = sorted(record["latency_ms"] for record in details)
        return {
            "dataset": {"counts": dataset.get("counts", {}),
                        "total": len(dataset.get("items", []))},
            "config": {"top_k": top_k, "method": method},
            "layers": _finalize(layers),
            "by_source": _finalize(by_source),
            "by_qtype": _finalize(by_qtype),
            "performance": {
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "p95_latency_ms": (latencies[int(len(latencies) * 0.95)] if latencies else 0.0),
            },
            "details": details,
        }

    # -- 评测报告 -----------------------------------------------------------
    def build_report(self, engine, top_k: int = 3, method: str = "hybrid",
                     rebuild: bool = False, verbose: bool = False) -> Dict:
        """
        生成可复现的评测报告

        报告固定包含：环境与语料概览、评测集构成、分层指标、按来源/问题类型的召回、
        拒答阈值标定、以及已知边界说明 —— 便于把同一份报告直接贴进论文或 PR。
        """
        from pure_rag.datasets import load_dataset

        engine.ensure_loaded()
        dataset = load_dataset(self.data_dir, rebuild=rebuild)
        layered = self.evaluate_layers(engine, dataset, top_k=top_k, method=method,
                                       verbose=verbose)
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "markdown": self._render_report(engine, dataset, layered, top_k, method),
            "layers": layered,
            "stats": engine.stats(),
            "dataset": dataset.get("counts", {}),
        }

    def _render_report(self, engine, dataset: Dict, layered: Dict,
                       top_k: int, method: str) -> str:
        import platform

        from pure_rag import __version__
        from pure_rag.datasets import LAYERS

        stats = engine.stats()
        lines = [
            "# WindRAG 检索评测报告",
            "",
            f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 检索引擎：纯 Python 内核 v{__version__}（零第三方依赖，Python {platform.python_version()}）",
            f"- 语料：{stats['source_documents']} 篇 → {stats['chunks']} 个分块，词表 {stats['vocabulary']} 词",
            f"- 评测集：{layered['dataset']['total']} 条（{len(dataset.get('counts', {}))} 层，"
            f"由语料自动推导）",
            f"- 参数：top_k={top_k}、method={method}、融合权重 α "
            f"{'自适应' if engine.alpha_override is None else f'固定 {engine.alpha_override}'}、"
            f"拒答阈值 {engine.min_coverage:.2f}",
            "",
            "## 评测集构成",
            "",
            "| 层 | 说明 | 条数 | 期望行为 |",
            "|---|---|---|---|",
        ]
        for layer, label in LAYERS.items():
            count = dataset.get("counts", {}).get(layer, 0)
            if not count:
                continue
            bucket = layered["layers"].get(layer)
            if bucket and bucket["answer_count"] and bucket["refuse_count"]:
                expect = "命中目标 / 拒答"
            elif bucket and bucket["answer_count"]:
                expect = "命中目标"
            else:
                expect = "拒答"
            lines.append(f"| `{layer}` | {label} | {count} | {expect} |")

        lines += ["", "## 分层指标", "",
                  f"| 层 | 条数 | Recall@{top_k} | MRR | nDCG@{top_k} | 拒答率 |",
                  "|---|---|---|---|---|---|"]
        for layer, label in LAYERS.items():
            bucket = layered["layers"].get(layer)
            if not bucket:
                continue
            # 有可回答条目的层就显示召回（不再硬编码 ANSWER_LAYERS，便于混合层也正常呈现）
            if bucket["answer_count"]:
                lines.append(f"| {label} | {bucket['count']} | {bucket['recall']:.2%} "
                             f"| {bucket['mrr']:.4f} | {bucket['ndcg']:.4f} "
                             f"| {bucket['refusal_rate']:.2%} |")
            else:
                lines.append(f"| {label} | {bucket['count']} | — | — | — "
                             f"| {bucket['refusal_rate']:.2%} |")

        for title, key in (("按数据来源", "by_source"), ("按问题类型", "by_qtype")):
            groups = layered[key]
            if not groups:
                continue
            lines += ["", f"## {title}", "",
                      f"| 分组 | 回答条数 | Recall@{top_k} | MRR | 拒答条数 | 拒答率 |",
                      "|---|---|---|---|---|---|"]
            for name, bucket in sorted(groups.items(),
                                       key=lambda kv: -kv[1]["count"]):
                lines.append(
                    f"| {name or '未知'} | {bucket['answer_count']} | {bucket['recall']:.2%} "
                    f"| {bucket['mrr']:.4f} | {bucket['refuse_count']} "
                    f"| {bucket['refusal_rate']:.2%} |")

        performance = layered.get("performance", {})
        if performance:
            lines += ["", "## 性能与规模", "",
                      "| 指标 | 数值 |", "|---|---|",
                      f"| 语料 / 分块 | {stats['source_documents']} 篇 / {stats['chunks']} 块 |",
                      f"| 平均分块长度 | {stats.get('avg_chunk_chars', '-')} 字符 |",
                      f"| 平均查询耗时（含生成） | {performance['avg_latency_ms']:.2f} ms |",
                      f"| P95 查询耗时 | {performance['p95_latency_ms']:.2f} ms |",
                      f"| 评测总查询数 | {layered['dataset']['total']} |"]

        rows = self.calibrate_threshold(engine, num_questions=15, top_k=top_k, method=method)
        if rows:
            lines += ["", "## 拒答阈值标定", "",
                      f"| 拒答阈值 | 正例回答率 | 正例召回@{top_k} | 反例拒答率 |",
                      "|---|---|---|---|"]
            for row in rows:
                lines.append(f"| {row['threshold']:.2f} | {row['answer_rate']:.2%} "
                             f"| {row['positive_recall']:.2%} "
                             f"| {row['negative_refusal_rate']:.2%} |")

        lines += [
            "", "## 说明与已知边界", "",
            "- 评测集由语料自动推导（QA 对原句/同义改写、故障案例「部件+故障类型」、"
            "文档标题、近邻负例、人工负例），**无需人工标注**，因此可随语料一起演进、可复现生成。",
            "- 召回类指标衡量的是「目标块是否进入 top-k」，不是「答案是否正确」；"
            "答案质量目前只用领域术语覆盖率作代理。",
            "- `near_miss`（近邻负例）用的是语料里不存在的「部件 × 故障」组合。"
            "覆盖率判据对它无能为力（查询词全是领域实词），改由知识图谱的 "
            "`部件 --HAS_FAULT--> 故障` 归属关系校验：搭配不存在时直接拒答，"
            "并指出该故障实际属于哪个部件。",
            "- `negative_missing`（域内缺资料）同样只能被部分识别，这是覆盖率指标的固有边界。",
            "- `human_like`（口语化查询）是本评测集唯一带「真人问法」性质的层：问题由人写下、"
            "标注弱相关部件+故障，专门暴露合成评测高估可用率的盲区（详见 README「口语化查询验证」）。"
            "它目前仍是自标注，应逐步替换为真实用户问法 + 相关性标注，但会一直作为常驻层存在。",
        ]
        return "\n".join(lines)

    def save_markdown_report(self, report: Dict, path: str = "data/pure_index/report.md") -> str:
        """落盘报告：Markdown 给人看，同名的 .json 给后续对比用"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report["markdown"], encoding="utf-8")
        structured = {key: value for key, value in report.items() if key != "markdown"}
        with open(output.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        return str(output)

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
            rewritten = rewrite_query(pair["question"])
            if rewritten and rewritten != pair["question"]:
                samples.append({"question": rewritten, "target": pair["question"]})
        return samples[:limit] if limit else samples

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
                mark = "hit" if refused else "miss"
                print(f"  {mark:<4} 覆盖率={result['confidence']['coverage']:.2f} {item['query'][:36]}")

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

        for item in items:
            # _resolve_target 固定返回五元组（查询/目标文本/目标文档/目标部件/目标故障），
            # 这里只按目标文本人工判定命中，其余字段置空不影响排名计算
            query, _target_text, _target_doc, _component, _fault = self._resolve_target(item)
            retrieval = engine.retrieve(query, top_k=top_k, method=method)
            rank = self.rank_of_item(engine, item,
                                     [doc["doc_id"] for doc in retrieval["documents"]])
            recall += 1.0 if rank else 0.0
            mrr += 1.0 / rank if rank else 0.0
            ndcg += 1.0 / math.log2(rank + 1) if rank else 0.0
            details.append({"query": query, "rank": rank})
            if verbose:
                mark = "hit" if rank else "miss"
                print(f"  {mark:<4} rank={rank or '-'}  {query[:40]}")

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
                mark = "hit" if 0 < rank <= top_k else "miss"
                print(f"  {mark:<4} [{index}/{len(pairs)}] rank={rank or '-'} "
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
