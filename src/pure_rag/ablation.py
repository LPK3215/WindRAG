"""
消融实验（零第三方依赖）

在纯 Python 内核上逐项开关各模块，量化每一层对检索与答案质量的贡献。
每组配置都在独立的临时索引目录里重建索引（同义词打通、新词发现会改变词表，
必须与检索时的分词口径一致），再跑同一批评测数据：

- 原句自召回：Recall@k / MRR / nDCG@k
- 同义改写查询：Recall@k（考察同义词打通与查询扩展）
- 答案质量：领域术语覆盖率（与 QA 参考答案比对）
- 拒答能力：域外问题的拒答率
- 平均检索耗时

用法：
    python src/rag_cli.py ablation            # 打印 Markdown 表格
    python src/rag_cli.py ablation --json     # 输出完整 JSON
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from pure_rag.datasets import ANSWER_LAYERS
from pure_rag.engine import RAGEngine
from pure_rag.evaluator import Evaluator

# 消融阶梯：每一行只比上一行多一项（后两组为替换型对照，用于单独评估该模块）
CONFIGURATIONS: List[Dict] = [
    {
        "name": "① 仅 BM25",
        "note": "关键词检索基线",
        "build": {"use_synonyms": False, "discover": True},
        "method": "bm25", "expand": False, "rerank": False,
    },
    {
        "name": "② 仅 TF-IDF 向量",
        "note": "语义检索基线",
        "build": {"use_synonyms": False, "discover": True},
        "method": "vector", "expand": False, "rerank": False,
    },
    {
        "name": "③ 混合 · 固定 α=0.7",
        "note": "两路加权融合",
        "build": {"use_synonyms": False, "discover": True},
        "method": "hybrid", "expand": False, "rerank": False, "alpha": 0.7,
    },
    {
        "name": "④ + 自适应 α",
        "note": "按查询特征预测权重",
        "build": {"use_synonyms": False, "discover": True},
        "method": "hybrid", "expand": False, "rerank": False,
    },
    {
        "name": "⑤ + 同义词打通",
        "note": "索引期标准词↔别名扩展",
        "build": {"use_synonyms": True, "discover": True},
        "method": "hybrid", "expand": False, "rerank": False,
    },
    {
        "name": "⑥ + 查询扩展",
        "note": "别名/关键词变体多路召回 + RRF",
        "build": {"use_synonyms": True, "discover": True},
        "method": "hybrid", "expand": True, "rerank": False,
    },
    {
        "name": "⑦ + MMR 重排",
        "note": "抑制内容重复的文档",
        "build": {"use_synonyms": True, "discover": True},
        "method": "hybrid", "expand": True, "rerank": True,
    },
    {
        "name": "⑧ + 知识图谱（完整）",
        "note": "图谱上下文并入答案",
        "build": {"use_synonyms": True, "discover": True},
        "method": "hybrid", "expand": True, "rerank": True, "kg": True,
    },
    {
        "name": "⑨ 完整 - 新词发现",
        "note": "对照：关闭语料新词发现",
        "build": {"use_synonyms": True, "discover": False},
        "method": "hybrid", "expand": True, "rerank": True, "kg": True,
    },
]


# 固定 α 扫描的取值（含自适应 α 作对照）
ALPHA_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

# 固定 α 扫描的基础配置：只变 α，其余固定为「同义词 + 无扩展 + 无 MMR + 无图谱」
ALPHA_BASE = {
    "build": {"use_synonyms": True, "discover": True},
    "method": "hybrid", "expand": False, "rerank": False,
}


class AblationRunner:
    """逐配置重建索引并评测"""

    def __init__(self, data_dir: str = "data", top_k: int = 3, num_questions: int = 15,
                 index_dir: str = None):
        self.data_dir = data_dir
        self.top_k = top_k
        self.index_dir = Path(index_dir) if index_dir else Path(data_dir) / "pure_index"
        self.evaluator = Evaluator(data_dir)

        self.pairs = self.evaluator.load_qa_pairs(num_questions)
        self.negatives = self.evaluator.load_negatives()
        # 分层评测集（由语料自动推导，条数随语料变化）比原来的 15 条 QA 更能区分配置差异
        from pure_rag.datasets import load_dataset

        self.dataset = load_dataset(data_dir)

    # -- 单组配置 -----------------------------------------------------------
    def _evaluate(self, config: Dict, work_dir: Path) -> Dict:
        build = {"use_synonyms": True, "discover": True}
        build.update(config.get("build", {}))

        engine = RAGEngine(
            data_dir=self.data_dir,
            index_dir=str(work_dir / "index"),
            use_kg=config.get("kg", False),
            use_llm=False,
            use_synonyms=build["use_synonyms"],
        )
        engine.build(discover=build["discover"], verbose=False)

        # 逐项开关通过引擎属性表达，与 query() 内部逻辑保持一致
        engine.use_expansion = config.get("expand", True)
        engine.use_rerank = config.get("rerank", True)
        engine.alpha_override = config.get("alpha")

        method = config.get("method", "hybrid")
        # 检索指标走分层评测集，与 eval --report 口径完全一致
        layered = self.evaluator.evaluate_layers(engine, self.dataset, self.top_k, method=method)
        overall = self._aggregate(layered["layers"], ANSWER_LAYERS)
        layers = layered["layers"]

        # 答案质量与耗时走完整问答路径（含生成与拒答判定）
        coverage, latencies = 0.0, []
        for pair in self.pairs:
            start = time.time()
            result = engine.query(pair["question"], top_k=self.top_k, method=method)
            latencies.append(time.time() - start)
            coverage += self.evaluator._keyword_coverage(pair.get("answer", ""), result["answer"])
        total = len(self.pairs) or 1

        return {
            "name": config["name"],
            "note": config.get("note", ""),
            "config": {
                "method": method,
                "synonyms": build["use_synonyms"],
                "discover": build["discover"],
                "expand": config.get("expand", True),
                "rerank": config.get("rerank", True),
                "kg": config.get("kg", False),
                "alpha": config.get("alpha"),
            },
            "metrics": {
                "recall_at_k": overall["recall"],
                "mrr": overall["mrr"],
                "ndcg_at_k": overall["ndcg"],
                "rewrite_recall": layers.get("qa_rewrite", {}).get("recall", 0.0),
                "case_recall": layers.get("fault_case", {}).get("recall", 0.0),
                "near_miss_refusal": layers.get("near_miss", {}).get("refusal_rate", 0.0),
                "ood_refusal": layers.get("negative_ood", {}).get("refusal_rate", 0.0),
                "keyword_coverage": coverage / total,
                "avg_latency_ms": sum(latencies) / total * 1000,
            },
        }

    @staticmethod
    def _aggregate(layer_buckets: Dict, layer_names) -> Dict:
        """把若干回答层的指标按条目数加权聚合（用累计值算，避免二次平均）"""
        buckets = [layer_buckets[name] for name in layer_names if name in layer_buckets]
        answers = sum(bucket["answer_count"] for bucket in buckets) or 1
        return {
            "recall": sum(bucket["hits"] for bucket in buckets) / answers,
            "mrr": sum(bucket["mrr_sum"] for bucket in buckets) / answers,
            "ndcg": sum(bucket["ndcg_sum"] for bucket in buckets) / answers,
        }

    # -- 全量 ---------------------------------------------------------------
    def run(self, verbose: bool = True) -> Dict:
        """依次跑完所有配置，返回结果（含 Markdown 表格）"""
        started = time.time()
        results: List[Dict] = []
        work_root = tempfile.mkdtemp(prefix="pure_rag_ablation_")
        try:
            for index, config in enumerate(CONFIGURATIONS, 1):
                if verbose:
                    print(f"  [{index}/{len(CONFIGURATIONS)}] {config['name']} ...")
                results.append(self._evaluate(config, Path(work_root)))
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": round(time.time() - started, 2),
            "data": {"dataset_total": len(self.dataset.get("items", [])),
                     "dataset_counts": self.dataset.get("counts", {}),
                     "top_k": self.top_k},
            "results": results,
            "markdown": self.to_markdown(results),
        }
        return payload

    # -- 固定 α 扫描 --------------------------------------------------------
    def sweep_alpha(self, verbose: bool = True) -> List[Dict]:
        """
        固定 α 扫描：与「自适应 α」对照，回答"自适应到底比固定值好多少"

        只变 α，其余配置固定；adaptive 行即引擎按查询特征预测权重的结果。
        """
        configs = []
        for alpha in ALPHA_SWEEP:
            configs.append({
                **ALPHA_BASE, "alpha": alpha,
                "name": f"α = {alpha:.1f}", "note": "固定融合权重",
            })
        configs.append({
            **ALPHA_BASE, "alpha": None,
            "name": "α = 自适应", "note": "按查询特征预测（默认）",
        })

        results = []
        work_root = tempfile.mkdtemp(prefix="pure_rag_alpha_")
        try:
            for index, config in enumerate(configs, 1):
                if verbose:
                    print(f"  [{index}/{len(configs)}] {config['name']} ...")
                results.append(self._evaluate(config, Path(work_root)))
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
        return results

    # -- 输出 ---------------------------------------------------------------
    def to_markdown(self, results: List[Dict]) -> str:
        header = ("| 配置 | 说明 | Recall@k | 改写召回 | 案例召回 | MRR | "
                  "术语覆盖率 | 域外拒答 | 近邻拒答 | 耗时 |")
        divider = "|---|---|---|---|---|---|---|---|---|---|"
        lines = [header, divider]
        for item in results:
            metrics = item["metrics"]
            lines.append(
                f"| {item['name']} | {item['note']} "
                f"| {metrics['recall_at_k']:.2%} | {metrics['rewrite_recall']:.2%} "
                f"| {metrics['case_recall']:.2%} | {metrics['mrr']:.4f} "
                f"| {metrics['keyword_coverage']:.2%} | {metrics['ood_refusal']:.2%} "
                f"| {metrics['near_miss_refusal']:.2%} "
                f"| {metrics['avg_latency_ms']:.2f} ms |"
            )
        return "\n".join(lines)

    def save(self, payload: Dict) -> str:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        path = self.index_dir / "ablation.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in payload.items() if k != "markdown"},
                      f, ensure_ascii=False, indent=2)
        return str(path)
