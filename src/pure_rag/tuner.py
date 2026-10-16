"""
超参数网格搜索（零第三方依赖）

在 QA 自召回指标上搜索分块与检索参数，选出综合表现最好的组合，
写入 <index_dir>/tuning.json，引擎下次启动时自动读取生效（分块参数在下次 build 生效）。

综合得分 = 0.40·Recall@k + 0.35·MRR + 0.25·nDCG@k
"""

import itertools
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, List

from pure_rag.datasets import ANSWER_LAYERS, load_dataset
from pure_rag.engine import RAGEngine
from pure_rag.evaluator import Evaluator

# 完整网格：144 组，小语料约 10 秒
#
# 说明：这里不搜索 use_expansion —— 索引期同义词扩展已经覆盖了大部分场景，
# 查询扩展在自召回/改写指标上的收益低于这批小样本的噪声水平，
# 因此保持开启（可用 engine.use_expansion = False 手动关闭）。
DEFAULT_GRID: Dict[str, List] = {
    "chunk_size": [300, 400, 600],
    "overlap": [60, 100],
    # 消融实验显示本语料更依赖 BM25 的精确匹配，因此把低 α 也纳入搜索
    "alpha_override": [None, 0.3, 0.5, 0.65, 0.8],
    "mmr_lambda": [0.6, 0.75, 1.0],
    "discover": [True, False],
}

# 快速网格：4 组，用于冒烟检查
QUICK_GRID: Dict[str, List] = {
    "chunk_size": [400],
    "overlap": [80],
    "alpha_override": [None, 0.3],
    "mmr_lambda": [0.75, 1.0],
    "discover": [True],
}

TUNING_FILE = "tuning.json"


class Tuner:
    """网格搜索调参器"""

    def __init__(self, data_dir: str = "data", index_dir: str = None,
                 top_k: int = 3, num_questions: int = 15):
        self.data_dir = data_dir
        self.index_dir = Path(index_dir) if index_dir else Path(data_dir) / "pure_index"
        self.top_k = top_k
        self.evaluator = Evaluator(data_dir)

        # 目标集用分层评测集里的"回答层"（共 80 条：QA 原句/改写/案例/文档），
        # 比只用 15 条 QA 更能区分配置差异
        self.dataset = load_dataset(data_dir)
        self.positives = [item for item in self.dataset.get("items", [])
                          if item["layer"] in ANSWER_LAYERS]

    # -- 网格 ---------------------------------------------------------------
    @staticmethod
    def combinations(grid: Dict[str, List]) -> Iterator[Dict]:
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            yield dict(zip(keys, values))

    # -- 单组评估 -----------------------------------------------------------
    def evaluate_params(self, params: Dict, work_dir: Path) -> Dict:
        """在临时索引目录上构建并评测一组参数"""
        engine = RAGEngine(
            data_dir=self.data_dir,
            index_dir=str(work_dir / "index"),
            use_kg=False,          # 知识图谱不参与文档排序，评测时关闭以加速
            use_llm=False,
        )
        engine.apply_tuning(params)
        engine.build(chunk_size=params["chunk_size"], overlap=params["overlap"],
                     discover=params["discover"], verbose=False)

        layered = self.evaluator.evaluate_layers(engine, self.dataset, self.top_k)
        buckets = [bucket for name, bucket in layered["layers"].items()
                   if name in ANSWER_LAYERS]
        answers = sum(bucket["answer_count"] for bucket in buckets) or 1

        metrics = {
            "recall": sum(bucket["hits"] for bucket in buckets) / answers,
            "mrr": sum(bucket["mrr_sum"] for bucket in buckets) / answers,
            "ndcg": sum(bucket["ndcg_sum"] for bucket in buckets) / answers,
        }
        metrics["rewrite_recall"] = layered["layers"].get("qa_rewrite", {}).get("recall", 0.0)
        metrics["score"] = round(0.40 * metrics["recall"] + 0.35 * metrics["mrr"]
                                 + 0.25 * metrics["ndcg"], 6)
        engine._cache.clear()      # 释放缓存，避免多轮搜索堆积内存
        return metrics

    # -- 搜索 ---------------------------------------------------------------
    def run(self, grid: Dict[str, List] = None, quick: bool = False,
            verbose: bool = True) -> Dict:
        """执行网格搜索并保存结果"""
        grid = grid or (QUICK_GRID if quick else DEFAULT_GRID)
        combos = list(self.combinations(grid))
        if verbose:
            print(f"开始网格搜索：{len(combos)} 组参数 × {len(self.positives)} 条回答层查询")
            print(f"搜索空间：{grid}")

        start = time.time()
        results: List[Dict] = []
        step = max(1, len(combos) // 10)

        with tempfile.TemporaryDirectory(prefix="pure_rag_tune_") as tmp:
            work_dir = Path(tmp)
            for index, params in enumerate(combos, 1):
                results.append({**params, **self.evaluate_params(params, work_dir)})
                if verbose and index % step == 0:
                    best = max(results, key=lambda item: item["score"])
                    print(f"  [{index}/{len(combos)}] score={best['score']:.4f} "
                          f"召回={best['recall']:.0%} "
                          f"改写={best['rewrite_recall']:.0%} mrr={best['mrr']:.4f} "
                          f"(chunk={best['chunk_size']} α={best['alpha_override']} "
                          f"λ={best['mmr_lambda']} disc={best['discover']})")

        results.sort(key=lambda item: (-item["score"], -item["recall"], -item["mrr"]))
        best = results[0]
        elapsed = round(time.time() - start, 2)
        self.save(best, results, elapsed)

        if verbose:
            print(f"\n搜索完成（{elapsed}s），最优参数：")
            print(f"  chunk_size={best['chunk_size']} overlap={best['overlap']} "
                  f"discover={best['discover']}")
            print(f"  alpha_override={best['alpha_override']} mmr_lambda={best['mmr_lambda']} "
                  f"use_expansion={best.get('use_expansion', True)}（默认开启，未参与搜索）")
            print(f"  Recall@{self.top_k}：回答层 {best['recall']:.2%} / "
                  f"改写层 {best['rewrite_recall']:.2%}")
            print(f"  MRR={best['mrr']:.4f} nDCG={best['ndcg']:.4f} score={best['score']:.4f}")
            print(f"  已写入 {self.index_dir / TUNING_FILE}（分块参数在下次 build 时生效）")

        return {"best": best, "top": results[:10], "evaluated": len(results), "elapsed": elapsed}

    def save(self, best: Dict, results: List[Dict], elapsed: float = 0.0) -> None:
        """保存调优结果"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "best": best,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluated": len(results),
            "elapsed": elapsed,
            "grid": DEFAULT_GRID,
            "top": results[:5],
        }
        with open(self.index_dir / TUNING_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
