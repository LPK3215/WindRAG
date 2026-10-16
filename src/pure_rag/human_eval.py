"""
用一批"像人会问"的口语化问题，验证 α（BM25 / TF-IDF 权重）结论是否稳健。

这批问题不来自文档字段，而是模拟一线运维工程师的自然提问：只描述现象、
省略或替换部件名、带冗余。每条标注它对应的 (部件, 故障类型) 作为弱标注，
判定"检索是否把同类工单送进前 3"。

这是上一条讨论里提议的"最小可行验证"：自己当标注员写 50 条真实问法，
看之前在小语料/合成语料上得到的「α 越小越好」结论会不会翻转。
如果是口语化查询让语义通道更有价值，高 α 应该反超低 α。

直接运行：
    python src/pure_rag/human_eval.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
QUESTIONS_PATH = ROOT / "data" / "human_questions.json"


def _field(content: str, label: str) -> str:
    match = re.search(rf"{label}：(.+)", content)
    return match.group(1).strip() if match else ""


def _ensure_corpus() -> None:
    if (ROOT / "data" / "generated").exists():
        return
    sys.path.insert(0, str(ROOT / "src"))
    from simulate_corpus import generate, save

    save(generate(scale=1.0, seed=20261016), str(ROOT / "data" / "generated"))


def main(argv=None) -> int:
    _ensure_corpus()
    from pure_rag.engine import RAGEngine

    engine = RAGEngine(use_llm=False)
    engine.ensure_loaded()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    answerable = [q for q in questions if q.get("kind") != "refuse"]

    # weighted_fusion = α·向量 + (1-α)·BM25，所以 α=1.0 是纯向量、α=0.0 是纯 BM25，
    # None 走自适应规则（当前默认）
    alphas = [("BM25-only (α=0.0)", 0.0), ("α=0.3", 0.3), ("α=0.5", 0.5),
              ("α=0.7", 0.7), ("α=0.8", 0.8), ("向量-only (α=1.0)", 1.0),
              ("adaptive (默认)", None)]

    stats = {name: {"hit": 0, "mrr": 0.0, "n": 0} for name, _ in alphas}

    for question in answerable:
        comp, fault = question["component"], question["fault"]
        for name, alpha in alphas:
            engine.alpha_override = alpha
            docs = engine.retrieve(question["q"], top_k=3)["documents"]
            matched = False
            best_rank = 0
            for rank, doc in enumerate(docs, 1):
                content = doc.get("content", "")
                if ("故障ID：" in content and _field(content, "部件") == comp
                        and _field(content, "故障类型") == fault):
                    matched = True
                    best_rank = rank
                    break
            stats[name]["n"] += 1
            if matched:
                stats[name]["hit"] += 1
                stats[name]["mrr"] += 1.0 / best_rank

    refusals = []
    for question in questions:
        if question.get("kind") != "refuse":
            continue
        engine.alpha_override = None
        result = engine.query(question["q"], top_k=3)
        refusals.append((question["q"], bool(result["confidence"].get("refused")),
                         result["confidence"].get("reason")))

    count = stats["α=0.3"]["n"]
    print(f"口语化问题（弱标注 = 部件+故障，判定前 3 是否含同类工单）：{count} 条\n")
    print(f"{'配置':<22}{'Recall@3':>10}{'MRR':>10}")
    best = max(stats, key=lambda k: stats[k]["hit"] / stats[k]["n"])
    for name, _ in alphas:
        s = stats[name]
        mark = "  ← Recall 最高" if name == best else ""
        print(f"{name:<22}{s['hit'] / s['n']:>9.1%}{s['mrr'] / s['n']:>10.4f}{mark}")

    print("\n无效组合拒答检查：")
    for q, refused, reason in refusals:
        print(f"  [{'拒答' if refused else '未拒'} ] {q}   ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
