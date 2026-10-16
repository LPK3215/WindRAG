"""
命令行入口

    python src/rag_cli.py build                 构建索引
    python src/rag_cli.py add 文件/目录...       增量追加文档
    python src/rag_cli.py ask "问题"            单次问答（文档命中词高亮）
    python src/rag_cli.py chat                  交互式多轮问答（/save 可导出记录）
    python src/rag_cli.py eval                  跑评测
    python src/rag_cli.py explain "问题"        查看检索打分明细
    python src/rag_cli.py tune                  网格搜索最优参数
    python src/rag_cli.py ablation              消融实验（各模块贡献对比表）
    python src/rag_cli.py stats                 查看索引统计
    python src/rag_cli.py serve                 启动零依赖 Web 界面
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pure_rag.engine import RAGEngine
from pure_rag.evaluator import Evaluator
from pure_rag.highlight import ansi, clip_spans, to_markdown

METHODS = ("hybrid", "vector", "bm25")


def print_card(card) -> None:
    """打印故障结构化卡片（检索到故障案例时才有）"""
    if not card:
        return
    severity = f"（严重程度：{card['severity']}）" if card.get("severity") else ""
    print(f"\n【故障卡片】{card['title']}{severity}")
    for field in card.get("fields", []):
        print(f"  {field['label']}：{field['value']}")


def print_claim_locations(claims) -> None:
    """打印要点在原文中的定位（文档编号 + 字符区间）"""
    if not claims:
        return
    print("\n要点定位（原文中的字符区间）：")
    for claim in claims:
        print(f"  {claim['index']}. doc{claim['doc_id']}"
              f"（{claim['source']}/{claim['title']}）字符 {claim['start']}–{claim['end']}")


def save_transcript(history, name: str = "", data_dir: str = "data") -> str:
    """把多轮对话导出为 Markdown 文件，返回文件路径（无对话时返回空串）"""
    if not history:
        return ""

    log_dir = Path(data_dir) / "chat_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{(name or 'chat_' + stamp)}.md"

    lines = [
        "# WindRAG 问答记录",
        "",
        f"- 导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 对话轮次：{len(history)}",
        "",
    ]
    for index, turn in enumerate(history, 1):
        lines += [f"## {index}. {turn['question']}", "", turn["answer"], ""]

        card = turn.get("card")
        if card:
            severity = f"（严重程度：{card['severity']}）" if card.get("severity") else ""
            lines += [f"> **故障卡片：{card['title']}**{severity}", ">"]
            for field in card.get("fields", []):
                lines.append(f"> - **{field['label']}**：{field['value']}")
            lines.append("")

        documents = turn.get("documents") or []
        if documents:
            lines += ["<details><summary>参考文档</summary>", ""]
            for doc_index, document in enumerate(documents, 1):
                metadata = document.get("metadata", {})
                preview = document["content"][:200].replace("\n", " ")
                snippet = to_markdown(preview, clip_spans(document.get("spans", []), 200))
                lines.append(f"{doc_index}. `{metadata.get('source')}/"
                             f"{metadata.get('title', '')}` score={document.get('score', 0):.4f}")
                lines.append(f"   - {snippet}")
            lines += ["", "</details>", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def create_engine(args) -> RAGEngine:
    """按命令行参数创建引擎"""
    engine = RAGEngine(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        use_kg=not args.no_kg,
        use_llm=(False if args.no_llm else None),
    )
    if getattr(args, "min_coverage", None) is not None:
        engine.min_coverage = float(args.min_coverage)
    return engine


# ----------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------
def cmd_build(args) -> int:
    engine = create_engine(args)
    engine.build(chunk_size=args.chunk_size, overlap=args.overlap, discover=not args.no_discover)
    return 0


def cmd_stats(args) -> int:
    engine = create_engine(args)
    stats = engine.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args) -> int:
    engine = create_engine(args)
    result = engine.query(args.question, top_k=args.top_k, method=args.method,
                          alpha=args.alpha, use_kg=not args.no_kg)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("\n" + "=" * 70)
    print(f"问题：{result['question']}")
    print("=" * 70)
    confidence = result["confidence"]
    label = "拒答" if result["mode"] == "refused" else f"{result['mode']} 生成"
    print(f"答案（{label} | α={result['alpha']:.3f} | "
          f"覆盖率 {confidence['coverage']:.0%} | "
          f"{result['timings']['total'] * 1000:.1f} ms）：")
    print(result["answer"])
    print_card(result.get("card"))

    if result["kg_entities"]:
        print("\n知识图谱命中：" + "、".join(e["name"] for e in result["kg_entities"]))

    if args.show_docs:
        print_claim_locations(result.get("claims"))
        print("\n" + "-" * 70)
        print("检索到的文档（命中词高亮）：")
        for doc in result["retrieved_docs"]:
            metadata = doc["metadata"]
            print(f"\n[doc{doc['doc_id']} | {doc['score']:.4f}] {metadata.get('source')}"
                  f" / {metadata.get('title', '')} / chunk {metadata.get('chunk_id')}")
            limit = 300
            preview = doc["content"][:limit].replace("\n", " ")
            print(ansi(preview, clip_spans(doc.get("spans", []), limit), args.color))
            if doc.get("matched"):
                print(f"    命中词：{'、'.join(doc['matched'])}")
    return 0


def _finish_chat(history, args) -> int:
    """退出前按 --log 配置导出会话记录"""
    if args.log is not None and history:
        path = save_transcript(history, args.log, args.data_dir)
        print(f"\n会话已保存到 {path}")
    print("再见！")
    return 0


def cmd_chat(args) -> int:
    engine = create_engine(args)
    engine.ensure_loaded()

    print("=" * 70)
    print("风电领域智能问答（纯 Python 版）—— 输入 /help 查看命令，/exit 退出")
    print("=" * 70)

    history = []
    while True:
        try:
            question = input("\n你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            return _finish_chat(history, args)

        if not question:
            continue
        if question in ("/exit", "/quit", "exit", "quit"):
            return _finish_chat(history, args)
        if question == "/help":
            print("  /docs    切换是否显示检索文档（当前：%s）"
                  % ("显示" if args.show_docs else "隐藏"))
            print("  /alpha N 固定融合权重 α（0~1），/alpha auto 恢复自适应")
            print("  /method  切换检索方法：" + "/".join(METHODS))
            print("  /save N  导出本次对话为 Markdown（可指定文件名）")
            print("  /exit    退出")
            continue
        if question.startswith("/save"):
            saved = save_transcript(history, question.replace("/save", "").strip(),
                                    args.data_dir)
            print(f"会话已保存到 {saved}" if saved else "当前还没有可保存的对话")
            continue
        if question == "/docs":
            args.show_docs = not args.show_docs
            print(f"检索文档：{'显示' if args.show_docs else '隐藏'}")
            continue
        if question.startswith("/alpha"):
            value = question.replace("/alpha", "").strip()
            if value in ("", "auto"):
                args.alpha = None
                print("已恢复自适应 α 预测")
            else:
                try:
                    args.alpha = max(0.0, min(1.0, float(value)))
                    print(f"已固定 α = {args.alpha}")
                except ValueError:
                    print("用法：/alpha 0.6 或 /alpha auto")
            continue
        if question == "/method":
            index = METHODS.index(args.method)
            args.method = METHODS[(index + 1) % len(METHODS)]
            print(f"检索方法：{args.method}")
            continue

        result = engine.query(question, top_k=args.top_k, method=args.method,
                              alpha=args.alpha, history=history)
        print(f"\n助手 > {result['answer']}")
        print_card(result.get("card"))
        print(f"\n(α={result['alpha']:.3f} | 覆盖率 {result['confidence']['coverage']:.0%} | "
              f"{result['timings']['total'] * 1000:.0f} ms | "
              f"参考文档 {len(result['retrieved_docs'])} 篇)")

        if args.show_docs:
            for doc in result["retrieved_docs"]:
                metadata = doc["metadata"]
                print(f"  [{doc['score']:.4f}] {metadata.get('source')} / "
                      f"{metadata.get('title', '')}：{doc['content'][:120]}".replace("\n", " "))

        history.append({
            "question": question,
            "answer": result["answer"],
            "documents": result["retrieved_docs"],
            "card": result.get("card"),
        })


def cmd_eval(args) -> int:
    engine = create_engine(args)
    evaluator = Evaluator(args.data_dir)
    if args.report:
        report = evaluator.build_report(engine, top_k=args.top_k, method=args.method,
                                        rebuild=args.rebuild)
        print(report["markdown"])
        if not args.no_save:
            print(f"\n报告已保存到 {evaluator.save_markdown_report(report, args.report_path)}")
        return 0

    result = evaluator.evaluate(engine, num_questions=args.num, top_k=args.top_k,
                               method=args.method, robust=args.robust,
                               negative=args.negative)
    evaluator.print_summary(result)
    if not args.no_save:
        evaluator.save_report(result)

    # 质量门禁：供 CI 使用，指标不达标时以非零退出码结束
    if args.min_recall is not None:
        recall = result["metrics"]["recall_at_k"]
        if recall < args.min_recall:
            print(f"\n[gate] Recall@{args.top_k} = {recall:.2%} < 要求 {args.min_recall:.2%}，构建失败")
            return 1
        print(f"\n[gate] Recall@{args.top_k} = {recall:.2%} ≥ 要求 {args.min_recall:.2%}，通过")
    return 0


def cmd_explain(args) -> int:
    engine = create_engine(args)
    result = engine.explain(args.question, top_k=args.top_k, method=args.method,
                            alpha=args.alpha)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("\n" + "=" * 70)
    print(f"查询：{result['question']}")
    print(f"分词：{' | '.join(result['tokens'])}")
    print(f"融合权重 α = {result['alpha']:.3f}（{result['method']}，仅两路基础融合）")
    print("=" * 70)

    for index, doc in enumerate(result["documents"], 1):
        print(f"\n[{index}] 融合分 {doc['score']:.4f}  {doc['source']} / {doc['title']}")
        print(f"    内容：{doc['preview']}")
        contributions = "  ".join(
            f"{term['term']}(bm25={term['bm25']:.3f} vec={term['tfidf']:.4f})"
            for term in doc["terms"]
        )
        print(f"    词项贡献：{contributions}")
    return 0


def cmd_tune(args) -> int:
    from pure_rag.tuner import Tuner

    tuner = Tuner(data_dir=args.data_dir, index_dir=args.index_dir,
                  top_k=args.top_k, num_questions=args.num)
    outcome = tuner.run(quick=args.quick)

    # 用最优分块参数重建生产索引，使调优结果立即生效
    print("\n按最优参数重建索引...")
    engine = create_engine(args)
    engine.apply_tuning(outcome["best"])
    engine.build(verbose=True)
    return 0


def cmd_add(args) -> int:
    engine = create_engine(args)
    summary = engine.add_files(args.paths, verbose=not args.json)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_ablation(args) -> int:
    from pure_rag.ablation import AblationRunner

    runner = AblationRunner(data_dir=args.data_dir, top_k=args.top_k,
                            num_questions=args.num, index_dir=args.index_dir)
    print("开始消融实验（每组配置都重建索引后评测）…")
    payload = runner.run()
    print("\n固定 α 扫描…")
    payload["alpha_sweep"] = runner.sweep_alpha()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\n## 消融实验\n")
        print(payload["markdown"])
        print("\n## 固定 α 扫描（与自适应 α 对照）\n")
        print(runner.to_markdown(payload["alpha_sweep"]))
        print(f"\n数据：{payload['data']}；总耗时 {payload['elapsed']}s")

    if not args.no_save:
        print(f"\n结果已保存到 {runner.save(payload)}")
    return 0


def cmd_serve(args) -> int:
    from pure_rag.server import serve

    engine = create_engine(args)
    engine.ensure_loaded()
    serve(engine, host=args.host, port=args.port)
    return 0


# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_cli", description="WindRAG 纯 Python 检索内核（零第三方依赖）"
    )
    parser.add_argument("--data-dir", default="data", help="数据目录（默认 data）")
    parser.add_argument("--index-dir", default=None, help="索引目录（默认 data/pure_index）")
    parser.add_argument("--no-kg", action="store_true", help="关闭知识图谱增强")
    parser.add_argument("--no-llm", action="store_true", help="强制使用抽取式生成")
    parser.add_argument("--no-color", action="store_true", help="关闭终端高亮")
    parser.add_argument("--min-coverage", type=float, default=None,
                        help="拒答阈值：查询词覆盖率低于该值时拒答（默认 0.35，0 为关闭）")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="构建索引")
    build.add_argument("--chunk-size", type=int, default=None,
                       help="分块长度（默认取调优结果，否则 400）")
    build.add_argument("--overlap", type=int, default=None,
                       help="分块重叠长度（默认取调优结果，否则 80）")
    build.add_argument("--no-discover", action="store_true", help="关闭语料新词发现")
    build.set_defaults(func=cmd_build)

    ask = subparsers.add_parser("ask", help="单次问答")
    ask.add_argument("question", help="用户问题")
    ask.add_argument("-k", "--top-k", type=int, default=3, help="检索文档数")
    ask.add_argument("-m", "--method", choices=METHODS, default="hybrid")
    ask.add_argument("-a", "--alpha", type=float, default=None, help="固定融合权重")
    ask.add_argument("--show-docs", action="store_true", help="打印检索到的文档")
    ask.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    ask.set_defaults(func=cmd_ask)

    chat = subparsers.add_parser("chat", help="交互式多轮问答")
    chat.add_argument("-k", "--top-k", type=int, default=3)
    chat.add_argument("-m", "--method", choices=METHODS, default="hybrid")
    chat.add_argument("-a", "--alpha", type=float, default=None)
    chat.add_argument("--show-docs", action="store_true")
    chat.add_argument("--log", nargs="?", const="", default=None,
                      help="退出时自动导出会话（可跟文件名）")
    chat.set_defaults(func=cmd_chat)

    add = subparsers.add_parser("add", help="增量追加文档（不重建整个索引）")
    add.add_argument("paths", nargs="+", help="待追加的文件或目录")
    add.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    add.set_defaults(func=cmd_add)

    evaluate = subparsers.add_parser("eval", help="运行检索评测")
    evaluate.add_argument("-n", "--num", type=int, default=15, help="评测问题数")
    evaluate.add_argument("-k", "--top-k", type=int, default=3)
    evaluate.add_argument("-m", "--method", choices=METHODS, default="hybrid")
    evaluate.add_argument("--robust", action="store_true",
                          help="额外评测同义词/关键词改写查询的鲁棒性")
    evaluate.add_argument("--negative", action="store_true",
                          help="额外评测拒答能力与拒答阈值标定")
    evaluate.add_argument("--min-recall", type=float, default=None,
                          help="质量门禁：Recall@k 低于该值时以非零退出码结束")
    evaluate.add_argument("--report", action="store_true",
                          help="生成分层评测报告（含按来源/问题类型的召回与阈值标定）")
    evaluate.add_argument("--rebuild", action="store_true",
                          help="重新从语料生成评测集（默认复用已生成的 eval_set.json）")
    evaluate.add_argument("--report-path", default="data/pure_index/report.md",
                          help="报告输出路径（同名 .json 一并写出）")
    evaluate.add_argument("--no-save", action="store_true", help="不保存报告文件")
    evaluate.set_defaults(func=cmd_eval)

    explain = subparsers.add_parser("explain", help="查看检索打分明细（词项级贡献）")
    explain.add_argument("question", help="用户问题")
    explain.add_argument("-k", "--top-k", type=int, default=3)
    explain.add_argument("-m", "--method", choices=METHODS, default="hybrid")
    explain.add_argument("-a", "--alpha", type=float, default=None)
    explain.add_argument("--json", action="store_true", help="以 JSON 输出")
    explain.set_defaults(func=cmd_explain)

    tune = subparsers.add_parser("tune", help="网格搜索最优检索参数")
    tune.add_argument("--quick", action="store_true", help="只跑 8 组快速网格")
    tune.add_argument("-k", "--top-k", type=int, default=3, help="评测用 top-k")
    tune.add_argument("-n", "--num", type=int, default=15, help="评测问题数")
    tune.set_defaults(func=cmd_tune)

    ablation = subparsers.add_parser("ablation", help="消融实验：逐项评估各模块贡献")
    ablation.add_argument("-n", "--num", type=int, default=15, help="评测问题数")
    ablation.add_argument("-k", "--top-k", type=int, default=3, help="评测用 top-k")
    ablation.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    ablation.add_argument("--no-save", action="store_true", help="不落盘")
    ablation.set_defaults(func=cmd_ablation)

    stats = subparsers.add_parser("stats", help="查看索引统计")
    stats.set_defaults(func=cmd_stats)

    serve_parser = subparsers.add_parser("serve", help="启动 Web 界面")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # 仅在真实终端下启用 ANSI 高亮，避免重定向到文件时出现转义字符
    args.color = not args.no_color and sys.stdout.isatty()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
