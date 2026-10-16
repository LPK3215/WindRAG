"""
纯 Python RAG 引擎（零第三方依赖）

检索流程：
    查询
      ├─ 分词（领域词典 + 新词发现补充）
      ├─ 语义路：TF-IDF 稀疏向量余弦相似度
      ├─ 关键词路：BM25
      ├─ 扩展路：同义词/关键词变体查询多路召回
      ↓
    自适应 α 归一化加权融合 → RRF 合并扩展结果 → MMR 多样性重排
      ↓
    知识图谱增强 → 抽取式/LLM 答案生成
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from pure_rag.bm25 import BM25Index
from pure_rag.chunker import build_index_text, chunk_documents
from pure_rag.env import load_env
from pure_rag.fusion import (merge_scores, min_max, mmr_select, reciprocal_rank_fusion,
                             scale_by_max, weighted_fusion)
from pure_rag.generator import AnswerGenerator, LLMClient
from pure_rag.highlight import matched_terms
from pure_rag.kg import KnowledgeGraphRetriever
from pure_rag.loader import (SUPPORTED_SUFFIXES, USER_DIR_NAME, ensure_dir,
                             load_documents, read_text_file)
from pure_rag.query_analysis import adaptive_alpha, expand_queries
from pure_rag.text_utils import Tokenizer, discover_terms, find_spans
from pure_rag.tfidf import TfidfIndex

# 多轮对话中指代消解的触发词
PRONOUNS = ("它", "他", "这个", "那个", "该", "此", "上述", "刚才", "还有", "那么")

# 查询扩展召回结果的融合权重（越小越保守）
EXPANSION_WEIGHT = 0.3
# MMR 中相关性与多样性的权衡
MMR_LAMBDA = 0.75
# 分块默认参数
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 80
# 拒答阈值：IDF 加权查询词覆盖率低于该值时判定为「知识库无相关内容」
# 注意：它衡量的是「查询词在语料中的覆盖程度」，不是「语料里是否有答案」，
# 因此对「域内但资料缺失」的问题并不敏感，实测标定见 README。
DEFAULT_MIN_COVERAGE = 0.35


class RAGEngine:
    """零依赖 RAG 引擎：构建索引、混合检索、答案生成"""

    def __init__(self, data_dir: str = "data", index_dir: str = None, use_kg: bool = True,
                 use_llm: Optional[bool] = None, kg_threshold: float = 0.2,
                 cache_size: int = 64, config_path: str = None,
                 use_synonyms: bool = True):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir) if index_dir else self.data_dir / "pure_index"

        # 加载 .env 配置（零依赖实现，替代 python-dotenv）
        config_path = config_path or (self.data_dir.parent / "config.env")
        load_env(config_path)

        self.use_synonyms = use_synonyms
        self.tokenizer = Tokenizer(use_synonyms=use_synonyms)
        self.terms: List[str] = []
        self.bm25 = BM25Index()
        self.tfidf = TfidfIndex()
        self.chunks: List[Dict] = []
        self._token_lists: List[List[str]] = []   # 与 chunks 一一对应的词序列（增量追加复用）
        self.source_count = 0
        self.built_at: Optional[str] = None

        self.kg = KnowledgeGraphRetriever(
            str(self.data_dir / "knowledge_graph.json"), self.tokenizer, kg_threshold
        ) if use_kg else None

        # 可调参数：优先取 tune 子命令搜索出的结果，否则用默认值
        self.tuning: Dict = self._load_tuning()
        self.mmr_lambda = float(self.tuning.get("mmr_lambda", MMR_LAMBDA))
        self.expansion_weight = float(self.tuning.get("expansion_weight", EXPANSION_WEIGHT))
        # 查询扩展默认关闭：规模化语料（1269 篇 / 1420 条评测）实测它是负收益 ——
        # 多路召回经 RRF 加回后稀释了精确匹配，案例召回 -5.33pp、整体召回 -1.94pp，
        # 延迟还涨了约 0.9ms；它唯一的收益是改写查询 +2.1pp，不划算。
        # 需要时用 apply_tuning({"use_expansion": True}) 或 tune 子命令打开。
        self.use_expansion = bool(self.tuning.get("use_expansion", False))
        self.use_rerank = bool(self.tuning.get("use_rerank", True))
        self.alpha_override = self.tuning.get("alpha_override")
        self.chunk_size = int(self.tuning.get("chunk_size", DEFAULT_CHUNK_SIZE))
        self.overlap = int(self.tuning.get("overlap", DEFAULT_OVERLAP))
        self.min_coverage = float(self.tuning.get("min_coverage", DEFAULT_MIN_COVERAGE))

        self.generator = AnswerGenerator(self.tokenizer, llm_client=self._build_llm_client(use_llm))
        self._cache: Dict[tuple, Dict] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------
    def _build_llm_client(self, use_llm: Optional[bool]) -> Optional[LLMClient]:
        """按配置决定是否启用 LLM（默认自动探测环境变量）"""
        if use_llm is False or os.getenv("DISABLE_LLM", "false").lower() == "true":
            return None
        if use_llm is None and not os.getenv("MODELSCOPE_API_KEY"):
            return None

        api_key = os.getenv("MODELSCOPE_API_KEY", "")
        if not api_key or api_key.startswith("your_"):
            return None

        from model_manager import create_default_model_manager   # 复用既有模型池（零依赖）

        manager = create_default_model_manager()
        if not manager.models:
            return None
        return LLMClient(manager)

    def _load_tuning(self) -> Dict:
        """读取超参数调优结果（data/pure_index/tuning.json），不存在则返回空"""
        path = self.index_dir / "tuning.json"
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("best", {})
        except (OSError, ValueError) as exc:
            print(f"[warn] 调优参数读取失败，使用默认值：{exc}")
            return {}

    def apply_tuning(self, params: Dict) -> None:
        """应用一组超参数（tune 子命令与外部脚本使用）"""
        self.tuning = dict(params or {})
        self.mmr_lambda = float(self.tuning.get("mmr_lambda", MMR_LAMBDA))
        self.expansion_weight = float(self.tuning.get("expansion_weight", EXPANSION_WEIGHT))
        self.use_expansion = bool(self.tuning.get("use_expansion", False))
        self.use_rerank = bool(self.tuning.get("use_rerank", True))
        self.alpha_override = self.tuning.get("alpha_override")
        self.min_coverage = float(self.tuning.get("min_coverage", DEFAULT_MIN_COVERAGE))
        self._cache.clear()

    def _apply_terms(self, terms: List[str]) -> None:
        """用语料发现的新词重建分词器，并同步到图谱与生成器"""
        self.terms = list(terms)
        self.tokenizer = Tokenizer(extra_words=self.terms, use_synonyms=self.use_synonyms)
        self.generator.tokenizer = self.tokenizer
        if self.kg:
            self.kg.tokenizer = self.tokenizer
            if self.kg.kg:
                self.kg._build_index()

    # ------------------------------------------------------------------
    # 索引构建与持久化
    # ------------------------------------------------------------------
    def build(self, chunk_size: int = None, overlap: int = None,
              discover: bool = None, verbose: bool = True) -> Dict:
        """
        从 data/ 目录重新构建全部索引并落盘

        不传参数时优先使用 tune 子命令搜索出的最优分块参数，其次使用默认值。
        """
        start = time.time()
        chunk_size = int(chunk_size or self.tuning.get("chunk_size", DEFAULT_CHUNK_SIZE))
        overlap = int(overlap if overlap is not None
                      else self.tuning.get("overlap", DEFAULT_OVERLAP))
        discover = self.tuning.get("discover", True) if discover is None else discover
        self.chunk_size, self.overlap = chunk_size, overlap

        documents = load_documents(str(self.data_dir))

        terms = discover_terms([d["content"] for d in documents]) if discover else []
        self._apply_terms(terms)

        self.chunks = chunk_documents(documents, chunk_size, overlap)
        self._token_lists = [
            self.tokenizer.cut_for_index(build_index_text(chunk)) for chunk in self.chunks
        ]
        self._rebuild_indexes()

        self.source_count = len(documents)
        self.built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._cache.clear()
        self.save()

        user_docs = sum(1 for d in documents if d.get("source") == "user_docs")
        summary = {
            "documents": len(documents),
            "chunks": len(self.chunks),
            "vocabulary": len(self.tfidf.df),
            "discovered_terms": len(terms),
            "user_documents": user_docs,
            "elapsed": round(time.time() - start, 3),
        }
        if verbose:
            print(f"[build] 文档 {summary['documents']} 篇（自有文档 {user_docs} 篇）"
                  f" → 分块 {summary['chunks']} 个")
            print(f"[build] 词表 {summary['vocabulary']} 词（新词发现补充 {summary['discovered_terms']} 个）")
            print(f"[build] 索引已保存到 {self.index_dir}（耗时 {summary['elapsed']}s）")
            if terms[:12]:
                print(f"[build] 新词示例：{'、'.join(terms[:12])}")
        return summary

    def _rebuild_indexes(self) -> None:
        """用当前词序列重建 BM25 与 TF-IDF（增量追加时只需重建索引结构，无需重新分词）"""
        self.bm25 = BM25Index()
        for tokens in self._token_lists:
            self.bm25.add(tokens)
        self.bm25.build()
        self.tfidf = TfidfIndex().build(self._token_lists)
        self._cache.clear()

    def save(self) -> None:
        """持久化索引：分块、词序列、BM25、TF-IDF、元信息"""
        ensure_dir(self.index_dir)
        with open(self.index_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        # 词序列一并落盘：后续追加文档时可直接复用，无需对旧文档重新分块分词
        with open(self.index_dir / "tokens.json", "w", encoding="utf-8") as f:
            json.dump(self._token_lists, f, ensure_ascii=False)
        self.bm25.save(self.index_dir / "bm25.json")
        self.tfidf.save(self.index_dir / "tfidf.json")
        meta = {
            "built_at": self.built_at,
            "source_count": self.source_count,
            "chunk_count": len(self.chunks),
            "terms": self.terms,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "version": 1,
        }
        with open(self.index_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """加载已落盘的索引，返回是否成功"""
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(self.index_dir / "documents.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self._apply_terms(meta.get("terms", []))

        tokens_path = self.index_dir / "tokens.json"
        if tokens_path.exists():
            with open(tokens_path, "r", encoding="utf-8") as f:
                self._token_lists = json.load(f)
        else:
            # 兼容旧索引：没有词序列缓存时按分块重新分词一次
            self._token_lists = [
                self.tokenizer.cut_for_index(build_index_text(chunk)) for chunk in self.chunks
            ]

        self.bm25 = BM25Index.load(self.index_dir / "bm25.json")
        self.tfidf = TfidfIndex.load(self.index_dir / "tfidf.json")
        self.source_count = meta.get("source_count", 0)
        self.built_at = meta.get("built_at")
        self.chunk_size = int(meta.get("chunk_size", self.chunk_size))
        self.overlap = int(meta.get("overlap", self.overlap))
        self._cache.clear()
        return True

    def append(self, documents: List[Dict], verbose: bool = True) -> Dict:
        """
        增量追加文档

        只对新增文档做分块与分词（旧文档复用持久化的词序列），
        再重建 BM25 / TF-IDF 的统计结构（IDF 依赖全局文档频率，无法局部更新）。
        """
        self.ensure_loaded()
        new_chunks = chunk_documents(documents, self.chunk_size, self.overlap)
        new_tokens = [
            self.tokenizer.cut_for_index(build_index_text(chunk)) for chunk in new_chunks
        ]

        self.chunks.extend(new_chunks)
        self._token_lists.extend(new_tokens)
        self._rebuild_indexes()

        self.source_count += len(documents)
        self.built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

        summary = {
            "added_documents": len(documents),
            "added_chunks": len(new_chunks),
            "total_chunks": len(self.chunks),
            "vocabulary": len(self.tfidf.df),
        }
        if verbose:
            print(f"[add] 追加文档 {summary['added_documents']} 篇 → 新增分块 "
                  f"{summary['added_chunks']} 个（累计 {summary['total_chunks']} 个）")
        return summary

    def add_files(self, paths: List[str], copy_to_user_dir: bool = True,
                  verbose: bool = True) -> Dict:
        """
        把外部文件追加进索引

        默认同时复制到 data/user_docs/，保证以后整体 build 能得到一致结果；
        已入库的文件（按文件名判断）会跳过。
        """
        self.ensure_loaded()
        user_dir = ensure_dir(self.data_dir / USER_DIR_NAME)
        indexed_files = self._indexed_files()
        documents, skipped = [], []

        # 目录自动展开为其中的文件
        expanded: List[Path] = []
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                expanded.extend(sorted(p for p in path.rglob("*") if p.is_file()))
            else:
                expanded.append(path)

        for path in expanded:
            if not path.is_file():
                skipped.append(f"{path}（文件不存在）")
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped.append(f"{path.name}（不支持的格式）")
                continue
            if path.name in indexed_files:
                skipped.append(f"{path.name}（已在索引中）")
                continue
            try:
                content = read_text_file(path).strip()
            except Exception as exc:                 # noqa: BLE001 - 单文件失败不影响其它文件
                skipped.append(f"{path.name}（解析失败：{exc}）")
                continue
            if not content:
                skipped.append(f"{path.name}（内容为空）")
                continue

            if copy_to_user_dir and path.parent != user_dir:
                shutil.copy2(path, user_dir / path.name)
            documents.append({
                "source": USER_DIR_NAME,
                "content": content,
                "metadata": {"type": "user_doc", "title": path.stem, "file": path.name},
            })
            indexed_files.add(path.name)

        if documents:
            summary = self.append(documents, verbose=verbose)
        else:
            summary = {"added_documents": 0, "added_chunks": 0,
                       "total_chunks": len(self.chunks), "vocabulary": len(self.tfidf.df)}
            if verbose:
                print("[add] 没有可追加的新文件")
        for item in skipped:
            if verbose:
                print(f"[add] 跳过 {item}")
        summary["skipped"] = skipped
        return summary

    def _indexed_files(self) -> set:
        """已入库的自有文档文件名集合"""
        return {
            chunk["metadata"]["file"]
            for chunk in self.chunks
            if chunk.get("metadata", {}).get("source") == USER_DIR_NAME
            and chunk["metadata"].get("file")
        }

    def ensure_loaded(self) -> None:
        """没有索引就自动构建（首次使用时省去手动 build）"""
        if self.chunks:
            return
        if not self.load():
            self.build(verbose=True)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 3, method: str = "hybrid",
                 alpha: float = None, use_kg: bool = True, use_rerank: bool = None,
                 use_expansion: bool = None) -> Dict:
        """
        混合检索

        method: hybrid（向量+BM25）/ vector（仅向量）/ bm25（仅关键词）
        alpha:  显式指定融合权重，None 则按查询特征自适应预测（或取调优的固定值）
        use_rerank / use_expansion: 不传则用引擎级开关（消融实验会逐项切换）
        """
        use_rerank = self.use_rerank if use_rerank is None else use_rerank
        use_expansion = self.use_expansion if use_expansion is None else use_expansion
        if not self.chunks:
            raise RuntimeError("索引尚未构建，请先运行：python src/rag_cli.py build")

        timings = {}
        begin = time.time()
        tokens = self.tokenizer.cut_for_index(question)
        timings["tokenize"] = round(time.time() - begin, 4)

        # 1) 融合权重：显式指定 > 调优固定值 > 查询特征自适应
        if alpha is None:
            alpha = (self.alpha_override if self.alpha_override is not None
                     else adaptive_alpha(question, self.tokenizer))
        else:
            alpha = float(alpha)

        # 2) 两路召回
        begin = time.time()
        vector_scores = self.tfidf.score(tokens) if method in ("hybrid", "vector") else {}
        bm25_scores = self.bm25.score(tokens) if method in ("hybrid", "bm25") else {}
        scores = weighted_fusion(vector_scores, bm25_scores, alpha)
        timings["recall"] = round(time.time() - begin, 4)

        # 3) 查询扩展：用变体查询多路召回，RRF 合并后按权重加回
        variants = expand_queries(question) if (use_expansion and self.use_expansion
                                                and method == "hybrid") else []
        if variants:
            begin = time.time()
            ranked_lists = []
            for variant in variants:
                variant_tokens = self.tokenizer.cut_for_index(variant)
                variant_scores = weighted_fusion(
                    self.tfidf.score(variant_tokens), self.bm25.score(variant_tokens), alpha
                )
                ranked = sorted(variant_scores.items(), key=lambda kv: (-kv[1], kv[0]))
                ranked_lists.append(ranked[:top_k * 3])
            scores = merge_scores(scores, min_max(reciprocal_rank_fusion(ranked_lists)),
                                  self.expansion_weight)
            timings["expansion"] = round(time.time() - begin, 4)

        # 4) MMR 多样性重排：先缩放到 (0,1]，保证与相似度同尺度且保留相对差距
        scores = scale_by_max(scores)
        candidate_size = max(top_k * 3, top_k)
        candidates = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:candidate_size]

        begin = time.time()
        if use_rerank and len(candidates) > top_k:
            selected = mmr_select(candidates, top_k, self.tfidf.similarity, self.mmr_lambda)
            # MMR 决定"选哪些文档"，最终仍按融合分数降序展示，保证分数单调可比
            selected.sort(key=lambda item: (-item[1], item[0]))
        else:
            selected = candidates[:top_k]
        timings["rerank"] = round(time.time() - begin, 4)

        # 5) 高亮定位：查询词在原文中的字符区间（单字查询词噪声大，不参与高亮）
        highlight_terms = {token for token in tokens if len(token) >= 2}

        documents = []
        for doc_id, score in selected:
            chunk = self.chunks[doc_id]
            spans = find_spans(chunk["content"], highlight_terms)
            documents.append({
                "doc_id": doc_id,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": round(float(score), 6),
                "spans": spans,
                "matched": matched_terms(spans),
            })

        # 6) 知识图谱增强
        kg_result = {"text": "", "entities": []}
        if use_kg and self.kg and self.kg.available:
            begin = time.time()
            kg_result = self.kg.build_context(question)
            timings["kg"] = round(time.time() - begin, 4)

        return {
            "question": question,
            "method": method,
            "alpha": round(alpha, 4),
            "documents": documents,
            "kg_context": kg_result["text"],
            "kg_entities": kg_result["entities"],
            "candidates": len(scores),
            "variants": variants,
            "timings": timings,
        }

    # ------------------------------------------------------------------
    # 可解释性
    # ------------------------------------------------------------------
    def explain(self, question: str, top_k: int = 3, method: str = "hybrid",
                alpha: float = None, max_terms: int = 8) -> Dict:
        """
        展示每个查询词对文档分数的贡献（用于调试检索效果）

        注意：这里展示的是「两路召回 + 加权融合」本身的结果，
        不包含查询扩展加分与 MMR 重排，因此更适合分析打分是否合理。
        """
        self.ensure_loaded()
        if not self.chunks:
            raise RuntimeError("索引尚未构建，请先运行：python src/rag_cli.py build")

        tokens = self.tokenizer.cut_for_index(question)
        if alpha is None:
            alpha = (self.alpha_override if self.alpha_override is not None
                     else adaptive_alpha(question, self.tokenizer))
        else:
            alpha = float(alpha)

        bm25_scores = self.bm25.score(tokens) if method in ("hybrid", "bm25") else {}
        vector_scores = self.tfidf.score(tokens) if method in ("hybrid", "vector") else {}
        fused = scale_by_max(weighted_fusion(vector_scores, bm25_scores, alpha))
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

        documents = []
        for doc_id, score in ranked:
            bm25_parts = self.bm25.explain(tokens, doc_id)
            tfidf_parts = self.tfidf.explain(tokens, doc_id)
            max_bm25 = max(bm25_parts.values(), default=0.0) or 1.0
            max_tfidf = max(tfidf_parts.values(), default=0.0) or 1.0

            terms = [
                {
                    "term": term,
                    "bm25": round(bm25_parts.get(term, 0.0), 4),
                    "tfidf": round(tfidf_parts.get(term, 0.0), 4),
                    "impact": round(max(bm25_parts.get(term, 0.0) / max_bm25,
                                        tfidf_parts.get(term, 0.0) / max_tfidf), 4),
                }
                for term in set(bm25_parts) | set(tfidf_parts)
            ]
            terms.sort(key=lambda item: (-item["impact"], item["term"]))

            chunk = self.chunks[doc_id]
            documents.append({
                "doc_id": doc_id,
                "score": round(float(score), 4),
                "source": chunk["metadata"].get("source"),
                "title": chunk["metadata"].get("title"),
                "preview": chunk["content"][:120].replace("\n", " "),
                "terms": terms[:max_terms],
            })

        return {
            "question": question,
            "tokens": sorted(set(tokens)),
            "alpha": round(alpha, 4),
            "method": method,
            "documents": documents,
        }

    # ------------------------------------------------------------------
    # 置信度与拒答
    # ------------------------------------------------------------------
    def confidence(self, question: str, documents: List[Dict]) -> Dict:
        """
        评估检索结果对查询词的覆盖程度（IDF 加权）

            coverage = Σ IDF(命中的查询词) / Σ IDF(全部查询词)

        查询词只要在检索到的文档里出现过就算命中（含同义词别名），
        因此 coverage 衡量的是「问题是否落在知识库的用词范围内」。

        它**不**判断「资料里是否真的有答案」——域内但缺资料的问题（如询问具体价格、
        某风场实际发电量）仍会被判为高覆盖，这是该指标的已知边界。
        """
        all_terms = self.tokenizer.cut(question)
        if not all_terms:
            return {"coverage": 0.0, "matched": [], "missing": [],
                    "matched_content": [], "terms": []}

        covered: set = set()
        for document in documents:
            doc_id = document.get("doc_id")
            if isinstance(doc_id, int) and doc_id < len(self._token_lists):
                covered.update(self._token_lists[doc_id])

        total_idf = sum(self.tfidf.idf_of(term) for term in all_terms)
        hit_idf = sum(self.tfidf.idf_of(term) for term in all_terms if term in covered)

        return {
            "coverage": round(hit_idf / total_idf, 4) if total_idf > 0 else 0.0,
            "matched": list(dict.fromkeys(t for t in all_terms if t in covered)),
            "missing": list(dict.fromkeys(t for t in all_terms if t not in covered)),
            # 命中的多字词：单字命中不算数（「科幻电影」的「电」也能碰巧命中）
            "matched_content": list(dict.fromkeys(
                t for t in all_terms if len(t) >= 2 and t in covered)),
            "terms": list(dict.fromkeys(all_terms)),
        }

    @staticmethod
    def should_refuse(confidence: Dict, threshold: float) -> bool:
        """
        拒答判据（两条同时满足才回答）：

        1. 至少命中一个多字词 —— 多字词才代表实词概念，避免靠单字巧合蒙混过关
        2. IDF 加权覆盖率达标 —— 控制与知识库用词的整体重叠程度

        threshold <= 0 表示完全关闭拒答（两条判据都跳过）。
        """
        if threshold <= 0:
            return False
        if not confidence.get("matched_content"):
            return True
        return confidence["coverage"] < threshold

    def _refusal_answer(self, confidence: Dict, documents: List[Dict], threshold: float,
                        reason: str = "low_coverage", combination: Dict = None) -> str:
        """拒答话术：说明判断依据，并给出最接近的资料/可行的问法"""
        if reason == "invalid_combination" and combination and combination["invalid"]:
            pair = combination["invalid"][0]
            lines = [
                f"知识库中没有收录「{pair['component']}」的「{pair['fault']}」故障"
                f"（该故障在知识库里属于「{pair['owner']}」）。"
            ]
            others = [item for item in combination["invalid"]
                      if item["component"] != pair["component"]]
            if others:
                lines.append("同样未收录的组合：" + "、".join(
                    f"{item['component']}的{item['fault']}" for item in others[:3]))
            lines.append(f"可以改问「{pair['owner']}{pair['fault']}怎么处理？」，"
                         "或换一个与该部件相关的故障描述。")
            return "\n".join(lines)

        lines = [
            f"知识库中没有检索到足以回答该问题的资料"
            f"（查询词覆盖率 {confidence['coverage']:.0%}，拒答阈值 {threshold:.0%}）。"
        ]
        if confidence.get("missing"):
            lines.append("未覆盖到的关键词：" + "、".join(confidence["missing"][:6]))
        if documents:
            metadata = documents[0]["metadata"]
            title = metadata.get("title") or metadata.get("source") or "未知来源"
            lines.append(f"最接近的资料是「{title}」，但它未必能回答这个问题。")
        lines.append("可以补充部件名称、故障现象或标准编号后重试。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = 3, method: str = "hybrid",
              alpha: float = None, use_kg: bool = True, use_rerank: bool = None,
              history: List[Dict] = None, use_cache: bool = True,
              min_coverage: float = None) -> Dict:
        """
        完整问答：检索 + 生成（含置信度评估与拒答）

        history: 多轮对话历史 [{"question":..., "answer":...}]，用于指代消解
        min_coverage: 本次调用的拒答阈值，None 时使用引擎配置（min_coverage=0 可关闭拒答）
        """
        question = (question or "").strip()
        if not question:
            # 保持与正常返回一致的字段集，调用方（CLI / Web）不必为空问题做特例
            return {
                "question": "",
                "resolved_question": "",
                "answer": "请输入有效的问题。",
                "mode": "empty",
                "retrieved_docs": [],
                "citations": [],
                "claims": [],
                "card": None,
                "confidence": {"coverage": 0.0, "matched": [], "missing": [],
                               "matched_content": [], "terms": [],
                               "refused": False, "reason": "empty_question"},
                "combination": {"components": [], "faults": [], "invalid": [], "valid": []},
                "alpha": 0.0,
                "candidates": 0,
                "variants": [],
                "kg_context": "",
                "kg_entities": [],
                "timings": {"total": 0.0},
            }

        resolved = self._resolve_coreference(question, history)
        threshold = self.min_coverage if min_coverage is None else float(min_coverage)
        use_rerank = self.use_rerank if use_rerank is None else use_rerank
        cache_key = (resolved, top_k, method, alpha, use_kg, use_rerank, threshold)
        if use_cache and cache_key in self._cache:
            return dict(self._cache[cache_key])

        self.ensure_loaded()
        begin = time.time()
        retrieval = self.retrieve(resolved, top_k, method, alpha, use_kg, use_rerank)
        confidence = self.confidence(resolved, retrieval["documents"])
        confidence["threshold"] = round(threshold, 4)

        # 组合校验：查询点名的部件与故障搭配是否成立（覆盖率判据看不出来的那类伪问题）
        combination = (self.kg.validate_combination(resolved)
                       if (use_kg and self.kg and self.kg.available)
                       else {"components": [], "faults": [], "invalid": [], "valid": []})

        if combination["invalid"] and not combination["valid"]:
            reason = "invalid_combination"
        elif self.should_refuse(confidence, threshold):
            reason = "low_coverage"
        else:
            reason = ""

        confidence["reason"] = reason
        if reason:
            # 判为不可答时直接拒答：不把弱相关文档包装成答案，也省下一次 LLM 调用
            confidence["refused"] = True
            generation = {
                "answer": self._refusal_answer(confidence, retrieval["documents"],
                                               threshold, reason, combination),
                "mode": "refused",
                "citations": [],
            }
        else:
            confidence["refused"] = False
            generation = self.generator.generate(
                resolved, retrieval["documents"], retrieval["kg_context"], self.tfidf.idf_of
            )

        result = {
            "question": question,
            "resolved_question": resolved,
            "answer": generation["answer"],
            "mode": generation["mode"],
            "retrieved_docs": retrieval["documents"],
            "citations": generation["citations"],
            "claims": generation.get("claims", []),
            "card": generation.get("card"),
            "confidence": confidence,
            "combination": combination,
            "alpha": retrieval["alpha"],
            # 候选规模与变体数：Web 界面与 JSON 调用方需要据此判断检索面有多宽
            "candidates": retrieval["candidates"],
            "variants": retrieval["variants"],
            "kg_context": retrieval["kg_context"],
            "kg_entities": retrieval["kg_entities"],
            "timings": {**retrieval["timings"],
                        "total": round(time.time() - begin, 4)},
        }

        if use_cache:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
        return result

    @staticmethod
    def _resolve_coreference(question: str, history: List[Dict] = None) -> str:
        """多轮追问补全：短问句含指代词时，拼接上一轮问题作为检索上下文"""
        if not history:
            return question
        last_question = (history[-1].get("question") or "").strip()
        if not last_question:
            return question
        if len(question) <= 15 and any(word in question for word in PRONOUNS):
            return f"{last_question} {question}"
        return question

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        """返回索引与图谱的统计信息"""
        if not self.chunks:
            self.ensure_loaded()
        kg_stats = {"nodes": 0, "edges": 0}
        if self.kg and self.kg.kg:
            kg_stats = {
                "nodes": len(self.kg.kg.nodes),
                "edges": sum(len(edges) for edges in self.kg.kg.edges.values()),
            }
        return {
            "source_documents": self.source_count,
            "chunks": len(self.chunks),
            "vocabulary": len(self.tfidf.df),
            "discovered_terms": len(self.terms),
            "avg_chunk_chars": round(
                sum(len(c["content"]) for c in self.chunks) / len(self.chunks), 1
            ) if self.chunks else 0,
            "built_at": self.built_at,
            "index_dir": str(self.index_dir),
            "llm_enabled": bool(self.generator.llm_client and self.generator.llm_client.available),
            "kg": kg_stats,
            "tuning": self.tuning or "默认参数（可运行 tune 子命令搜索）",
        }
