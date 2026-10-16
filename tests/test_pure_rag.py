"""
纯 Python RAG 内核单元测试（标准库 unittest）

运行：
    python -m unittest tests.test_pure_rag -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pure_rag.ablation import ALPHA_SWEEP, CONFIGURATIONS, AblationRunner
from pure_rag.bm25 import BM25Index
from pure_rag.chunker import build_index_text, chunk_text
from pure_rag.cli import save_transcript
from pure_rag.datasets import build_dataset
from pure_rag.engine import RAGEngine
from pure_rag.evaluator import Evaluator
from pure_rag.fusion import (min_max, mmr_select, reciprocal_rank_fusion,
                             weighted_fusion)
from pure_rag.generator import build_fault_card, parse_fault_cases
from pure_rag.highlight import ansi, clip_spans, to_html, to_markdown
from pure_rag.kg import KnowledgeGraphRetriever
from pure_rag.loader import load_user_documents, read_text_file
from pure_rag.query_analysis import adaptive_alpha, classify_question
from pure_rag.server import RAGRequestHandler
from pure_rag.text_utils import (Tokenizer, discover_terms, expand_query, find_spans,
                                 split_sentences, split_sentences_with_offsets)
from pure_rag.tfidf import TfidfIndex
from pure_rag.tuner import Tuner


_FIXTURE_DIR = None


def fixture_data_dir() -> str:
    """
    测试用小数据目录：项目自带合成数据 + 知识图谱 + 极小规模的生成语料

    直接用仓库的 data/ 会因为规模化语料（1269 篇）让单测慢十几倍，
    这里取 scale=0.02 的生成语料，既覆盖「生成语料加载」这条路径又保持快速。
    """
    global _FIXTURE_DIR
    if _FIXTURE_DIR:
        return _FIXTURE_DIR

    import shutil as _shutil

    from simulate_corpus import generate, save

    tmp = tempfile.mkdtemp(prefix="pure_rag_fixture_")
    synthetic = Path(tmp) / "synthetic"
    synthetic.mkdir(parents=True)
    for name in ("fault_cases", "technical_docs", "operations_manual",
                 "qa_pairs", "negative_queries"):
        source = Path(ROOT) / "data" / "synthetic" / f"{name}.json"
        if source.exists():
            _shutil.copy2(source, synthetic / source.name)
    _shutil.copy2(Path(ROOT) / "data" / "knowledge_graph.json", Path(tmp) / "knowledge_graph.json")
    save(generate(scale=0.02), str(Path(tmp) / "generated"))

    _FIXTURE_DIR = tmp
    return tmp


def tearDownModule():
    import shutil as _shutil

    if _FIXTURE_DIR:
        _shutil.rmtree(_FIXTURE_DIR, ignore_errors=True)


def _make_docx(path, paragraphs):
    """用标准库生成一个最小可用的 docx，用于验证解析逻辑"""
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = ("<?xml version='1.0' encoding='UTF-8'?>"
           "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
           f"<w:body>{body}</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)

CORPUS = [
    ["齿轮箱", "温度", "过高", "齿轮箱"],
    ["叶片", "裂纹", "更换"],
    ["偏航", "系统", "原理"],
]


class TextUtilsTest(unittest.TestCase):
    """分词、切句、同义词、新词发现"""

    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_domain_words_kept_whole(self):
        tokens = self.tokenizer.cut("齿轮箱温度过高怎么办")
        self.assertIn("齿轮箱", tokens)
        self.assertIn("温度过高", tokens)

    def test_stopwords_filtered(self):
        tokens = self.tokenizer.cut("齿轮箱的润滑油")
        self.assertNotIn("的", tokens)
        self.assertIn("润滑油", tokens)

    def test_ascii_token_kept(self):
        self.assertIn("iso", self.tokenizer.cut("ISO VG 320 齿轮油"))

    def test_split_sentences(self):
        self.assertEqual(len(split_sentences("第一句。第二句！第三句？")), 3)

    def test_expand_query_synonym(self):
        variants = expand_query("变速箱漏油怎么办")
        self.assertTrue(any("齿轮箱" in variant for variant in variants))

    def test_discover_terms(self):
        texts = ["检查散热风扇接线正常。更换散热风扇轴承正常。清洁散热风扇叶片正常。"] * 6
        terms = discover_terms(texts, min_freq=4, min_pmi=2.0, min_entropy=0.5)
        self.assertIn("散热风扇", terms)

    def test_punctuation_not_tokenized(self):
        self.assertNotIn("，", self.tokenizer.cut("齿轮箱，温度高"))


class IndexTest(unittest.TestCase):
    """BM25 与 TF-IDF 索引"""

    def test_bm25_ranks_relevant_doc_first(self):
        index = BM25Index()
        for tokens in CORPUS:
            index.add(tokens)
        index.build()
        ranked = index.search(["齿轮箱", "温度"])
        self.assertEqual(ranked[0][0], 0)
        self.assertGreater(ranked[0][1], 0)

    def test_bm25_rare_terms_weight_more(self):
        corpus = [["齿轮箱", "温度"], ["齿轮箱", "振动"], ["齿轮箱", "噪声"], ["叶片"]]
        index = BM25Index()
        for tokens in corpus:
            index.add(tokens)
        index.build()
        # 叶片仅出现于 1 篇文档，稀有度高于遍布所有文档的齿轮箱
        self.assertGreater(index.idf("叶片"), index.idf("齿轮箱"))

    def test_bm25_round_trip(self):
        index = BM25Index()
        for tokens in CORPUS:
            index.add(tokens)
        index.build()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bm25.json")
            index.save(path)
            restored = BM25Index.load(path)
        self.assertEqual(restored.search(["偏航"])[0][0], 2)
        self.assertAlmostEqual(restored.avgdl, index.avgdl)

    def test_tfidf_cosine(self):
        index = TfidfIndex().build(CORPUS)
        self.assertAlmostEqual(index.similarity(0, 0), 1.0, places=6)
        self.assertLess(index.similarity(0, 1), 0.2)

    def test_tfidf_search_and_round_trip(self):
        index = TfidfIndex().build(CORPUS)
        self.assertEqual(index.search(["叶片", "裂纹"])[0][0], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tfidf.json")
            index.save(path)
            restored = TfidfIndex.load(path)
        self.assertEqual(restored.search(["偏航", "原理"])[0][0], 2)

    def test_tfidf_no_match_returns_empty(self):
        index = TfidfIndex().build(CORPUS)
        self.assertEqual(index.search(["量子力学"]), [])


class FusionTest(unittest.TestCase):
    """融合与重排"""

    def test_min_max(self):
        self.assertEqual(min_max({1: 5.0, 2: 5.0}), {1: 1.0, 2: 1.0})
        self.assertEqual(min_max({1: 0.0, 2: 3.0, 3: 6.0}), {1: 0.0, 2: 0.5, 3: 1.0})

    def test_weighted_fusion_extremes(self):
        vector = {0: 0.9, 1: 0.1}
        bm25 = {0: 0.1, 1: 0.9}
        pure_vector = weighted_fusion(vector, bm25, alpha=1.0)
        pure_bm25 = weighted_fusion(vector, bm25, alpha=0.0)
        self.assertGreater(pure_vector[0], pure_vector[1])
        self.assertGreater(pure_bm25[1], pure_bm25[0])

    def test_rrf_prefers_consensus(self):
        list_a = [(0, 9.0), (1, 5.0)]
        list_b = [(0, 0.9), (2, 0.5)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        self.assertEqual(max(fused, key=fused.get), 0)

    def test_mmr_penalizes_duplicates(self):
        # 1 与 0 高度相似，2 不相似；λ 较小时应优先选到 2
        similarity = lambda a, b: 0.95 if {a, b} == {0, 1} else 0.0
        picked = mmr_select([(0, 1.0), (1, 0.98), (2, 0.9)], top_k=2,
                            similarity=similarity, lambda_=0.5)
        self.assertEqual([doc_id for doc_id, _ in picked], [0, 2])


class ChunkerTest(unittest.TestCase):
    """分块"""

    def test_chunk_size_bounded(self):
        text = "\n".join(f"第{i}段：" + "内容说明" * 30 for i in range(6))
        chunks = chunk_text(text, chunk_size=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200 + 40 + 2)

    def test_long_sentence_hard_split(self):
        chunks = chunk_text("长" * 500, chunk_size=100, overlap=0)
        self.assertGreaterEqual(len(chunks), 5)

    def test_index_text_boosts_title(self):
        chunk = {"content": "正文内容", "metadata": {"title": "齿轮箱", "keywords": "润滑"}}
        text = build_index_text(chunk)
        self.assertGreaterEqual(text.count("齿轮箱"), 2)
        self.assertIn("润滑", text)


class QueryAnalysisTest(unittest.TestCase):
    """查询理解"""

    def test_alpha_by_standard_code(self):
        self.assertAlmostEqual(adaptive_alpha("ISO 81400 标准对齿轮箱的要求"), 0.3)

    def test_alpha_by_unit(self):
        self.assertAlmostEqual(adaptive_alpha("轴承温度超过85℃怎么处理"), 0.4)

    def test_alpha_by_question_type(self):
        self.assertAlmostEqual(adaptive_alpha("偏航系统的工作原理是什么"), 0.8)

    def test_classify(self):
        scores = classify_question("齿轮箱温度过高怎么处理？")
        self.assertGreater(scores["故障诊断"], 0)


class EngineTest(unittest.TestCase):
    """端到端：构建索引、混合检索、答案生成"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pure_rag_test_")
        cls.engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(cls.tmp_dir, "index"),
                               use_llm=False)
        cls.summary = cls.engine.build(verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_build_summary(self):
        self.assertGreater(self.summary["documents"], 40)
        self.assertGreater(self.summary["chunks"], 40)
        self.assertGreater(self.summary["vocabulary"], 200)

    def test_hybrid_query(self):
        result = self.engine.query("齿轮箱温度过高应该怎么处理？")
        self.assertTrue(result["answer"])
        self.assertTrue(result["retrieved_docs"])
        self.assertIn("齿轮箱", "".join(d["content"] for d in result["retrieved_docs"]))
        self.assertTrue(0.0 <= result["alpha"] <= 1.0)

    def test_all_retrieval_methods_work(self):
        for method in ("hybrid", "vector", "bm25"):
            result = self.engine.query("偏航系统的作用", method=method)
            self.assertTrue(result["retrieved_docs"], msg=f"method={method} 无结果")

    def test_scores_sorted_desc(self):
        result = self.engine.query("叶片裂纹怎么处理", top_k=4)
        scores = [doc["score"] for doc in result["retrieved_docs"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_knowledge_graph_synonym_match(self):
        result = self.engine.query("变速箱异响怎么排查？")
        names = [entity["name"] for entity in result["kg_entities"]]
        self.assertTrue(any("齿轮箱" in name for name in names), msg=f"实际命中：{names}")

    def test_no_match_query_degrades_gracefully(self):
        result = self.engine.query("zzzz qqqq 量子纠缠")
        self.assertTrue(result["answer"])

    def test_empty_query(self):
        self.assertTrue(self.engine.query("   ")["answer"])

    def test_load_from_disk_gives_same_top_doc(self):
        before = self.engine.query("齿轮箱润滑油更换周期")["retrieved_docs"][0]["doc_id"]
        reloaded = RAGEngine(data_dir=fixture_data_dir(),
                             index_dir=self.engine.index_dir, use_llm=False)
        self.assertTrue(reloaded.load())
        after = reloaded.query("齿轮箱润滑油更换周期")["retrieved_docs"][0]["doc_id"]
        self.assertEqual(before, after)

    def test_cache_returns_equal_result(self):
        first = self.engine.query("发电机过热的处理步骤")
        second = self.engine.query("发电机过热的处理步骤")
        self.assertEqual(first["answer"], second["answer"])

    def test_stats(self):
        stats = self.engine.stats()
        self.assertGreater(stats["chunks"], 0)
        self.assertGreater(stats["kg"]["nodes"], 0)


class UserDocsTest(unittest.TestCase):
    """自有文档导入：txt / md / csv / json / docx"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pure_rag_docs_")
        self.user_dir = Path(self.tmp) / "user_docs"
        self.user_dir.mkdir(parents=True)
        (self.user_dir / "手册.md").write_text("# 齿轮箱\n润滑油更换周期为2-3年。", encoding="utf-8")
        (self.user_dir / "数据.csv").write_text("部件,故障\n齿轮箱,齿轮磨损\n", encoding="utf-8")
        (self.user_dir / "问答.json").write_text(
            json.dumps([{"title": "偏航", "content": "偏航系统用于对准风向。"}], ensure_ascii=False),
            encoding="utf-8",
        )
        _make_docx(self.user_dir / "规范.docx",
                   ["叶片检查规范", "每季度检查叶片前缘腐蚀情况。"])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_formats_loaded(self):
        documents = load_user_documents(self.tmp)
        titles = sorted(d["metadata"]["title"] for d in documents)
        self.assertEqual(titles, ["手册", "数据", "规范", "问答"])
        self.assertTrue(all(d["source"] == "user_docs" for d in documents))

    def test_docx_parsed_by_zipfile(self):
        text = read_text_file(self.user_dir / "规范.docx")
        self.assertIn("叶片检查规范", text)
        self.assertIn("每季度检查叶片前缘腐蚀情况", text)
        self.assertNotIn("<w:p>", text)

    def test_csv_keeps_column_names(self):
        text = read_text_file(self.user_dir / "数据.csv")
        self.assertIn("部件: 齿轮箱", text)
        self.assertIn("故障: 齿轮磨损", text)

    def test_json_uses_content_field(self):
        text = read_text_file(self.user_dir / "问答.json")
        self.assertIn("偏航系统用于对准风向", text)

    def test_empty_dir_returns_empty_list(self):
        empty = tempfile.mkdtemp(prefix="pure_rag_empty_")
        try:
            self.assertEqual(load_user_documents(empty), [])
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class ExplainTest(unittest.TestCase):
    """检索可解释性：逐查询词的分数贡献"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pure_rag_explain_")
        cls.engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(cls.tmp_dir, "index"),
                               use_llm=False)
        cls.engine.build(verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_term_contributions(self):
        result = self.engine.explain("齿轮箱温度过高", top_k=2)
        self.assertEqual(len(result["documents"]), 2)
        self.assertIn("齿轮箱", result["tokens"])
        terms = {term["term"]: term for term in result["documents"][0]["terms"]}
        self.assertIn("齿轮箱", terms)
        self.assertGreater(terms["齿轮箱"]["impact"], 0)
        self.assertGreater(terms["齿轮箱"]["bm25"] + terms["齿轮箱"]["tfidf"], 0)

    def test_terms_sorted_by_impact(self):
        document = self.engine.explain("齿轮箱温度过高", top_k=1)["documents"][0]
        impacts = [term["impact"] for term in document["terms"]]
        self.assertEqual(impacts, sorted(impacts, reverse=True))


class RobustnessTest(unittest.TestCase):
    """改写查询构造与评测"""

    def setUp(self):
        self.evaluator = Evaluator(fixture_data_dir())

    def test_rewrites_differ_from_original(self):
        samples = self.evaluator.build_rewrites()
        self.assertTrue(samples)
        for sample in samples:
            self.assertNotEqual(sample["question"], sample["target"])
            self.assertTrue(sample["target"].strip())

    def test_synonym_rewrite_present(self):
        questions = [s["question"] for s in self.evaluator.build_rewrites()]
        self.assertTrue(any(alias in q for q in questions for alias in ("变速箱", "桨叶", "刹车")))


class TunerTest(unittest.TestCase):
    """超参数网格搜索"""

    def test_quick_grid_writes_tuning_file(self):
        tmp = tempfile.mkdtemp(prefix="pure_rag_tune_")
        try:
            tuner = Tuner(data_dir=fixture_data_dir(),
                          index_dir=os.path.join(tmp, "index"), num_questions=5)
            outcome = tuner.run(quick=True, verbose=False)
            self.assertEqual(outcome["evaluated"], 4)
            self.assertGreaterEqual(outcome["best"]["score"], 0.0)
            self.assertLessEqual(outcome["best"]["score"], 1.0)
            self.assertTrue((Path(tmp) / "index" / "tuning.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class HighlightTest(unittest.TestCase):
    """字符级定位与高亮渲染"""

    def test_find_spans_case_insensitive(self):
        spans = find_spans("ISO VG 320 齿轮油", ["iso", "齿轮油"])
        self.assertEqual(sorted(term for _, _, term in spans), ["iso", "齿轮油"])

    def test_find_spans_merges_overlap(self):
        spans = find_spans("齿轮箱维护", ["齿轮箱", "齿轮"])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][:2], (0, 3))

    def test_find_spans_no_match(self):
        self.assertEqual(find_spans("齿轮箱维护", ["叶片"]), [])

    def test_ansi_wraps_match(self):
        rendered = ansi("齿轮箱温度", find_spans("齿轮箱温度", ["齿轮箱"]))
        self.assertIn("\033[1;33m齿轮箱\033[0m", rendered)

    def test_html_escapes_document_content(self):
        text = "<script>齿轮箱</script>"
        rendered = to_html(text, find_spans(text, ["齿轮箱"]))
        self.assertNotIn("<script>", rendered)
        self.assertIn("<mark>齿轮箱</mark>", rendered)

    def test_markdown_bold(self):
        rendered = to_markdown("齿轮箱温度", find_spans("齿轮箱温度", ["齿轮箱"]))
        self.assertEqual(rendered, "**齿轮箱**温度")

    def test_clip_spans_drops_out_of_range(self):
        spans = find_spans("齿轮箱温度过高", ["温度过高"])
        self.assertEqual(clip_spans(spans, 3), [])


class IncrementalAddTest(unittest.TestCase):
    """增量追加：只对新文档分块分词，词序列随索引持久化"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pure_rag_add_")
        self.engine = RAGEngine(data_dir=self.tmp,
                                index_dir=os.path.join(self.tmp, "index"),
                                use_llm=False, use_kg=False)
        self.engine.build(verbose=False)
        self.file = Path(self.tmp) / "补充说明.md"
        self.file.write_text("# 变流器滤网\n变流器进风口滤网每月清洗一次。", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_then_retrieve(self):
        summary = self.engine.add_files([str(self.file)], verbose=False)
        self.assertEqual(summary["added_documents"], 1)
        self.assertEqual(summary["added_chunks"], 1)
        result = self.engine.query("变流器滤网多久清洗一次？")
        self.assertIn("每月清洗", result["answer"])
        self.assertEqual(result["retrieved_docs"][0]["metadata"]["source"], "user_docs")

    def test_duplicate_file_skipped(self):
        self.engine.add_files([str(self.file)], verbose=False)
        again = self.engine.add_files([str(self.file)], verbose=False)
        self.assertEqual(again["added_documents"], 0)
        self.assertTrue(again["skipped"])

    def test_file_copied_into_user_docs(self):
        self.engine.add_files([str(self.file)], verbose=False)
        self.assertTrue((Path(self.tmp) / "user_docs" / "补充说明.md").exists())

    def test_directory_expanded(self):
        directory = Path(self.tmp) / "batch"
        directory.mkdir()
        (directory / "a.md").write_text("塔筒螺栓力矩检查要求。", encoding="utf-8")
        (directory / "b.txt").write_text("机舱罩密封条更换周期。", encoding="utf-8")
        summary = self.engine.add_files([str(directory)], verbose=False)
        self.assertEqual(summary["added_documents"], 2)

    def test_tokens_persisted_and_reloaded(self):
        self.engine.add_files([str(self.file)], verbose=False)
        reloaded = RAGEngine(data_dir=self.tmp, index_dir=self.engine.index_dir,
                             use_llm=False, use_kg=False)
        self.assertTrue(reloaded.load())
        self.assertEqual(len(reloaded._token_lists), len(reloaded.chunks))
        self.assertEqual(reloaded.query("变流器滤网多久清洗一次？")["retrieved_docs"][0]["doc_id"], 0)

    def test_markdown_markers_removed(self):
        self.engine.add_files([str(self.file)], verbose=False)
        content = self.engine.chunks[0]["content"]
        self.assertNotIn("#", content)


class SimulateCorpusTest(unittest.TestCase):
    """规模化语料生成：确定性 + 能被 loader 正确加载"""

    def test_generation_is_deterministic(self):
        from simulate_corpus import generate

        first = generate(scale=0.01, seed=42)
        second = generate(scale=0.01, seed=42)
        other = generate(scale=0.01, seed=43)
        self.assertEqual(first["work_orders"][0]["content"], second["work_orders"][0]["content"])
        self.assertNotEqual(first["work_orders"][0]["content"], other["work_orders"][0]["content"])

    def test_generated_documents_are_loaded(self):
        from pure_rag.loader import load_documents

        documents = load_documents(fixture_data_dir())
        sources = {document["source"] for document in documents}
        self.assertIn("work_orders", sources)
        self.assertIn("monthly_reports", sources)
        for document in documents:
            if document["source"] == "work_orders":
                self.assertTrue(document["metadata"]["title"])
                self.assertIn("故障ID：", document["content"])
                break

    def test_work_order_fields_are_consistent(self):
        """严重程度与停机时长不应自相矛盾（低严重度配 168 小时这类）"""
        from simulate_corpus import SEVERITY_DOWNTIME, generate

        corpus = generate(scale=0.05)
        for document in corpus["work_orders"]:
            severity = document["content"].split("严重程度：")[1].splitlines()[0].strip()
            downtime = int(document["content"].split("停机时长：")[1].split(" ")[0])
            self.assertIn(downtime, SEVERITY_DOWNTIME[severity])


class DatasetTest(unittest.TestCase):
    """分层评测集：构成、目标完整性、确定性"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset(fixture_data_dir())
        cls.items = cls.dataset["items"]

    def test_all_layers_present(self):
        counts = self.dataset["counts"]
        for layer in ("qa_verbatim", "qa_rewrite", "fault_case", "doc_title",
                      "near_miss", "negative_ood", "negative_missing"):
            self.assertGreater(counts.get(layer, 0), 0, msg=f"{layer} 层为空")
        self.assertGreaterEqual(len(self.items), 100)

    def test_answer_items_have_target(self):
        for item in self.items:
            if item["expect"] == "answer":
                self.assertTrue(item["target_text"] or item["target_doc"],
                                msg=f"缺少目标：{item['query']}")

    def test_refuse_items_have_no_target(self):
        for item in self.items:
            if item["expect"] == "refuse":
                self.assertIsNone(item["target_text"])
                self.assertIsNone(item["target_doc"])

    def test_build_is_deterministic(self):
        again = build_dataset(fixture_data_dir())
        self.assertEqual([item["query"] for item in again["items"]],
                         [item["query"] for item in self.items])
        self.assertEqual(again["counts"], self.dataset["counts"])

    def test_rewrite_layer_differs_from_verbatim(self):
        verbatim = {item["query"] for item in self.items if item["layer"] == "qa_verbatim"}
        rewritten = {item["query"] for item in self.items if item["layer"] == "qa_rewrite"}
        self.assertTrue(rewritten)
        self.assertFalse(rewritten & verbatim)

    def test_fault_case_queries_avoid_duplicate_component(self):
        queries = [item["query"] for item in self.items if item["layer"] == "fault_case"]
        self.assertTrue(queries)
        for query in queries:
            self.assertNotIn("发电机发电机", query)
            self.assertIn("怎么处理", query)


class LayeredEvalTest(unittest.TestCase):
    """分层评测：目标判定与分母口径"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pure_rag_layers_")
        cls.engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(cls.tmp_dir, "index"),
                               use_llm=False)
        cls.engine.build(verbose=False)
        cls.evaluator = Evaluator(fixture_data_dir())
        cls.dataset = build_dataset(fixture_data_dir())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_rank_by_text_and_by_document(self):
        item_text = next(i for i in self.dataset["items"] if i["layer"] == "fault_case")
        item_doc = next(i for i in self.dataset["items"] if i["layer"] == "doc_title")
        ranked = [doc["doc_id"] for doc in
                  self.engine.retrieve(item_text["query"], top_k=5)["documents"]]
        self.assertGreaterEqual(
            self.evaluator.rank_of_item(self.engine, item_text, ranked), 1)
        ranked_doc = [doc["doc_id"] for doc in
                      self.engine.retrieve(item_doc["query"], top_k=5)["documents"]]
        self.assertGreaterEqual(
            self.evaluator.rank_of_item(self.engine, item_doc, ranked_doc), 1)

    def test_unknown_target_returns_zero(self):
        item = {"query": "齿轮箱", "target_text": "不存在的签名"}
        ranked = [doc["doc_id"] for doc in
                  self.engine.retrieve("齿轮箱", top_k=3)["documents"]]
        self.assertEqual(self.evaluator.rank_of_item(self.engine, item, ranked), 0)

    def test_buckets_use_their_own_denominators(self):
        layered = self.evaluator.evaluate_layers(self.engine, self.dataset, top_k=3)
        fault = layered["layers"]["fault_case"]
        near_miss = layered["layers"]["near_miss"]
        self.assertEqual(fault["answer_count"], fault["count"])
        self.assertEqual(near_miss["refuse_count"], near_miss["count"])
        self.assertEqual(near_miss["answer_count"], 0)
        self.assertGreaterEqual(fault["recall"], 0.9)

    def test_report_contains_all_sections(self):
        report = self.evaluator.build_report(self.engine)
        for section in ("## 评测集构成", "## 分层指标", "## 按数据来源",
                        "## 按问题类型", "## 拒答阈值标定", "## 说明与已知边界"):
            self.assertIn(section, report["markdown"])


class CombinationTest(unittest.TestCase):
    """知识图谱组合校验：部件 × 故障的搭配是否成立"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pure_rag_combo_")
        cls.kg = KnowledgeGraphRetriever(os.path.join(fixture_data_dir(), "knowledge_graph.json"))
        cls.engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(cls.tmp_dir, "index"),
                               use_llm=False)
        cls.engine.build(verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_valid_combination(self):
        result = self.kg.validate_combination("齿轮箱齿轮磨损怎么处理？")
        self.assertTrue(result["valid"])
        self.assertFalse(result["invalid"])
        self.assertEqual(result["valid"][0]["owner"], "齿轮箱")

    def test_invalid_combination_reports_owner(self):
        result = self.kg.validate_combination("偏航系统制动器失效怎么处理？")
        self.assertEqual(result["valid"], [])
        self.assertEqual(result["invalid"][0]["component"], "偏航系统")
        self.assertEqual(result["invalid"][0]["owner"], "制动系统")

    def test_query_without_fault_is_not_judged(self):
        result = self.kg.validate_combination("偏航系统的作用是什么？")
        self.assertEqual(result["faults"], [])
        self.assertFalse(result["invalid"])

    def test_alias_expansion_not_used_for_matching(self):
        # 「偏航偏差过大」里的"偏航"若被扩展成「偏航系统」，会凭空造出一个有效搭配
        result = self.kg.validate_combination("发电机偏航偏差过大怎么处理？")
        self.assertEqual(result["valid"], [])
        self.assertTrue(result["invalid"])

    def test_invalid_combination_refused_with_guidance(self):
        result = self.engine.query("齿轮箱制动器失效怎么处理？")
        self.assertTrue(result["confidence"]["refused"])
        self.assertEqual(result["confidence"]["reason"], "invalid_combination")
        self.assertIn("制动系统", result["answer"])

    def test_valid_combination_still_answered(self):
        result = self.engine.query("齿轮箱齿轮磨损怎么处理？")
        self.assertFalse(result["confidence"]["refused"])
        self.assertEqual(result["confidence"]["reason"], "")

    def test_near_miss_dataset_is_genuinely_invalid(self):
        dataset = build_dataset(fixture_data_dir())
        items = [item for item in dataset["items"] if item["layer"] == "near_miss"]
        self.assertTrue(items)
        for item in items:
            result = self.kg.validate_combination(item["query"])
            self.assertTrue(result["invalid"], msg=f"{item['query']} 未被判为无效组合")
            self.assertEqual(result["valid"], [], msg=f"{item['query']} 意外命中有效搭配")


class AblationTest(unittest.TestCase):
    """消融实验：配置可跑通、指标结构完整、开关确实生效"""

    def test_run_reports_all_configurations(self):
        runner = AblationRunner(data_dir=fixture_data_dir(), num_questions=5)
        payload = runner.run(verbose=False)
        self.assertEqual(len(payload["results"]), len(CONFIGURATIONS))
        for item in payload["results"]:
            for key in ("recall_at_k", "mrr", "ndcg_at_k", "rewrite_recall",
                        "case_recall", "near_miss_refusal", "ood_refusal",
                        "keyword_coverage", "avg_latency_ms"):
                self.assertIn(key, item["metrics"])
            self.assertLessEqual(item["metrics"]["recall_at_k"], 1.0)
            self.assertGreaterEqual(item["metrics"]["recall_at_k"], 0.0)
        self.assertIn("| 配置 |", payload["markdown"])

    def test_alpha_sweep_includes_adaptive_row(self):
        runner = AblationRunner(data_dir=fixture_data_dir(), num_questions=5)
        rows = runner.sweep_alpha(verbose=False)
        self.assertEqual(len(rows), len(ALPHA_SWEEP) + 1)
        self.assertIsNone(rows[-1]["config"]["alpha"])
        self.assertEqual(rows[0]["config"]["alpha"], ALPHA_SWEEP[0])

    def test_synonym_switch_controls_tokenizer(self):
        enabled = Tokenizer(use_synonyms=True)
        disabled = Tokenizer(use_synonyms=False)
        self.assertIn("变速箱", enabled.cut_for_index("齿轮箱"))
        self.assertNotIn("变速箱", disabled.cut_for_index("齿轮箱"))
        self.assertNotIn("变速箱", disabled.cut("变速箱"))
        self.assertIn("变速箱", enabled.cut("变速箱"))


class SentenceOffsetTest(unittest.TestCase):
    """带偏移的句子切分"""

    def test_offsets_map_back_to_source(self):
        text = "齿轮箱温度异常升高。\n解决方案：立即停机检查；更换润滑油！"
        spans = split_sentences_with_offsets(text)
        self.assertTrue(spans)
        for start, end, sentence in spans:
            self.assertEqual(text[start:end], sentence)

    def test_terminator_kept_newline_dropped(self):
        spans = split_sentences_with_offsets("第一句。第二句！\n第三行")
        self.assertEqual([s for _a, _b, s in spans], ["第一句。", "第二句！", "第三行"])

    def test_consistent_with_plain_splitter(self):
        text = "甲。乙；丙？\n丁"
        self.assertEqual([s for _a, _b, s in split_sentences_with_offsets(text)],
                         split_sentences(text))

    def test_empty_text(self):
        self.assertEqual(split_sentences_with_offsets(""), [])


class ClaimLocationTest(unittest.TestCase):
    """答案要点的引用定位"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pure_rag_claim_")
        cls.engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(cls.tmp_dir, "index"),
                               use_llm=False, use_kg=False)
        cls.engine.build(verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_every_claim_maps_back_to_source(self):
        result = self.engine.query("齿轮箱温度过高应该怎么处理？")
        self.assertTrue(result["claims"])
        for claim in result["claims"]:
            content = self.engine.chunks[claim["doc_id"]]["content"]
            self.assertEqual(content[claim["start"]:claim["end"]], claim["text"])
            self.assertLess(claim["start"], claim["end"])

    def test_claims_ordered_within_each_document(self):
        claims = self.engine.query("齿轮箱温度过高", top_k=5)["claims"]
        by_doc = {}
        for claim in claims:
            by_doc.setdefault(claim["doc_id"], []).append(claim["start"])
        for starts in by_doc.values():
            self.assertEqual(starts, sorted(starts))

    def test_claim_index_is_sequential(self):
        claims = self.engine.query("偏航系统的作用是什么？")["claims"]
        self.assertEqual([c["index"] for c in claims], list(range(1, len(claims) + 1)))


class FaultCardTest(unittest.TestCase):
    """故障案例解析与结构化卡片"""

    CASE = ("故障ID：FC001\n部件：齿轮箱\n故障类型：齿轮磨损\n症状：温度异常升高\n"
            "原因：润滑油不足\n诊断：振动分析\n解决方案：更换润滑油\n"
            "预防措施：定期检查\n严重程度：高")

    def test_parse_single_case(self):
        cases = parse_fault_cases(self.CASE)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["部件"], "齿轮箱")
        self.assertEqual(cases[0]["严重程度"], "高")

    def test_parse_multiple_cases_in_one_chunk(self):
        cases = parse_fault_cases(self.CASE + "\n" + self.CASE.replace("FC001", "FC002"))
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[1]["故障ID"], "FC002")

    def test_plain_text_is_not_a_case(self):
        self.assertEqual(parse_fault_cases("原因：这只是普通文档里的一句话"), [])

    def test_card_returns_none_without_case(self):
        documents = [{"doc_id": 0, "content": "标题：规范\n内容：普通技术文档。"}]
        self.assertIsNone(build_fault_card(documents))

    def test_card_from_real_query(self):
        tmp = tempfile.mkdtemp(prefix="pure_rag_card_")
        try:
            engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(tmp, "index"), use_llm=False)
            engine.build(verbose=False)
            card = engine.query("齿轮箱温度过高应该怎么处理？")["card"]
            self.assertIsNotNone(card)
            self.assertIn("齿轮箱", card["title"])
            labels = [field["label"] for field in card["fields"]]
            self.assertIn("解决方案", labels)
            self.assertIn("预防措施", labels)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ServerRenderTest(unittest.TestCase):
    """服务端高亮渲染：锚点、要点摘录与转义"""

    def test_documents_get_anchor_and_html(self):
        result = {"retrieved_docs": [{"doc_id": 3, "content": "齿轮箱温度过高。",
                                      "spans": [(0, 3, "齿轮箱")], "score": 1.0,
                                      "metadata": {}}], "claims": []}
        RAGRequestHandler._render_highlights(result)
        document = result["retrieved_docs"][0]
        self.assertEqual(document["anchor"], "doc-3")
        self.assertIn("<mark>齿轮箱</mark>", document["content_html"])
        self.assertNotIn("spans", document)

    def test_claims_get_excerpt_with_inner_marks(self):
        result = {
            "retrieved_docs": [{"doc_id": 3, "content": "甲。齿轮箱温度过高。",
                                "spans": [(2, 5, "齿轮箱")], "score": 1.0, "metadata": {}}],
            "claims": [{"index": 1, "text": "齿轮箱温度过高。", "doc_id": 3,
                        "start": 2, "end": 10, "source": "x", "title": "y"}],
        }
        RAGRequestHandler._render_highlights(result)
        claim = result["claims"][0]
        self.assertEqual(claim["anchor"], "doc-3")
        self.assertIn("<mark>齿轮箱</mark>", claim["excerpt_html"])

    def test_rendering_escapes_document_content(self):
        result = {"retrieved_docs": [{"doc_id": 0, "content": "<script>alert(1)</script>",
                                      "spans": [], "score": 1.0, "metadata": {}}],
                  "claims": [{"index": 1, "text": "<img onerror=1>", "doc_id": 0,
                              "start": 0, "end": 3, "source": "x", "title": "y"}]}
        RAGRequestHandler._render_highlights(result)
        self.assertNotIn("<script>", result["retrieved_docs"][0]["content_html"])
        self.assertNotIn("<img", result["claims"][0]["excerpt_html"])


class ConfidenceTest(unittest.TestCase):
    """置信度评估与拒答"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pure_rag_confidence_")
        cls.engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(cls.tmp_dir, "index"),
                               use_llm=False, use_kg=False)
        cls.engine.build(verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_in_domain_query_answered(self):
        result = self.engine.query("齿轮箱温度过高应该怎么处理？")
        self.assertFalse(result["confidence"]["refused"])
        self.assertGreaterEqual(result["confidence"]["coverage"], 0.8)
        self.assertIn("齿轮箱", result["confidence"]["matched_content"])

    def test_out_of_domain_query_refused(self):
        result = self.engine.query("如何做红烧肉？")
        self.assertTrue(result["confidence"]["refused"])
        self.assertEqual(result["mode"], "refused")
        self.assertIn("知识库", result["answer"])

    def test_shared_word_query_still_refused(self):
        # 与语料共有「天气」一词，靠「必须命中多字词 + 覆盖率」双判据拦住
        result = self.engine.query("明天北京天气怎么样？")
        self.assertTrue(result["confidence"]["refused"])

    def test_threshold_zero_disables_refusal(self):
        result = self.engine.query("如何做红烧肉？", min_coverage=0)
        self.assertFalse(result["confidence"]["refused"])
        self.assertNotEqual(result["mode"], "refused")

    def test_coverage_reflects_relevance(self):
        relevant = self.engine.query("偏航系统的作用是什么？")["confidence"]["coverage"]
        irrelevant = self.engine.query("推荐几部科幻电影")["confidence"]["coverage"]
        self.assertGreater(relevant, irrelevant)

    def test_should_refuse_rules(self):
        self.assertTrue(self.engine.should_refuse(
            {"matched_content": [], "coverage": 1.0}, 0.35))
        self.assertFalse(self.engine.should_refuse(
            {"matched_content": ["齿轮箱"], "coverage": 1.0}, 0.35))
        self.assertFalse(self.engine.should_refuse(
            {"matched_content": [], "coverage": 0.0}, 0.0))

    def test_refusal_message_mentions_coverage(self):
        answer = self.engine.query("今晚吃什么好")["answer"]
        self.assertIn("覆盖率", answer)


class NegativeEvalTest(unittest.TestCase):
    """拒答评测集与阈值标定"""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pure_rag_negative_")
        cls.engine = RAGEngine(data_dir=fixture_data_dir(),
                               index_dir=os.path.join(cls.tmp_dir, "index"),
                               use_llm=False, use_kg=False)
        cls.engine.build(verbose=False)
        cls.evaluator = Evaluator(fixture_data_dir())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_negative_set_well_formed(self):
        negatives = self.evaluator.load_negatives()
        self.assertGreaterEqual(len(negatives), 10)
        for item in negatives:
            self.assertTrue(item["query"].strip())
            self.assertIn(item["type"], ("out_of_domain", "missing_in_kb"))

    def test_evaluate_negatives_groups_by_type(self):
        result = self.evaluator.evaluate_negatives(self.engine)
        self.assertEqual(result["count"], len(self.evaluator.load_negatives()))
        self.assertIn("out_of_domain", result["by_type"])
        self.assertGreaterEqual(result["by_type"]["out_of_domain"]["refusal_rate"], 0.8)

    def test_calibration_is_monotonic(self):
        rows = self.evaluator.calibrate_threshold(self.engine,
                                                  thresholds=(0.0, 0.35, 0.9))
        refusal_rates = [row["negative_refusal_rate"] for row in rows]
        self.assertEqual(refusal_rates, sorted(refusal_rates))
        self.assertLessEqual(rows[-1]["answer_rate"], rows[0]["answer_rate"])

    def test_engine_confidence_is_public_api(self):
        retrieval = self.engine.retrieve("齿轮箱温度过高", top_k=2)
        confidence = self.engine.confidence("齿轮箱温度过高", retrieval["documents"])
        self.assertIn("matched_content", confidence)
        self.assertIn("missing", confidence)


class TranscriptTest(unittest.TestCase):
    """会话记录导出"""

    def test_save_transcript_with_sources(self):
        tmp = tempfile.mkdtemp(prefix="pure_rag_log_")
        try:
            history = [{
                "question": "齿轮箱怎么维护？",
                "answer": "定期更换润滑油。",
                "documents": [{
                    "content": "齿轮箱润滑油更换周期2-3年",
                    "metadata": {"source": "technical_docs", "title": "齿轮箱规范"},
                    "score": 0.9123,
                    "spans": [(0, 3, "齿轮箱")],
                }],
            }]
            path = save_transcript(history, "单元测试", tmp)
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("齿轮箱怎么维护？", text)
            self.assertIn("定期更换润滑油", text)
            self.assertIn("**齿轮箱**润滑油更换周期", text)
            self.assertIn("technical_docs/齿轮箱规范", text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_history_returns_empty(self):
        self.assertEqual(save_transcript([], "x", "data"), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
