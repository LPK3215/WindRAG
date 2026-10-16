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

from pure_rag.bm25 import BM25Index
from pure_rag.chunker import build_index_text, chunk_text
from pure_rag.cli import save_transcript
from pure_rag.engine import RAGEngine
from pure_rag.evaluator import Evaluator
from pure_rag.fusion import (min_max, mmr_select, reciprocal_rank_fusion,
                             weighted_fusion)
from pure_rag.highlight import ansi, clip_spans, to_html, to_markdown
from pure_rag.loader import load_user_documents, read_text_file
from pure_rag.query_analysis import adaptive_alpha, classify_question
from pure_rag.text_utils import (Tokenizer, discover_terms, expand_query, find_spans,
                                 split_sentences)
from pure_rag.tfidf import TfidfIndex
from pure_rag.tuner import Tuner


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
        cls.engine = RAGEngine(data_dir=os.path.join(ROOT, "data"),
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
        reloaded = RAGEngine(data_dir=os.path.join(ROOT, "data"),
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
        cls.engine = RAGEngine(data_dir=os.path.join(ROOT, "data"),
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
        self.evaluator = Evaluator(os.path.join(ROOT, "data"))

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
            tuner = Tuner(data_dir=os.path.join(ROOT, "data"),
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
