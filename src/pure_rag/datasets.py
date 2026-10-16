"""
评测集构建（零第三方依赖，完全确定：同一份语料必然得到同一份评测集）

评测数据全部由语料自身推导，不需要人工标注，因此可以随语料一起演进：

| 层 | 来源 | 目标 | 期望行为 |
|----|------|------|---------|
| qa_verbatim      | QA 对原句            | 该 QA 对所在块（按问题文本定位） | 回答且命中 |
| qa_rewrite       | 原句的同义改写        | 同上 | 回答且命中 |
| fault_case       | 故障案例「部件 + 故障类型」 | 该案例块（按故障 ID 定位） | 回答且命中 |
| doc_title        | 技术文档 / 运维手册标题 | 该文档的任意块（按 source + title 定位） | 回答且命中 |
| near_miss        | 部件 × 故障的无效组合 | 无（语料里不存在该搭配） | 严格拒答 |
| negative_ood     | 域外问题（人工少量）  | 无 | 拒答 |
| negative_missing | 域内但缺资料（人工少量） | 无 | 拒答（当前机制只能部分识别） |
| human_like       | 模拟一线工程师的口语化提问 | 同类工单（按部件+故障类型匹配任意一条） | 命中目标 / 拒答 |

目标定位方式有三种，按数据特点选用：
- `target_text`：块内容里必须包含该子串（用于 QA 对、故障案例这类唯一标识）
- `target_doc`：块的 metadata 命中 source + title（用于整篇文档，避免长文档只有首块被算命中）
- `target_component` / `target_fault`：块为工单且部件与故障类型同时命中（用于口语化层——
  同一故障会有很多条工单，只要召回「同类」即算对，而非指定某一条）

目标定位方式有两种，按数据特点选用：
- `target_text`：块内容里必须包含该子串（用于 QA 对、故障案例这类唯一标识）
- `target_doc`：块的 metadata 命中 source + title（用于整篇文档，避免长文档只有首块被算命中）
"""

import json
from pathlib import Path
from typing import Dict, List

from pure_rag.generator import parse_fault_cases
from pure_rag.loader import load_documents
from pure_rag.query_analysis import classify_question
from pure_rag.text_utils import SYNONYMS, normalize

DATASET_PATH = "eval_set.json"

LAYERS = {
    "qa_verbatim": "QA 原句",
    "qa_rewrite": "同义改写",
    "fault_case": "案例查询",
    "doc_title": "文档查询",
    "near_miss": "近邻负例",
    "negative_ood": "域外问题",
    "negative_missing": "域内缺资料",
    "human_like": "口语化查询",
}
ANSWER_LAYERS = ("qa_verbatim", "qa_rewrite", "fault_case", "doc_title")

# 自带「部件/故障类型」字段、可按案例推导查询的语料来源
FAULT_SOURCES = ("fault_cases", "work_orders")
# 按标题推导查询、按 source + title 定位的整篇文档来源
DOC_SOURCES = ("technical_docs", "operations_manual", "inspections",
               "monthly_reports", "spare_parts", "sops", "safety_rules")
# 各来源贴合内容形态的提问方式（都用同一个模板会让月报、备件这类查询很别扭）
DOC_QUERIES = {
    "technical_docs": "{title}有哪些要求？",
    "operations_manual": "{title}包括哪些内容？",
    "inspections": "{title}的结果如何？",
    "monthly_reports": "{title}的可利用率是多少？",
    "spare_parts": "{title}的更换周期是多久？",
    "sops": "{title}的作业步骤有哪些？",
    "safety_rules": "{title}有哪些要求？",
}
REFUSE_LAYERS = ("near_miss", "negative_ood", "negative_missing")

# 近邻负例每个部件最多取几个无效组合，控制规模
NEAR_MISS_PER_COMPONENT = 2


def rewrite_query(question: str) -> str:
    """
    把一个问句改写成"换个说法"的版本（用于同义改写层）

    优先做同义词替换（齿轮箱 → 变速箱），否则退化为关键词式提问（去掉疑问词）。
    改写不成功时返回空串，由调用方跳过。
    """
    for alias, standards in SYNONYMS.items():
        for standard in standards:
            if standard in question:
                return question.replace(standard, alias, 1)

    for phrase in ("怎么", "如何", "什么", "为什么", "哪些", "是否", "怎样"):
        if phrase in question:
            return question.replace(phrase, "").replace("？", "").replace("?", "").strip()
    return ""


def _item(query: str, layer: str, source: str = "", qtype: str = "",
          target_text: str = None, target_doc: Dict = None,
          target_component: str = None, target_fault: str = None,
          expect: str = None, note: str = "") -> Dict:
    """统一构造评测条目

    expect 默认按层决定（拒答层→refuse，其余→answer），调用方可显式覆盖，
    口语化层里「无效组合」条目就借此标记为 refuse。
    """
    if expect is None:
        expect = "refuse" if layer in REFUSE_LAYERS else "answer"
    return {
        "query": query.strip(),
        "layer": layer,
        "expect": expect,
        "source": source,
        "qtype": qtype or _main_qtype(query),
        "target_text": target_text,
        "target_doc": target_doc,
        "target_component": target_component,
        "target_fault": target_fault,
        "note": note,
    }


def _main_qtype(query: str) -> str:
    """取问题类型软分类里的主类型，用于分层统计"""
    scores = classify_question(query)
    if not scores:
        return "其他"
    best = max(scores, key=lambda key: scores[key])
    return best if scores[best] > 0 else "其他"


# ---------------------------------------------------------------------------
# 各层构建
# ---------------------------------------------------------------------------
def _qa_layer(documents: List[Dict]) -> List[Dict]:
    """QA 对：原句 + 同义改写两层"""
    items = []
    for document in documents:
        if document.get("source") != "qa_pairs":
            continue
        question = _extract_qa_question(document["content"])
        if not question:
            continue
        items.append(_item(question, "qa_verbatim", "qa_pairs",
                           target_text=question))
        rewritten = rewrite_query(question)
        if rewritten and rewritten != question:
            # 改写后仍以原问题所在块为目标（同一个问题的另一种问法）
            items.append(_item(rewritten, "qa_rewrite", "qa_pairs",
                               target_text=question, note=f"原句：{question}"))
    return items


def _extract_qa_question(content: str) -> str:
    """QA 对文本形如「问题：xxx\n答案：yyy」，取出问题"""
    for line in content.splitlines():
        if line.startswith("问题："):
            return line[len("问题："):].strip()
    return ""


def _fault_case_layer(documents: List[Dict]) -> List[Dict]:
    """故障案例：用「部件 + 故障类型」构造查询，目标按故障 ID 定位"""
    items = []
    for document in documents:
        source = document.get("source")
        if source not in FAULT_SOURCES:
            continue
        cases = parse_fault_cases(document["content"])
        if not cases:
            continue
        case = cases[0]
        component, fault_type = case.get("部件", ""), case.get("故障类型", "")
        if not (component and fault_type) or not case.get("故障ID"):
            continue
        # 故障类型常常已经带了部件名（发电机过热），避免拼成「发电机发电机过热」
        subject = fault_type if fault_type.startswith(component) else f"{component}{fault_type}"

        # 规模化语料里同一个故障会有很多条工单，只问「部件+故障」会撞成一条，
        # 因此带上风场与机组编号 —— 既能区分实例，也顺带考察"近重复文档的区分能力"
        metadata = document.get("metadata", {})
        farm, turbine = metadata.get("farm", ""), metadata.get("turbine", "")
        prefix = f"{farm} {turbine} 机组" if farm and turbine else ""
        items.append(_item(
            f"{prefix}{subject}怎么处理？", "fault_case", source,
            target_text=f"故障ID：{case['故障ID']}",
            note=f"症状：{case.get('症状', '')[:40]}",
        ))
    return items


def _doc_layer(documents: List[Dict]) -> List[Dict]:
    """技术文档 / 运维手册：用标题构造查询，目标为该文档的任意块"""
    items = []
    for document in documents:
        source = document.get("source")
        if source not in DOC_SOURCES:
            continue
        title = (document.get("metadata") or {}).get("title") or ""
        if not title:
            continue
        template = DOC_QUERIES.get(source, "{title}有哪些要求？")
        items.append(_item(
            template.format(title=title), "doc_title", source,
            target_doc={"source": source, "title": title},
        ))
    return items


def _near_miss_layer(data_dir: str) -> List[Dict]:
    """
    近邻负例：语料里不存在的「部件 × 故障类型」组合

    查询用的词都是领域实词，因此覆盖率判据抓不住它，用于暴露
    "系统会不会把别部件的方案套过来" 这一失效模式。
    """
    kg_path = Path(data_dir) / "knowledge_graph.json"
    negatives_path = Path(data_dir) / "synthetic" / "negative_queries.json"
    if not kg_path.exists():
        return []

    try:
        from knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph.load(str(kg_path))
    except Exception:            # noqa: BLE001 - 图谱不可用时该层整层跳过
        return []

    components = sorted(node["id"] for node in kg.nodes.values()
                        if node["type"] == "component")
    fault_names = {}
    for node in kg.nodes.values():
        if node["type"] != "fault_type":
            continue
        parents = kg.get_reverse_neighbors(node["id"], "HAS_FAULT")
        owner = parents[0]["node_id"] if parents else ""
        fault_names[node["id"]] = (node["properties"].get("name", node["id"]), owner)

    items = []
    for index, component in enumerate(components):
        candidates = [
            (fault_name, owner)
            for _fault_id, (fault_name, owner) in sorted(fault_names.items())
            # 排除两类：该故障本就属于这个部件；故障名里含任一部件名
            # （如「叶片裂纹」自带「叶片」，拼出来会带一个有效搭配，就不算近邻负例了）
            if owner != component and fault_name
            and not any(name in fault_name for name in components)
        ]
        if not candidates:
            continue
        # 轮转取样，避免所有部件都配上同一个故障（否则近邻负例会退化成同一模式）
        for offset in range(min(NEAR_MISS_PER_COMPONENT, len(candidates))):
            position = (index * NEAR_MISS_PER_COMPONENT + offset) % len(candidates)
            fault_name, owner = candidates[position]
            items.append(_item(
                f"{component}{fault_name}怎么处理？", "near_miss", "fault_cases",
                note=f"语料中该故障只属于 {owner}",
            ))

    # 人工负例：域外与域内缺资料
    if negatives_path.exists():
        with open(negatives_path, "r", encoding="utf-8") as f:
            for entry in json.load(f).get("negative_queries", []):
                layer = ("negative_ood" if entry.get("type") == "out_of_domain"
                         else "negative_missing")
                items.append(_item(entry["query"], layer, "manual",
                                   note=entry.get("note", "")))
    return items


def build_human_like_layer(data_dir: str) -> List[Dict]:
    """口语化评测层：用 data/human_questions.json 里模拟一线工程师的提问

    每条问题标注了对应的 (部件, 故障类型) 作弱标注，判定「检索是否把同类工单送进前 k」。
    这是人工标注 / 真实问法的占位常驻层 —— 目前由本项目自己写下 50 条口语问法
    （详见 human_eval.py 与 README「口语化查询验证」一节），后续应替换为真实用户问法
    + 相关性标注；但这一层会一直作为评测集的一部分存在，专门暴露「合成评测高估可用率」的盲区。
    """
    path = Path(data_dir) / "human_questions.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    items = []
    for entry in questions:
        query = (entry.get("q") or "").strip()
        if not query:
            continue
        if entry.get("kind") == "refuse":
            items.append(_item(query, "human_like", "manual", qtype="口语",
                               expect="refuse", note=entry.get("expect", "")))
        else:
            items.append(_item(query, "human_like", "work_orders", qtype="口语",
                               target_component=entry.get("component"),
                               target_fault=entry.get("fault"),
                               note="弱标注：命中同类工单即算对"))
    return items


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def corpus_fingerprint(documents: List[Dict], data_dir: str = "data") -> str:
    """语料指纹：语料一变，评测集就该重建（避免评测集与语料脱节）

    同时把评测侧文件 human_questions.json 纳入哈希：口语化问题集一旦改动，
    评测集也应重建。
    """
    import hashlib

    hasher = hashlib.sha256()
    for document in documents:
        hasher.update(str(document.get("source", "")).encode("utf-8"))
        hasher.update(document.get("content", "").encode("utf-8"))
    human_path = Path(data_dir) / "human_questions.json"
    if human_path.exists():
        hasher.update(human_path.read_bytes())
    return hasher.hexdigest()[:16]


def build_dataset(data_dir: str = "data") -> Dict:
    """从语料构建完整评测集"""
    documents = load_documents(data_dir)
    items = []
    items += _qa_layer(documents)
    items += _fault_case_layer(documents)
    items += _doc_layer(documents)
    items += _near_miss_layer(data_dir)
    items += build_human_like_layer(data_dir)

    # 去重：同层同查询只留一条
    seen, unique = set(), []
    for item in items:
        key = (item["layer"], normalize(item["query"]))
        if key in seen or not item["query"]:
            continue
        seen.add(key)
        unique.append(item)

    return {
        "description": "由语料自动推导的分层评测集，可用 eval --report --rebuild 重新生成",
        "fingerprint": corpus_fingerprint(documents),
        "layers": LAYERS,
        "counts": {layer: sum(1 for item in unique if item["layer"] == layer)
                   for layer in LAYERS},
        "items": unique,
    }


def save_dataset(dataset: Dict, data_dir: str = "data") -> str:
    path = Path(data_dir) / DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    return str(path)


def load_dataset(data_dir: str = "data", rebuild: bool = False, verbose: bool = False) -> Dict:
    """加载评测集；不存在、语料变了或显式要求重建时，重新构建并落盘"""
    path = Path(data_dir) / DATASET_PATH
    if not rebuild and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        current = corpus_fingerprint(load_documents(data_dir))
        if cached.get("fingerprint") == current:
            return cached
        if verbose:
            print("[dataset] 语料已变化，重建评测集")

    dataset = build_dataset(data_dir)
    save_dataset(dataset, data_dir)
    return dataset
