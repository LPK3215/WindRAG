"""
知识图谱检索增强（零第三方依赖）

直接复用 src/knowledge_graph.py 的轻量图谱实现，只替换实体匹配方式：
原版用 sentence-transformers 的 embedding 余弦相似度；这里改为
「TF-IDF 稀疏向量相似度 + 字面命中加权」，同样解决"变速箱→齿轮箱"这类同义问题。
"""

from pathlib import Path
from typing import Dict, List, Tuple

from knowledge_graph import KnowledgeGraph

from pure_rag.text_utils import Tokenizer, expand_query
from pure_rag.tfidf import TfidfIndex

# 可作为检索入口的实体类型
ENTRY_TYPES = ("component", "fault_type")


class KnowledgeGraphRetriever:
    """知识图谱检索器：实体匹配 + 子图文本化"""

    def __init__(self, kg_path: str = "data/knowledge_graph.json",
                 tokenizer: Tokenizer = None, threshold: float = 0.2):
        self.path = Path(kg_path)
        self.tokenizer = tokenizer or Tokenizer()
        self.threshold = threshold
        self.kg: KnowledgeGraph = None
        self._index = TfidfIndex()
        self._node_ids: List[str] = []
        self._component_ids: List[str] = []
        self._fault_owners: Dict[str, str] = {}

        if self.path.exists():
            self.kg = KnowledgeGraph.load(str(self.path))
            self._build_index()
            self._build_fault_owners()

    @property
    def available(self) -> bool:
        return self.kg is not None and bool(self._node_ids)

    def _node_text(self, node: Dict) -> str:
        """节点的可检索文本：名称 + 描述/标题 + 严重程度"""
        props = node.get("properties", {})
        parts = [node["id"], props.get("name", ""), props.get("description", ""),
                 props.get("title", ""), props.get("severity", "")]
        return " ".join(p for p in parts if p)

    def _build_index(self) -> None:
        """对所有入口实体建 TF-IDF 索引"""
        texts, ids = [], []
        for node_id, node in self.kg.nodes.items():
            if node["type"] not in ENTRY_TYPES:
                continue
            ids.append(node_id)
            texts.append(self.tokenizer.cut_for_index(self._node_text(node)))
        self._node_ids = ids
        self._component_ids = [node_id for node_id in ids
                               if self.kg.nodes[node_id]["type"] == "component"]
        self._index.build(texts)

    def _build_fault_owners(self) -> None:
        """建立「故障名 → 所属部件」映射，用于校验部件与故障的搭配是否成立"""
        for node_id, node in self.kg.nodes.items():
            if node["type"] != "fault_type":
                continue
            name = node.get("properties", {}).get("name") or node_id
            parents = self.kg.get_reverse_neighbors(node_id, "HAS_FAULT")
            if parents:
                self._fault_owners[name] = parents[0]["node_id"]

    # -- 组合校验 -----------------------------------------------------------
    def validate_combination(self, query: str) -> Dict:
        """
        校验查询里「部件 + 故障」的搭配在图谱中是否存在

        覆盖率判据只看得懂"词在不在语料里"，对「偏航系统制动器失效」这类
        **词对组合不存在**的伪问题无能为力（实测拒答率仅 5.56%）。
        这里用图谱的 `部件 --HAS_FAULT--> 故障` 归属关系补上这一环：
        只有当查询同时点名了部件与故障、且所有搭配都不成立时才判为无效组合。
        """
        result = {"components": [], "faults": [], "invalid": [], "valid": []}
        if not self.available or not query:
            return result

        # 只在原文上匹配，不做同义词扩展：扩展会把「偏航偏差过大」里的“偏航”
        # 补成「偏航系统」，凭空造出一个有效搭配，反而漏掉该拦的伪问题。
        result["components"] = [node_id for node_id in self._component_ids
                                if node_id in query]
        result["faults"] = [name for name in self._fault_owners if name in query]
        for fault in result["faults"]:
            owner = self._fault_owners[fault]
            for component in result["components"]:
                pair = {"component": component, "fault": fault, "owner": owner}
                key = "valid" if component == owner else "invalid"
                result[key].append(pair)
        return result

    # -- 实体匹配 -----------------------------------------------------------
    def match_nodes(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        语义匹配实体：字面命中直接给满分，其余用 TF-IDF 余弦相似度

        同义词变体（如"变速箱"→"齿轮箱"）会一并参与字面匹配，
        等价于原版 embedding 相似度解决的"同义词匹配不上"问题。
        返回 [(node_id, score)]，仅保留超过阈值的实体
        """
        if not self.available:
            return []

        candidates = [query] + expand_query(query)
        tokens = self.tokenizer.cut_for_index(" ".join(candidates))
        literal_hits = [
            node_id for node_id in self._node_ids
            if any(node_id in variant for variant in candidates)
        ]

        scores: Dict[str, float] = {}
        for doc_id, score in self._index.score(tokens).items():
            scores[self._node_ids[doc_id]] = float(score)
        for node_id in literal_hits:
            scores[node_id] = max(scores.get(node_id, 0.0), 1.0)

        matched = [(node_id, score) for node_id, score in scores.items() if score >= self.threshold]
        matched.sort(key=lambda item: (-item[1], item[0]))
        return matched[:top_k]

    # -- 上下文构造 ---------------------------------------------------------
    def _neighbor_names(self, node_id: str, relation: str, limit: int = 6) -> List[str]:
        """把邻居节点转成可读文本（名称 > 标题 > 描述 > 节点 id）"""
        names = []
        for neighbor in self.kg.get_neighbors(node_id, relation):
            node = self.kg.nodes.get(neighbor["node_id"], {})
            props = node.get("properties", {})
            name = (props.get("name") or props.get("title")
                    or props.get("description") or node.get("id", ""))
            severity = props.get("severity", "")
            names.append(f"{name}({severity})" if severity else name)
        return names[:limit]

    def describe(self, node_id: str) -> str:
        """把单个实体的邻域信息文本化"""
        node = self.kg.nodes.get(node_id)
        if not node:
            return ""

        lines: List[str] = []
        if node["type"] == "component":
            faults = self._neighbor_names(node_id, "HAS_FAULT")
            if faults:
                lines.append(f"{node_id}的常见故障：{'、'.join(faults)}")
            ops = self._neighbor_names(node_id, "REQUIRES", limit=3)
            if ops:
                lines.append(f"{node_id}相关运维操作：{'、'.join(ops)}")
        elif node["type"] == "fault_type":
            symptoms = self._neighbor_names(node_id, "HAS_SYMPTOM", limit=5)
            causes = self._neighbor_names(node_id, "CAUSED_BY", limit=5)
            solutions = self._neighbor_names(node_id, "SOLVED_BY", limit=2)
            if symptoms:
                lines.append(f"{node_id}症状：{'、'.join(symptoms)}")
            if causes:
                lines.append(f"{node_id}原因：{'、'.join(causes)}")
            for solution in solutions:
                lines.append(f"{node_id}解决方案：{solution}")
            # 反向找到所属部件
            parents = self.kg.get_reverse_neighbors(node_id, "HAS_FAULT")
            if parents:
                lines.append(f"所属部件：{parents[0]['node_id']}")
        return "\n".join(lines)

    def build_context(self, query: str, top_k: int = 2) -> Dict:
        """
        查询知识图谱，返回 {"text": 上下文文本, "entities": [实体名]}

        未命中任何实体时 text 为空字符串，调用方据此跳过图谱增强。
        """
        matched = self.match_nodes(query, top_k=top_k)
        if not matched:
            return {"text": "", "entities": []}

        blocks, entities = [], []
        for node_id, score in matched:
            description = self.describe(node_id)
            if description:
                blocks.append(description)
                entities.append({"name": node_id, "score": round(score, 4)})

        return {"text": "\n".join(blocks), "entities": entities}
