"""
知识图谱模块
轻量级实现，无需Neo4j
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


class KnowledgeGraph:
    """轻量级知识图谱"""
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # node_id -> node_data
        self.edges: Dict[str, List[Dict]] = defaultdict(list)  # source -> [(target, relation, metadata)]
        self.reverse_edges: Dict[str, List[Dict]] = defaultdict(list)  # target -> [(source, relation, metadata)]
    
    def add_node(self, node_id: str, node_type: str, properties: Dict = None):
        """添加节点"""
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": properties or {}
        }
    
    def add_edge(self, source: str, target: str, relation: str, properties: Dict = None):
        """添加边"""
        edge = {
            "source": source,
            "target": target,
            "relation": relation,
            "properties": properties or {}
        }
        self.edges[source].append(edge)
        self.reverse_edges[target].append(edge)
    
    def get_neighbors(self, node_id: str, relation: str = None) -> List[Dict]:
        """获取邻居节点"""
        neighbors = []
        for edge in self.edges[node_id]:
            if relation is None or edge["relation"] == relation:
                neighbors.append({
                    "node_id": edge["target"],
                    "relation": edge["relation"],
                    "properties": edge["properties"]
                })
        return neighbors
    
    def get_reverse_neighbors(self, node_id: str, relation: str = None) -> List[Dict]:
        """获取反向邻居节点"""
        neighbors = []
        for edge in self.reverse_edges[node_id]:
            if relation is None or edge["relation"] == relation:
                neighbors.append({
                    "node_id": edge["source"],
                    "relation": edge["relation"],
                    "properties": edge["properties"]
                })
        return neighbors
    
    def find_nodes_by_type(self, node_type: str) -> List[Dict]:
        """按类型查找节点"""
        return [node for node in self.nodes.values() if node["type"] == node_type]
    
    def find_nodes_by_property(self, key: str, value: Any) -> List[Dict]:
        """按属性查找节点"""
        return [node for node in self.nodes.values() 
                if node["properties"].get(key) == value]
    
    def get_subgraph(self, start_node: str, max_depth: int = 2) -> 'KnowledgeGraph':
        """获取子图（广度优先遍历）"""
        subgraph = KnowledgeGraph()
        visited = set()
        queue = [(start_node, 0)]
        
        while queue:
            node_id, depth = queue.pop(0)
            if node_id in visited or depth > max_depth:
                continue
            
            visited.add(node_id)
            node_data = self.nodes[node_id]
            subgraph.add_node(node_id, node_data["type"], node_data["properties"])
            
            # 添加边和邻居节点
            for edge in self.edges[node_id]:
                subgraph.add_edge(node_id, edge["target"], edge["relation"], edge["properties"])
                if edge["target"] not in visited:
                    queue.append((edge["target"], depth + 1))
        
        return subgraph
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "nodes": list(self.nodes.values()),
            "edges": [edge for edges in self.edges.values() for edge in edges]
        }
    
    def save(self, filepath: str):
        """保存图谱到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """从文件加载图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls()
        for node in data["nodes"]:
            kg.add_node(node["id"], node["type"], node["properties"])
        
        for edge in data["edges"]:
            kg.add_edge(edge["source"], edge["target"], edge["relation"], edge["properties"])
        
        return kg


class WindPowerKnowledgeGraphBuilder:
    """风电知识图谱构建器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.kg = KnowledgeGraph()
    
    def build_from_data(self):
        """从现有数据构建知识图谱"""
        print("开始构建风电知识图谱...")
        
        # 添加部件节点
        self._add_components()
        
        # 从故障案例构建图谱
        self._build_from_fault_cases()
        
        # 从运维手册构建图谱
        self._build_from_operations_manual()
        
        print(f"知识图谱构建完成！节点数: {len(self.kg.nodes)}, 边数: {len([e for edges in self.kg.edges.values() for e in edges])}")
        
        return self.kg
    
    def _add_components(self):
        """添加部件节点"""
        components = [
            ("齿轮箱", "component", {"description": "传动系统核心部件"}),
            ("发电机", "component", {"description": "机械能转换为电能"}),
            ("叶片", "component", {"description": "捕获风能"}),
            ("变桨系统", "component", {"description": "调节叶片角度"}),
            ("偏航系统", "component", {"description": "对准风向"}),
            ("液压系统", "component", {"description": "提供液压动力"}),
            ("制动系统", "component", {"description": "安全制动"}),
            ("塔筒", "component", {"description": "支撑结构"}),
            ("变压器", "component", {"description": "电压变换"})
        ]
        
        for comp_id, comp_type, props in components:
            self.kg.add_node(comp_id, comp_type, props)
    
    def _build_from_fault_cases(self):
        """从故障案例构建图谱"""
        fault_cases_path = self.data_dir / "synthetic" / "fault_cases.json"
        if not fault_cases_path.exists():
            return
        
        with open(fault_cases_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for case in data["fault_cases"]:
            component = case["component"]
            fault_type = case["fault_type"]
            
            # 添加故障节点
            fault_id = f"{component}_{fault_type}"
            self.kg.add_node(fault_id, "fault_type", {
                "name": fault_type,
                "severity": case["severity"]
            })
            
            # 部件 -> 故障
            self.kg.add_edge(component, fault_id, "HAS_FAULT")
            
            # 添加症状节点
            symptoms = case["symptoms"].split("、")
            for symptom in symptoms:
                symptom_id = f"symptom_{symptom[:4]}"
                if symptom_id not in self.kg.nodes:
                    self.kg.add_node(symptom_id, "symptom", {"name": symptom})
                self.kg.add_edge(fault_id, symptom_id, "HAS_SYMPTOM")
            
            # 添加原因节点
            causes = case["causes"].split("、")
            for cause in causes:
                cause_id = f"cause_{cause[:4]}"
                if cause_id not in self.kg.nodes:
                    self.kg.add_node(cause_id, "cause", {"name": cause})
                self.kg.add_edge(fault_id, cause_id, "CAUSED_BY")
            
            # 添加解决方案节点
            solution_id = f"solution_{fault_id}"
            self.kg.add_node(solution_id, "solution", {
                "description": case["solution"]
            })
            self.kg.add_edge(fault_id, solution_id, "SOLVED_BY")
    
    def _build_from_operations_manual(self):
        """从运维手册构建图谱"""
        ops_manual_path = self.data_dir / "synthetic" / "operations_manual.json"
        if not ops_manual_path.exists():
            return
        
        with open(ops_manual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for doc in data["operations_manual"]:
            category = doc["category"]
            title = doc["title"]
            
            # 添加维护操作节点
            op_id = f"op_{doc['id']}"
            self.kg.add_node(op_id, "maintenance_operation", {
                "title": title,
                "category": category
            })
            
            # 根据标题关键词建立部件关系（配置驱动，避免硬编码if-elif链）
            component_keywords = {
                "齿轮箱": ["齿轮箱"],
                "发电机": ["发电机", "电气"],
                "叶片": ["叶片"],
                "变桨系统": ["变桨"],
                "偏航系统": ["偏航"],
                "液压系统": ["液压"],
                "制动系统": ["制动"],
                "塔筒": ["塔筒"],
                "变压器": ["变压器"],
                "控制系统": ["控制", "PLC", "传感器"],
                "轴承": ["轴承"],
            }
            for comp, keywords in component_keywords.items():
                if any(kw in title for kw in keywords):
                    self.kg.add_edge(comp, op_id, "REQUIRES")
                    break


def build_and_save_kg(data_dir: str = "data", output_path: str = "data/knowledge_graph.json"):
    """构建并保存知识图谱"""
    builder = WindPowerKnowledgeGraphBuilder(data_dir)
    kg = builder.build_from_data()
    kg.save(output_path)
    print(f"知识图谱已保存到 {output_path}")
    return kg


if __name__ == "__main__":
    build_and_save_kg()
