"""
数据处理模块
加载和预处理风电领域数据
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import re


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.synthetic_dir = self.data_dir / "synthetic"
        self.processed_dir = self.data_dir / "processed"
        
        # 创建处理后的数据目录
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "chunks").mkdir(exist_ok=True)
    
    def load_raw_text(self) -> List[Dict]:
        """加载原始文本数据"""
        documents = []
        
        # 加载OpenWindSCADA README
        readme_path = self.raw_dir / "openscada_readme.txt"
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "source": "openscada_readme",
                    "content": content,
                    "metadata": {"type": "documentation"}
                })
        
        return documents
    
    def load_synthetic_data(self) -> List[Dict]:
        """加载合成数据"""
        documents = []
        
        # 加载故障案例
        fault_cases_path = self.synthetic_dir / "fault_cases.json"
        if fault_cases_path.exists():
            with open(fault_cases_path, 'r', encoding='utf-8') as f:
                fault_data = json.load(f)
                for case in fault_data["fault_cases"]:
                    content = self._format_fault_case(case)
                    documents.append({
                        "source": "fault_cases",
                        "content": content,
                        "metadata": {
                            "type": "fault_case",
                            "component": case["component"],
                            "fault_type": case["fault_type"],
                            "severity": case["severity"]
                        }
                    })
        
        # 加载问答对
        qa_pairs_path = self.synthetic_dir / "qa_pairs.json"
        if qa_pairs_path.exists():
            with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
                for qa in qa_data["qa_pairs"]:
                    content = f"问题：{qa['question']}\n答案：{qa['answer']}"
                    documents.append({
                        "source": "qa_pairs",
                        "content": content,
                        "metadata": {
                            "type": "qa_pair",
                            "category": qa["category"]
                        }
                    })
        
        # 加载技术文档
        technical_docs_path = self.synthetic_dir / "technical_docs.json"
        if technical_docs_path.exists():
            with open(technical_docs_path, 'r', encoding='utf-8') as f:
                tech_data = json.load(f)
                for doc in tech_data["technical_docs"]:
                    content = f"标题：{doc['title']}\n类别：{doc['category']}\n内容：{doc['content']}"
                    documents.append({
                        "source": "technical_docs",
                        "content": content,
                        "metadata": {
                            "type": "technical_doc",
                            "category": doc["category"],
                            "title": doc["title"]
                        }
                    })
        
        # 加载运维手册
        operations_manual_path = self.synthetic_dir / "operations_manual.json"
        if operations_manual_path.exists():
            with open(operations_manual_path, 'r', encoding='utf-8') as f:
                ops_data = json.load(f)
                for doc in ops_data["operations_manual"]:
                    content = f"标题：{doc['title']}\n类别：{doc['category']}\n内容：{doc['content']}"
                    documents.append({
                        "source": "operations_manual",
                        "content": content,
                        "metadata": {
                            "type": "operations_manual",
                            "category": doc["category"],
                            "title": doc["title"]
                        }
                    })
        
        return documents
    
    def _format_fault_case(self, case: Dict) -> str:
        """格式化故障案例"""
        return f"""
故障ID：{case['id']}
部件：{case['component']}
故障类型：{case['fault_type']}
症状：{case['symptoms']}
原因：{case['causes']}
诊断：{case['diagnosis']}
解决方案：{case['solution']}
预防措施：{case['prevention']}
严重程度：{case['severity']}
"""
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """将文档分割成块"""
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # 按段落分割
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            
            current_chunk = ""
            chunk_id = 0
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n"
                else:
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "metadata": {
                                **metadata,
                                "chunk_id": chunk_id,
                                "source": doc["source"]
                            }
                        })
                        chunk_id += 1
                    current_chunk = para + "\n"
            
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_id,
                        "source": doc["source"]
                    }
                })
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict]):
        """保存文档块"""
        chunks_path = self.processed_dir / "chunks" / "documents.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(chunks)} 个文档块到 {chunks_path}")
    
    def process_all(self):
        """处理所有数据"""
        print("开始处理数据...")
        
        # 加载数据
        raw_docs = self.load_raw_text()
        synthetic_docs = self.load_synthetic_data()
        
        all_docs = raw_docs + synthetic_docs
        print(f"加载了 {len(all_docs)} 个文档")
        
        # 分块
        chunks = self.chunk_documents(all_docs)
        print(f"分割成 {len(chunks)} 个文档块")
        
        # 保存
        self.save_chunks(chunks)
        
        print("数据处理完成！")
        return chunks


if __name__ == "__main__":
    processor = DataProcessor()
    chunks = processor.process_all()
