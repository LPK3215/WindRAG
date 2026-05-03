"""
索引构建模块
构建向量索引和BM25索引
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import jieba


class IndexBuilder:
    """索引构建器"""
    
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese", data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.processed_dir / "index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"加载向量模型: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def load_chunks(self) -> List[Dict]:
        """加载文档块"""
        chunks_path = self.processed_dir / "chunks" / "documents.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks
    
    def build_vector_index(self, chunks: List[Dict]):
        """构建向量索引"""
        print("构建向量索引...")
        
        # 提取文本
        texts = [chunk["content"] for chunk in chunks]
        
        # 生成嵌入
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 保存嵌入
        embeddings_path = self.index_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # 保存元数据
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump([chunk["metadata"] for chunk in chunks], f, ensure_ascii=False, indent=2)
        
        print(f"向量索引已保存到 {embeddings_path}")
        return embeddings
    
    def build_bm25_index(self, chunks: List[Dict]):
        """构建BM25索引"""
        print("构建BM25索引...")
        
        # 使用jieba进行中文分词
        tokenized_corpus = []
        for chunk in chunks:
            tokens = list(jieba.cut_for_search(chunk["content"]))
            tokenized_corpus.append(tokens)
        
        # 构建BM25
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 保存BM25索引
        bm25_path = self.index_dir / "bm25.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25, f)
        
        print(f"BM25索引已保存到 {bm25_path}")
        return bm25
    
    def build_all(self):
        """构建所有索引"""
        print("开始构建索引...")
        
        # 加载文档块
        chunks = self.load_chunks()
        print(f"加载了 {len(chunks)} 个文档块")
        
        # 构建向量索引
        self.build_vector_index(chunks)
        
        # 构建BM25索引
        self.build_bm25_index(chunks)
        
        print("索引构建完成！")


if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build_all()
