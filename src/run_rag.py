"""
RAG系统主程序
检索增强生成问答系统
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import os
import jieba
from dotenv import load_dotenv
from knowledge_graph import KnowledgeGraph
from model_manager import create_default_model_manager
from adaptive_fusion import AdaptiveFusionModel

# 加载环境变量
load_dotenv(os.path.join(Path(__file__).parent.parent, 'config.env'))

# 标记LLM可用性（用于本地模型加载，当前使用API方式）
LLM_AVAILABLE = True


class WindTurbineRAG:
    """风电RAG系统"""
    
    def __init__(self, data_dir: str = "data", model_name: str = "shibing624/text2vec-base-chinese", use_llm: bool = True):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.processed_dir / "index"
        
        print(f"加载向量模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 加载索引
        self._load_indexes()
        
        # 加载知识图谱
        self.kg = self._load_knowledge_graph()
        
        # 初始化模型管理器
        self.model_manager = create_default_model_manager()
        print(f"模型池初始化完成，共{len(self.model_manager.models)}个模型")
        
        # 初始化自适应融合模型
        self.adaptive_fusion = AdaptiveFusionModel()
        self.adaptive_fusion_loaded = self.adaptive_fusion.load()
        if self.adaptive_fusion_loaded:
            print("自适应融合模型已加载")
        else:
            print("自适应融合模型未找到，将使用固定α=0.7")
        
        # 检查是否禁用LLM
        disable_llm = os.getenv('DISABLE_LLM', 'false').lower() == 'true'
        if disable_llm:
            print("LLM已禁用，将使用模板生成")
            self.use_llm = False
        else:
            # 加载LLM
            self.use_llm = use_llm and LLM_AVAILABLE
            self.llm_model = None
            self.llm_tokenizer = None
            
            if self.use_llm:
                self._load_llm()
    
    def _load_indexes(self):
        """加载索引"""
        print("加载索引...")
        
        # 加载向量索引
        embeddings_path = self.index_dir / "embeddings.npy"
        self.embeddings = np.load(embeddings_path)
        
        # 加载元数据
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 加载BM25索引
        bm25_path = self.index_dir / "bm25.pkl"
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        print("索引加载完成！")
    
    def _load_knowledge_graph(self):
        """加载知识图谱"""
        kg_path = self.data_dir / "knowledge_graph.json"
        if kg_path.exists():
            print("加载知识图谱...")
            kg = KnowledgeGraph.load(str(kg_path))
            print(f"知识图谱加载完成！节点数: {len(kg.nodes)}, 边数: {len([e for edges in kg.edges.values() for e in edges])}")
            return kg
        else:
            print("知识图谱文件不存在，跳过加载")
            return None
    
    def _load_llm(self):
        """加载ModelScope LLM"""
        api_key = os.getenv('MODELSCOPE_API_KEY')
        model_name = os.getenv('MODELSCOPE_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
        
        if not api_key or api_key == 'your_modelscope_api_key_here':
            print("警告: 未配置ModelScope API Key，将使用模板生成答案")
            self.use_llm = False
            return
        
        try:
            print(f"ModelScope API已配置，模型: {model_name}")
            print("ModelScope集成完成，将使用API生成答案")
            self.use_llm = True
        except Exception as e:
            print(f"ModelScope配置失败: {e}")
            self.use_llm = False
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """向量检索（L2归一化后计算余弦相似度）"""
        # 生成查询向量
        query_embedding = self.model.encode([query])
        
        # L2归一化文档向量
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)  # 避免除零
        normalized_embeddings = self.embeddings / norms
        
        # L2归一化查询向量
        q_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        q_norm = np.where(q_norm == 0, 1e-8, q_norm)
        normalized_query = query_embedding / q_norm
        
        # 计算余弦相似度（归一化后点积=余弦相似度）
        similarities = np.dot(normalized_embeddings, normalized_query.T).flatten()
        
        # 获取top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return list(zip(top_indices.tolist(), top_scores.tolist()))
    
    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """BM25检索（使用jieba中文分词）"""
        # 使用jieba进行中文分词
        query_tokens = list(jieba.cut_for_search(query))
        
        # 检索
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return list(zip(top_indices.tolist(), top_scores.tolist()))
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = None) -> List[Tuple[int, float]]:
        """混合检索（支持自适应α）"""
        # 自适应α预测
        if alpha is None:
            if self.adaptive_fusion_loaded:
                query_embedding = self.model.encode([query])[0]
                alpha = self.adaptive_fusion.predict_alpha(query, query_embedding)
            else:
                alpha = 0.7
        
        # 向量检索
        vector_results = self.vector_search(query, top_k=top_k * 2)
        vector_scores = {idx: score for idx, score in vector_results}
        
        # BM25检索
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # 归一化
        all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        normalized_vector = {}
        normalized_bm25 = {}
        
        if vector_scores:
            v_max = max(vector_scores.values())
            v_min = min(vector_scores.values())
            for idx in all_indices:
                if idx in vector_scores:
                    if v_max - v_min > 0:
                        normalized_vector[idx] = (vector_scores[idx] - v_min) / (v_max - v_min)
                    else:
                        normalized_vector[idx] = 1.0
                else:
                    normalized_vector[idx] = 0.0
        
        if bm25_scores:
            b_max = max(bm25_scores.values())
            b_min = min(bm25_scores.values())
            for idx in all_indices:
                if idx in bm25_scores:
                    if b_max - b_min > 0:
                        normalized_bm25[idx] = (bm25_scores[idx] - b_min) / (b_max - b_min)
                    else:
                        normalized_bm25[idx] = 1.0
                else:
                    normalized_bm25[idx] = 0.0
        
        # 融合
        fused_scores = {}
        for idx in all_indices:
            fused_scores[idx] = alpha * normalized_vector.get(idx, 0) + (1 - alpha) * normalized_bm25.get(idx, 0)
        
        # 获取top-k
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results
    
    def retrieve_documents(self, query: str, top_k: int = 5, method: str = "hybrid", use_rerank: bool = True, alpha: float = None) -> List[Dict]:
        """检索文档"""
        if method == "vector":
            results = self.vector_search(query, top_k * 2)  # 检索更多文档用于重排序
        elif method == "bm25":
            results = self.bm25_search(query, top_k * 2)
        else:
            results = self.hybrid_search(query, top_k * 2, alpha=alpha)
        
        # 加载文档内容
        chunks_path = self.processed_dir / "chunks" / "documents.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # 返回结果
        retrieved = []
        for idx, score in results:
            doc = chunks[idx]
            retrieved.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": score
            })
        
        # 重排序
        if use_rerank and len(retrieved) > top_k:
            retrieved = self._rerank_documents(query, retrieved, top_k)
        else:
            retrieved = retrieved[:top_k]
        
        # 知识图谱增强
        if self.kg:
            kg_context = self._query_knowledge_graph(query)
            if kg_context:
                retrieved.append({
                    "content": kg_context,
                    "metadata": {"source": "knowledge_graph", "type": "kg_enhancement"},
                    "score": 0.8
                })
        
        return retrieved
    
    def _rerank_documents(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """重排序文档"""
        # 计算查询和文档的相似度
        query_embedding = self.model.encode([query])
        
        reranked = []
        for doc in documents:
            doc_embedding = self.model.encode([doc["content"]])
            similarity = np.dot(query_embedding, doc_embedding.T)[0][0]
            
            # 结合原始检索分数和语义相似度
            combined_score = 0.6 * doc["score"] + 0.4 * similarity
            reranked.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": combined_score
            })
        
        # 按综合分数排序
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
    
    def _query_knowledge_graph(self, query: str) -> str:
        """查询知识图谱获取相关上下文（基于embedding语义匹配）"""
        if not self.kg:
            return ""
        
        # 编码查询
        query_embedding = self.model.encode([query])
        q_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        q_norm = np.where(q_norm == 0, 1e-8, q_norm)
        normalized_query = query_embedding / q_norm
        
        best_match = None
        best_score = 0.0
        
        # 在所有部件节点中做语义匹配
        for node_id, node_data in self.kg.nodes.items():
            if node_data["type"] == "component":
                # 编码节点描述文本
                node_text = node_data["properties"].get("description", node_id)
                node_embedding = self.model.encode([node_text])
                n_norm = np.linalg.norm(node_embedding, axis=1, keepdims=True)
                n_norm = np.where(n_norm == 0, 1e-8, n_norm)
                normalized_node = node_embedding / n_norm
                
                # 计算余弦相似度
                similarity = float(np.dot(normalized_query, normalized_node.T)[0][0])
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = node_id
        
        # 只有当相似度超过阈值时才使用图谱上下文
        if best_match and best_score > 0.5:
            context_parts = []
            neighbors = self.kg.get_neighbors(best_match, "HAS_FAULT")
            for neighbor in neighbors:
                fault_node = self.kg.nodes[neighbor["node_id"]]
                context_parts.append(f"{best_match}故障: {fault_node['properties']['name']}")
            return " | ".join(context_parts) if context_parts else ""
        
        return ""
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """生成答案"""
        # 构建上下文
        context = "\n\n".join([doc["content"] for doc in retrieved_docs[:3]])
        
        # 如果启用了LLM且可用
        if self.use_llm:
            return self._generate_with_llm(query, context)
        else:
            # 使用模板生成
            return f"""基于检索到的相关信息，回答您的问题：

问题：{query}

参考信息：
{context}

答案：
根据检索到的风电领域知识，建议您参考上述相关信息。如需更详细的解答，请提供更多上下文或具体场景。"""
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """使用LLM生成答案（使用模型管理器）"""
        from openai import OpenAI
        
        # 获取当前可用模型
        model_info = self.model_manager.get_current_model()
        if not model_info:
            print("没有可用的模型，使用模板生成")
            return self._fallback_generate(query, context)
        
        prompt = f"""你是一个风电领域的专家助手。请基于以下参考信息回答用户的问题。

参考信息：
{context}

用户问题：{query}

请用简洁、专业的语言回答问题。如果参考信息中没有相关内容，请说明。"""
        
        # 尝试使用当前模型
        for attempt in range(3):  # 最多尝试3次
            try:
                current_model = self.model_manager.get_current_model()
                if not current_model:
                    break
                
                # 添加延迟以避免速率限制
                import time
                time.sleep(0.5)  # 500ms延迟
                
                # 使用OpenAI客户端调用ModelScope
                client = OpenAI(
                    base_url=current_model.base_url,
                    api_key=current_model.api_key
                )
                
                response = client.chat.completions.create(
                    model=current_model.name,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
                    temperature=0.7,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                # 报告成功
                self.model_manager.report_success(current_model.name)
                return answer
                
            except Exception as e:
                print(f"模型 {current_model.name} 调用失败 (尝试 {attempt + 1}/3): {e}")
                # 报告失败，触发模型切换
                self.model_manager.report_failure(current_model.name, str(e))
                continue
        
        # 所有模型都失败，使用模板生成
        print("所有模型都不可用，使用模板生成")
        return self._fallback_generate(query, context)
    
    def _fallback_generate(self, query: str, context: str) -> str:
        """回退到模板生成"""
        return f"""基于检索到的相关信息，回答您的问题：

问题：{query}

参考信息：
{context}

答案：
根据检索到的风电领域知识，建议您参考上述相关信息。"""
    
    def query(self, question: str, top_k: int = 5, method: str = "hybrid") -> Dict:
        """查询接口"""
        print(f"\n问题：{question}")
        
        # 检索
        retrieved = self.retrieve_documents(question, top_k, method)
        print(f"检索到 {len(retrieved)} 个相关文档")
        
        # 生成答案
        answer = self.generate_answer(question, retrieved)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved
        }


def main():
    """主函数"""
    # 初始化RAG系统
    rag = WindTurbineRAG()
    
    # 测试查询
    test_questions = [
        "风电机的齿轮箱温度过高应该怎么处理？",
        "如何判断风电机叶片是否需要更换？",
        "风电机组的偏航系统有什么作用？"
    ]
    
    for question in test_questions:
        result = rag.query(question, top_k=3)
        print(f"\n{'='*80}")
        print(f"答案：{result['answer']}")
        print(f"\n检索到的文档（前2个）：")
        for i, doc in enumerate(result['retrieved_docs'][:2]):
            print(f"\n文档{i+1} (评分: {doc['score']:.4f}):")
            print(f"来源: {doc['metadata']['source']}")
            print(f"内容: {doc['content'][:200]}...")


if __name__ == "__main__":
    main()
