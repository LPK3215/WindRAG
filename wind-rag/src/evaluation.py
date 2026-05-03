"""
RAG系统评估模块
自定义评估指标，避免RAGAS依赖冲突
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import re


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.test_questions = self._load_test_questions()
    
    def _load_test_questions(self) -> List[Dict]:
        """加载测试问题"""
        qa_pairs_path = self.data_dir / "synthetic" / "qa_pairs.json"
        if qa_pairs_path.exists():
            with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
                return qa_data["qa_pairs"]
        return []
    
    def evaluate_retrieval(self, retrieved_docs: List[Dict], expected_keywords: List[str]) -> Dict:
        """评估检索质量"""
        if not retrieved_docs:
            return {
                "retrieved_count": 0,
                "keyword_match_rate": 0.0,
                "avg_relevance_score": 0.0
            }
        
        # 关键词匹配率：至少有一个关键词出现在至少一个文档中的比例
        keyword_match_rate = 0.0
        if expected_keywords and retrieved_docs:
            matched_queries = 0
            for keyword in expected_keywords:
                for doc in retrieved_docs:
                    if keyword.lower() in doc["content"].lower():
                        matched_queries += 1
                        break
            keyword_match_rate = matched_queries / len(expected_keywords)
        elif not expected_keywords:
            keyword_match_rate = 1.0
        
        # 平均相关性评分
        avg_relevance_score = sum(doc.get("score", 0) for doc in retrieved_docs) / len(retrieved_docs)
        
        return {
            "retrieved_count": len(retrieved_docs),
            "keyword_match_rate": keyword_match_rate,
            "avg_relevance_score": avg_relevance_score
        }
    
    def evaluate_answer(self, answer: str, reference_answer: str = None) -> Dict:
        """评估答案质量"""
        if not answer:
            return {
                "answer_length": 0,
                "has_content": False,
                "keyword_coverage": 0.0
            }
        
        # 答案长度
        answer_length = len(answer)
        
        # 是否有实质内容
        has_content = len(answer.strip()) > 10
        
        # 关键词覆盖率（如果有参考答案）
        keyword_coverage = 0.0
        if reference_answer:
            ref_words = set(re.findall(r'\w+', reference_answer.lower()))
            ans_words = set(re.findall(r'\w+', answer.lower()))
            if ref_words:
                keyword_coverage = len(ans_words & ref_words) / len(ref_words)
        
        return {
            "answer_length": answer_length,
            "has_content": has_content,
            "keyword_coverage": keyword_coverage
        }
    
    def evaluate_response_time(self, query_time: float) -> Dict:
        """评估响应时间"""
        return {
            "response_time": query_time,
            "is_acceptable": query_time < 10.0  # 10秒内为可接受
        }
    
    def evaluate_single_query(self, rag_system, question: str, expected_keywords: List[str] = None) -> Dict:
        """评估单个查询"""
        start_time = time.time()
        result = rag_system.query(question, top_k=3, method="hybrid")
        query_time = time.time() - start_time
        
        # 评估检索
        retrieval_metrics = self.evaluate_retrieval(result["retrieved_docs"], expected_keywords or [])
        
        # 评估答案
        answer_metrics = self.evaluate_answer(result["answer"])
        
        # 评估响应时间
        time_metrics = self.evaluate_response_time(query_time)
        
        return {
            "question": question,
            "answer": result["answer"],
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "time_metrics": time_metrics
        }
    
    def run_evaluation(self, rag_system, num_questions: int = 10) -> Dict:
        """运行完整评估"""
        print(f"开始评估RAG系统，测试 {num_questions} 个问题...")
        
        results = []
        test_qs = self.test_questions[:num_questions]
        
        for i, qa in enumerate(test_qs, 1):
            print(f"测试问题 {i}/{len(test_qs)}: {qa['question'][:30]}...")
            
            # 提取期望关键词
            expected_keywords = []
            if "category" in qa:
                expected_keywords.append(qa["category"])
            
            result = self.evaluate_single_query(rag_system, qa["question"], expected_keywords)
            results.append(result)
        
        # 计算总体指标
        overall_metrics = self._calculate_overall_metrics(results)
        
        return {
            "overall_metrics": overall_metrics,
            "detailed_results": results
        }
    
    def _calculate_overall_metrics(self, results: List[Dict]) -> Dict:
        """计算总体指标"""
        total = len(results)
        if total == 0:
            return {}
        
        # 检索指标
        avg_retrieved_count = sum(r["retrieval_metrics"]["retrieved_count"] for r in results) / total
        avg_keyword_match = sum(r["retrieval_metrics"]["keyword_match_rate"] for r in results) / total
        avg_relevance = sum(r["retrieval_metrics"]["avg_relevance_score"] for r in results) / total
        
        # 答案指标
        avg_answer_length = sum(r["answer_metrics"]["answer_length"] for r in results) / total
        content_rate = sum(r["answer_metrics"]["has_content"] for r in results) / total
        
        # 时间指标
        avg_response_time = sum(r["time_metrics"]["response_time"] for r in results) / total
        acceptable_rate = sum(r["time_metrics"]["is_acceptable"] for r in results) / total
        
        return {
            "total_queries": total,
            "retrieval": {
                "avg_retrieved_count": avg_retrieved_count,
                "avg_keyword_match_rate": avg_keyword_match,
                "avg_relevance_score": avg_relevance
            },
            "answer": {
                "avg_answer_length": avg_answer_length,
                "content_rate": content_rate
            },
            "performance": {
                "avg_response_time": avg_response_time,
                "acceptable_rate": acceptable_rate
            }
        }
    
    def save_evaluation_report(self, evaluation_result: Dict, output_path: str = "data/evaluation_report.json"):
        """保存评估报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        print(f"评估报告已保存到 {output_file}")
    
    def print_summary(self, evaluation_result: Dict):
        """打印评估摘要"""
        metrics = evaluation_result["overall_metrics"]
        
        print("\n" + "="*50)
        print("RAG系统评估报告")
        print("="*50)
        print(f"测试查询数: {metrics['total_queries']}")
        
        print("\n【检索质量】")
        print(f"  平均检索数量: {metrics['retrieval']['avg_retrieved_count']:.2f}")
        print(f"  关键词匹配率: {metrics['retrieval']['avg_keyword_match_rate']:.2%}")
        print(f"  平均相关性评分: {metrics['retrieval']['avg_relevance_score']:.4f}")
        
        print("\n【答案质量】")
        print(f"  平均答案长度: {metrics['answer']['avg_answer_length']:.0f} 字符")
        print(f"  有效内容率: {metrics['answer']['content_rate']:.2%}")
        
        print("\n【性能】")
        print(f"  平均响应时间: {metrics['performance']['avg_response_time']:.2f} 秒")
        print(f"  可接受响应率: {metrics['performance']['acceptable_rate']:.2%}")
        print("="*50)


if __name__ == "__main__":
    from run_rag import WindTurbineRAG
    
    # 初始化RAG系统
    print("初始化RAG系统...")
    rag = WindTurbineRAG()
    
    # 运行评估
    evaluator = RAGEvaluator()
    result = evaluator.run_evaluation(rag, num_questions=5)
    
    # 打印摘要
    evaluator.print_summary(result)
    
    # 保存报告
    evaluator.save_evaluation_report(result)
