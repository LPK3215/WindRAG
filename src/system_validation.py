"""
系统稳定性与可行性验证
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict
from run_rag import WindTurbineRAG


class SystemValidator:
    """系统验证器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.rag = None
        self.results = {}
    
    def init_rag(self):
        """初始化RAG系统"""
        print("初始化RAG系统...")
        self.rag = WindTurbineRAG()
        print("RAG系统初始化完成")
    
    def test_long_running(self, num_queries: int = 100) -> Dict:
        """长时间运行测试"""
        print(f"\n开始长时间运行测试，共{num_queries}次查询...")
        
        # 加载测试问题
        qa_pairs_path = self.data_dir / "synthetic" / "qa_pairs.json"
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        questions = [qa["question"] for qa in qa_data["qa_pairs"]]
        
        # 扩充问题列表
        test_queries = questions * (num_queries // len(questions) + 1)
        test_queries = test_queries[:num_queries]
        
        start_time = time.time()
        success_count = 0
        fail_count = 0
        error_messages = []
        response_times = []
        
        for i, query in enumerate(test_queries, 1):
            try:
                query_start = time.time()
                result = self.rag.query(query, top_k=3, method="hybrid")
                query_time = time.time() - query_start
                response_times.append(query_time)
                success_count += 1
                
                if i % 20 == 0:
                    print(f"  进度: {i}/{num_queries}, 成功率: {success_count/i:.1%}")
            except Exception as e:
                fail_count += 1
                error_messages.append(f"查询{i}: {str(e)}")
        
        total_time = time.time() - start_time
        
        result = {
            "total_queries": num_queries,
            "success_count": success_count,
            "fail_count": fail_count,
            "success_rate": success_count / num_queries,
            "total_time": total_time,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "errors": error_messages[:10]  # 只保存前10个错误
        }
        
        print(f"长时间运行测试完成: 成功率 {result['success_rate']:.1%}")
        return result
    
    def test_concurrent_queries(self, num_concurrent: int = 10, num_rounds: int = 5) -> Dict:
        """并发查询测试"""
        print(f"\n开始并发查询测试，并发数: {num_concurrent}, 轮次: {num_rounds}...")
        
        import threading
        
        # 加载测试问题
        qa_pairs_path = self.data_dir / "synthetic" / "qa_pairs.json"
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        questions = [qa["question"] for qa in qa_data["qa_pairs"]]
        
        results = {"success": 0, "fail": 0, "errors": [], "response_times": []}
        lock = threading.Lock()
        
        def query_worker(query):
            try:
                start = time.time()
                result = self.rag.query(query, top_k=3, method="hybrid")
                elapsed = time.time() - start
                
                with lock:
                    results["success"] += 1
                    results["response_times"].append(elapsed)
            except Exception as e:
                with lock:
                    results["fail"] += 1
                    results["errors"].append(str(e))
        
        total_start = time.time()
        for round_num in range(num_rounds):
            print(f"  轮次 {round_num + 1}/{num_rounds}")
            threads = []
            for i in range(num_concurrent):
                query = questions[i % len(questions)]
                thread = threading.Thread(target=query_worker, args=(query,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        total_time = time.time() - total_start
        total_queries = num_concurrent * num_rounds
        
        result = {
            "total_queries": total_queries,
            "success_count": results["success"],
            "fail_count": results["fail"],
            "success_rate": results["success"] / total_queries,
            "total_time": total_time,
            "avg_response_time": sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0,
            "errors": results["errors"][:5]
        }
        
        print(f"并发查询测试完成: 成功率 {result['success_rate']:.1%}")
        return result
    
    def test_data_coverage(self) -> Dict:
        """数据覆盖度分析"""
        print("\n开始数据覆盖度分析...")
        
        # 加载所有数据
        fault_cases_path = self.data_dir / "synthetic" / "fault_cases.json"
        with open(fault_cases_path, 'r', encoding='utf-8') as f:
            fault_data = json.load(f)
        
        technical_docs_path = self.data_dir / "synthetic" / "technical_docs.json"
        with open(technical_docs_path, 'r', encoding='utf-8') as f:
            tech_data = json.load(f)
        
        ops_manual_path = self.data_dir / "synthetic" / "operations_manual.json"
        with open(ops_manual_path, 'r', encoding='utf-8') as f:
            ops_data = json.load(f)
        
        # 分析部件覆盖
        components = set()
        for case in fault_data["fault_cases"]:
            components.add(case["component"])
        
        # 分析故障类型覆盖
        fault_types = set()
        for case in fault_data["fault_cases"]:
            fault_types.add(case["fault_type"])
        
        # 分析文档类别覆盖
        doc_categories = set()
        for doc in tech_data["technical_docs"]:
            doc_categories.add(doc["category"])
        for doc in ops_data["operations_manual"]:
            doc_categories.add(doc["category"])
        
        result = {
            "fault_cases_count": len(fault_data["fault_cases"]),
            "technical_docs_count": len(tech_data["technical_docs"]),
            "operations_manual_count": len(ops_data["operations_manual"]),
            "total_documents": len(fault_data["fault_cases"]) + len(tech_data["technical_docs"]) + len(ops_data["operations_manual"]),
            "components_covered": list(components),
            "components_count": len(components),
            "fault_types_count": len(fault_types),
            "doc_categories_count": len(doc_categories)
        }
        
        print(f"数据覆盖度分析完成: {result['total_documents']}个文档，覆盖{result['components_count']}个部件")
        return result
    
    def test_question_type_coverage(self) -> Dict:
        """问题类型覆盖测试"""
        print("\n开始问题类型覆盖测试...")
        
        # 定义不同类型的问题
        test_questions = {
            "故障诊断": [
                "风电机的齿轮箱温度过高应该怎么处理？",
                "叶片出现裂纹应该怎么处理？",
                "发电机绝缘击穿怎么办？"
            ],
            "维护操作": [
                "风电机组的正常维护周期是多久？",
                "如何进行齿轮箱维护？",
                "变桨系统如何检查？"
            ],
            "系统原理": [
                "风电机组的偏航系统有什么作用？",
                "变桨系统如何工作？",
                "液压系统的作用是什么？"
            ],
            "性能优化": [
                "如何提高风电机的发电效率？",
                "如何降低故障率？"
            ]
        }
        
        results = {}
        for qtype, questions in test_questions.items():
            type_results = []
            for q in questions:
                try:
                    start = time.time()
                    result = self.rag.query(q, top_k=3, method="hybrid")
                    elapsed = time.time() - start
                    type_results.append({
                        "success": True,
                        "response_time": elapsed,
                        "has_answer": len(result["answer"]) > 10
                    })
                except Exception as e:
                    type_results.append({
                        "success": False,
                        "error": str(e)
                    })
            
            results[qtype] = {
                "total": len(type_results),
                "success": sum(r["success"] for r in type_results),
                "success_rate": sum(r["success"] for r in type_results) / len(type_results),
                "avg_response_time": sum(r.get("response_time", 0) for r in type_results) / len(type_results)
            }
        
        print("问题类型覆盖测试完成")
        return results
    
    def test_robustness(self) -> Dict:
        """鲁棒性测试：异常输入处理"""
        print("\n开始鲁棒性测试...")
        
        test_cases = [
            ("空输入", ""),
            ("超长输入", "风电" * 500),
            ("特殊字符", "风电@#$%^&*()"),
            ("乱码", "㐀㐁㐂㐃"),
            ("英文输入", "How to maintain wind turbine?"),
            ("混合输入", "风力发电机wind turbine如何维护？")
        ]
        
        results = {}
        for name, query in test_cases:
            try:
                start = time.time()
                result = self.rag.query(query, top_k=3, method="hybrid")
                elapsed = time.time() - start
                results[name] = {
                    "success": True,
                    "response_time": elapsed,
                    "has_answer": len(result["answer"]) > 0
                }
            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e)
                }
        
        success_count = sum(r["success"] for r in results.values())
        print(f"鲁棒性测试完成: 成功率 {success_count/len(results):.1%}")
        return results
    
    def run_all_validation(self):
        """运行所有验证测试"""
        print("="*60)
        print("开始系统稳定性与可行性验证")
        print("="*60)
        
        self.init_rag()
        
        # 运行各项测试
        self.results["long_running"] = self.test_long_running(50)
        self.results["concurrent"] = self.test_concurrent_queries(5, 3)
        self.results["data_coverage"] = self.test_data_coverage()
        self.results["question_coverage"] = self.test_question_type_coverage()
        self.results["robustness"] = self.test_robustness()
        
        # 保存结果
        output_path = self.data_dir / "validation_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        self.print_summary()
        print(f"\n验证报告已保存到 {output_path}")
    
    def print_summary(self):
        """打印验证摘要"""
        print("\n" + "="*60)
        print("系统验证摘要")
        print("="*60)
        
        print(f"\n【长时间运行测试】")
        lr = self.results["long_running"]
        print(f"  成功率: {lr['success_rate']:.1%}")
        print(f"  平均响应时间: {lr['avg_response_time']:.2f}秒")
        print(f"  最大响应时间: {lr['max_response_time']:.2f}秒")
        
        print(f"\n【并发查询测试】")
        cc = self.results["concurrent"]
        print(f"  成功率: {cc['success_rate']:.1%}")
        print(f"  平均响应时间: {cc['avg_response_time']:.2f}秒")
        
        print(f"\n【数据覆盖度】")
        dc = self.results["data_coverage"]
        print(f"  总文档数: {dc['total_documents']}")
        print(f"  覆盖部件数: {dc['components_count']}")
        print(f"  故障类型数: {dc['fault_types_count']}")
        print(f"  文档类别数: {dc['doc_categories_count']}")
        
        print(f"\n【问题类型覆盖】")
        qc = self.results["question_coverage"]
        for qtype, data in qc.items():
            print(f"  {qtype}: 成功率 {data['success_rate']:.1%}")
        
        print(f"\n【鲁棒性测试】")
        rb = self.results["robustness"]
        success_count = sum(r["success"] for r in rb.values())
        print(f"  成功率: {success_count/len(rb):.1%}")
        
        print("="*60)
        
        # 总体评估
        print("\n【总体评估】")
        if lr['success_rate'] > 0.95 and cc['success_rate'] > 0.9:
            print("  ✓ 系统稳定性：良好")
        else:
            print("  ✗ 系统稳定性：需改进")
        
        if dc['components_count'] >= 5 and dc['total_documents'] >= 50:
            print("  ✓ 数据支撑度：良好")
        else:
            print("  ✗ 数据支撑度：不足")
        
        if success_count/len(rb) > 0.8:
            print("  ✓ 系统鲁棒性：良好")
        else:
            print("  ✗ 系统鲁棒性：需改进")


if __name__ == "__main__":
    validator = SystemValidator()
    validator.run_all_validation()
