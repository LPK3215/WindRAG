"""
RAG系统测试脚本
非交互式测试多个查询
"""

import sys
sys.path.append('.')

from run_rag import WindTurbineRAG


def main():
    """测试主函数"""
    print("=" * 80)
    print("风电领域RAG系统测试")
    print("=" * 80)
    print()
    
    # 初始化RAG系统
    print("正在初始化系统...")
    rag = WindTurbineRAG()
    print("系统初始化完成！")
    print()
    
    # 测试问题列表
    test_questions = [
        "风电机的齿轮箱温度过高应该怎么处理？",
        "如何判断风电机叶片是否需要更换？",
        "风电机组的偏航系统有什么作用？",
        "风电机的变桨系统是如何工作的？",
        "风电机组在极端天气下应该如何运行？"
    ]
    
    # 依次测试每个问题
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 80}")
        print(f"测试 {i}/{len(test_questions)}")
        print(f"问题：{question}")
        print("=" * 80)
        
        # 查询
        result = rag.query(question, top_k=3)
        
        # 显示答案
        print("\n答案：")
        print(result['answer'])
        
        # 显示检索到的文档摘要
        print(f"\n检索到 {len(result['retrieved_docs'])} 个相关文档")
        for j, doc in enumerate(result['retrieved_docs'][:2], 1):
            print(f"  文档{j}: {doc['metadata']['source']} (评分: {doc['score']:.4f})")
        
        print()
    
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
