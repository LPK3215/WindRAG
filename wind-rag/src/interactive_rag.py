"""
交互式RAG系统
支持用户输入问题并获取答案
"""

import sys
sys.path.append('.')

from run_rag import WindTurbineRAG


def main():
    """交互式主函数"""
    print("=" * 80)
    print("风电领域智能问答系统 (RAG)")
    print("=" * 80)
    print()
    
    # 初始化RAG系统
    print("正在初始化系统...")
    rag = WindTurbineRAG()
    print("系统初始化完成！")
    print()
    
    print("使用说明：")
    print("- 输入您的问题，系统会检索相关文档并生成答案")
    print("- 输入 'quit' 或 'exit' 退出系统")
    print("- 输入 'help' 查看帮助")
    print()
    
    # 交互式循环
    while True:
        try:
            # 获取用户输入
            question = input("请输入您的问题: ").strip()
            
            # 退出命令
            if question.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            # 帮助命令
            if question.lower() == 'help':
                print("\n帮助信息：")
                print("- 可以询问风电机的故障诊断、维护流程、技术参数等问题")
                print("- 示例问题：")
                print("  * 风电机的齿轮箱温度过高应该怎么处理？")
                print("  * 如何判断风电机叶片是否需要更换？")
                print("  * 风电机组的偏航系统有什么作用？")
                print("  * 风电机的变桨系统是如何工作的？")
                print()
                continue
            
            # 空输入
            if not question:
                continue
            
            # 查询
            print(f"\n正在查询: {question}")
            result = rag.query(question, top_k=3)
            
            # 显示答案
            print("\n" + "=" * 80)
            print("答案：")
            print("=" * 80)
            print(result['answer'])
            print()
            
            # 显示检索到的文档
            print("=" * 80)
            print("检索到的相关文档：")
            print("=" * 80)
            for i, doc in enumerate(result['retrieved_docs'], 1):
                print(f"\n【文档{i}】评分: {doc['score']:.4f}")
                print(f"来源: {doc['metadata']['source']}")
                if 'component' in doc['metadata']:
                    print(f"部件: {doc['metadata']['component']}")
                if 'category' in doc['metadata']:
                    print(f"类别: {doc['metadata']['category']}")
                print(f"内容: {doc['content'][:300]}...")
                print()
            
            print("=" * 80)
            print()
            
        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入 'quit' 退出")
            print()


if __name__ == "__main__":
    main()
