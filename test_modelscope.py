"""
测试ModelScope API调用 - 使用OpenAI客户端
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv(Path(__file__).parent / 'config.env')

api_key = os.getenv('MODELSCOPE_API_KEY')
print(f"API Key: {api_key[:20]}..." if api_key else "API Key: 未配置")

if not api_key or api_key == 'your_modelscope_api_key_here':
    print("未配置API Key，无法测试")
    exit(1)

try:
    from openai import OpenAI
    
    print("\n使用OpenAI客户端调用ModelScope API...")
    
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1',
        api_key=api_key
    )
    
    # 尝试不同的模型
    test_models = [
        'Qwen/Qwen3-VL-8B-Instruct',  # 用户示例中的模型
        'qwen-turbo',
        'qwen-plus',
        'qwen-max',
    ]
    
    for model in test_models:
        print(f"\n尝试模型: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': '你好，请简单介绍一下风电。'
                }],
                temperature=0.7,
                max_tokens=100
            )
            
            answer = response.choices[0].message.content
            print(f"✓ 成功！")
            print(f"回答: {answer}")
            print(f"\n正确的模型: {model}")
            exit(0)
        except Exception as e:
            print(f"✗ 失败: {str(e)[:100]}")
    
    print("\n所有模型测试失败")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
