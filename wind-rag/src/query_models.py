"""
查询ModelScope支持的模型列表
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
config_path = Path(__file__).parent.parent / 'config.env'
load_dotenv(config_path)

API_KEY = os.getenv('MODELSCOPE_API_KEY')
BASE_URL = "https://api-inference.modelscope.cn/v1"

def test_model(model_name):
    """测试特定模型是否可用"""
    from openai import OpenAI
    
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': '你好'
            }],
            max_tokens=10
        )
        print(f"✓ {model_name} 可用")
        return True
    except Exception as e:
        error_msg = str(e)
        if "no provider supported" in error_msg:
            print(f"✗ {model_name} - 无提供者")
        else:
            print(f"✗ {model_name} - {error_msg[:50]}")
        return False

if __name__ == "__main__":
    print("开始测试ModelScope API支持的模型...\n")
    
    # 尝试不同的模型名称格式
    test_models = [
        # 尝试简短格式
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        # 尝试带版本号
        "qwen-turbo-latest",
        "qwen-plus-latest",
        # 尝试其他可能的名字
        "Qwen/Qwen2.5-7B-Instruct",
        "qwen/qwen2.5-7b-instruct",
        "Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct",
        # 尝试API推理服务可能使用的格式
        "Qwen/Qwen2-7B-Instruct",
        "qwen/qwen-7b-instruct",
    ]
    
    available_models = []
    for model in test_models:
        if test_model(model):
            available_models.append(model)
    
    print(f"\n可用模型列表：")
    for model in available_models:
        print(f"  - {model}")
    
    if not available_models:
        print("\n未找到可用模型。建议访问 ModelScope API 文档查看支持的模型列表：")
        print("https://api-inference.modelscope.cn/docs")
