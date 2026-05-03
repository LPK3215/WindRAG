"""
模型管理器
实现模型池和自动切换机制，带线程安全
"""

import os
import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    api_key: str
    base_url: str = "https://api-inference.modelscope.cn/v1"
    max_retries: int = 3
    cooldown: int = 60  # 冷却时间（秒）
    priority: int = 0  # 优先级，数字越小优先级越高


class ModelManager:
    """线程安全的模型管理器"""
    
    def __init__(self):
        self.models: List[ModelInfo] = []
        self.current_index = 0
        self.model_status: Dict[str, Dict] = {}  # 记录每个模型的状态
        self.request_history: deque = deque(maxlen=100)  # 请求历史
        self._lock = threading.Lock()  # 线程锁保护共享状态
    
    def add_model(self, model: ModelInfo):
        """添加模型到池"""
        with self._lock:
            self.models.append(model)
            self.model_status[model.name] = {
                "available": True,
                "fail_count": 0,
                "last_failure_time": 0,
                "total_requests": 0,
                "successful_requests": 0
            }
            # 按优先级排序
            self.models.sort(key=lambda m: m.priority)
    
    def get_current_model(self) -> Optional[ModelInfo]:
        """获取当前可用模型（线程安全）"""
        with self._lock:
            if not self.models:
                return None
            
            # 检查当前模型是否可用
            current = self.models[self.current_index]
            if self._is_model_available(current):
                return current
            
            # 如果当前模型不可用，查找下一个可用模型
            for i, model in enumerate(self.models):
                if self._is_model_available(model):
                    self.current_index = i
                    return model
            
            return None
    
    def _is_model_available(self, model: ModelInfo) -> bool:
        """检查模型是否可用（持有锁后调用）"""
        status = self.model_status[model.name]
        current_time = time.time()
        
        # 如果在冷却期，返回False
        if not status["available"]:
            if current_time - status["last_failure_time"] < model.cooldown:
                return False
            else:
                # 冷却期结束，恢复可用
                status["available"] = True
                status["fail_count"] = 0
        
        return True
    
    def report_success(self, model_name: str):
        """报告模型调用成功（线程安全）"""
        with self._lock:
            if model_name in self.model_status:
                self.model_status[model_name]["successful_requests"] += 1
                self.model_status[model_name]["total_requests"] += 1
                self.model_status[model_name]["fail_count"] = 0
                self.model_status[model_name]["available"] = True
    
    def report_failure(self, model_name: str, error: str = ""):
        """报告模型调用失败（线程安全）"""
        with self._lock:
            if model_name in self.model_status:
                status = self.model_status[model_name]
                status["total_requests"] += 1
                status["fail_count"] += 1
                status["last_failure_time"] = time.time()
                
                # 如果失败次数超过阈值，标记为不可用
                if status["fail_count"] >= 3:
                    status["available"] = False
                    print(f"模型 {model_name} 连续失败{status['fail_count']}次，标记为不可用")
                
                # 切换到下一个模型
                self._switch_to_next_model_locked()
    
    def _switch_to_next_model_locked(self):
        """切换到下一个模型（持有锁后调用）"""
        original_index = self.current_index
        for i in range(len(self.models)):
            next_index = (self.current_index + 1) % len(self.models)
            next_model = self.models[next_index]
            if self._is_model_available(next_model):
                self.current_index = next_index
                print(f"切换模型: {self.models[original_index].name} -> {next_model.name}")
                return
        
        print("警告：所有模型都不可用")
    
    def get_model_stats(self) -> Dict:
        """获取模型统计信息（线程安全）"""
        with self._lock:
            stats = {}
            for model in self.models:
                status = self.model_status[model.name]
                total = status["total_requests"]
                stats[model.name] = {
                    "available": status["available"],
                    "fail_count": status["fail_count"],
                    "total_requests": total,
                    "successful_requests": status["successful_requests"],
                    "success_rate": status["successful_requests"] / total if total > 0 else 0
                }
            return stats
    
    def reset_model_status(self, model_name: str = None):
        """重置模型状态（线程安全）"""
        with self._lock:
            reset_state = {
                "available": True,
                "fail_count": 0,
                "last_failure_time": 0,
                "total_requests": 0,
                "successful_requests": 0
            }
            if model_name:
                if model_name in self.model_status:
                    self.model_status[model_name] = reset_state
            else:
                for name in self.model_status:
                    self.model_status[name] = reset_state


def create_default_model_manager() -> ModelManager:
    """创建默认的模型管理器（从环境变量动态读取模型列表）"""
    manager = ModelManager()
    api_key = os.getenv('MODELSCOPE_API_KEY', '')
    
    # 从环境变量读取模型列表，逗号分隔，格式: name:priority
    model_env = os.getenv('MODELSCOPE_MODELS', 'Qwen/Qwen2.5-7B-Instruct:1')
    
    models_to_add = []
    for entry in model_env.split(','):
        entry = entry.strip()
        if ':' in entry:
            name, priority_str = entry.rsplit(':', 1)
            try:
                priority = int(priority_str)
            except ValueError:
                priority = 0
        else:
            name = entry
            priority = 0
        
        if name:
            models_to_add.append(ModelInfo(
                name=name,
                api_key=api_key,
                priority=priority
            ))
    
    for model in models_to_add:
        manager.add_model(model)
    
    return manager


if __name__ == "__main__":
    manager = create_default_model_manager()
    print("模型池初始化完成")
    for model in manager.models:
        print(f"  - {model.name} (优先级: {model.priority})")
    
    # 测试获取模型
    current = manager.get_current_model()
    print(f"\n当前模型: {current.name if current else 'None'}")
