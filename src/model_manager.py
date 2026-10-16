"""
模型管理器
实现模型池和自动切换机制，带线程安全
"""

import os
import json
import threading
import time
from pathlib import Path
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
    # 思维链模型：请求里带 thinking.type=disabled 让它直接出结论，
    # 否则推理写在 content 里，max_tokens 用完就没有正文（glm-z1-flash、glm-4.7-flash 实测如此）
    thinking_toggle: bool = False


# 名字前命中这些模式的模型默认关掉思维链
THINKING_TOGGLE_PREFIXES = ("glm-z1", "glm-4.5", "glm-4.7")


# OpenAI 兼容服务的默认入口；新增供应商只需在这里登记
PROVIDER_BASE_URLS: Dict[str, str] = {
    "modelscope": "https://api-inference.modelscope.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}

# 智谱免费档模型池（实测能正常出正文的都放进去轮流用，不逮着一个模型死用）
# 排序按关掉思维链后的实测延迟：250414 约 2.4s / 4.7 约 4.0s / 4v 约 5.9s / flash 约 6.1s / 4.5 约 27s
# 需要关推理才能用的是 4.5 与 4.7 这两个链式模型
ZHIPU_FREE_MODELS = (
    "glm-4-flash-250414:1",
    "glm-4.7-flash:2",
    "glm-4v-flash:3",
    "glm-4-flash:4",
    "glm-4.5-flash:5",
)

# 不放进池子的：glm-z1-flash（实测 thinking.type=disabled 也关不住推理，17s 后被 token 上限截断，
#               拿不到正文）、glm-4.6-flash（403/1220 无权限）、
#               glm-5.x 与 glm-4-air/airx/long/plus（收费档，429/1113）


def _parse_model_entries(model_env: str) -> List[tuple]:
    """解析 name:priority 形式的模型列表（逗号分隔，忽略空白项）"""
    entries = []
    for entry in model_env.split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ':' in entry:
            name, priority_str = entry.rsplit(':', 1)
            try:
                priority = int(priority_str)
            except ValueError:
                priority = 0
        else:
            name, priority = entry, 0
        if name:
            entries.append((name, priority))
    return entries


class ModelManager:
    """线程安全的模型管理器

    rotation_file：把“轮到哪个模型”落盘的可选路径。CLI 是一次一个新进程，
    只记在内存里的话每次都会从首选模型开始，等于还是在单着用一个模型。
    """
    
    def __init__(self, rotation_file: str = None):
        self.models: List[ModelInfo] = []
        self.current_index = 0
        self.model_status: Dict[str, Dict] = {}  # 记录每个模型的状态
        self.request_history: deque = deque(maxlen=100)  # 请求历史
        self._lock = threading.Lock()  # 线程锁保护共享状态
        self.rotation_file = Path(rotation_file) if rotation_file else None
        self._pending_cursor = self._read_cursor()
    
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
    
    def _read_cursor(self) -> Optional[str]:
        """读回上次轮到的模型名（文件不存在或损坏时当新开始）"""
        if not self.rotation_file or not self.rotation_file.exists():
            return None
        try:
            with open(self.rotation_file, "r", encoding="utf-8") as f:
                return json.load(f).get("model") or None
        except (OSError, ValueError):
            return None

    def _save_cursor_locked(self):
        """写入当前轮到的模型名（调用方已持锁）"""
        if not self.rotation_file or not self.models:
            return
        payload = {"model": self.models[self.current_index].name,
                   "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            self.rotation_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.rotation_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"[warn] 轮转状态无法写入 {self.rotation_file}：{exc}")

    def restore_cursor(self) -> bool:
        """模型全部加完后恢复上次的轮转位；返回是否命中"""
        if not self._pending_cursor:
            return False
        for index, model in enumerate(self.models):
            if model.name == self._pending_cursor:
                self.current_index = index
                return True
        return False

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
    
    def _advance_round_robin_locked(self):
        """把当前模型指向下一个仍可用的模型（轮流使用，调用方已持锁）"""
        if len(self.models) < 2:
            return
        for offset in range(1, len(self.models) + 1):
            index = (self.current_index + offset) % len(self.models)
            if self._is_model_available(self.models[index]):
                self.current_index = index
                return

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
                status = self.model_status[model_name]
                status["successful_requests"] += 1
                status["total_requests"] += 1
                status["fail_count"] = 0
                status["available"] = True
            # 轮流用：一次成功后把当前位往后推一位，否则只要首选模型没挂
            # 所有流量就全落在它身上，容易被单模型限流掐住
            self._advance_round_robin_locked()
            self._save_cursor_locked()
    
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
                self._save_cursor_locked()
    
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


def _needs_thinking_toggle(name: str) -> bool:
    """按名字前缀判断该模型是否需要显式关掉思维链"""
    lowered = name.lower()
    return any(lowered.startswith(prefix) for prefix in THINKING_TOGGLE_PREFIXES)


def create_default_model_manager(rotation_file: str = None) -> ModelManager:
    """创建默认模型管理器（从环境变量动态读取供应商、地址与模型列表）

    优先读通用键 LLM_*，没配时兼容旧的 MODELSCOPE_*：
      LLM_PROVIDER   modelscope | zhipu（决定默认 base_url 与免费档模型池）
      LLM_BASE_URL   显式指定 OpenAI 兼容服务地址，优先于 provider 默认值
      LLM_API_KEY    接口密钥，缺省回退 MODELSCOPE_API_KEY
      LLM_MODELS     name:priority 逗号分隔，缺省回退 MODELSCOPE_MODELS，
                     zhipu 供应商下默认只用免费档模型
      LLM_ROTATION_FILE 轮转状态落盘路径（CLI 单进程一次一个新起时也能接着上次的模型用）
    """
    manager = ModelManager(rotation_file=rotation_file or os.getenv('LLM_ROTATION_FILE', ''))
    provider = os.getenv('LLM_PROVIDER', '').strip().lower()

    api_key = os.getenv('LLM_API_KEY', '').strip() or os.getenv('MODELSCOPE_API_KEY', '').strip()
    base_url = (os.getenv('LLM_BASE_URL', '').strip()
                or PROVIDER_BASE_URLS.get(provider) or PROVIDER_BASE_URLS["modelscope"])

    model_env = (os.getenv('LLM_MODELS', '').strip()
                 or os.getenv('MODELSCOPE_MODELS', '').strip())
    if not model_env:
        model_env = (','.join(ZHIPU_FREE_MODELS) if provider == 'zhipu'
                     else 'Qwen/Qwen2.5-7B-Instruct:1')

    models_to_add = [
        ModelInfo(name=name, api_key=api_key, base_url=base_url, priority=priority,
                  thinking_toggle=_needs_thinking_toggle(name))
        for name, priority in _parse_model_entries(model_env)
    ]

    for model in models_to_add:
        manager.add_model(model)

    manager.restore_cursor()
    return manager


if __name__ == "__main__":
    manager = create_default_model_manager()
    print("模型池初始化完成")
    for model in manager.models:
        print(f"  - {model.name} (优先级: {model.priority})  {model.base_url}")
    
    # 测试获取模型
    current = manager.get_current_model()
    print(f"\n当前模型: {current.name if current else 'None'}")
