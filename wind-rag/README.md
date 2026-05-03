<div align="center">

# 🌬️ WindRAG

**风电领域智能诊断问答系统 — 基于多策略混合检索与知识图谱增强的RAG**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented-orange.svg)]()
[![Wind Power](https://img.shields.io/badge/Domain-Wind%20Power-9cf.svg)]()

</div>

---

## 📖 项目简介

WindRAG 是一个面向风力发电领域的智能故障诊断问答系统，采用**检索增强生成（RAG）**架构，融合向量语义检索、BM25关键词检索和知识图谱结构检索三种策略，实现高质量的风电领域知识问答。

### ✨ 核心特性

- 🔀 **自适应混合检索** — 根据查询特征动态调整向量检索与BM25检索的融合权重
- 🕸️ **知识图谱增强** — 基于embedding语义匹配的知识图谱查询，解决同义词问题
- 🔄 **模型池管理** — 线程安全的多LLM自动切换，支持失败冷却与优雅降级
- 🇨🇳 **中文深度优化** — jieba分词、中文向量模型、领域术语识别
- 🛡️ **高鲁棒性** — 异常输入、并发查询、LLM不可用等场景全部测试通过

---

## 🏗️ 系统架构

<img src="docs/images/architecture.png" width="800" alt="WindRAG Architecture">

系统采用四层架构设计：

| 层级 | 功能 | 关键技术 |
|:---:|------|---------|
| 数据层 | 多源异构数据整合 | 故障案例/技术文档/运维手册/问答对 |
| 索引层 | 三种互补索引 | 向量索引 · BM25索引 · 知识图谱 |
| 检索层 | 多策略融合检索 | 自适应α融合 · 语义重排序 · 图谱增强 |
| 生成层 | 鲁棒答案生成 | 模型池管理 · LLM生成 · 模板降级 |

---

## 🔀 检索流程

<img src="docs/images/retrieval_flow.png" width="700" alt="Retrieval Flow">

### 自适应融合公式

$$S_{hybrid}(q, d) = \alpha \cdot \hat{S}_{vec}(q, d) + (1-\alpha) \cdot \hat{S}_{BM25}(q, d)$$

其中 $\alpha$ 不再是固定值，而是根据查询特征**动态预测**：

| 查询特征 | 推荐α值 | 含义 |
|---------|:---:|------|
| 含标准编号 (ISO/GB) | 0.3 | BM25精确匹配更重要 |
| 含数值规格 (mm/℃) | 0.4 | 精确数值检索更关键 |
| 术语密集 (≥3个) | 0.5 | BM25召回效果好 |
| 维护操作类 | 0.6 | 语义+关键词均衡 |
| 故障诊断类 | 0.7 | 语义理解更重要 |
| 性能优化类 | 0.75 | 语义理解为主 |
| 系统原理类 | 0.8 | 纯语义理解 |

---

## 🕸️ 知识图谱

<img src="docs/images/knowledge_graph.png" width="500" alt="Knowledge Graph">

- **162个节点**：12个部件节点 + 30+故障节点 + 维护操作节点
- **160条边**：`HAS_FAULT`（部件→故障）+ `REQUIRES`（部件→维护操作）
- **语义查询**：基于embedding余弦相似度匹配，阈值τ=0.5

---

## 📊 实验结果

<img src="docs/images/experiment_results.png" width="800" alt="Experiment Results">

### 系统稳定性

| 指标 | 长时间运行(50次) | 并发查询(15次) |
|------|:---:|:---:|
| 成功率 | **100%** | **100%** |
| 平均响应时间 | 0.65s | 2.59s |

### 鲁棒性测试

| 异常输入类型 | 成功 | 响应时间 |
|------------|:---:|:---:|
| 空输入 | ✓ | 0.66s |
| 超长输入 | ✓ | 0.82s |
| 特殊字符 | ✓ | 0.62s |
| 乱码 | ✓ | 0.62s |
| 英文输入 | ✓ | 0.61s |
| 混合输入 | ✓ | 0.64s |

### 数据覆盖度

<img src="docs/images/data_coverage.png" width="400" alt="Data Coverage">

| 数据类型 | 数量 |
|---------|:---:|
| 故障案例 | 30 |
| 技术文档 | 10 |
| 运维手册 | 10 |
| 问答对 | 10 |
| 覆盖部件 | 12 |
| 故障类型 | 30 |
| 知识图谱节点 | 162 |

### 自适应α分布

<img src="docs/images/alpha_distribution.png" width="700" alt="Alpha Distribution">

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 2GB+ 内存（向量模型加载）

### 安装

```bash
git clone https://github.com/your-username/WindRAG.git
cd WindRAG
pip install -r requirements.txt
```

### 配置

创建 `config.env` 文件：

```env
# ModelScope API（可选，不配置则使用模板生成）
MODELSCOPE_API_KEY=your_api_key_here
MODELSCOPE_MODELS=Qwen/Qwen2.5-7B-Instruct:1,Qwen/Qwen2.5-72B-Instruct:2

# 禁用LLM（仅使用模板生成）
DISABLE_LLM=false
```

### 构建索引

```bash
# 1. 数据预处理
python src/data_processing.py

# 2. 构建向量+BM25索引
python src/build_index.py

# 3. 构建知识图谱
python src/knowledge_graph.py

# 4. 训练自适应融合模型（可选）
python src/generate_training_data.py
python src/adaptive_fusion.py
```

### 启动Web界面

```bash
python src/gradio_app.py
# 访问 http://localhost:7865
```

### 命令行使用

```python
from src.run_rag import WindTurbineRAG

rag = WindTurbineRAG()
result = rag.query("齿轮箱温度过高怎么处理？")

print(result["answer"])
```

---

## 📁 项目结构

```
WindRAG/
├── src/                          # 核心源码
│   ├── run_rag.py                # RAG系统主程序
│   ├── data_processing.py        # 数据预处理
│   ├── build_index.py            # 索引构建（向量+BM25）
│   ├── knowledge_graph.py        # 知识图谱构建与查询
│   ├── model_manager.py          # 线程安全模型池管理
│   ├── adaptive_fusion.py        # 自适应检索融合
│   ├── query_features.py         # 查询特征提取
│   ├── evaluation.py             # 评估模块
│   ├── system_validation.py      # 系统验证
│   ├── ablation_experiment.py    # 消融实验
│   ├── gradio_app.py             # Web界面
│   └── query_models.py           # 模型可用性查询
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据
│   ├── synthetic/                # 合成数据
│   └── processed/                # 处理后数据+索引
├── docs/                         # 文档与图片
│   └── images/                   # 科研绘图
├── config.env                    # 环境配置
├── requirements.txt              # 依赖
└── README.md                     # 本文件
```

---

## 🔧 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 向量模型 | text2vec-base-chinese | 768维中文文本向量 |
| 中文分词 | jieba | 搜索模式分词 |
| BM25 | rank-bm25 | Okapi BM25 |
| 向量计算 | numpy | L2归一化余弦相似度 |
| LLM API | OpenAI兼容接口 | ModelScope API |
| Web界面 | Gradio | 交互式问答 |
| 知识图谱 | 自研轻量级实现 | 无需图数据库 |
| 模型管理 | 自研ModelManager | 线程安全+自动切换 |
| 自适应融合 | MLP+规则先验 | 查询感知权重预测 |

---

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<div align="center">

**WindRAG** — 让风电运维更智能 🌬️

</div>
