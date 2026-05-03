# 代码审查报告 - wind-rag 项目

## 严重问题（必须修复）

### 1. 知识图谱查询逻辑完全错误 [P0]
**文件**: `src/run_rag.py:265-274`
**问题**: `_query_knowledge_graph` 方法使用简单的字符串包含匹配 (`if node_id in query`)，无法处理同义词、语义变化。例如查询"变速箱"无法匹配"齿轮箱"节点。
**影响**: 知识图谱增强功能形同虚设，完全依赖字面匹配。

### 2. BM25中文分词完全错误 [P0]
**文件**: `src/run_rag.py:132`, `src/build_index.py:64`
**问题**: 中文分词使用 `list(query)` 按字符分割，破坏了词语语义。BM25基于词频统计，按字分词会导致所有文档的TF-IDF分布严重失真。
**影响**: BM25检索结果完全不可信。

### 3. 模型池硬编码不可用模型 [P0]
**文件**: `src/model_manager.py:153-175`
**问题**: 三个模型名称（Qwen2.5-7B, Qwen2.5-72B, Qwen2-72B）在ModelScope API上全部返回 "no provider supported"。
**影响**: 模型切换功能完全无效，所有调用都会失败降级。

### 4. 系统验证并发测试存在线程安全问题 [P1]
**文件**: `src/system_validation.py:93-118`
**问题**: 多线程共享同一个RAG实例，模型管理器的 `current_index` 和 `model_status` 会被并发修改，没有线程锁保护。
**影响**: 并发测试结果不可靠，可能产生竞态条件。

### 5. 知识图谱构建硬编码条件分支 [P1]
**文件**: `src/knowledge_graph.py:230-246`
**问题**: 使用一长串 `if "齿轮箱" in title` 硬编码匹配运维手册与部件的关系，无法扩展，新增部件需要修改代码。
**影响**: 可维护性极差，新增数据类型必须改代码。

## 中等问题

### 6. 向量检索未归一化 [P2]
**文件**: `src/run_rag.py:121`
**问题**: 直接使用 `np.dot` 计算点积，没有L2归一化。长文档的embedding模长天然更大，会获得不公平的高分。
**影响**: 检索结果偏向长文档，短文档即使高度相关也可能被压制。

### 7. Gradio界面信息过时 [P2]
**文件**: `src/gradio_app.py:20-21`
**问题**: 硬编码 "Qwen/Qwen3-VL-8B-Instruct" 与实际配置不符，用户看到的模型信息和实际运行的不一致。

### 8. 评价指标公式不严谨 [P2]
**文件**: `src/evaluation.py:50`
**问题**: `keyword_match_rate = keyword_matches / min(len(retrieved_docs), total_keywords)` 这个公式在边界条件下会产生误导性结果。

## 次要问题

### 9. 冗余导入 [P3]
**文件**: `src/run_rag.py:22-26`
**问题**: 导入未使用的 ModelScope SDK 模块（GenerationConfig, AutoModelForCausalLM, AutoTokenizer），增加依赖负担。

### 10. 数据块分割逻辑不够健壮 [P3]
**文件**: `src/data_processing.py:131-171`
**问题**: 如果一个段落长度超过chunk_size，该段落会被直接放入一个chunk，可能导致chunk远超设定大小。且chunk_id未在source级别唯一。

---

## 修复计划
1. 重写 `_query_knowledge_graph` - 使用embedding语义匹配而非字符串包含
2. 集成中文分词（jieba）替代字符级分词
3. 移除硬编码模型，改为从配置动态读取
4. 为模型管理器添加线程锁
5. 重构知识图谱构建逻辑，使用配置驱动的映射表
6. 向量检索添加L2归一化
7. 修复Gradio界面动态读取实际模型信息
8. 修正评价指标公式
9. 清理冗余导入
10. 优化数据块分割边界处理
