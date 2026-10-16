# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [Unreleased]

### 新增
- 添加项目全景观览 HTML 页面
- 新增零依赖纯 Python 检索内核 `src/pure_rag/`（仅用标准库）：
  - 手写 Okapi BM25 与 TF-IDF 稀疏向量索引（倒排表 + 余弦相似度）
  - 领域词典分词 + 正向最大匹配 + 无监督新词发现（互信息 + 左右熵）
  - 自适应 α 融合 + RRF 多路召回合并 + MMR 多样性重排
  - 知识图谱检索增强（TF-IDF 语义匹配 + 同义词表）
  - 抽取式答案生成，可选 LLM（标准库 urllib 直连 OpenAI 兼容接口）
  - 命令行入口 `src/rag_cli.py`（build / ask / chat / eval / stats / serve）
  - 零依赖 Web 界面 `src/pure_rag/server.py`（http.server + 单页应用）
  - 评测模块 `src/pure_rag/evaluator.py`（Recall@k / MRR / nDCG@k）
  - 34 项单元测试 `tests/test_pure_rag.py`
- 检索内核扩展：
  - 自有文档导入：自动纳入 `data/user_docs/` 的 txt / md / json / csv / docx（zipfile + 正则解析，无需 python-docx）
  - 检索可解释性：`explain` 子命令输出每个查询词对 BM25 / TF-IDF 的分数贡献
  - 超参数调优：`tuner.py` 网格搜索分块、融合权重、MMR 参数，结果写入 `tuning.json` 并自动生效
  - 语义鲁棒性评测：把 QA 问题改写为同义词表达/关键词查询后再评测（`eval --robust`）
  - 同义词双向扩展：查询侧别名→标准词，索引侧标准词→别名，配合标点过滤与虚词停用
  - 单元测试扩充到 44 项
- 检索内核再扩展：
  - 增量追加：`add` 子命令支持文件/目录，只对新增文档分块分词（词序列随索引持久化复用）
  - 命中高亮：`find_spans` 字符级定位 + `highlight.py` 三态渲染（终端 ANSI / Web `<mark>` / Markdown 加粗）
  - 会话记录：`chat /save` 与 `--log` 导出带来源标注的 Markdown 记录
  - Markdown 文档解析时清理标题井号与强调符号
  - 单元测试扩充到 59 项
- 可靠性与持续集成：
  - 置信度与拒答：IDF 加权查询词覆盖率 + 「至少命中一个多字词」双判据，
    覆盖率不足时明确回「知识库没有相关资料」，阈值可由 `--min-coverage` 调整
  - 拒答评测集 `data/synthetic/negative_queries.json`：域外 10 条 + 域内缺资料 6 条，
    按类型分组统计拒答率，暴露当前机制的边界
  - 阈值标定：`eval --negative` 输出「阈值 → 正例回答率 / 正例召回 / 反例拒答率」权衡表
  - 质量门禁：`eval --min-recall` 指标不达标时以非零退出码结束
  - 新增 `.cnb.yml`：PR 门禁与 main 回归自动跑单元测试 + 评测（Recall 门禁 0.9）
  - 单元测试扩充到 70 项

### 变更
- README 补充零依赖内核章节、实测指标与项目结构

## [1.0.0] - 2026-01

### 新增
- 面向风电领域的智能故障诊断问答系统
- 自适应混合检索（向量检索 + BM25 + 知识图谱）
- 知识图谱增强（基于 embedding 语义匹配）
- 模型池管理（多 LLM 自动切换、失败冷却、优雅降级）
- 中文深度优化（jieba 分词、中文向量模型、领域术语识别）
- 高鲁棒性设计（异常输入、并发查询、LLM 不可用等场景）
