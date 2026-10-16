# diagram-input.md — WindRAG 出图前输入蓝图

> 用途：本文件是后续所有出图（架构 / 流程 / 时序 / 数据流）的唯一输入源。
> 只整理、只提炼，不含任何图形、JSON 蓝图或 HTML。
> 真源：`src/pure_rag/` 代码为主，`README.md` 为辅；`paper/main.tex` 的拔高叙事按干扰项处理。
> 出图对象锁定：纯标准库检索主线（`src/pure_rag/`），不含遗留重依赖研究版。

## 1. 组件清单（主节点 12 个）

- `cli` | CLI 入口 | `rag_cli.py` 子命令分发 build/ask/chat/add/explain/eval/tune/ablation/stats/serve（类型：前端）
- `web_server` | Web 服务 | 标准库 http.server，路由 /api/query、/api/session/reset、/api/stats，含 ConversationStore 会话上下文（类型：后端）
- `web_ui` | Web 前端页 | `src/pure_rag/web/` 的 index.html/style.css/app.js 对话式单页（类型：前端）
- `engine` | RAGEngine 编排核心 | `engine.py`，串联检索与生成，内置置信度与拒答判定（类型：后端）
- `tokenizer` | 分词与词处理 | `text_utils.py`，Tokenizer、cut_for_index、discover_terms、find_spans、同义词扩展（类型：后端）
- `bm25` | BM25 关键词检索 | `bm25.py`，BM25Index.score（类型：后端）
- `tfidf` | TF-IDF 稀疏语义检索 | `tfidf.py`，TfidfIndex.score / similarity / idf_of（类型：后端）
- `fusion` | 融合与重排 | `fusion.py` + `query_analysis.py`，weighted_fusion、adaptive_alpha、reciprocal_rank_fusion、mmr_select（类型：后端）
- `kg` | 知识图谱增强 | `kg.py`，KnowledgeGraphRetriever.build_context / validate_combination，组合校验 HAS_FAULT（类型：后端）
- `generator` | 答案生成 | `generator.py`，AnswerGenerator.generate，产出 answer / mode / citations / claims / card（类型：后端）
- `llm_client` | LLM 客户端 | `generator.py` 内 LLMClient（urllib 直连 OpenAI 兼容接口）+ `model_manager` 模型池（类型：外部）
- `store` | 语料与索引存储 | `data/pure_index/`（documents.json/tokens.json/bm25.json/tfidf.json/meta.json）、`data/knowledge_graph.json`、`data/synthetic/`、`data/user_docs/`；经 loader.py / chunker.py 进入 build 路径（类型：存储）

## 2. 关系清单

- `web_ui → web_server`：POST /api/query，入参 {question, top_k, method, alpha, session_id}（实线主路径）
- `web_server → engine`：调用 engine.query(question, top_k, method, alpha, history)（实线主路径）
- `web_server → web_server`：ConversationStore 按 session_id 读写最近 8 轮，供指代消解（虚线反馈）
- `cli → engine`：按子命令调用 engine.build / query / explain / stats（实线主路径）
- `cli → web_server`：serve 子命令导入并启动 serve(engine)（实线）
- `engine → tokenizer`：cut_for_index(question) 得到查询词（实线主路径）
- `engine → bm25`：bm25.score(tokens) 关键词路召回（实线主路径）
- `engine → tfidf`：tfidf.score(tokens) 语义路召回；mmr_select 用 tfidf.similarity（实线主路径）
- `engine → fusion`：adaptive_alpha 预测 α，weighted_fusion(vector_scores, bm25_scores, alpha)，可选 expand_queries + reciprocal_rank_fusion + mmr_select（实线主路径）
- `engine → kg`：build_context(question) 生成 kg_context；validate_combination(question) 供拒答判定（实线；图谱不参与打分，仅上下文与组合校验）
- `engine → generator`：generate(resolved, documents, kg_context, tfidf.idf_of)（实线主路径）
- `generator → llm_client`：chat(prompt)（虚线可选增强，失败自动回退抽取式并把原因写入 llm_error）
- `llm_client → store`：模型池轮转位读写 `data/pure_index/llm_rotation.json`（虚线）
- `engine → store`：load() 读索引 / save() 落盘 / build() 经 loader+chunker 重建 bm25.json、tfidf.json、meta.json（实线双向）
- `tokenizer → store`：build 期 discover_terms 新词发现结果写入 meta.json（虚线反馈）

## 3. 分层分组（供 boundaries 用）

- 入口层：`cli`、`web_server`、`web_ui`
- 编排核心：`engine`（置信度拒答判定内联于此，不单列节点）
- 检索层：`tokenizer`、`bm25`、`tfidf`、`fusion`、`kg`
- 生成层：`generator`、`llm_client`
- 存储层：`store`

## 4. 主流程（编号步骤）

- ① 用户在 CLI 或 Web 页提问，入口层接收 question / top_k / method / alpha / session_id
- ② 入口层调用 engine.query，多轮时先经 _resolve_coreference 补全指代
- ③ **[核心] 检索**：tokenizer 分词 → bm25 与 tfidf 双路召回 → fusion 用 adaptive_alpha + weighted_fusion 融合 →（可选 expand_queries + reciprocal_rank_fusion + mmr_select）→ 得到 top_k 文档
- ④ kg.build_context 产出 kg_context，kg.validate_combination 产出搭配校验
- ⑤ **[核心] 拒答判定**（生成之前）：confidence 的 coverage 与 matched_content，经 should_refuse + 组合校验决定 answer 走拒答还是继续
- ⑥ **[核心] 生成**：generator.generate 出抽取式答案；配了 LLM 且可用时走 llm，失败回退 extractive；产出 citations / claims / card
- ⑦ 结果回流入口层：highlight 渲染命中高亮，Web 侧把本轮问答写入 ConversationStore 供下轮指代消解（虚线反馈到 ②）

## 5. 关键术语（须原样上图，逐字照抄）

- 模块与类：`RAGEngine`、`engine.py`、`server.py`、`cli.py`、`ConversationStore`、`Tokenizer`、`BM25Index`、`TfidfIndex`、`KnowledgeGraphRetriever`、`AnswerGenerator`、`LLMClient`、`model_manager`
- 方法：`cut_for_index`、`discover_terms`、`find_spans`、`adaptive_alpha`、`expand_queries`、`weighted_fusion`、`reciprocal_rank_fusion`、`min_max`、`merge_scores`、`scale_by_max`、`mmr_select`、`build_context`、`validate_combination`、`generate`、`score`、`similarity`、`idf_of`、`_resolve_coreference`、`should_refuse`
- 关系与判据：`HAS_FAULT`、`coverage`、`matched_content`、`min_coverage`
- 接口与字段：`/api/query`、`/api/session/reset`、`/api/stats`、`question`、`top_k`、`method`、`alpha`、`session_id`、`answer`、`mode`、`citations`、`claims`、`card`、`llm_error`、`kg_context`
- 取值/枚举：`hybrid`、`vector`、`bm25`、`llm`、`extractive`、`refused`
- 子命令：`build`、`ask`、`chat`、`add`、`explain`、`eval`、`tune`、`ablation`、`stats`、`serve`

## 6. 明确排除项（不画什么及原因）

- 评测与调参模块 `evaluator.py`、`datasets.py`、`tuner.py`、`ablation.py`、`human_eval.py`：属离线评测分支，不在对话检索主链路上，画进去会喧宾夺主。
- 基建模块 `env.py`（.env 解析）、`stdio.py`（Windows 编码加固）、`highlight.py`（ANSI/HTML/Markdown 渲染）：横切辅助，折进 cli / web_server 的职责描述，不单列节点。
- `loader.py`、`chunker.py`：仅服务 build，已合并进 `store` 节点的"经 loader/chunker 进入 build 路径"，不单独成点（合并内容已在此说明，非笼统"其它"）。
- 遗留重依赖研究版 `run_rag.py`、`build_index.py`、`gradio_app.py`、`adaptive_fusion.py`、`query_features.py`、`evaluation.py`、`system_validation.py`、`ablation_experiment.py`、`generate_images.py`：AGENTS 约定只作对照、不进主线，不画。
- `simulate_corpus.py`、`gen_readme_facts.py`：测试语料生成与文档数字生成脚本，属工具产物，不画。
- README / `paper/main.tex` 中的自适应 α 公式表、消融负面结果表、口语化验证结论、四层架构宣传图、知识图谱 162 节点统计：是文档叙事与实验结论，不是系统结构组件，不画。
- `data/processed/`、`data/adaptive_model/`、`data/raw/`：遗留向量索引与训练产物，与纯标准库主线无关，不画。

## 待确认清单（材料不足以判断，≤5 条，含默认处理）

1. 出图对象：默认画纯标准库主线（`src/pure_rag/`）。若你要的是论文/遗留版那套（含向量模型、MLP 自适应融合、Gradio），需另行取源——默认按主线出图。
2. `kg` 的地位：代码里图谱不参与打分，只做 build_context 上下文与 validate_combination 拒答校验。默认按此弱化画成旁路（虚线/次要），不按论文里的"图谱增强检索"画成主链一环。
3. 单图节点预算：当前主节点为 12 个。若某一类图（尤其时序图）需要更细，可把 `fusion` 内的 weighted_fusion / RRF / MMR 拆成子步骤——默认保持合并，仅在时序图展开。
4. 是否要体现"拒答"分支的成败两路：默认在 ⑤ 标为核心判定并画出"拒答 → 直接返回 refusal"与"通过 → 进生成"两条去向；若只想要单线主流程可省去拒答出边。
5. `llm_client` 作为外部依赖：默认用"外部"类型并虚线（可选增强、失败回退）。若你希望图里淡化大模型，可整体折叠进 `generator`——默认保留，以体现"检索与生成分离、LLM 属可选"这一真实约束。
