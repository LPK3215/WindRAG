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
- 答案可追溯与结构化输出：
  - 引用定位：`split_sentences_with_offsets` 切句时保留字符区间，每条要点带上
    `doc_id` 与原文区间（CLI 打印「要点定位」，Web 点击即跳转到来源文档并高亮）
  - 服务端渲染文档片段与要点摘录（含转义），前端只切换高亮，不做字符串拼装
  - 故障卡片：检索到故障案例时输出「症状 / 原因 / 诊断 / 解决方案 / 预防措施」结构化卡片，
    Web 渲染为表格，`chat --log` 与 `/save` 导出时一并写入记录
  - 单元测试扩充到 85 项
- 消融实验：
  - 新增 `src/pure_rag/ablation.py` 与 `ablation` 子命令：逐项开关各模块，
    每组配置都在独立目录重建索引后评测（检索 / 改写召回 / 答案覆盖 / 拒答 / 耗时）
  - 固定 α 扫描（0.3~0.8 + 自适应对照），用于回答"自适应比固定值好多少"
  - 分词器与引擎新增 `use_synonyms` 开关，便于做同义词打通的对照实验
  - README 记录完整消融表与结论，含负面结果（查询扩展冗余、MMR 影响有限、
    自适应 α 未跑赢最优固定 α、新词发现有取舍），并标注样本量不足的边界
  - 单元测试扩充到 88 项
- 评测集扩充（把结论从"样本不足"推进到"趋势一致"）：
  - `src/pure_rag/datasets.py`：由语料自动推导的 7 层评测集（QA 原句/同义改写/故障案例/
    文档标题/近邻负例/域外/域内缺资料），共 114 条，确定性构建、可随语料演进
  - 目标定位支持两种方式：文本签名（QA 对、故障案例按 ID）与整篇文档（按 source + title），
    避免长文档只有首块被算命中
  - 新增 `near_miss` 近邻负例：部件 × 故障的无效组合，量化"把别部件方案套过来"的失效模式
  - `evaluate_layers` 分层评测：召回与拒答各自使用正确分母，并按数据来源/问题类型分组
  - `eval --report` 生成可复现的分层报告（Markdown + JSON，含阈值标定与已知边界说明）
  - 消融实验与超参调优都迁移到新评测集上，README 表格与结论按新数据重写
  - 单元测试扩充到 98 项
- 拒答能力补强（近邻负例拒答率 5.56% → 100%）：
  - 知识图谱组合校验：`kg.validate_combination` 查 `部件 --HAS_FAULT--> 故障` 是否成立，
    搭配不存在时直接拒答，并指出该故障在知识库里实际属于哪个部件
  - 组合校验只在原文上匹配、不做同义词扩展：否则「偏航偏差过大」里的"偏航"会被
    扩展成「偏航系统」，凭空造出一个有效搭配，反而漏掉该拦的伪问题
  - 近邻负例生成改为轮转取样并排除故障名里含部件名的组合，
    修正了此前 16/18 条都配同一个「偏航」故障的偏差
  - 修复引擎以 use_kg=False 构造时、query() 默认 use_kg=True 导致的空指针
  - 单元测试扩充到 105 项
- 语料规模化（这一步把前面的结论推翻了）：
  - 新增 `src/simulate_corpus.py`：确定性生成 1203 篇运维语料（工单/巡检/月报/备件/SOP/安全规程/QA），
    风场名取自公开数据集 OpenWindSCADA 的真实场站，机型用真实型号，
    部件与故障沿用知识图谱词表；其余为模板生成并在文档中明确标注
  - loader 支持 `data/generated/`：每个 JSON 条目作为独立文档加载，保留结构化元数据
  - 评测集从 114 条扩到 1420 条，语料从 65 篇扩到 1269 篇
  - **结论反转**：1269 篇下仅 BM25 反超完整配置（Recall 98.05% vs 96.68%）；
    查询扩展从"中性"变为负收益（召回 -1.94pp、案例召回 -5.33pp、延迟 +0.9ms），
    已改为默认关闭；α 扫描呈单调趋势（α=0.3 最优，MRR 0.9583 vs 自适应 0.9250）
  - 报告新增「性能与规模」段（1269 篇下平均 1.74ms / P95 2.24ms）
  - CI 门禁改为按分层报告的「回答层汇总召回」判定（原 15 条 QA 口径在规模化后只剩 80%），
    CI 增加语料生成步骤
  - 单测改用小规模夹具语料，避免每次都加载 1269 篇（77s → 5s）
  - 单元测试扩充到 108 项
- 口语化查询验证（用真问法戳破合成评测的「假象」）：
  - 新增 `data/human_questions.json`：50 条模拟一线工程师口语提问 + 3 条无效组合，
    自己当标注员标上对应部件+故障（弱标注）
  - 新增 `src/pure_rag/human_eval.py`：在多个 α 下测「口语查询 Recall@3 / MRR」
  - 结论：合成评测 97.62% 是假象（查询从文档字段派生、共享稀有词），
    口语查询掉到 50~68%；两套语料都偏向低 α / BM25（方向没反转），但绝对质量差距来自
    **词汇鸿沟**——本内核「向量」通道实为 TF-IDF，做不了真语义匹配
- 口语化查询成为常驻评测层 `human_like`：
  - `datasets.py` 新增 `build_human_like_layer()`：从 `data/human_questions.json` 读取人写的
    口语提问，可回答条目标注 (部件, 故障类型) 作弱标注、命中判据为「同类工单（任意一条）」，
    无效组合条目标记为拒答；一并纳入语料指纹（问题集变了评测集即重建）
  - `evaluator.py` 扩展 `rank_of_item` 支持「按部件+故障类型匹配任意同类工单」，报告渲染改为
    按层内是否有可回答条目决定显示召回（不再硬编码 ANSWER_LAYERS），混合层也能正常呈现
  - 该层**故意不进** `ANSWER_LAYERS`，因此不影响 CI 门禁与 `tune` 的优化目标（用户要求先只挂层、保留什么后续再定）
  - 评测集由 7 层 1420 条扩到 8 层 1473 条；README/CHANGELOG 同步

### 新增
- 编码与文档一致性（本轮维护）：
  - `src/pure_rag/stdio.py`：标准流编码加固。终端保持原编码，被重定向时切 UTF-8，
    编码不了的字符降级为转义（backslashreplace）、解码不了的字节降级为 U+FFFD，
    命令行入口 `cli.main()` 统一调用
  - 自有文档导入支持 GB18030：`loader.read_plain_text()` 按 UTF-8(含 BOM) → GB18030
    依次尝试，两种都不行才报错（不再用 `errors="ignore"` 静默丢字节），
    txt / md / json / csv 均走该路径
  - `src/gen_readme_facts.py`：README「当前实测数据」块生成器（仅标准库），
    从 `report.json` / `meta.json` / `knowledge_graph.json` / unittest 发现结果取数，
    支持 `--check` 校验漂移；`.cnb.yml` 增加「文档事实一致性」阶段
  - `tests/__init__.py`：包标记，避开 site-packages 同名 `tests` 常规包对仓内命名空间包的阴影
  - `RAGEngine.query()` 返回体补上 `candidates` 与 `variants`（Web 页面徽章与 JSON 调用方需要）
- 大模型生成接入（智谱免费档，仍零第三方依赖）：
  - `create_default_model_manager()` 改为供应商感知：新增通用键 `LLM_PROVIDER` /
    `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODELS`，未配时兼容旧的 `MODELSCOPE_*`；
    zhipu 默认地址 `https://open.bigmodel.cn/api/paas/v4`，默认模型池只放实测可用的免费档
    （glm-4-flash-250414 / glm-4-flash / glm-4.5-flash）
  - `LLMClient` 错误可读化：回读服务端错误体里的 code/msg（上限 300 字符），
    能区分 429/1305 过载、403/1220 无权限、429/1113 余额不足，不再只报 "HTTP Error 429"
  - 降级不静默：`llm_error` 随结果返回，命令行打 `[warn]` 行、Web 显示红色提示；
    模型池全部处于冷却期时也写明原因（之前这一分支是静默的）
  - `stats` 拆分 `llm_configured`（配没配）与 `llm_enabled`（当下可用，冷却中为 false）
  - 提示词重构：知识图谱作为最后一条编号资料入列，不再单独开【知识图谱信息】段落
    （模型会把这类内部标签当正文引用出来）；同时切除 `<think>` 推理段、
    `content` 为空时回读 `reasoning_content`（glm-z1-flash 实测会把 token 预算全花在推理上，
    默认不入池）
  - Web 引用对齐：来源行首标 `[N]`（按 `citations` 的 doc_id 回映射，未被引用标“未引用”），
    知识图谱补一行同编号条目，徽章新增「图谱引用 [N]」与「耗时（缓存）」

### 修复
- LLM 答案路径上的两个结构缺陷：
  - `query()` 命中缓存时返回的是缓存里的同一个对象，Web 层就地 `pop("spans")` 会把缓存
    改坏，下一个相同问题就没有高亮；现改为进出缓存都用深副本，并给出 `cached` 标记
  - `citations` 缺 `doc_id`，导致界面无法把答案里的 `[N]` 对应到具体文档
- Windows 中文代码页下命令行崩溃（重定向输出时必现）：
  - `eval --robust` / `eval --negative`：`UnicodeEncodeError: 'gbk' codec can't encode '\u2713'`
    ——逐条命中记号由 ✓/✗ 改为 ASCII 的 hit/miss，并在入口统一配好标准流
  - `ablation`：配置名里的数学减号 U+2212 同样不能进 GBK，改为 ASCII `-`；进度输出的 `…` 改 `...`
  - `chat /save`：管道输入的孤立代理项导致写 UTF-8 时报 `surrogates not allowed`，
    现由 stdin 的 `errors="replace"` 从根上消除
- `eval --robust` 改写层评测结构性报错：`Evaluator.measure()` 按三个名字解包
  `_resolve_target()` 的五元组，招致 `ValueError: too many values to unpack`（整个鲁棒性评测跑不完）
- `python -m unittest tests.test_pure_rag` 在装有同名 `tests` 包的环境下报
  `ModuleNotFoundError`（CI 用的 clean 镜像不受影响，本地 Windows 必现）
- Web 界面「候选」徽章永远显示 `undefined`：`query()` 未返回 `candidates` 字段
- 空问题走 `query()` 时返回体缺 `alpha` / `timings` 等字段，CLI 与 Web 会 KeyError
- `tune` 输出误导：进度行打印的是「历史最优」而非本轮参数（看起来像重复行），
  收尾行把 `use_expansion` 写成「默认开启」，与引擎实际默认（已关闭）和消融结论相左；
  同时 `num_questions` 参数原本被丢弃，现按「取前 N 条回答层查询」生效
- `datasets.corpus_fingerprint()` 未把 `data_dir` 传给 `human_questions.json` 那一路，
  用 `--data-dir` 指向其他目录时语料指纹会漏算该文件
- `datasets.py` 模块注释里「目标定位方式」重复了两遍（三种/两种并存），保留三种的正确版本

### 新增
- 页面包成对话式界面（仍然零依赖）：
  - `src/pure_rag/web/` 三个静态文件（`index.html` / `style.css` / `app.js`），由
    `server.py` 按白名单读取，不再把 HTML 塑在 Python 字符串里；无框架、无 CDN、可离线
  - 上方会话流 + 下方输入区：用户/助手气泡、Enter 发送、Shift+Enter 换行、自适应行高、
    “检索中 Ns”计时、自动吸底
  - 多轮会话：服务端 `ConversationStore`（线程安全 + LRU，每会话保留最近 8 轮）按
    `session_id` 维护上下文，新接口 `POST /api/session/reset`；不带 `session_id` 仍是单轮无状态
  - 引用可用：答案里的 `[1]`、`[2, 3]`、`[2-5]` 渲染成可点击引用，多编号同时描边全部来源并滚到第一条；
    拒答轮来源区改名“检索候选（N 篇，未作为回答依据）”
  - 顶栏常驻模型状态（已启用 / 不可用（带最近一次失败原因）/ 未启用），每轮问答后刷新；
    抽取式要点点击后的摘录带前后可视上下文（超出部分用 `…` 标记）
  - 刷新不丢记录：本地存最近 10 轮并重建气泡，轮次计数以实际可恢复的气泡数为准

### 修复
- 对话界面首轮验收发现的问题（逐项复测通过）：
  - 刷新后界面历史清空而服务端会话仍在 → 出现“屏幕上没有上一轮却被补全”的幽灵提示与错误轮次；
    改为会话与记录同进同退，旧版只存轮次的本地数据也不再造成不一致
  - 模型池进入冷却期时静默退回抽取式（不告知原因）；现在顶栏徽章与提示条都带上失败原因
  - 拒答文案与判据错配：`invalid_combination` 被套上“覆盖率 100% 低于阈值 35%”这种矛盾句子；
    引擎给出 `reason = no_content_term`（实词零命中）与 `low_coverage`（覆盖率不足）两种子原因，
    命令行与界面分别措词；单字片段（“几”“科”“幻”）不再被列为“未覆盖关键词”
  - 追问被误拒答：上一轮拼进检索词后覆盖率分母被稀释（实测 0.7598 降到 0.3335）；
    现在合并查询与原话查询各跑一次，取覆盖率更高的那次作为本轮依据
  - 省略式追问得不到补全：“那多久换一次？”一类开头词不在指代表里 → 补充触发词
  - 向量/关键词单路检索仍展示自适应 alpha；拒答轮展示无用的 alpha 与“耗时”（实为检索耗时）
  - 图谱占位编号即使未被引用也挂在界面上；现在只有答案真的引用了该编号才显示
  - LLM 偶尔在末尾多吐一行光秃秃的编号（可带“资料编号：”前缀）→ 输出清洗
- 静态路由只查白名单，`/static/../server.py` 与百分号编码穿越均返回 404；未知路径与
  `GET /api/query` 统一返回 JSON 错误，不再可能溢出源码；缓存命中返回深副本，
  Web 层加字段不再污染引擎缓存

### 变更
- 免费档模型不再只用一个：模型池放进 5 个实测可用免费档（glm-4-flash-250414 / glm-4.7-flash / glm-4v-flash /
  glm-4-flash / glm-4.5-flash），一次调用成功就轮到下一个（`report_success` 里推进轮转位），
  并把轮转位存 `data/pure_index/llm_rotation.json`，让命令行一次一个新进程时也能接着转
- 思维链模型（4.5 / 4.7）请求自动带 `thinking.type=disabled`，服务端不认该参数（400）时去掉再试一次；
  `finish_reason=length` 的回答（只有半段思考）不再当答案给用户，而是报错转下一个模型；
  实测 glm-4.7-flash 关推理后从“32s 且无正文”变成约 4s 出正常答案；glm-z1-flash 关不住推理（实测 17s
  仍被 token 截断），因此仍不入池，但代码保留截断检测
- 错误码转成可执行提示：1113 提示“该模型属收费档，请从 LLM_MODELS 移除”、1220 提示换免费档、
  1305 提示“服务端临时过载，稍后重试”
- 答案可看到是谁答的：结果新增 `llm_model`，CLI 答案头部与 Web meta 行都会显示模型名；
  `stats` 新增 `llm_pool`（每个模型的可用/失败/请求数），顶栏显示“池内 N 个”
- Web 轮数计数不再封顶：`/api/query` 的 `turns` 改名 `context_turns`（它是“上下文保留了几轮”，
  有 8 轮上限），页面轮数由前端自己数，超过 8 轮不再停在“共 8 轮”
- 项目主线收敛为纯 Python：`AGENTS.md` 写入「项目自身需求」（只用标准库、检索与生成分离、
  重依赖模块只作对照、入口必须走 `stdio.configure_streams()`、文档数字不写死）
- README 结构调整：新增「本地部署（纯 Python，主线）」与脚本生成的「当前实测数据」块，
  原「快速开始（完整研究版）」改为「快速开始（遗留研究版）」；技术栈分主线/遗留两张表；
  贡献章节改写为本仓库实际的 main 单分支 + 三远端同步流程；文档不再用 emoji 装饰
- `requirements.txt` 顶部注明只服务遗留研究版，主线零依赖
- `.cnb.yml`：评测阶段不再 `--no-save`（需落 `report.json` 供文档一致性校验），
  新增「文档事实一致性」阶段
- `eval` 默认强制抽取式生成（新增 `--use-llm` 才走大模型）：保证指标可复现，
  也不让上千条评测查询去砸免费额度；`ablation` 与 `tune` 本就 `use_llm=False`
- 单元测试由 108 项增到 109 项（项数以 README 自动生成块为准，不再手写）

### 新增
- 开源规范文件补全（增量，不破坏既有内容）：
  - README 拆分为英文主版 `README.md` + 中文配套版 `README.zh-CN.md`，两版章节一一对应、
    顶部互链；`src/gen_readme_facts.py` 实测块改为按目标文件语言渲染（README.md 英文标签、
    README.zh-CN.md 中文标签），数字仍全部取自真源不手写
  - CONTRIBUTING 同样拆分为英文主版 + `CONTRIBUTING.zh-CN.md`，补充文档数字与中英双版校验命令
  - 新增 `SECURITY.md`（漏洞报告政策）、`FAQ.md`（常见问题）、`AUTHORS`（作者与维护者）
  - 新增 `Makefile`（build/ask/chat/serve/eval/ablation/tune/stats/test/corpus/facts 便捷目标，
    仅调用 `python src/rag_cli.py`，零依赖）与 `.gitattributes`（行尾规范化 + 数据/索引/HTML 的
    linguist 归类）
  - `.gitignore` 补充 `.env`、`.tmp_*`、`*.bak` 与 archify 取证侧车产物

### 变更
- `LICENSE` 版权行补上作者身份（LPK3215 and WindRAG contributors，年份 2026 不变）
- `.cnb.yml` 文档事实一致性阶段追加对 `README.zh-CN.md` 的 `--check`，中英双版数字同受门禁
- 项目内新增交互式图集 `docs/archify/`（architecture / workflow / sequence / dataflow 四张，
  均由 archify 生成并通过 showcase 校验）与出图输入蓝图 `diagram-input.md`

## [1.0.0] - 2026-01

### 新增
- 面向风电领域的智能故障诊断问答系统
- 自适应混合检索（向量检索 + BM25 + 知识图谱）
- 知识图谱增强（基于 embedding 语义匹配）
- 模型池管理（多 LLM 自动切换、失败冷却、优雅降级）
- 中文深度优化（jieba 分词、中文向量模型、领域术语识别）
- 高鲁棒性设计（异常输入、并发查询、LLM 不可用等场景）
