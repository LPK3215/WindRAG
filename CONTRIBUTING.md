# 贡献指南

感谢您关注 WindRAG！请阅读以下指南参与贡献。

## 开发环境搭建

请先阅读 [README.md](./README.md) 了解项目架构和安装步骤。

纯 Python 检索内核（`src/pure_rag/`）零第三方依赖，有 Python 3.10+ 即可运行，
不需要 `pip install -r requirements.txt`（那是研究版才需要的重依赖）。

## 分支策略

- 本项目为单人维护，**改动一律直接提交到 `main`**，不另开 `feature/*`、`fix/*` 分支，也不走 PR
- 确有必要开分支时，请先说明原因，并在合并后立即删除，避免仓库里留下多余分支

## 提交规范

| 前缀 | 用途 |
|---|---|
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |
| `docs:` | 文档更新 |
| `perf:` | 性能优化 |
| `refactor:` | 代码重构 |
| `chore:` | 构建/工具变更 |
| `test:` | 测试相关 |

标题控制在 50 字以内、只说改了什么；「为什么改、怎么改」写进提交正文。

## 提交前自检

```bash
python -m unittest tests.test_pure_rag -v        # 单元测试
python src/rag_cli.py build                      # 索引能正常构建
python src/rag_cli.py eval -n 15 --robust --negative --no-save --min-recall 0.9
```

以上三条与 `.cnb.yml` 流水线一致，推送到 `main` 后会由流水线自动再验证一遍。

## 代码规范

- 使用 Python 3.10+ 风格
- 遵循 PEP 8 规范
- 不要提交 `config.env` 或任何 API 密钥
- 索引目录（`data/pure_index/`、`data/processed/`）与 `data/chat_logs/` 可重新生成，不入库
- 不要把构建环境的回收站目录（`.Trash-0/`）带进版本库
