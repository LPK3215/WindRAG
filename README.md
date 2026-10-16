<div align="center">

<img src="./docs/assets/banner.svg" alt="WindRAG" width="100%"/>

[![license](https://img.shields.io/github/license/LPK3215/WindRAG)](LICENSE)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/third--party%20deps-0-brightgreen)
![install](https://img.shields.io/badge/pip%20install-not%20required-informational)
![ui](https://img.shields.io/badge/web%20UI-http.server%20SPA-8ab4f8)

**WindRAG** — a lightweight, pure-standard-library RAG for wind-turbine O&M knowledge bases.

Conversational retrieval plus a zero-dependency web page that renders results, source locations, and matched terms. Runs straight in a clean interpreter with plain `python`.

[Features](#-features) · [Quick Start](#quick-start) · [CLI](#cli) · [Web UI](#web-ui) · [Visualization](#-visualization) · [Evaluation](#evaluation-and-known-limits) · [FAQ](FAQ.md)

[中文](README.zh-CN.md)

</div>

## ✨ Features

- **Zero dependencies** — the entire mainline (`src/pure_rag/`) is the Python standard library only; `build` / `ask` / `eval` / `serve` run in a clean interpreter with no `pip install`.
- **Hybrid retrieval** — hand-written Okapi BM25 + sparse TF-IDF, fused with a rule-prior adaptive alpha, RRF merging and MMR reranking, plus TF-IDF-based knowledge-graph context.
- **Retrieval / generation separated** — without an LLM it returns a readable extractive answer; the LLM is optional, fails back to extractive, and surfaces `llm_error` instead of degrading silently.
- **Traceable answers** — per-point character spans back to the source, clickable citations, fault cards, and hit-term highlighting (terminal ANSI / web `<mark>` / Markdown bold).
- **Honest evaluation** — a deterministic, layered eval set auto-derived from the corpus, with ablation tables reporting negative results and the real (much lower) usability of colloquial queries.

## What it is / what it is not

| Is | Is not |
|---|---|
| Conversational retrieval QA (`ask` / `chat`) | Dense-vector semantic search (substituted by TF-IDF) |
| Zero-dependency web page for results and sources | Heavy front-end framework (native HTML/CSS/JS, no CDN) |
| Keyword + sparse semantic + rule-based fusion rerank | Neural retrieval model / training pipeline |
| Optional LLM enhancement (stdlib `urllib`) | A required LLM (extractive answers work without one) |
| Import your own docs (txt/md/json/csv/docx) | Online crawling / data pipelines |

## Quick start

No `pip install`, no model required:

```bash
python src/rag_cli.py build                                  # build index (tokenize + BM25 + TF-IDF + new-word discovery)
python src/rag_cli.py ask "齿轮箱温度过高应该怎么处理？" --show-docs
python src/rag_cli.py chat                                   # interactive multi-turn QA
python src/rag_cli.py serve --port 8000                      # open http://127.0.0.1:8000
```

Retrieval and generation are two separate layers. Without `LLM_API_KEY` (or with `DISABLE_LLM=true`), retrieval still returns documents and matched terms, and generation degrades to an extractive answer (`mode` is `extractive`) — it never errors, hangs, or returns an empty shell just because no model is configured.

To import your own material: drop files into `data/user_docs/` and re-run `build`, or add them incrementally with `python src/rag_cli.py add <file/dir>` (only new documents are chunked and tokenized; existing ones reuse the persisted token sequences).

## CLI

Entry point `python src/rag_cli.py <subcommand>`, all real implementations:

| Subcommand | Purpose |
|---|---|
| `build` | Build the index (tokenize + BM25 + TF-IDF + unsupervised new-word discovery) |
| `ask "<question>" [--show-docs]` | Single-shot QA; `--show-docs` prints each point's character span in the source |
| `chat [--log]` | Interactive multi-turn QA; `--log` or `/save` exports a Markdown transcript with sources |
| `add <file/dir>` | Append documents incrementally without rebuilding the whole index |
| `explain "<question>"` | Per-query-term contribution to BM25 / TF-IDF scores |
| `stats` | Index statistics and model availability status |
| `serve --port <p>` | Start the zero-dependency web UI |
| `eval [-n N] [--report] [--negative] [--robust]` | Retrieval evaluation / layered report / refusal calibration |
| `tune [--quick]` | Grid-search chunk and fusion parameters, persist and apply |
| `ablation` | Ablation table: toggle each module and compare |

As a library:

```python
import sys; sys.path.insert(0, "src")
from pure_rag import RAGEngine

engine = RAGEngine(use_llm=False)
result = engine.query("齿轮箱温度过高应该怎么处理？")
print(result["answer"])
for doc in result["retrieved_docs"]:
    print(doc["score"], doc["matched"], doc["metadata"]["title"])   # matched = hit terms
```

## Web UI

After `serve`, open `http://127.0.0.1:8000`. It looks like an ordinary chatbot: conversation stream on top, input box at the bottom, Enter to send, Shift+Enter for a newline, and you can keep following up ("how often should it be replaced?", "what about it?").

- Top bar: model status (enabled / unavailable with the failure reason / not enabled) and corpus size, refreshed each turn.
- Answer bubble: natural-language answer, a meta line (generation mode / alpha / method / latency / candidates / graph hits), and refusal / degradation hints.
- Citations: `[1]` `[2, 3]` in the answer are clickable; multi-number citations outline all sources at once and scroll to the first.
- Collapsible sections: fault card, point locations (click to jump to the source and show an excerpt), citation sources.
- Sessions: the server keeps the last 8 turns per `session_id` for coreference resolution; the retrieval layer itself stays stateless.

Routes (no framework, no CDN; the three front-end files live in `src/pure_rag/web/`):

```text
GET  /                   chat page
GET  /static/style.css   styles       GET /static/app.js   logic (whitelisted files only)
POST /api/query          QA, body {question, top_k, method, alpha, session_id}
POST /api/session/reset  clear this session's multi-turn context
GET  /api/stats          index and model status
```

## 📊 Visualization

The architecture, workflow, sequence, and data-flow diagrams are generated with Archify and shipped as self-contained interactive HTML under [`docs/archify/`](docs/archify/) (validated to the showcase profile). Open them in a browser — they support zoom, pan, search, and relationship tracing:

- [Interactive architecture panorama](docs/archify/architecture-panorama.html) — every module, layer boundaries, external service, and the mainline-vs-legacy split.
- [Q&A workflow](docs/archify/workflow-panorama.html) — the full path from a question to an answer, including the pre-generation refusal gate and the graph side branch.
- [/api/query call sequence](docs/archify/sequence-panorama.html) — one request traced across every component in time order.
- [Data-flow panorama](docs/archify/dataflow-panorama.html) — corpus -> ingest -> index -> consumption, all reproducible.

The banner at the top is generated by [`scripts/visualization/generate_banner.py`](scripts/visualization/generate_banner.py), which reads version, test count, corpus size and Recall from the real sources at runtime (nothing is hard-coded).

## Current measured figures

Every value below is generated by a script from the real sources — never hand-written. After the corpus or evaluation changes, re-run `python src/rag_cli.py eval --report`, then `python src/gen_readme_facts.py` (use `--check` to detect drift).

<!-- BEGIN AUTO:FACTS (generated by: python src/gen_readme_facts.py) -->

| Item | Value | Source |
|---|---|---|
| Kernel version | 1.0.0 | `pure_rag.__version__` |
| Corpus | 1269 docs / 1273 chunks / vocab 4179 (276 discovered) | `data/pure_index/meta.json` |
| Knowledge graph | 162 nodes / 160 edges | `data/knowledge_graph.json` |
| Eval set | 1473 items (8 layers) | `data/eval_set.json` |
| Answer-layer Recall@3 | 97.62% | `python src/rag_cli.py eval --report` |
| Colloquial-layer Recall@3 | 58.00% (53 items) | same |
| Near-miss refusal | 100.00% | same |
| Out-of-domain refusal | 100.00% | same |
| Unit tests | 110 | `python -m unittest tests.test_pure_rag` |

<!-- END AUTO:FACTS -->

## Evaluation and known limits

The eval set is derived from the corpus and built deterministically, with no manual labeling (`python src/rag_cli.py eval --report` writes `data/pure_index/report.md`). To be honest about it:

- **Synthetic-eval scores are inflated.** Synthetic queries are derived from document fields and share rare terms (farm name / unit id / fault name) with the answers — effectively "reading the answer to find the document" — so the answer layer reaches 97.62%. Measured with human-written colloquial questions in `data/human_questions.json` (symptoms only, component names omitted), real usability drops to 50-68%. The gap is not about tuning but the lack of true semantic representation: this kernel's "vector" channel is only TF-IDF, so it cannot bridge the lexical gap between "buzzing noise" vs "abnormal noise", or "overheating curtailment" vs "winding temperature 130C". Reproduce: `python src/pure_rag/human_eval.py` (deterministic).
- **Scale overturns small-corpus conclusions.** The same code shows "answer layer 100%, modules only help" on 65 docs, but on 1269 docs plain BM25 beats the full configuration. Query expansion diluted exact matches after RRF merging and is now off by default. Toggle each module with `python src/rag_cli.py ablation` (includes fixed-alpha sweeps and negative results, reported as-is).
- **Refusal has two criteria:** IDF-weighted query-term coverage (blocks out-of-domain questions, 100% measured) and knowledge-graph component-fault combination validation (blocks pseudo-questions like "yaw-system brake failure" whose terms all exist but the pair does not, and reports which component the fault actually belongs to).
- **Remaining boundary:** in-domain-but-missing material (prices, a specific farm's actual output, spare-part model numbers) has full coverage and a valid combination, the info simply is not in the corpus — the current criteria cannot detect this.

## Optional LLM

The generation layer connects to an OpenAI-compatible endpoint via stdlib `urllib` — no SDK. A few keys in `config.env` (already gitignored, never committed):

```env
LLM_PROVIDER=zhipu            # zhipu | modelscope; sets default base URL and free-tier model pool
LLM_API_KEY=your_api_key_here
LLM_MODELS=glm-4-flash-250414:1,glm-4v-flash:3,glm-4-flash:4
DISABLE_LLM=false
```

Conventions:

- `LLM_MODELS` format is `name:priority`; only models the account can call for free go into the pool. On failure it switches to the next and enters a 60s cooldown; the rotation index is stored in `data/pure_index/llm_rotation.json`.
- **Evaluation never calls the model:** `eval` / `ablation` / `tune` force extractive by default, so thousands of queries do not burn free quota and non-deterministic generation cannot pollute metrics.
- **Degradation is not silent:** when the LLM fails or the whole pool is cooling, the answer falls back to extractive and `llm_error` carries the server's original code/msg; the CLI prints a `[warn]` line and the web UI shows a notice under the answer.

## Design trade-offs (zero dependency)

`src/pure_rag/` implements the whole chain with the standard library; the heavy research version coexists but is never imported on the mainline path:

| Capability | Usual dependency | Pure-Python mainline |
|---|---|---|
| Chinese tokenization | jieba | Domain dictionary + forward maximum match + bigrams, unsupervised new-word discovery |
| Keyword retrieval | rank-bm25 | Hand-written Okapi BM25 (inverted list + smoothed IDF) |
| Semantic retrieval | text2vec (768-dim) | Hand-written sparse TF-IDF vectors + inverted index + cosine |
| Fusion weight | MLP + rule prior | Rule-prior adaptive alpha + RRF merge + MMR rerank |
| Knowledge graph | embedding cosine match | TF-IDF semantic match + synonym table ("gearbox <-> transmission") |
| Answer generation | LLM API | Extractive summary (default), optional LLM via urllib |
| Web UI | Gradio | `http.server` chat single page, native HTML/CSS/JS |
| Env config | python-dotenv | Self-written `.env` parser (`env.py`) |
| Third-party deps | 10+ packages | 0 |

The modules at `src/` root — `run_rag.py`, `build_index.py`, `gradio_app.py`, `adaptive_fusion.py`, `query_features.py`, `evaluation.py`, `system_validation.py`, `ablation_experiment.py`, `generate_images.py` — are the heavy research version, kept only for reference: not on the mainline, never imported on the pure-Python path. All new changes land in `src/pure_rag/`.

## Project layout

```text
src/
├── rag_cli.py            # pure-Python CLI entry
├── gen_readme_facts.py   # README measured-figures block generator (from real sources, not hand-written)
├── simulate_corpus.py    # scaled test-corpus generator (deterministic, reproducible)
└── pure_rag/             # zero-dependency retrieval kernel (mainline)
    ├── text_utils.py     # tokenization, new-word discovery, sentence split, synonyms
    ├── query_analysis.py # question classification, adaptive alpha, query expansion
    ├── bm25.py / tfidf.py / fusion.py / kg.py   # retrieval and fusion
    ├── chunker.py / loader.py                    # chunking and document import
    ├── generator.py      # extractive generation + optional LLM
    ├── highlight.py      # hit highlighting (ANSI / HTML / Markdown)
    ├── datasets.py / evaluator.py / tuner.py / ablation.py / human_eval.py   # evaluation
    ├── env.py / stdio.py # config loader / stdio encoding hardening (survives Windows code pages)
    ├── engine.py / cli.py # RAGEngine core / subcommands
    └── web/              # chat front-end static assets (read by server.py via a whitelist)
data/
├── user_docs/            # your own docs (txt/md/json/csv/docx, indexed automatically)
├── synthetic/            # synthetic data + refusal eval set
├── pure_index/           # pure-Python index and tuning params (regenerable, not committed)
└── generated/            # scaled test corpus (from simulate_corpus.py, regenerable)
tests/test_pure_rag.py    # pure-Python kernel unit tests
scripts/
└── visualization/        # README asset generators (stdlib only)
    └── generate_banner.py # renders docs/assets/banner.svg from real sources
docs/
├── assets/banner.svg     # generated hero banner (English text)
└── archify/              # interactive architecture / workflow / sequence / dataflow HTML
.cnb.yml                  # CI: unit tests + eval gate + README figures consistency
Makefile                  # convenience targets wrapping python src/rag_cli.py
README.md / README.zh-CN.md          # English main + Chinese companion
CONTRIBUTING.md / CONTRIBUTING.zh-CN.md
CHANGELOG.md / FAQ.md / SECURITY.md / AUTHORS / LICENSE
```

To reproduce the scaled corpus and metrics behind the tables above:

```bash
python src/simulate_corpus.py        # deterministic test corpus (65 docs -> 1269 docs)
python src/rag_cli.py build
python src/rag_cli.py eval --report
python src/gen_readme_facts.py       # refresh the "Current measured figures" block above
```

> Corpus fidelity note: farm names come from the 25 real sites in the public OpenWindSCADA dataset, models use real designations, and the component / fault vocabulary follows the project knowledge graph; everything else (unit ids, timestamps, symptoms, downtime, report numbers) is template-generated and is not a measured distribution. Its purpose is to make the sample size large and the corpus noisy, not to serve as real data.

## Contributing, license and contact

- Repository: [GitHub](https://github.com/LPK3215/WindRAG) (origin) — mirrored to [Gitee](https://gitee.com/LPK-Gitee/WindRAG) and [CNB](https://cnb.cool/lpk3215/WindRAG); all three track `main`.
- Contributions: read [CONTRIBUTING.md](CONTRIBUTING.md) ([中文](CONTRIBUTING.zh-CN.md)); change history lives in [CHANGELOG.md](CHANGELOG.md); common questions in [FAQ.md](FAQ.md); vulnerability reporting in [SECURITY.md](SECURITY.md).
- Authors and maintainers: see [AUTHORS](AUTHORS). Maintainer: **LPK3215** <17538703215@163.com>.
- License: released under the [MIT License](LICENSE), © 2026 LPK3215 and WindRAG contributors.
