# WindRAG Project Analysis

## Scope and Evidence

This document describes the repository state inspected on 2026-10-17. The conclusions are based on the source files listed below and on direct execution of the mainline commands.

- Main entry: `src/rag_cli.py`
- Main package: `src/pure_rag/`
- Shared standard-library modules: `src/data_processing.py`, `src/knowledge_graph.py`, `src/model_manager.py`
- Runtime configuration: `config.env` and `config.env.example`
- Persisted index: `data/pure_index/`
- CI and container definitions: `.cnb.yml`, `Dockerfile`, `docker-compose.yml`

Measured values in this document come from the commands shown in the verification section. Re-run those commands after changing the corpus or index configuration.

## One-Line Summary

WindRAG is a wind-turbine operations RAG system whose production path is a pure-Python hybrid retriever with optional knowledge-graph context, extractive answers by default, optional OpenAI-compatible LLM generation, a CLI, and a zero-dependency HTTP chat UI.

## Five-Minute Orientation

### Primary tasks

The mainline supports these tasks through `python src/rag_cli.py <command>`:

- `build`: load corpus files, chunk documents, tokenize, discover terms, build BM25 and TF-IDF indexes, and persist them.
- `ask`: retrieve documents and return an answer with confidence, citations, source spans, and optional fault-card data.
- `chat`: keep a local conversation history for coreference-aware follow-up questions.
- `add`: append supported user documents and rebuild global index statistics.
- `explain`: expose per-term BM25 and TF-IDF contributions.
- `eval`: run deterministic retrieval and refusal evaluations.
- `tune`: search chunk and fusion parameters in temporary index directories and persist the selected parameters.
- `ablation`: compare retrieval configurations with each module enabled or disabled.
- `stats`: report index, knowledge-graph, and LLM-pool state.
- `serve`: start the standard-library HTTP server and static web UI.

### Inputs

- CLI arguments and question text enter through `src/pure_rag/cli.py`.
- Corpus inputs come from `data/raw/`, `data/synthetic/`, `data/generated/`, and `data/user_docs/`.
- Optional LLM settings are loaded from `config.env` by `src/pure_rag/env.py` and `src/pure_rag/engine.py`.
- Web requests enter through `POST /api/query` in `src/pure_rag/server.py`.

### Outputs

- Index artifacts are JSON files under `data/pure_index/`.
- CLI answers are text or JSON.
- Evaluation writes JSON and Markdown reports when saving is enabled.
- The HTTP API returns JSON; the static UI renders the same result with HTML highlights and source anchors.

### Main path

```text
src/rag_cli.py
  -> pure_rag.cli.main
  -> RAGEngine
  -> loader / chunker / tokenizer
  -> BM25 + TF-IDF + fusion + optional MMR
  -> knowledge graph context and combination validation
  -> refusal gate
  -> extractive generator or optional LLM client
  -> CLI output or HTTP JSON response
```

## Repository Shape

| Path | Role | Runtime status |
|---|---|---|
| `src/pure_rag/` | Production retrieval kernel, generation, evaluation, and HTTP server | Mainline |
| `src/rag_cli.py` | Repository-root CLI wrapper | Mainline entry |
| `src/data_processing.py` | Shared corpus parser used by the pure loader | Mainline dependency, standard library only |
| `src/knowledge_graph.py` | Shared dictionary and adjacency-list graph implementation | Mainline dependency, standard library only |
| `src/model_manager.py` | Optional LLM model pool, rotation, cooldown, and provider defaults | Mainline dependency when LLM is enabled |
| `src/*.py` research scripts | Sentence-transformers, NumPy, jieba, Gradio, and related research implementation | Legacy/reference path |
| `data/` | Source corpus, generated corpus, graph, evaluation data, and persisted index | Runtime data |
| `tests/test_pure_rag.py` | Pure-Python test suite | Mainline test coverage |
| `tests/test_modelscope.py` | External SDK based integration test | Optional legacy/dependency path |
| `requirements.txt` | Third-party dependencies for the legacy research path | Not required by the mainline |
| `Dockerfile` and `docker-compose.yml` | Container build and HTTP deployment | Mainline deployment |

The repository is therefore a mixed workspace rather than a single dependency profile. The current production boundary is `src/rag_cli.py` plus `src/pure_rag/` and the three shared root modules listed above. The research modules remain present but are not imported by the pure CLI path.

## Entry Points and Dispatch

### CLI entry

`src/rag_cli.py` inserts `src/` into `sys.path`, imports `pure_rag.cli.main`, and exits with its return code. `pure_rag.cli.main` first calls `configure_streams()` and then dispatches an `argparse` subcommand.

Global options must appear before the subcommand. For example:

```text
python src/rag_cli.py --no-llm ask "gearbox overheating handling"
python src/rag_cli.py --no-llm --index-dir <temporary-index> build
```

The package entry `python -m pure_rag` requires `PYTHONPATH=src`; the repository-root wrapper is the normal invocation.

### Web entry

`cmd_serve` creates an engine, calls `ensure_loaded()`, and passes it to `pure_rag.server.serve()`. The server uses `ThreadingHTTPServer` with a handler class bound to that engine instance.

## Core Modules and Ownership

### Ingestion and indexing

- `src/pure_rag/loader.py` reads plain text, Markdown, JSON, CSV, and DOCX using standard-library parsers. It combines `DataProcessor.load_raw_text()` and `DataProcessor.load_synthetic_data()` with generated and user documents.
- `src/data_processing.py` formats built-in fault cases, QA pairs, technical documents, and operations manuals into normalized document records.
- `src/pure_rag/chunker.py` splits paragraphs into sentences and hard windows, carries tail context into overlapping chunks, and repeats title/keywords in the index-only text for field weighting.
- `src/pure_rag/text_utils.py` owns normalization, domain vocabulary, stopwords, synonym aliases, forward maximum matching, bigram supplementation, sentence offsets, span finding, and unsupervised new-word discovery.
- `RAGEngine.build()` discovers terms, rebuilds the tokenizer, chunks all documents, tokenizes every chunk, builds both indexes, and writes the persisted artifacts.

The persisted index consists of `documents.json`, `tokens.json`, `bm25.json`, `tfidf.json`, and `meta.json`. `RAGEngine.load()` restores the tokenizer terms, chunks, token sequences, indexes, source count, and chunk parameters. `ensure_loaded()` loads this state or builds it when the index is absent.

### Retrieval

- `src/pure_rag/bm25.py` implements an inverted-list Okapi BM25 index.
- `src/pure_rag/tfidf.py` implements sparse TF-IDF scoring and cosine similarity.
- `src/pure_rag/query_analysis.py` classifies questions and predicts the vector/BM25 fusion weight.
- `src/pure_rag/fusion.py` implements min-max scaling, weighted fusion, reciprocal-rank fusion, score merging, and MMR selection.
- `RAGEngine.retrieve()` executes the following stages:
  1. Tokenize the question.
  2. Select explicit alpha, tuned alpha, or adaptive alpha.
  3. Score the TF-IDF and BM25 channels selected by `method`.
  4. Optionally retrieve synonym variants and merge them with RRF. The engine default is expansion disabled unless tuning enables it.
  5. Scale scores, select candidates, and optionally run MMR.
  6. Locate query-term spans in original chunk text.
  7. Query the knowledge graph for context and entities.

### Knowledge graph

`src/knowledge_graph.py` stores nodes and adjacency lists in JSON-backed Python dictionaries. `src/pure_rag/kg.py` builds a TF-IDF index over graph node descriptions, matches query entities, expands neighbors, builds context text, and validates component/fault combinations. `RAGEngine.query()` uses invalid combinations as a separate refusal reason so a query containing individually known terms does not automatically receive a cross-component answer.

### Query and answer generation

`RAGEngine.query()` is the application-level orchestration boundary:

1. Empty input returns a structured `mode=empty` result.
2. Short follow-up questions containing configured pronouns are combined with the previous question for retrieval.
3. Retrieval and IDF-weighted query-term coverage are computed.
4. If coreference changed the query, the original question is also retrieved and the higher-coverage result is selected.
5. Knowledge-graph combination validation and the coverage threshold decide whether to refuse.
6. Accepted queries enter `AnswerGenerator.generate()`.

`src/pure_rag/generator.py` first filters low-scoring documents, then builds a fault card when a chunk contains the required fault-case fields. Without an LLM it scores source sentences by query-term IDF, structure labels, document rank, and sentence length; it returns ordered claims, source citations, and character offsets. This is the default answer path.

When enabled, `LLMClient` uses `urllib.request` against an OpenAI-compatible `/chat/completions` endpoint. `model_manager.py` supplies the configured model pool, priority ordering, round-robin rotation, failure counts, and cooldown. A failed call returns to extractive generation and exposes the failure as `llm_error` in the result.

### HTTP server

`src/pure_rag/server.py` exposes:

| Route | Behavior |
|---|---|
| `GET /` and `GET /index.html` | Static chat page |
| `GET /static/style.css` | Whitelisted CSS asset |
| `GET /static/app.js` | Whitelisted JavaScript asset |
| `GET /api/stats` | Engine statistics and LLM state |
| `POST /api/query` | Query, retrieval, generation, highlights, and optional session context |
| `POST /api/session/reset` | Remove one in-memory session |

`ConversationStore` keeps up to eight turns per session and up to 64 sessions. The request handler serializes engine calls with a class-level lock while the HTTP server itself is threaded. The server only serves the three whitelisted static paths and does not concatenate arbitrary URL paths.

## Data and Evaluation Flow

### Corpus flow

```text
data/raw + data/synthetic + data/generated + data/user_docs
  -> pure_rag.loader.load_documents
  -> DataProcessor formatting for built-in sources
  -> chunk_documents
  -> Tokenizer.cut_for_index
  -> BM25Index + TfidfIndex
  -> data/pure_index/*.json
```

`add` copies external supported files into `data/user_docs/` by default, appends only new chunks, and rebuilds the global BM25/TF-IDF statistics because IDF depends on all indexed documents.

### Evaluation flow

`src/pure_rag/datasets.py` derives a deterministic dataset from the corpus. It includes QA queries, fault-case queries, document-title queries, near-miss component/fault combinations, manual negative queries, and human-like queries. `src/pure_rag/evaluator.py` measures target-block recall, MRR, nDCG, answer-term coverage, latency, and refusal behavior. `eval --report` saves the layered report and its structured JSON counterpart when saving is enabled.

The evaluation is a retrieval proxy, not a human answer-quality benchmark. The target for many positive cases is a text signature or metadata match in the indexed chunk. The answer-quality metric is domain-term coverage. The evaluator itself documents that synthetic queries can share terms with their target answers and that domain-in-scope but missing information is difficult to refuse reliably.

## Configuration and Deployment

- `src/pure_rag/env.py` parses `KEY=VALUE` files without overriding already-exported environment variables.
- `config.env` is ignored by Git and contains local LLM settings. `config.env.example` documents the supported keys.
- `requirements.txt` contains the legacy research dependencies and is not imported by the pure path.
- `Dockerfile` copies `src/`, `data/`, and tests, builds the index during image construction, exposes port 8000, and starts `serve` on `0.0.0.0`.
- `docker-compose.yml` maps host port 8000 to the container, mounts `config.env` read-only, and does not mount the host `data/` directory.
- `.cnb.yml` runs pure tests, generates a corpus, builds an index, runs a Recall gate, and checks generated README facts.

## Verified Runtime Evidence

The following commands were run on 2026-10-17 with Python 3.11.8. The `-I` flag was used for isolated interpreter startup, and `--no-llm` was used for deterministic local verification.

### Existing persisted index

```text
python -I src/rag_cli.py stats
```

Returned a loaded index with 1269 source documents, 1273 chunks, 4179 vocabulary terms, 276 discovered terms, and a 162-node/160-edge knowledge graph. The values are runtime output, not a hand-maintained configuration source.

### Build, ask, and eval in a temporary index

```text
python -I src/rag_cli.py --no-llm --index-dir <temporary-index> build
python -I src/rag_cli.py --no-llm --index-dir <temporary-index> ask "齿轮箱温度过高应该怎么处理？" --json
python -I src/rag_cli.py --no-llm --index-dir <temporary-index> eval -n 3 --no-save
```

The temporary build completed with the same 1269-document/1273-chunk corpus. The ask command returned `mode=extractive`, coverage `1.0`, knowledge-graph context, citations, claims, and source spans. The three-query evaluation completed with Recall@3 `100.00%`, MRR `1.0000`, nDCG@3 `1.0000`, and no LLM calls.

The temporary index was removed after verification.

### Web surface

The server was started on temporary port 8765 with the temporary index. Direct requests returned:

- `GET /`: HTTP 200 and `text/html; charset=utf-8`.
- `GET /api/stats`: the same loaded index counts.
- `POST /api/query`: `mode=extractive` and one stored session context turn.
- `POST /api/session/reset`: `reset=true`.

The server process was stopped and port 8765 was released after the requests.

Browser-level visual inspection was not completed. The repository browser skill was available, but the local Chrome process did not expose CDP on port 9222, so navigation and screenshot automation could not connect. No browser result is claimed beyond the direct HTTP checks above.

## Boundaries and Known Limits

- The pure mainline does not use dense embeddings, a vector database, a web framework, or an SDK. Its semantic channel is sparse TF-IDF.
- Query expansion is implemented but disabled by default in the engine configuration because the checked-in comments and README record negative results on the scaled corpus.
- Confidence is IDF-weighted lexical coverage of retrieved documents. It can detect out-of-domain vocabulary and graph-invalid component/fault pairs, but it cannot reliably prove that an in-domain question has no answer in the corpus.
- `RAGEngine` persists JSON indexes and rebuilds BM25/TF-IDF statistics on append; there is no separate database or background worker.
- The HTTP server keeps sessions in process memory. Restarting the process removes conversational context.
- LLM availability is environment-dependent. The pure path remains usable without an API key or with `--no-llm`; when enabled, network failures are surfaced through `llm_error` after extractive fallback.
- The root research scripts and `tests/test_modelscope.py` require packages listed in `requirements.txt` or installed SDKs. They are outside the pure CLI runtime contract.

## Reproduction Commands

Use these commands to refresh the runtime facts after corpus or code changes:

```text
python src/rag_cli.py build
python src/rag_cli.py --no-llm ask "齿轮箱温度过高应该怎么处理？" --json
python src/rag_cli.py --no-llm eval --report
python src/rag_cli.py stats
python src/gen_readme_facts.py --check
```

`build` writes `data/pure_index/`; run it only when rebuilding the persisted index is intended. `eval --report` writes the report files unless `--no-save` is supplied. `serve` is a blocking process and should be stopped after manual verification.
