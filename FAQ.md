# FAQ

Frequently asked questions about WindRAG.

## Can I really run this with no `pip install`?

Yes. The mainline in `src/pure_rag/` (plus `src/rag_cli.py`) uses only the Python standard library. In a clean Python 3.10+ interpreter:

```bash
python src/rag_cli.py build
python src/rag_cli.py ask "What causes gearbox overheating?"
python src/rag_cli.py serve --port 8000
```

`requirements.txt` is only for the legacy research version (`src/run_rag.py`, `gradio_app.py`, and friends), which is kept for reference and is not on the mainline path.

## Is a large language model required?

No. Retrieval and generation are separate layers. Without `LLM_API_KEY` (or with `DISABLE_LLM=true`) the generation layer falls back to an extractive answer (`mode` = `extractive`) and never errors out because a model is missing. The LLM is an optional enhancement reached through stdlib `urllib` against an OpenAI-compatible endpoint — no vendor SDK.

## Does "semantic search" mean vector embeddings?

No. There is no dense-vector model. The "semantic" channel is hand-written sparse TF-IDF with cosine similarity. This is an honest limit, not a shortcut in the docs: the evaluation section of the README explains why lexical gaps (symptom phrasing vs. document phrasing) hurt real colloquial queries.

## Why does the README show 97% Recall but also 50-68% for colloquial queries?

Both are real and come from different eval layers. The high number is from synthetic queries derived from document fields (they share rare terms with the answer). The lower number is from human-written colloquial questions in `data/human_questions.json`, measured by `python src/pure_rag/human_eval.py`. The gap is the point: synthetic evaluation overstates usability, which is why the `human_like` layer is kept in the eval set.

## Where do the numbers in the README come from?

They are generated, never hand-written. `python src/rag_cli.py eval --report` produces the report, then `python src/gen_readme_facts.py` (and `--readme README.zh-CN.md` for the Chinese edition) writes the `AUTO:FACTS` block from `report.json` / `meta.json` / `knowledge_graph.json` / unittest discovery / `pure_rag.__version__`. CI runs `--check` to fail on drift.

## Can I add my own documents?

Yes. Put files into `data/user_docs/` and re-run `build`, or use `python src/rag_cli.py add <file/dir>` to append incrementally (only new documents are chunked and tokenized; existing ones reuse persisted token sequences). Supported formats: txt, md, json, csv, and docx (parsed with `zipfile`, no python-docx). Text files are read as UTF-8 (with BOM) or GB18030.

## Is the corpus real wind-farm data?

Partly. Farm names come from the public OpenWindSCADA dataset's real sites, models use real designations, and the component/fault vocabulary follows the project knowledge graph. Everything else (unit ids, timestamps, symptoms, downtime, numbers) is template-generated via `src/simulate_corpus.py` to make the sample size large and the corpus noisy — it is not a measured distribution and should not be treated as real operational data.

## Why does `eval` not call the LLM even when one is configured?

Because `eval`, `ablation`, and `tune` force extractive generation by default. Metrics must be reproducible, and thousands of evaluation queries must not burn free-tier quota or be polluted by non-deterministic generation. Pass `eval --use-llm` if you explicitly want the model in the loop.

## How is the project versioned?

The kernel version is the single source of truth in `pure_rag.__version__`; the README `AUTO:FACTS` block and `stats` read it from there, and it is never hand-written. Release history is tracked in `CHANGELOG.md` following Keep a Changelog.

## How do I run the tests?

```bash
python -m unittest tests.test_pure_rag -v
```

The count is auto-reported in the README; do not hand-copy it.
