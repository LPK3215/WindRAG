# Contributing to WindRAG

Thanks for your interest in WindRAG! Please read this guide before contributing.

Language: this file is the English main version; a Chinese companion lives at [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md).

## Development environment

Read [README.md](README.md) ([中文](README.zh-CN.md)) for the architecture and run steps first.

The pure-Python retrieval kernel (`src/pure_rag/`) has zero third-party dependencies: a plain Python 3.10+ interpreter is enough. You do **not** need `pip install -r requirements.txt` — that file is only for the heavy research version.

## Branch strategy

- This is a single-maintainer project; **all changes are committed directly to `main`**. No `feature/*` or `fix/*` branches, no pull-request round-trip.
- If a branch is genuinely necessary, state why up front and delete it right after merging so no stray branches are left behind.

## Commit convention

| Prefix | Use |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `perf:` | Performance |
| `refactor:` | Refactoring |
| `chore:` | Build / tooling |
| `test:` | Tests |

Keep the subject under 50 characters and say only what changed; put the "why and how" in the commit body.

## Pre-commit self-check

```bash
python -m unittest tests.test_pure_rag -v        # unit tests
python src/simulate_corpus.py                   # regenerate the deterministic corpus
python src/rag_cli.py build                     # index builds cleanly
python src/rag_cli.py eval --report --min-recall 0.9
python src/rag_cli.py ablation                  # refresh the overview source report
python src/gen_readme_facts.py --check           # README figures match the real sources
python src/gen_readme_facts.py --check --readme README.zh-CN.md
python scripts/visualization/generate_overview.py --check
```

These mirror the `.cnb.yml` pipeline, which re-runs corpus generation, unit tests, evaluation, ablation, and the generated-documentation gate on every push to `main`.

## Code style

- Python 3.10+ style, follow PEP 8.
- Do not commit `config.env` or any API key; keys live only in the gitignored config file.
- The mainline (`src/pure_rag/`, `src/rag_cli.py`) must stay standard-library-only; never import the heavy research-version modules on the pure-Python path.
- Index directories (`data/pure_index/`, `data/processed/`) and `data/chat_logs/` are regenerable and not committed.
- Do not let build-environment trash directories (`.Trash-0/`) slip into the repo.

## Documentation numbers

Corpus size, chunk counts, vocabulary, eval-item counts, metrics, and test counts must never be hand-written. Generate README values with `python src/gen_readme_facts.py` and dashboard values with `python scripts/visualization/generate_overview.py` (use each command's `--check` mode to detect drift); after the corpus changes, refresh the docs.
