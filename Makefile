# WindRAG convenience targets (mainline is pure Python, zero third-party deps).
# Every target just calls `python src/rag_cli.py ...`; nothing here pip-installs.
# Run `make help` to list targets.

PY ?= python
PORT ?= 8000
Q ?= gearbox overheating handling

.PHONY: help corpus build ask chat explain stats serve eval eval-report eval-negative robust tune ablation test facts check clean

help: ## Show this help
	@echo "WindRAG targets:"; \
	grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

corpus: ## Generate the deterministic scaled test corpus (65 -> 1269 docs)
	$(PY) src/simulate_corpus.py

build: ## Build the index (tokenize + BM25 + TF-IDF + new-word discovery)
	$(PY) src/rag_cli.py build

ask: ## Single-shot QA (override the question with Q="...")
	$(PY) src/rag_cli.py ask "$(Q)" --show-docs

chat: ## Interactive multi-turn QA
	$(PY) src/rag_cli.py chat --log

explain: ## Per-term retrieval score breakdown for Q
	$(PY) src/rag_cli.py explain "$(Q)"

stats: ## Index and model status (JSON)
	$(PY) src/rag_cli.py stats

serve: ## Start the zero-dependency web UI (override port with PORT=...)
	$(PY) src/rag_cli.py serve --port $(PORT)

eval: ## Quick retrieval evaluation (N via -n, default 15)
	$(PY) src/rag_cli.py eval -n 15

eval-report: ## Layered reproducible report (writes report.md + report.json)
	$(PY) src/rag_cli.py eval --report

eval-negative: ## Refusal-capability evaluation + threshold calibration
	$(PY) src/rag_cli.py eval --negative

robust: ## Robustness on synonym/keyword rewritten queries
	$(PY) src/rag_cli.py eval -n 15 --robust

tune: ## Grid-search best chunk/fusion parameters and apply
	$(PY) src/rag_cli.py tune --quick

ablation: ## Ablation table across modules
	$(PY) src/rag_cli.py ablation

test: ## Run the unit test suite
	$(PY) -m unittest tests.test_pure_rag -v

facts: ## Regenerate the AUTO:FACTS blocks in both READMEs
	$(PY) src/gen_readme_facts.py
	$(PY) src/gen_readme_facts.py --readme README.zh-CN.md

check: ## Verify README figures have not drifted from the real sources
	$(PY) src/gen_readme_facts.py --check
	$(PY) src/gen_readme_facts.py --check --readme README.zh-CN.md

clean: ## Remove regenerable index artifacts (data/pure_index, generated corpus)
	rm -rf data/pure_index/*.json data/pure_index/*.npy data/generated/*
