# Security Policy

## Reporting a vulnerability

If you find a security issue in WindRAG, please report it privately rather than opening a public issue.

- Email: 17538703215@163.com (maintainer LPK3215)
- Or use the private "Report a vulnerability" form on the GitHub repository: https://github.com/LPK3215/WindRAG

Please include a description of the problem, the affected subcommand or module, and — if possible — a minimal way to reproduce it. We aim to acknowledge reports within a few business days and will keep you informed of the fix timeline.

## Scope and relevant attack surface

The mainline (`src/pure_rag/`, `src/rag_cli.py`) runs locally with the Python standard library only and does not expose a remote service by default. Notes on the parts most worth care:

- **Web server binding**: `serve` binds to `127.0.0.1` by default. Only pass `--host 0.0.0.0` when you deliberately want LAN exposure; the web UI is not authenticated.
- **Static-file path traversal**: `/static/<file>` reads only from a fixed whitelist under `src/pure_rag/web/`; unknown paths and encoded traversals return 404. If you extend static serving, keep the whitelist approach.
- **Optional LLM egress**: when an LLM is configured, the generation layer sends question text and retrieved snippets to the OpenAI-compatible endpoint listed in `config.env`. Do not feed secrets into documents under `data/user_docs/` if a model is enabled, since relevant text may be included in the prompt.
- **Secrets handling**: API keys live only in the gitignored `config.env` / `.env`. Never commit real keys; `.env.example` and `config.env.example` are the templates.
- **Untrusted input**: query text is used for retrieval and may be echoed into answers and transcripts (`chat --log` / `/save`). Treat exported transcripts as untrusted if they came from others.

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x (mainline, pure Python) | Yes |

The heavy research version under `src/` root is kept for reference only and is not a supported deployment target.

## Dependency posture

The mainline has zero third-party runtime dependencies by design, so there is no dependency supply chain on the pure-Python path. `requirements.txt` is only for the legacy research version; keeping the mainline dependency-free is itself a security property.
