"""
WindRAG README banner generator (standard library only).

Purpose:
    Render docs/assets/banner.svg, the README hero strip: project name, one-line
    positioning, a wind-turbine motif, and live stat chips. All in-image text is
    English (GitHub SVG Chinese fonts are unreliable; Chinese captions live in the
    README body instead).

Data source (read at runtime, never hard-coded):
    - pure_rag.__version__                     kernel version
    - unittest discovery of tests/test_pure_rag.py  test count
    - data/pure_index/meta.json                corpus docs / chunks
    - data/pure_index/report.json              vocabulary + answer-layer Recall@3
    Regenerate the report first if needed: `python src/rag_cli.py eval --report`

Dependencies: Python 3.10+ standard library only (json, pathlib, sys, unittest).

Run:
    python scripts/visualization/generate_banner.py
    python scripts/visualization/generate_banner.py --check

Output:
    docs/assets/banner.svg
"""

import argparse
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT / "src"))

# Brand palette (shared with the README hero / badges visual language).
BG_TOP = "#0b1220"
BG_BOTTOM = "#123047"
CYAN = "#22d3ee"
TEAL = "#34d399"
WHITE = "#e5f0f5"
MUTED = "#8fb2c4"
CHIP_BG = "#0e1c2b"

ANSWER_LAYERS = ("qa_verbatim", "qa_rewrite", "fault_case", "doc_title")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"[banner] missing source file {path}; run: python src/rag_cli.py eval --report"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_stats() -> dict:
    from pure_rag import __version__

    meta = _load_json(ROOT / "data" / "pure_index" / "meta.json")
    report = _load_json(ROOT / "data" / "pure_index" / "report.json")

    suite = unittest.TestLoader().discover(
        str(ROOT / "tests"), pattern="test_pure_rag.py", top_level_dir=str(ROOT)
    )
    layers = report["layers"]["layers"]
    answers = sum(b["answer_count"] for n, b in layers.items() if n in ANSWER_LAYERS) or 1
    hits = sum(b["hits"] for n, b in layers.items() if n in ANSWER_LAYERS)

    return {
        "version": __version__,
        "docs": int(meta["source_count"]),
        "chunks": int(meta["chunk_count"]),
        "vocabulary": int(report["stats"]["vocabulary"]),
        "recall": hits / answers,
        "tests": suite.countTestCases(),
    }


def _chip(x: int, label: str, value: str) -> str:
    """One stat chip: value on top, small label below, inside a rounded rect."""
    width = max(len(value), len(label)) * 11 + 34
    return (
        f'  <g transform="translate({x},0)">'
        f'<rect x="0" y="150" width="{width}" height="64" rx="14" fill="{CHIP_BG}" '
        f'stroke="#1e3a52" stroke-width="1.5"/>'
        f'<text x="{width/2:.0f}" y="180" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="22" font-weight="700" fill="{CYAN}">{value}</text>'
        f'<text x="{width/2:.0f}" y="202" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" letter-spacing="1" fill="{MUTED}">{label}</text>'
        f'</g>\n'
    )


def render(stats: dict) -> str:
    w, h = 1200, 240
    recall = f"{stats['recall']*100:.2f}%"
    version = f"v{stats['version']}"
    tests = f"{stats['tests']} tests"
    deps = "0 third-party deps"
    corpus = f"{stats['docs']} docs"

    # A simple wind-turbine mark: mast + three blades from a hub.
    turbine = (
        f'<g transform="translate(96,88)">'
        f'<circle cx="0" cy="0" r="12" fill="{CYAN}"/>'
        f'<rect x="-3" y="0" width="6" height="92" rx="3" fill="{MUTED}"/>'
        f'<g stroke="{CYAN}" stroke-width="12" stroke-linecap="round">'
        f'<line x1="0" y1="0" x2="0" y2="-52"/>'
        f'<line x1="0" y1="0" x2="45" y2="26"/>'
        f'<line x1="0" y1="0" x2="-45" y2="26"/>'
        f'</g></g>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="WindRAG banner">\n'
        f'  <defs>'
        f'<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{BG_TOP}"/><stop offset="1" stop-color="{BG_BOTTOM}"/>'
        f'</linearGradient></defs>\n'
        f'  <rect width="{w}" height="{h}" rx="22" fill="url(#bg)"/>\n'
        f'  {turbine}\n'
        f'  <text x="170" y="86" font-family="Segoe UI, Arial, sans-serif" font-size="52" '
        f'font-weight="800" fill="{WHITE}">WindRAG</text>\n'
        f'  <text x="172" y="120" font-family="Segoe UI, Arial, sans-serif" font-size="20" '
        f'fill="{TEAL}">Wind-turbine O&amp;M RAG on the pure Python standard library</text>\n'
        + _chip(170, "kernel", version)
        + _chip(320, "runtime deps", deps)
        + _chip(560, "answer Recall@3", recall)
        + _chip(760, "corpus", corpus)
        + _chip(900, "unit tests", tests)
        + '</svg>\n'
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate or check the project banner")
    parser.add_argument("--check", action="store_true", help="fail when the generated banner drifts")
    args = parser.parse_args(argv)

    out_dir = ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "banner.svg"
    expected = render(collect_stats())
    if args.check:
        if not out.exists():
            print(f"[banner] missing generated asset {out}")
            return 1
        if out.read_text(encoding="utf-8") == expected:
            print("[banner] generated asset matches the real sources")
            return 0
        print("[banner] generated asset drifted; run `python scripts/visualization/generate_banner.py`")
        return 1
    out.write_text(expected, encoding="utf-8")
    print(f"[banner] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
