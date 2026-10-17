"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .engine import analyze
from .profiles import PROFILES
from .report import to_html, to_json, to_terminal
from .rules import RULES


def _parse_thresholds(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"--set expects KEY=VALUE, got {it!r}")
        out[k.strip()] = float(v)
    return out


def _list_rules() -> str:
    lines = []
    for r in sorted(RULES.values(), key=lambda r: (r.category, r.id)):
        flag = " [LLM only]" if r.llm_only else ""
        lines.append(f"{r.id}  {r.name}{flag}\n      {r.category} · severity {r.severity} · {r.detector}\n      {r.description}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-isms",
        description="ESLint for prose: find AI-isms and low-quality writing patterns, with an interactive HTML report.",
    )
    p.add_argument("inputs", nargs="*", help="Text/Markdown files to scan, or '-' for stdin (default: stdin).")
    p.add_argument("--html", metavar="PATH", nargs="?", const="", help="Write an interactive HTML report "
                   "(default name: <input>.ai-isms.html).")
    p.add_argument("--open", action="store_true", help="Open the HTML report in a browser.")
    p.add_argument("--json", metavar="PATH", nargs="?", const="-", help="Write the JSON report (to stdout by default).")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress the terminal summary.")
    p.add_argument("-n", "--max-findings", type=int, default=12, help="Findings to list in the terminal (default 12).")
    p.add_argument("-p", "--profile", default="general", choices=sorted(PROFILES), help="Genre profile (default: general).")
    p.add_argument("--disable", default="", help="Comma-separated rule ids to skip, e.g. S01,L04.")
    p.add_argument("--only", default="", help="Comma-separated rule ids to run exclusively.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a rule threshold, e.g. --set S01.per250=5 (repeatable).")
    g = p.add_argument_group("LLM assist (optional; pip install 'ai-isms[llm]')")
    g.add_argument("--llm", action="store_true", help="Add a semantic pass with a Claude model.")
    g.add_argument("--model", default=None, help="Model id (default: claude-haiku-5-5).")
    g.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"], help="Model effort (default: low).")
    g.add_argument("--no-review", action="store_true", help="Don't ask the LLM to review local findings.")
    p.add_argument("--fail-above", type=float, metavar="SCORE", help="Exit with status 2 if an overall score exceeds SCORE (for CI).")
    p.add_argument("--list-rules", action="store_true", help="Print the rule catalog and exit.")
    p.add_argument("--version", action="version", version=f"ai-isms {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_rules:
        print(_list_rules())
        return 0

    llm_opts = None
    if args.llm:
        from .llm import DEFAULT_MODEL, LLMOptions

        llm_opts = LLMOptions(model=args.model or DEFAULT_MODEL, effort=args.effort, review_local=not args.no_review)

    inputs = args.inputs or ["-"]
    thresholds = _parse_thresholds(args.set)
    disabled = [x for x in args.disable.split(",") if x.strip()]
    only = [x for x in args.only.split(",") if x.strip()] or None
    worst = 0.0
    for src in inputs:
        if src == "-":
            if sys.stdin.isatty():
                print("Reading from stdin (Ctrl-D to finish)…", file=sys.stderr)
            text, name = sys.stdin.read(), "stdin"
        else:
            path = Path(src)
            if not path.is_file():
                print(f"ai-isms: no such file: {src}", file=sys.stderr)
                return 1
            text, name = path.read_text(encoding="utf-8", errors="replace"), path.name
        if not text.strip():
            print(f"ai-isms: {name} is empty", file=sys.stderr)
            continue

        report = analyze(text, profile=args.profile, disabled=disabled, only=only, thresholds=thresholds,
                         llm=llm_opts, source=name)
        worst = max(worst, report.scores["overall"] or 0.0)

        if not args.quiet and args.json != "-":
            print(to_terminal(report, max_findings=args.max_findings), flush=True)
            if len(inputs) > 1:
                print()
        if args.json is not None:
            payload = to_json(report)
            if args.json == "-":
                print(payload)
            else:
                out = Path(args.json if len(inputs) == 1 else f"{Path(name).stem}.ai-isms.json")
                out.write_text(payload, encoding="utf-8")
                print(f"JSON report → {out}", file=sys.stderr)
        if args.html is not None or args.open:
            if args.html and len(inputs) == 1:
                out = Path(args.html)
            else:
                base = Path(src).with_suffix("") if src != "-" else Path("stdin")
                out = Path(f"{base}.ai-isms.html")
            out.write_text(to_html(report), encoding="utf-8")
            print(f"HTML report → {out}", file=sys.stderr)
            if args.open:
                webbrowser.open(out.resolve().as_uri())

    if args.fail_above is not None and worst > args.fail_above:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
