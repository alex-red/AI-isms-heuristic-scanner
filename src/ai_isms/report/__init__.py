"""Reporters: self-contained interactive HTML, terminal summary, JSON."""

from __future__ import annotations

import json
import os
import shutil
import sys
from importlib import resources

from ..models import DIMENSIONS, Report

_PLACEHOLDER = "/*__AI_ISMS_DATA__*/"


def to_json(report: Report, indent: int | None = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)


def to_html(report: Report) -> str:
    template = resources.files(__package__).joinpath("template.html").read_text(encoding="utf-8")
    data = json.dumps(report.to_dict(), ensure_ascii=False, separators=(",", ":"))
    # Safe embedding inside <script type="application/json">.
    data = data.replace("</", "<\\/").replace("<!--", "<\\!--").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace(_PLACEHOLDER, data, 1)


# ---------------------------------------------------------------------------
# Terminal

_COLORS = {
    "semantic_emptiness": "31", "repetition": "33", "formulaic_rhetoric": "35",
    "structure_rhythm": "36", "generic_vocabulary": "34", "performative_tone": "32",
}


class _Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def __call__(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s


def _use_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def to_terminal(report: Report, *, max_findings: int = 12, stream=None) -> str:
    stream = stream or sys.stdout
    c = _Style(_use_color(stream))
    width = min(shutil.get_terminal_size((100, 20)).columns, 110)
    S, M = report.scores, report.metrics
    lines: list[str] = []
    title = report.config.get("source") or "input"
    score = "—" if S["overall"] is None else f"{S['overall']:.0f}/100"
    lines.append(c("1", f"AI-isms · {title}"))
    lines.append(f"{c('1', score)}  {c('1;31' if (S['overall'] or 0) >= 45 else '1;33' if (S['overall'] or 0) >= 25 else '1;32', S['band'])}"
                 f"  {c('2', S['band_blurb'])}")
    mode = "local"
    if report.llm:
        mode = f"local + {report.llm.get('model')}" if report.llm.get("ok") else "local (LLM failed)"
    lines.append(c("2", f"{M['words']} words · {M['sentences']} sentences · profile {report.config['profile']} · {mode} · "
                        f"{M['elapsed_ms']} ms"))
    if report.llm and report.llm.get("summary"):
        lines.append("")
        lines.append(c("3", report.llm["summary"]))
    for e in (report.llm or {}).get("errors", []):
        lines.append(c("31", f"⚠ {e}"))
    lines.append("")

    bar_w = 24
    for key, meta in DIMENSIONS.items():
        d = S["dimensions"][key]
        filled = round(d["score"] / 100 * bar_w)
        bar = c(_COLORS[key], "█" * filled) + c("2", "░" * (bar_w - filled))
        est = c("2", " est.") if d["estimated"] else ""
        lines.append(f"  {meta['label'][:30]:<30} {bar} {d['score']:>5.1f}  {c('2', str(d['count']) + ' findings')}{est}")
    nc = S.get("noncontributing_share", 0)
    lines.append(c("2", f"  non-contributing sentences: {nc:.0%}"))
    lines.append("")

    active = [f for f in report.findings if not (f.review and f.review.get("verdict") == "dismiss")]
    active.sort(key=lambda f: -(f.severity * f.confidence * f.weight))
    if not active:
        lines.append(c("32", "  No findings. Clean copy."))
    for f in active[:max_findings]:
        rule = report.rules[f.rule_id]
        excerpt = " … ".join(" ".join(report.text[s.start:s.end].split()) for s in f.spans[:2])
        if len(excerpt) > width - 14:
            excerpt = excerpt[: width - 15] + "…"
        sev = "●" * f.severity + c("2", "○" * (5 - f.severity))
        tag = c("35", " LLM") if f.detector == "llm" else ""
        lines.append(f"  {c(_COLORS[f.category] + ';1', f.rule_id)} {sev} {c('1', f.message if f.message != rule.name else rule.name)}{tag}")
        lines.append(f"      {c('2', '“' + excerpt + '”')}")
        if f.suggestion:
            lines.append(f"      {c('2', '→ ' + f.suggestion)}")
    if len(active) > max_findings:
        lines.append(c("2", f"  … {len(active) - max_findings} more (see --html report)"))
    return "\n".join(lines)
