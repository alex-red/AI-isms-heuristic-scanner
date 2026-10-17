"""Optional semantic pass using a Claude model (default: Claude Haiku 5.5).

The LLM does three things the local engine can't do well:
  1. Rates every sentence's semantic contribution (2 substantive, 1 supporting, 0 noncontributing).
  2. Reports semantic findings (C01-C07, D01-D05, T02, T05), each citing sentence ids and an exact quote.
  3. Reviews the local findings, confirming or dismissing likely false positives.

Every LLM finding must quote the text so it can be highlighted; quotes that can't
be located fall back to whole-sentence highlights.

Two backends: the Anthropic SDK (default; requires ``pip install 'ai-isms[llm]'`` and
credentials it can resolve, such as ANTHROPIC_API_KEY or an ``ant auth login`` profile),
or ``backend="claude-cli"``, which shells out to ``claude -p`` and uses whatever login
Claude Code already has (no SDK or API key needed).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from .models import Finding, Span
from .rules import RULES
from .text import Document

DEFAULT_MODEL = "claude-haiku-5-5"
# USD per million tokens (input, output) for cost estimates in the report.
PRICING = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
SEMANTIC_RULES = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "D01", "D02", "D03", "D04", "D05", "T02", "T05"]
CHUNK_SENTENCES = 120


@dataclass
class LLMOptions:
    model: str = DEFAULT_MODEL
    effort: str = "low"
    review_local: bool = True
    max_tokens: int = 32000
    backend: str = "api"  # "api" (Anthropic SDK) or "claude-cli" (`claude -p`)
    timeout: int = 600


@dataclass
class LLMResult:
    findings: list[Finding] = field(default_factory=list)
    contributions: dict[int, tuple[int, str]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


SYSTEM_PROMPT = """You are a meticulous prose editor working inside a writing linter. You judge writing \
quality, not authorship: never speculate about whether a human or an AI wrote the text.

You will receive a document split into numbered sentences, the intended genre, and a list of findings \
produced by fast deterministic rules. Your three jobs:

1. CONTRIBUTION. For every sentence id, rate what it adds for the reader:
   2 = substantive: a distinct fact, mechanism, observation, example, decision, or useful distinction.
   1 = supporting: clarifies, connects, summarizes or contextualizes an existing idea in a useful way.
   0 = noncontributing: rephrases an earlier claim, empty rhetoric, filler, or performative framing.
   Judge in context. A brief direct answer, a needed definition, or a deliberate recap in a long document \
can be useful. Headings are not included.

2. SEMANTIC FINDINGS. Report only issues you can point to with an exact quote, using these rule ids:
{rules}
   Rules for findings:
   - "quote" must be copied character-for-character from the cited sentences (keep it short: the \
offending phrase or clause, not the whole paragraph unless the whole paragraph is the problem).
   - Do not duplicate a local finding that already covers the same text and issue; review it instead.
   - Prefer few, high-confidence findings. Polished, specific writing should get none.
   - "suggestion" is one concrete revision strategy for this specific text (not a generic tip).

3. REVIEW. For each local finding, decide: "confirm" (a real problem here), "dismiss" (a false positive \
or a legitimate stylistic choice in this context), or "uncertain". Keep notes under 20 words.

Finally write a 1-2 sentence "summary" of the text's main quality problems (or say it is clean). \
Be direct and specific; no praise, no hedging."""


def _rules_block() -> str:
    lines = []
    for rid in SEMANTIC_RULES:
        m = RULES[rid]
        lines.append(f"   {rid} {m.name}: {m.description}")
    return "\n".join(lines)


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "contribution": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "contribution", "reason"],
                    "additionalProperties": False,
                },
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "enum": SEMANTIC_RULES},
                        "sentence_ids": {"type": "array", "items": {"type": "integer"}},
                        "quote": {"type": "string"},
                        "severity": {"type": "integer"},
                        "confidence": {"type": "number"},
                        "explanation": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["rule_id", "sentence_ids", "quote", "severity", "confidence", "explanation", "suggestion"],
                    "additionalProperties": False,
                },
            },
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "finding_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["confirm", "dismiss", "uncertain"]},
                        "note": {"type": "string"},
                    },
                    "required": ["finding_id", "verdict", "note"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["sentences", "findings", "reviews", "summary"],
        "additionalProperties": False,
    }


def _build_user_message(doc: Document, sent_ids: list[int], local: list[Finding], profile: str) -> str:
    lines = [f"<genre>{profile}</genre>", "<document>"]
    for i in sent_ids:
        s = doc.sentences[i]
        lines.append(f"[S{i + 1}] {' '.join(s.text.split())}")
    lines.append("</document>")
    if local:
        lines.append("<local_findings>")
        for f in local:
            excerpt = " … ".join(" ".join(doc.text[sp.start:sp.end].split())[:120] for sp in f.spans[:3])
            lines.append(f"{f.id} | {f.rule_id} {RULES[f.rule_id].name} | {f.message} | “{excerpt}”")
        lines.append("</local_findings>")
    else:
        lines.append("<local_findings>none</local_findings>")
    lines.append("Return the JSON object described in your instructions. Sentence ids are the numbers after S.")
    return "\n".join(lines)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')).strip().lower()


def locate_quote(doc: Document, quote: str, sent_idx: list[int]) -> list[Span]:
    """Find the quote within the cited sentences; fall back to the sentences themselves."""
    sents = [doc.sentences[i] for i in sent_idx if 0 <= i < len(doc.sentences)]
    if not sents:
        return []
    lo, hi = min(s.start for s in sents), max(s.end for s in sents)
    q = quote.strip().strip('"“”')
    if q:
        pos = doc.text.find(q, lo, hi)
        if pos != -1:
            return [Span(pos, pos + len(q))]
        # Whitespace/quote-insensitive search via a character map.
        region = doc.text[lo:hi]
        norm_chars, mapping = [], []
        prev_space = False
        for k, ch in enumerate(region):
            c = _norm(ch) if not ch.isspace() else " "
            if c == " ":
                if prev_space:
                    continue
                prev_space = True
            else:
                prev_space = False
            norm_chars.append(c or " ")
            mapping.append(k)
        hay = "".join(norm_chars)
        needle = _norm(q)
        p = hay.find(needle)
        if p != -1 and needle:
            a = lo + mapping[p]
            b = lo + mapping[min(p + len(needle) - 1, len(mapping) - 1)] + 1
            return [Span(a, b)]
    return [Span(s.start, s.end) for s in sents]


def _call(client: Any, opts: LLMOptions, system: str, user: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    import anthropic

    meta: dict[str, Any] = {}
    try:
        with client.messages.stream(
            model=opts.model,
            max_tokens=opts.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": opts.effort, "format": {"type": "json_schema", "schema": _schema()}},
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        return None, {"error": "Authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`."}
    except anthropic.NotFoundError:
        return None, {"error": f"Model {opts.model!r} not found or not available to this account."}
    except anthropic.RateLimitError:
        return None, {"error": "Rate limited by the API; try again shortly."}
    except anthropic.APIStatusError as e:
        return None, {"error": f"API error {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError:
        return None, {"error": "Could not reach the Anthropic API (network error)."}

    meta["input_tokens"] = msg.usage.input_tokens
    meta["output_tokens"] = msg.usage.output_tokens
    meta["stop_reason"] = msg.stop_reason
    if msg.stop_reason == "refusal":
        meta["error"] = "The model declined to analyze this text."
        return None, meta
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return json.loads(text), meta
    except json.JSONDecodeError:
        meta["error"] = "Model output was not valid JSON" + (" (hit max_tokens)" if msg.stop_reason == "max_tokens" else "")
        return None, meta


def _call_cli(opts: LLMOptions, system: str, user: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Same contract as `_call`, but through `claude -p` with a JSON schema and no tools."""
    cmd = ["claude", "-p", "--model", opts.model, "--effort", opts.effort, "--output-format", "json",
           "--json-schema", json.dumps(_schema()), "--system-prompt", system, "--tools", "",
           "--no-session-persistence", "--disable-slash-commands"]
    try:
        # Run from an empty temp dir so no project CLAUDE.md or settings leak into the request.
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=cwd, timeout=opts.timeout)
    except FileNotFoundError:
        return None, {"error": "`claude` CLI not found on PATH."}
    except subprocess.TimeoutExpired:
        return None, {"error": f"`claude -p` timed out after {opts.timeout}s."}
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, {"error": f"`claude -p` exited {proc.returncode} without JSON: {(proc.stderr or proc.stdout)[:200].strip()}"}

    usage = out.get("usage", {})
    meta: dict[str, Any] = {
        "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": out.get("total_cost_usd"),
        "stop_reason": out.get("stop_reason"),
    }
    if out.get("is_error"):
        meta["error"] = f"`claude -p` error: {str(out.get('result', ''))[:200]}"
        return None, meta
    data = out.get("structured_output")
    if not isinstance(data, dict):
        try:
            data = json.loads(out.get("result", ""))
        except (json.JSONDecodeError, TypeError):
            meta["error"] = "Model output was not valid JSON"
            return None, meta
    return data, meta


def run_llm(doc: Document, local: list[Finding], opts: LLMOptions, *, profile: str) -> LLMResult:
    result = LLMResult(info={"model": opts.model, "effort": opts.effort, "ok": False, "errors": [], "summary": ""})
    cli = opts.backend == "claude-cli"
    client = None
    if not cli:
        try:
            import anthropic
        except ImportError:
            result.info["errors"].append("The 'anthropic' package is not installed: pip install 'ai-isms[llm]'")
            return result
        client = anthropic.Anthropic()
    result.info["backend"] = opts.backend
    system = SYSTEM_PROMPT.format(rules=_rules_block())
    scored = [s.index for s in doc.sentences if doc.paragraphs[s.para].kind != "heading"]
    if not scored:
        result.info["errors"].append("No sentences to analyze.")
        return result

    by_id = {f.id: f for f in local}
    in_tok = out_tok = 0
    cli_cost = 0.0
    summaries = []
    for c0 in range(0, len(scored), CHUNK_SENTENCES):
        chunk = scored[c0:c0 + CHUNK_SENTENCES]
        lo, hi = doc.sentences[chunk[0]].start, doc.sentences[chunk[-1]].end
        chunk_local = [f for f in local if f.spans and lo <= f.spans[0].start < hi] if opts.review_local else []
        message = _build_user_message(doc, chunk, chunk_local, profile)
        data, meta = _call_cli(opts, system, message) if cli else _call(client, opts, system, message)
        in_tok += meta.get("input_tokens", 0)
        out_tok += meta.get("output_tokens", 0)
        cli_cost += meta.get("cost_usd") or 0.0
        if data is None:
            result.info["errors"].append(meta.get("error", "unknown error"))
            continue
        valid = set(chunk)
        for row in data.get("sentences", []):
            idx = int(row.get("id", 0)) - 1
            if idx in valid:
                result.contributions[idx] = (max(0, min(2, int(row.get("contribution", 1)))), str(row.get("reason", ""))[:200])
        for row in data.get("findings", []):
            rid = row.get("rule_id")
            if rid not in RULES:
                continue
            idxs = [int(i) - 1 for i in row.get("sentence_ids", []) if int(i) - 1 in valid]
            spans = locate_quote(doc, str(row.get("quote", "")), idxs)
            if not spans:
                continue
            meta_r = RULES[rid]
            result.findings.append(Finding(
                rule_id=rid, category=meta_r.category, message=meta_r.name, spans=spans,
                severity=max(1, min(5, int(row.get("severity", meta_r.severity)))),
                confidence=round(max(0.0, min(1.0, float(row.get("confidence", 0.7)))), 2),
                scope="sentence" if len(spans) == 1 else "paragraph",
                suggestion=str(row.get("suggestion", "")) or meta_r.suggestion,
                action=meta_r.action, detector="llm", evidence=str(row.get("explanation", "")),
            ))
        for row in data.get("reviews", []):
            f = by_id.get(str(row.get("finding_id", "")))
            if f is not None:
                f.review = {"verdict": row.get("verdict", "uncertain"), "note": str(row.get("note", ""))[:200]}
        if data.get("summary"):
            summaries.append(data["summary"].strip())

    price = PRICING.get(opts.model)
    result.info.update({
        "ok": bool(result.contributions or result.findings),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cli_cost, 5) if cli else (round((in_tok * price[0] + out_tok * price[1]) / 1e6, 5) if price else None),
        "summary": " ".join(summaries),
        "reviewed": sum(1 for f in local if f.review),
        "dismissed": sum(1 for f in local if f.review and f.review.get("verdict") == "dismiss"),
    })
    return result
