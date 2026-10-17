"""Rule registry.

A rule is a function ``check(doc, ctx) -> list[Finding]`` registered with the
``@rule`` decorator together with its metadata. Rules are grouped into modules by
scoring dimension. Adding a rule is a matter of writing one function; see
CONTRIBUTING.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..models import Finding, RuleMeta, Span
from ..text import Document

CheckFn = Callable[[Document, "Context"], list[Finding]]

RULES: dict[str, RuleMeta] = {}
CHECKS: dict[str, CheckFn] = {}


@dataclass
class Context:
    """Per-run configuration visible to rules."""

    profile: str = "general"
    severity_scale: dict[str, float] = field(default_factory=dict)  # rule_id -> multiplier
    thresholds: dict[str, float] = field(default_factory=dict)  # "S01.per250" -> value
    disabled: set[str] = field(default_factory=set)

    def t(self, key: str, default: float) -> float:
        return self.thresholds.get(key, default)


def rule(
    id: str,
    name: str,
    category: str,
    severity: int,
    description: str,
    suggestion: str = "",
    action: str = "review",
    detector: str = "pattern",
    llm_only: bool = False,
) -> Callable[[CheckFn], CheckFn]:
    meta = RuleMeta(id, name, category, severity, description, suggestion, action, detector, llm_only)

    def deco(fn: CheckFn) -> CheckFn:
        if id in RULES:
            raise ValueError(f"duplicate rule id {id}")
        RULES[id] = meta
        if not llm_only:
            CHECKS[id] = fn
        return fn

    return deco


def make(
    rule_id: str,
    spans: list[tuple[int, int]] | list[Span],
    *,
    message: str | None = None,
    severity: int | None = None,
    confidence: float = 0.7,
    scope: str = "span",
    evidence: str = "",
    suggestion: str | None = None,
) -> Finding:
    """Build a Finding pre-filled from rule metadata."""
    meta = RULES[rule_id]
    sp = [s if isinstance(s, Span) else Span(*s) for s in spans]
    return Finding(
        rule_id=rule_id,
        category=meta.category,
        message=message or meta.name,
        spans=sp,
        severity=max(1, min(5, severity if severity is not None else meta.severity)),
        confidence=round(max(0.0, min(1.0, confidence)), 2),
        scope=scope,
        suggestion=suggestion if suggestion is not None else meta.suggestion,
        action=meta.action,
        detector=meta.detector,
        evidence=evidence,
    )


def llm_rule(id: str, name: str, category: str, severity: int, description: str, suggestion: str, action: str) -> None:
    """Register a rule that only the LLM pass can produce."""
    rule(id, name, category, severity, description, suggestion, action, "llm", llm_only=True)(lambda d, c: [])


# Import rule modules so they register themselves.
from . import rhetoric, lexical, structure, emptiness, redundancy, tone  # noqa: E402,F401
