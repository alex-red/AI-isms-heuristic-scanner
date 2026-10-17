"""Core data structures shared by the engine, scorer, LLM pass and reporters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Scoring dimensions and their default weights (sum to 1.0).
# Starting weights from the framework, not empirically validated.
DIMENSIONS: dict[str, dict[str, Any]] = {
    "semantic_emptiness": {
        "label": "Semantic emptiness",
        "weight": 0.30,
        "blurb": "Vague claims, low information density, tautology, unsupported generalizations",
    },
    "repetition": {
        "label": "Repetition & redundancy",
        "weight": 0.20,
        "blurb": "Repeated ideas, unnecessary restatements, circular explanation",
    },
    "formulaic_rhetoric": {
        "label": "Formulaic rhetoric",
        "weight": 0.15,
        "blurb": "Artificial contrasts, question-answer hooks, rhetorical templates",
    },
    "structure_rhythm": {
        "label": "Mechanical structure & rhythm",
        "weight": 0.15,
        "blurb": "Predictable cadence, repetitive paragraph structures, unnatural formatting",
    },
    "generic_vocabulary": {
        "label": "Generic vocabulary & clichés",
        "weight": 0.10,
        "blurb": "Corporate buzzwords, stock transitions, decorative metaphors",
    },
    "performative_tone": {
        "label": "Performative tone",
        "weight": 0.10,
        "blurb": "Unsolicited praise, contrived profundity, exaggerated emotional framing",
    },
}

ACTIONS = ("remove", "clarify", "consolidate", "substantiate", "vary", "review")
SCOPES = ("span", "sentence", "paragraph", "section", "document")


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def overlaps(self, other: "Span") -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return max(0, self.end - self.start)


@dataclass
class Finding:
    rule_id: str
    category: str
    message: str
    spans: list[Span]
    severity: int  # 1-5: how much the issue affects quality
    confidence: float  # 0-1: confidence in this rule match (NOT that AI wrote it)
    scope: str = "span"
    suggestion: str = ""
    action: str = "review"
    detector: str = "pattern"  # pattern | statistical | lexical | semantic-local | llm
    evidence: str = ""
    # Filled in by the merger / LLM review.
    id: str = ""
    weight: float = 1.0  # dedup factor applied in scoring
    related: list[str] = field(default_factory=list)
    review: dict[str, str] | None = None

    def to_dict(self, text: str | None = None) -> dict[str, Any]:
        d = asdict(self)
        d["spans"] = [{"start": s.start, "end": s.end} for s in self.spans]
        if text is not None and self.spans:
            d["excerpt"] = " … ".join(text[s.start : s.end] for s in self.spans[:3])
        return d


@dataclass(frozen=True)
class RuleMeta:
    id: str
    name: str
    category: str
    severity: int
    description: str
    suggestion: str
    action: str
    detector: str
    llm_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SentenceNote:
    """Per-sentence semantic contribution: 2 substantive, 1 supporting, 0 noncontributing."""

    index: int
    start: int
    end: int
    contribution: int
    source: str  # "local" (estimated) or "llm"
    reason: str = ""
    words: int = 0


@dataclass
class Report:
    text: str
    findings: list[Finding]
    sentences: list[SentenceNote]
    metrics: dict[str, Any]
    scores: dict[str, Any]
    rules: dict[str, RuleMeta]
    config: dict[str, Any]
    llm: dict[str, Any] | None = None
    paragraphs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "text": self.text,
            "config": self.config,
            "scores": self.scores,
            "metrics": self.metrics,
            "findings": [f.to_dict(self.text) for f in self.findings],
            "sentences": [asdict(s) for s in self.sentences],
            "rules": {k: v.to_dict() for k, v in self.rules.items()},
            "dimensions": DIMENSIONS,
            "llm": self.llm,
            "paragraphs": self.paragraphs,
        }
