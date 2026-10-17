"""Turn findings into dimension scores and an overall 0-100 AI-ism score.

Higher = more AI-isms. Scores are *illustrative* until calibrated against a
human-rated corpus; the findings are the product, the number is for comparison.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .models import DIMENSIONS, Finding, SentenceNote

MIN_WORDS_FOR_SCORE = 80
SEV_POINTS = {1: 1.0, 2: 2.2, 3: 3.8, 4: 5.8, 5: 8.0}
K = 8.0  # saturation constant: points at which a dimension reaches ~63

BANDS = [
    (12, "Clean", "Few or no AI-isms detected"),
    (25, "Light", "Isolated patterns; mostly stylistic"),
    (45, "Noticeable", "Repeated patterns materially affect quality"),
    (65, "Heavy", "Pervasive formulaic writing"),
    (101, "Saturated", "Reads like unedited AI output"),
]


def band(score: float) -> tuple[str, str]:
    for limit, name, blurb in BANDS:
        if score < limit:
            return name, blurb
    return BANDS[-1][1], BANDS[-1][2]


def finding_points(f: Finding, words: int) -> float:
    base = SEV_POINTS.get(f.severity, 1.0) * f.confidence * f.weight
    if f.scope == "document":
        return base * 1.2
    return base / max(words / 100.0, 1.0)


def score(findings: list[Finding], sentences: list[SentenceNote], words: int, *, llm: bool) -> dict[str, Any]:
    pts: dict[str, float] = defaultdict(float)
    by_rule: dict[str, float] = defaultdict(float)
    for f in findings:
        p = finding_points(f, words)
        pts[f.category] += p
        by_rule[f.rule_id] += p

    scored = [s for s in sentences if s.words >= 3]
    zero_share = sum(1 for s in scored if s.contribution == 0) / len(scored) if scored else 0.0
    pts["semantic_emptiness"] += 12.0 * zero_share * (1.0 if llm else 0.5)

    dims = {}
    for key, meta in DIMENSIONS.items():
        val = 100.0 * (1.0 - math.exp(-pts[key] / K))
        dims[key] = {
            "score": round(val, 1),
            "points": round(pts[key], 2),
            "weight": meta["weight"],
            "count": sum(1 for f in findings if f.category == key),
            "estimated": key in ("semantic_emptiness", "repetition") and not llm,
        }

    overall = sum(d["score"] * d["weight"] for d in dims.values())
    graded = words >= MIN_WORDS_FOR_SCORE
    name, blurb = band(overall)
    top = sorted(by_rule.items(), key=lambda kv: -kv[1])[:5]
    return {
        "overall": round(overall, 1) if graded else None,
        "band": name if graded else "Ungraded",
        "band_blurb": blurb if graded else f"Too short to grade reliably (<{MIN_WORDS_FOR_SCORE} words); findings shown only.",
        "dimensions": dims,
        "top_rules": [{"rule_id": r, "points": round(p, 2)} for r, p in top if p > 0],
        "noncontributing_share": round(zero_share, 3),
        "confidence": "low" if words < 200 else ("medium" if not llm else "high"),
        "bands": [{"max": lim, "name": n, "blurb": b} for lim, n, b in BANDS],
    }
