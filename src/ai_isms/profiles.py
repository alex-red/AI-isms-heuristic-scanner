"""Genre profiles: per-rule severity multipliers and threshold overrides.

A multiplier of 0 disables a rule for that genre. Thresholds use the
``RULE.key`` names read by rules via ``ctx.t()``.
"""

from __future__ import annotations

PROFILES: dict[str, dict] = {
    "general": {
        "description": "Balanced defaults for essays, posts and general prose.",
        "scale": {},
        "thresholds": {},
    },
    "chat": {
        "description": "AI assistant replies. Stricter on validation, familiarity and formatting habits.",
        "scale": {"T01": 1.3, "T06": 1.2, "S05": 1.2, "S06": 1.2, "D03": 1.2},
        "thresholds": {"S05.bold_per100": 2.0},
    },
    "technical": {
        "description": "Docs, READMEs, specs. Headings, lists and bold are expected; chattiness is not.",
        "scale": {"S05": 0.4, "S06": 0.4, "S07": 0.6, "D03": 0.7, "C04": 0.8, "T06": 1.5, "T03": 1.3, "R03": 0.7},
        "thresholds": {"S05.bold_per100": 4.0, "S05.words_per_heading": 30, "S06.share": 0.85, "S03.cv": 0.3},
    },
    "academic": {
        "description": "Papers and reports. Hedging and recaps are conventional; hype and familiarity are not.",
        "scale": {"T05": 0.4, "D03": 0.5, "L05": 0.6, "D04": 0.7, "T06": 1.5, "T03": 1.4, "C05": 1.2},
        "thresholds": {"S03.cv": 0.3},
    },
    "marketing": {
        "description": "Landing pages and announcements. Some enthusiasm is the genre; empty claims still count.",
        "scale": {"T03": 0.6, "S05": 0.7, "S02": 0.8, "L02": 1.2, "C03": 1.1},
        "thresholds": {"T03.bang_per100": 2.0},
    },
    "creative": {
        "description": "Fiction and personal essays. Rhythm, metaphor and fragments are craft choices here.",
        "scale": {"S02": 0.4, "R03": 0.5, "L06": 0.5, "S03": 0.7, "T02": 0.7, "C04": 0.5, "T03": 0.6, "S01": 0.6,
                  "C01": 0.6, "C06": 0.5, "R02": 0.7},
        "thresholds": {"S01.per250": 5.0},
    },
    "casual": {
        "description": "Messages, forum posts, social. Informality is fine.",
        "scale": {"T06": 0.0, "T04": 0.5, "T03": 0.6, "L04": 0.6, "S01": 0.8, "C04": 0.6},
        "thresholds": {},
    },
}


def get_profile(name: str) -> dict:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}") from None
