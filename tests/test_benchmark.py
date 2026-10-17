"""Sanity checks for the benchmark manifest and the committed AI-style samples (no network needed)."""

import json
from pathlib import Path

from ai_isms import analyze
from ai_isms.profiles import PROFILES

BENCH = Path(__file__).resolve().parent.parent / "benchmark"
MANIFEST = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))["samples"]
LABELS = {"human", "ai-typical", "ai-polished"}


def test_manifest_is_well_formed():
    ids = [s["id"] for s in MANIFEST]
    assert len(ids) == len(set(ids))
    for s in MANIFEST:
        assert s["label"] in LABELS
        assert s["profile"] in PROFILES
        assert ("file" in s) != ("source" in s), f"{s['id']} needs exactly one of file/source"
        assert s.get("license")


def test_committed_samples_exist_and_are_long_enough_to_grade():
    for s in MANIFEST:
        if "file" in s:
            text = (BENCH / s["file"]).read_text(encoding="utf-8")
            assert len(text.split()) >= 80, s["id"]


def test_typical_ai_samples_outscore_polished_ones():
    def mean_score(label):
        scores = [analyze((BENCH / s["file"]).read_text(encoding="utf-8"), profile=s["profile"]).scores["overall"]
                  for s in MANIFEST if s["label"] == label]
        return sum(scores) / len(scores)

    assert mean_score("ai-typical") > mean_score("ai-polished") + 5
