from ai_isms import analyze
from ai_isms.engine import consolidate
from ai_isms.models import Finding, Span


def test_short_text_is_ungraded():
    r = analyze("Great question! It's not about speed. It's about transformation.")
    assert r.scores["overall"] is None
    assert r.scores["band"] == "Ungraded"
    assert r.findings  # findings are still reported


def test_formulaic_scores_far_above_specific(formulaic, specific):
    s, h = analyze(formulaic).scores, analyze(specific).scores
    assert s["overall"] >= 40
    assert h["overall"] <= 12
    assert set(s["dimensions"]) == set(h["dimensions"])


def test_overlapping_findings_are_downweighted():
    a = Finding("L02", "generic_vocabulary", "a", [Span(0, 20)], 3, 0.9, id="F1")
    b = Finding("C05", "semantic_emptiness", "b", [Span(2, 18)], 2, 0.6, id="F2")
    consolidate([a, b])
    assert a.weight == 1.0 and b.weight == 0.35
    assert "F1" in b.related


def test_report_serializes(formulaic):
    d = analyze(formulaic).to_dict()
    assert d["findings"][0]["excerpt"]
    assert len(d["sentences"]) > 10
    assert d["paragraphs"]
