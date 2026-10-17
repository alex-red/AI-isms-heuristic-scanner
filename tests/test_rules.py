import pytest

from ai_isms import analyze
from ai_isms.rules import RULES


def ids(text, **kw):
    return {f.rule_id for f in analyze(text, **kw).findings}


def test_catalog_has_37_rules_across_six_dimensions():
    assert len(RULES) == 37
    assert len({r.category for r in RULES.values()}) == 6


@pytest.mark.parametrize("rule_id,text", [
    ("R01", "It's not about speed. It's about transformation."),
    ("R01", "AI isn't just changing how we work—it's reshaping what's possible."),
    ("R02", "We shipped the change last week. The result? Latency dropped across the board."),
    ("R04", "Here's what most people miss: the cache was never warm."),
    ("R05", "We rewrote the scheduler over two quarters and cut costs by a third.\n\nAnd that changes everything."),
    ("L03", "In today's rapidly evolving digital landscape, teams need tools."),
    ("L02", "Our robust, scalable, and innovative platform helps."),
    ("L06", "This feature is a double-edged sword for the team."),
    ("T01", "Great question! The answer is 42."),
    ("T04", "I'll be honest, the migration was painful."),
    ("T05", "This could potentially reduce costs in some cases."),
    ("T06", "Let's dive in and configure the server."),
    ("D04", "We did this in order to reduce latency."),
    ("C02", "This design matters because it is important."),
    ("C07", "To succeed, stay curious and embrace change."),
])
def test_rule_fires(rule_id, text):
    assert rule_id in ids(text)


@pytest.mark.parametrize("rule_id,text", [
    ("R01", "The service is not ready yet because the database migration failed on Tuesday."),
    ("R02", "Should we cache the result? We measured it, and the cache saved 40 ms per request on average, so yes."),
    ("L01", "We use a robust retry policy."),  # a single word is never flagged
    ("S01", "One dash — here. Then plain sentences follow for a while and nothing else happens at all in this text."),
])
def test_rule_does_not_fire(rule_id, text):
    assert rule_id not in ids(text)


def test_staccato_and_uniform_rhythm():
    stacc = "We tried. It failed. We retried. It failed again. Then we read the logs carefully and found the bug."
    assert "S02" in ids(stacc + " " + stacc)
    uniform = " ".join(["The team reviewed every open ticket this week."] * 2 + [
        "The service handled more traffic than last month.", "The new cache reduced load on the main database.",
        "The deploy pipeline now runs in under ten minutes.", "The on-call rotation added two more engineers today.",
        "The dashboard shows error rates for every single region.", "The budget review is scheduled for next Tuesday morning.",
        "The design doc covers the migration plan in detail.", "The release notes list every change we shipped.",
    ])
    assert "S03" in ids(uniform)


def test_em_dash_density():
    text = " ".join(["The plan — such as it was — worked for a while, then stalled."] * 4)
    assert "S01" in ids(text)


def test_code_blocks_are_ignored():
    text = "Run the script below.\n\n```python\n# In today's fast-paced world, delve into the tapestry\nprint('Great question!')\n```\n"
    assert ids(text) == set()


def test_disable_and_only():
    text = "Great question! It's not about speed. It's about transformation."
    assert "T01" not in ids(text, disabled=["T01"])
    assert ids(text, only=["R01"]) == {"R01"}


def test_profile_can_disable_rule():
    text = "Let's dive in and configure the server. Feel free to ask questions."
    assert "T06" in ids(text, profile="general")
    assert "T06" not in ids(text, profile="casual")


def test_findings_have_valid_spans(formulaic):
    r = analyze(formulaic)
    for f in r.findings:
        assert f.spans and all(0 <= s.start < s.end <= len(formulaic) for s in f.spans)
        assert 1 <= f.severity <= 5 and 0 <= f.confidence <= 1
        assert f.id


def test_human_text_is_clean(specific):
    assert analyze(specific).findings == []


def test_quoted_mentions_are_not_uses():
    text = 'Avoid openers like "In today\'s rapidly evolving landscape" and templates like “It\'s not X, it\'s Y”.'
    assert ids(text) == set()


def test_bold_label_paragraph_is_a_heading():
    from ai_isms.text import parse

    doc = parse("**Setup** (optional)\n\nInstall the package first.")
    assert doc.paragraphs[0].kind == "heading"
