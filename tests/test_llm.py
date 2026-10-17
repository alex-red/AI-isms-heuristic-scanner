"""LLM pass tested against a fake client (no network)."""

import json
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

from ai_isms import analyze  # noqa: E402
from ai_isms.llm import LLMOptions, locate_quote  # noqa: E402
from ai_isms.text import parse  # noqa: E402

TEXT = (
    "Consider the interface. It is intuitive. It is easy to use. "
    "It provides an intuitive experience that users find easy to use. Great question! "
    "We measured task time with 12 users and it fell from 40 to 25 seconds."
)


class FakeStream:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
        )


class FakeClient:
    def __init__(self, payload):
        self.calls = []
        self.messages = SimpleNamespace(stream=self._stream)
        self.payload = payload

    def _stream(self, **kw):
        self.calls.append(kw)
        return FakeStream(self.payload)


def test_llm_pass_maps_findings_reviews_and_contributions(monkeypatch):
    local = analyze(TEXT)
    t01 = next(f for f in local.findings if f.rule_id == "T01")
    payload = {
        "sentences": [{"id": i, "contribution": 0 if i in (2, 3, 4) else 2, "reason": "r"} for i in range(1, 7)],
        "findings": [{
            "rule_id": "D01", "sentence_ids": [2, 3, 4], "quote": "an intuitive experience",
            "severity": 4, "confidence": 0.9, "explanation": "Three claims, one idea.", "suggestion": "Keep one.",
        }],
        "reviews": [{"finding_id": t01.id, "verdict": "dismiss", "note": "test"}],
        "summary": "Repeats one idea three times.",
    }
    fake = FakeClient(payload)
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: fake)

    r = analyze(TEXT, llm=LLMOptions())
    assert r.llm["ok"] and r.llm["summary"].startswith("Repeats")
    assert r.llm["cost_usd"] == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1e6)
    call = fake.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"

    d01 = next(f for f in r.findings if f.detector == "llm")
    assert TEXT[d01.spans[0].start:d01.spans[0].end] == "an intuitive experience"
    dismissed = next(f for f in r.findings if f.rule_id == "T01")
    assert dismissed.review["verdict"] == "dismiss" and dismissed.weight == 0.0
    assert {n.source for n in r.sentences} == {"llm"}
    assert r.scores["dimensions"]["semantic_emptiness"]["estimated"] is False


def test_llm_errors_fall_back_to_local(monkeypatch):
    def boom(*a, **k):
        raise anthropic.APIConnectionError(request=None)

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: SimpleNamespace(messages=SimpleNamespace(stream=boom)))
    r = analyze(TEXT, llm=LLMOptions())
    assert r.llm["ok"] is False and r.llm["errors"]
    assert r.findings  # local findings survive


def test_locate_quote_is_whitespace_and_quote_insensitive():
    doc = parse("He said it’s   fine. Then left.")
    sp = locate_quote(doc, "it's fine", [0])
    assert doc.text[sp[0].start:sp[0].end] == "it’s   fine"
    assert locate_quote(doc, "not present", [1])[0].start == doc.sentences[1].start
