"""The `claude -p` backend, tested with a fake subprocess (no CLI, SDK or network needed)."""

import json
from types import SimpleNamespace

from ai_isms import analyze
from ai_isms.llm import LLMOptions

TEXT = (
    "Consider the interface. It is intuitive. It is easy to use. "
    "It provides an intuitive experience that users find easy to use. Great question! "
    "We measured task time with 12 users and it fell from 40 to 25 seconds."
)

PAYLOAD = {
    "sentences": [{"id": i, "contribution": 0 if i in (2, 3, 4) else 2, "reason": "r"} for i in range(1, 7)],
    "findings": [{
        "rule_id": "D01", "sentence_ids": [2, 3, 4], "quote": "an intuitive experience",
        "severity": 4, "confidence": 0.9, "explanation": "Three claims, one idea.", "suggestion": "Keep one.",
    }],
    "reviews": [],
    "summary": "Repeats one idea three times.",
}


def _fake_run(stdout, calls):
    def run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    return run


def test_cli_backend_parses_structured_output_and_cost(monkeypatch):
    out = json.dumps({
        "is_error": False, "structured_output": PAYLOAD, "total_cost_usd": 0.0025, "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 900, "cache_read_input_tokens": 0, "output_tokens": 300},
    })
    calls = []
    monkeypatch.setattr("ai_isms.llm.subprocess.run", _fake_run(out, calls))

    r = analyze(TEXT, llm=LLMOptions(backend="claude-cli", effort="medium"))
    cmd, kw = calls[0]
    assert cmd[:2] == ["claude", "-p"] and cmd[cmd.index("--effort") + 1] == "medium"
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-5-5" and cmd[cmd.index("--tools") + 1] == ""
    assert "Consider the interface" in kw["input"]

    assert r.llm["ok"] and r.llm["backend"] == "claude-cli"
    assert r.llm["cost_usd"] == 0.0025 and r.llm["input_tokens"] == 902
    d01 = next(f for f in r.findings if f.detector == "llm")
    assert TEXT[d01.spans[0].start:d01.spans[0].end] == "an intuitive experience"


def test_cli_backend_falls_back_to_result_json_string(monkeypatch):
    out = json.dumps({"is_error": False, "result": json.dumps(PAYLOAD), "usage": {}})
    monkeypatch.setattr("ai_isms.llm.subprocess.run", _fake_run(out, []))
    assert analyze(TEXT, llm=LLMOptions(backend="claude-cli")).llm["ok"]


def test_cli_backend_errors_keep_local_results(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr("ai_isms.llm.subprocess.run", missing)
    r = analyze(TEXT, llm=LLMOptions(backend="claude-cli"))
    assert r.llm["ok"] is False and "not found" in r.llm["errors"][0]
    assert r.findings

    monkeypatch.setattr("ai_isms.llm.subprocess.run", _fake_run(json.dumps({"is_error": True, "result": "boom"}), []))
    r = analyze(TEXT, llm=LLMOptions(backend="claude-cli"))
    assert r.llm["ok"] is False and "boom" in r.llm["errors"][0]
