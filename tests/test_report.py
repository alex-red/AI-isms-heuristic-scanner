import json
import re

from ai_isms import analyze
from ai_isms.report import to_html, to_json, to_terminal


def test_html_embeds_data_safely():
    text = "Great question! </script><script>alert(1)</script> " * 3
    html = to_html(analyze(text))
    payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "</script" not in payload
    data = json.loads(payload)
    assert data["text"].startswith("Great question!")


def test_json_and_terminal(formulaic):
    r = analyze(formulaic, source="formulaic.md")
    assert json.loads(to_json(r))["config"]["source"] == "formulaic.md"
    out = to_terminal(r)
    assert "formulaic.md" in out and "Semantic emptiness" in out
