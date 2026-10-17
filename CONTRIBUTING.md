# Contributing

Thanks for helping. The most valuable contributions are, in order:

1. **False positives with example text.** If the scanner flags good writing, open an issue with the passage, the rule id, and the genre. These drive threshold and profile changes.
2. **Missed patterns.** A short example of a formulaic construction the rules don't catch.
3. **Rated samples** for the calibration corpus (see the README roadmap).
4. Code: new rules, better detectors, reporters.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[llm,dev]'
pytest
```

The core package must stay dependency-free (standard library only). Optional features go behind extras.

## Adding a rule

Rules live in `src/ai_isms/rules/`, one module per scoring dimension. A rule is a function registered with `@rule`:

```python
@rule(
    "R07", "Name shown in reports", "formulaic_rhetoric", 3,   # id, name, dimension, default severity
    "One-sentence description of the pattern and why it hurts the writing.",
    "Concrete revision strategy.",
    "remove",            # action: remove | clarify | consolidate | substantiate | vary | review
    "pattern",           # detector: pattern | lexical | statistical | semantic-local
)
def r07(doc: Document, ctx: Context) -> list[Finding]:
    return [make("R07", [(m.start(), m.end())], confidence=0.7)
            for m in doc.finditer(MY_REGEX, prose_only=True)]
```

Guidelines:

- **Match on `doc.masked` (via `doc.finditer`)**, never on raw text, so code and URLs are ignored and offsets stay correct.
- **Report the smallest span that shows the problem.** Use several spans rather than one huge one.
- **Confidence is confidence in the match**, not in AI authorship. Lower it for ambiguous matches; raise severity when a pattern repeats.
- **Don't flag single common words.** Require a cluster, a density threshold, or a construction.
- Read thresholds through `ctx.t("R07.key", default)` so profiles and `--set` can tune them.
- Add a positive and a negative case to `tests/test_rules.py`.
- Run `python benchmark/run.py --compare benchmark/baseline.json` and check that human samples didn't gain findings. See [benchmark/README.md](benchmark/README.md).
- If the rule can only be judged semantically, register it with `llm_rule(...)` and add its id to `SEMANTIC_RULES` in `llm.py`.

## Phrase lists

`src/ai_isms/lexicon.py` holds the dictionaries. Patterns are case-insensitive; `*` matches one word, `(a|b)` alternates, and `'` matches straight and curly apostrophes. Add phrases that are *characteristic*, not merely common, and include a test.

## Style

Match the surrounding code. Keep functions small and rules self-contained. Run `pytest` before opening a PR.
