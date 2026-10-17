# AI-isms Scanner

AI-isms Scanner is a linter for prose. It finds formulaic LLM writing patterns and plain low-quality writing (vague claims, repetition, filler, mechanical rhythm), points at the exact characters responsible, explains each finding, and writes an interactive HTML report.

```
$ ai-isms examples/formulaic.md --html
AI-isms · formulaic.md
56/100  Heavy  Pervasive formulaic writing
206 words · 27 sentences · profile general · local · 6 ms

  Semantic emptiness             ██████████████████████░░  91.9  16 findings est.
  Repetition & redundancy        █████░░░░░░░░░░░░░░░░░░░  20.3  4 findings est.
  Formulaic rhetoric             █████████████████░░░░░░░  71.9  8 findings
  ...
  R01 ●●●○○ Artificial negation/contrast
      “AI isn't just changing how we work—it's”
      → State the substantive claim directly instead of contrasting it with a strawman.
```

Open [`examples/formulaic.ai-isms.html`](examples/formulaic.ai-isms.html) to see a report.

## What it is and isn't

The scanner is **not an AI detector**. It doesn't guess who wrote a text. It flags writing problems that readers associate with unedited LLM output, and it measures three things separately:

- **AI-style markers**: recognizable templates ("It's not X, it's Y", "The result?", "In today's rapidly evolving landscape").
- **Writing quality**: repetition, vagueness, inflation, sentences that add nothing.
- **Context**: genre profiles relax or tighten rules, because headings and bullets are normal in docs and fragments are normal in fiction.

Every warning has a rule id, a span, a severity (1–5), a confidence (0–1, confidence in *the match*, not in authorship), an explanation and a suggested action. The overall score is there for comparing drafts; the findings are what you act on.

## Install

```bash
pip install ai-isms            # core: pure standard library, no dependencies
pip install 'ai-isms[llm]'     # adds the optional Claude semantic pass
```

From a checkout: `pip install -e '.[llm,dev]'`. The command is `ai-isms`; the Python package is `ai_isms`.

## Usage

```bash
ai-isms draft.md                      # terminal summary
ai-isms draft.md --html --open        # write draft.ai-isms.html and open it
cat reply.txt | ai-isms -p chat       # stdin, AI-assistant profile
ai-isms docs/*.md -p technical -q --html
ai-isms post.md --json report.json    # machine-readable output
ai-isms post.md --fail-above 45       # exit 2 in CI if the overall score exceeds 45
ai-isms --list-rules
```

Options worth knowing:

| Flag | Purpose |
|---|---|
| `-p, --profile` | `general` (default), `chat`, `technical`, `academic`, `marketing`, `creative`, `casual` |
| `--disable S01,L04` / `--only R01,R02` | Turn rules off, or run a subset |
| `--set S01.per250=5` | Override a rule threshold (repeatable) |
| `--llm` | Add the semantic pass (see below) |
| `--model`, `--effort` | Model id (default `claude-haiku-5-5`) and effort (default `low`) |

### Python API

```python
from ai_isms import analyze
from ai_isms.report import to_html

report = analyze(text, profile="technical")
print(report.scores["overall"], report.scores["band"])
for f in report.findings:
    print(f.rule_id, f.severity, f.confidence, text[f.spans[0].start:f.spans[0].end])
open("report.html", "w").write(to_html(report))
```

## How it works

```
text ─► preprocess ─► deterministic rules ─► (optional) LLM pass ─► merge & dedupe ─► score ─► HTML / JSON / terminal
        sentences,     regex, phrase          contribution ratings,  overlapping
        paragraphs,    dictionaries,          semantic findings,     findings count
        Markdown,      statistics             review of local        once
        code masking                          findings
```

**Local first.** All 36 local rules run with the Python standard library in a few milliseconds per page: no model downloads, no network, reproducible output. Code blocks, inline code and URLs are masked so they never trigger rules, and every offset maps back to the original text.

**Semantic rules have local proxies.** "Says little despite its length" (C01) and "abstract filler" (C06) are approximated by comparing specificity signals (numbers, names, quotes, code, "for example") against abstract nouns (solutions, opportunities, growth). Redundancy (D01/D02) uses TF-IDF cosine and vocabulary novelty. These dimensions are marked `est.` in reports until the LLM pass confirms them.

**Semantic contribution.** Every sentence is rated 2 (substantive), 1 (supporting) or 0 (non-contributing). Locally this is estimated from vocabulary novelty, concrete detail and rhetorical-template coverage; with `--llm` the model rates each sentence. The report shows this as a strip across the top and as a "contribution lens" over the text.

### Optional LLM pass

```bash
export ANTHROPIC_API_KEY=...        # or `ant auth login`
ai-isms essay.md --llm --html
```

With `--llm`, one structured-output request per ~120 sentences goes to Claude Haiku 5.5 (configurable). The model:

1. rates every sentence's contribution,
2. reports semantic findings (tautology, circular reasoning, generic advice, paraphrased repetition…), each with an exact quote that must be found in the text to be highlighted,
3. reviews the local findings and confirms or dismisses each one. Dismissed findings stay visible (struck through) but stop counting toward the score.

A 1,000-word document costs about $0.001 on Haiku 5.5 at low effort; the report shows token usage and cost. If the call fails, the local results are still produced and the error appears in the report.

## Rules

37 rules in six dimensions. Severity is the default; profiles and repetition adjust it.

**Semantic emptiness** (weight 30%)

| ID | Rule | Detector | Sev |
|---|---|---|---|
| C01 | Says little despite its length | semantic-local | 4 |
| C02 | Tautology | semantic-local | 4 |
| C03 | Unsupported sweeping claim | semantic-local | 4 |
| C04 | Vague referents and actors | lexical | 2 |
| C05 | Manufactured significance | pattern | 3 |
| C06 | Abstract filler instead of specifics | semantic-local | 4 |
| C07 | Generic recommendation | semantic-local | 3 |

**Repetition & redundancy** (weight 20%)

| ID | Rule | Detector | Sev |
|---|---|---|---|
| D01 | Near-duplicate sentence | lexical | 4 |
| D02 | Paragraph adds little new information | lexical | 4 |
| D03 | Unnecessary recap | lexical | 3 |
| D04 | Verbose restatement | lexical | 1 |
| D05 | Circular reasoning | LLM only | 4 |

**Formulaic rhetoric** (weight 15%)

| ID | Rule | Detector | Sev |
|---|---|---|---|
| R01 | Artificial negation/contrast | pattern | 3 |
| R02 | Rhetorical question immediately answered | pattern | 3 |
| R03 | Forced triplets / rule-of-three | pattern | 2 |
| R04 | Manufactured revelation | pattern | 3 |
| R05 | Dramatic one-line conclusion | pattern | 3 |
| R06 | Repeated binary framing | pattern | 3 |

**Mechanical structure & rhythm** (weight 15%)

| ID | Rule | Detector | Sev |
|---|---|---|---|
| S01 | Excessive em-dash usage | statistical | 1 |
| S02 | Staccato rhythm | statistical | 3 |
| S03 | Uniform sentence length | statistical | 3 |
| S04 | Identical paragraph architecture | statistical | 4 |
| S05 | Unnecessary bolding, headings or emoji | statistical | 2 |
| S06 | Excessive listification | statistical | 2 |
| S07 | Repeated sentence starters | statistical | 1 |

**Generic vocabulary & clichés** (weight 10%)

| ID | Rule | Detector | Sev |
|---|---|---|---|
| L01 | AI-associated vocabulary cluster | lexical | 2 |
| L02 | Corporate buzzword stacking | lexical | 3 |
| L03 | Generic scene-setting introduction | pattern | 3 |
| L04 | Empty intensifiers | lexical | 1 |
| L05 | Stock transitional phrase | pattern | 1 |
| L06 | Decorative / cliché metaphor | pattern | 2 |

**Performative tone** (weight 10%)

| ID | Rule | Detector | Sev |
|---|---|---|---|
| T01 | Unsolicited validation | pattern | 3 |
| T02 | False profundity | pattern | 3 |
| T03 | Artificial emotional intensity | lexical | 3 |
| T04 | Performative honesty | pattern | 2 |
| T05 | Excessive hedging or false balance | pattern | 3 |
| T06 | Inappropriate conversational familiarity | pattern | 2 |

Single words are never penalized on their own: "robust" or "delve" only count as part of a cluster (L01) or a stack (L02). Most pattern rules escalate when a construction repeats, because repetition is what makes it conspicuous.

## Scoring

For each finding, `points = severity_points × confidence × dedup_weight`, divided by document length in hundreds of words (document-scope findings are not length-normalized). Each dimension saturates as `100 × (1 − e^(−points/8))`; the overall score is the weighted sum of dimensions. Findings that overlap a stronger finding from another rule count at 35%, so one phrase doesn't collect three full penalties.

| Score | Band |
|---|---|
| < 12 | Clean |
| 12–25 | Light |
| 25–45 | Noticeable |
| 45–65 | Heavy |
| ≥ 65 | Saturated |

Texts under 80 words get findings but no grade. **The weights, thresholds and bands are starting points, not calibrated values.** Calibrating them against human ratings is the main open task (see below).

## The HTML report

One self-contained file (data inlined; fonts load from Google Fonts when online and fall back to system serif offline):

- the text with each finding underlined in its dimension's colour; underline weight tracks severity, and stacked findings show a second line
- hover for the rule, click to open the note; click again to cycle through stacked findings
- a notes column with filters by dimension, severity, detector (local / LLM) and free text, sorted by position, impact or rule
- a sentence-by-sentence contribution strip, and a lens that tints the text by contribution
- metrics: sentence-rhythm chart, length variation, specificity, AI-vocabulary density, lexical diversity, reading ease
- the full rule catalog with match counts
- light/dark themes, JSON export, print stylesheet, keyboard navigation (`j`/`k`, `/`, `L`, `Esc`)

## Development

### Run it locally

You need Python 3.10 or newer. The core has no dependencies, so a virtual environment is the only setup.

```bash
git clone https://github.com/alex-red/AI-isms-heuristic-scanner.git
cd AI-isms-heuristic-scanner
python -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -e '.[llm,dev]'        # editable install: source changes apply immediately
```

Drop `llm` from the extras if you only need the local rules. Then try it on the bundled samples:

```bash
ai-isms examples/formulaic.md                  # terminal summary
ai-isms examples/formulaic.md --html --open    # write and open the HTML report
ai-isms examples/specific.md -p technical      # a well-grounded sample, should score low
ai-isms --list-rules
```

To run without installing anything, point Python at `src/`:

```bash
PYTHONPATH=src python -m ai_isms examples/formulaic.md
```

To try your own text, pipe it in or pass a file:

```bash
pbpaste | ai-isms -p chat                      # macOS clipboard
ai-isms ~/notes/draft.md --json -              # JSON to stdout
```

To try the LLM pass, set `ANTHROPIC_API_KEY` and add `--llm`. It calls the real API and costs a fraction of a cent per page. The tests never touch the network.

### Tests

```bash
pytest                      # whole suite
pytest tests/test_rules.py  # one file
pytest -k R01 -x            # filter by name, stop at first failure
```

`tests/test_llm.py` uses a fake client and is skipped if `anthropic` isn't installed.

CI runs the suite on Python 3.10, 3.12 and 3.14, then scans this README with the linter itself:

```bash
ai-isms README.md -p technical --fail-above 25
```

Run that command after editing the README. If the score goes over 25, CI fails.

### Project layout

| Path | Contents |
|---|---|
| `src/ai_isms/cli.py` | Argument parsing and the `ai-isms` entry point |
| `src/ai_isms/engine.py` | `analyze()`: runs rules, merges findings, scores |
| `src/ai_isms/text.py` | Sentence, paragraph and Markdown parsing; code masking |
| `src/ai_isms/rules/` | One module per dimension, rules registered with `@rule` |
| `src/ai_isms/lexicon.py` | Phrase and vocabulary lists |
| `src/ai_isms/profiles.py` | Genre profiles and their threshold overrides |
| `src/ai_isms/scoring.py` | Points, dimension saturation, bands |
| `src/ai_isms/llm.py` | Optional Claude semantic pass |
| `src/ai_isms/report/` | HTML (`template.html`), JSON and terminal output |
| `tests/` | pytest suite |
| `examples/` | Sample inputs and their generated reports |

### Common tasks

- **Add or tune a rule:** see [CONTRIBUTING.md](CONTRIBUTING.md). Write the positive and negative test first, then check the effect on all three examples.
- **Tune a threshold without editing code:** `ai-isms draft.md --set S01.per250=5`.
- **Change the report UI:** edit `src/ai_isms/report/template.html`. Report data is inlined in the output, so regenerate with `--html` and reload the file.
- **Refresh the example reports:** rerun the CLI on each file in `examples/` with `--html`. The reports are committed, so include them in the diff.

## Roadmap

- **Calibration corpus**: human-rated texts that include well-written AI output, poor human writing, academic, technical, creative and translated prose, so the scorer doesn't just learn to punish polish.
- Embedding-based paraphrase detection as an optional local extra.
- Cross-document structure checks (every section built the same way).
- SARIF output and an editor/LSP integration.
- Languages other than English (the phrase lists are English-only today).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Phrase lists live in [`src/ai_isms/lexicon.py`](src/ai_isms/lexicon.py), and false-positive reports with example text are especially welcome.

## License

MIT
