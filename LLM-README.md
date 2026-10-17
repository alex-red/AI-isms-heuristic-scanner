# LLM-README: running the scanner on a target

This file is for an LLM agent (Claude Code, a CI bot, or similar) that has a local checkout of this repository and needs to scan some text with it. Humans should read [README.md](README.md).

## What you are doing

`ai-isms` is a prose linter. It reads text, reports span-level findings with a rule id, severity, confidence and a suggested fix, and gives an overall 0-100 score. It does not decide who wrote the text. Treat the output as editing advice.

## Ground rules

- Work from the repository root (the directory containing `pyproject.toml`).
- Do not modify the target file unless the user asked you to. Scanning is read-only.
- Do not start servers or run builds. Nothing here needs either.
- Do not use `--llm` unless the user asked for it. It calls the Anthropic API, needs `ANTHROPIC_API_KEY`, and costs money. The default local scan needs no network.
- Write reports to a scratch location, not into the target's directory, unless the user says otherwise.

## 1. Get a working command

Try these in order and use the first that works.

```bash
# a) already installed
ai-isms --version

# b) no install needed: run straight from source (Python 3.10+, stdlib only)
PYTHONPATH=src python -m ai_isms --version

# c) install into a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # add ',llm' only if you will use --llm
ai-isms --version
```

Option (b) is the least invasive. In the rest of this file, `ai-isms` means whichever form worked; with (b), replace it with `PYTHONPATH=src python -m ai_isms`.

If the shell is fish, set the variable with `env PYTHONPATH=src python -m ai_isms ...`, and activate a venv with `source .venv/bin/activate.fish`.

## 2. Prepare the target

The scanner takes plain text or Markdown.

| Target | How to feed it |
|---|---|
| A file | `ai-isms path/to/file.md` |
| Several files | `ai-isms a.md b.md` (each is scored separately) |
| A string you have in memory | write it to a scratch file, or pipe it: `printf '%s' "$TEXT" \| ai-isms -p chat` |
| A web page, PDF or docx | extract the prose to a text file first; do not feed HTML or binary |
| A source-code file | scan only the comments and docstrings, or the Markdown docs; code itself is not prose |

Code blocks, inline code and URLs inside Markdown are masked automatically. Texts under 80 words get findings but no grade.

Empty input prints an error and is skipped. A missing path exits with status 1.

## 3. Pick a profile

Use `-p` to match the genre. A wrong profile produces false positives (for example, bullets in docs are normal).

| Profile | Use for |
|---|---|
| `general` | default, unknown or mixed prose |
| `chat` | assistant replies, support answers |
| `technical` | docs, READMEs, specs, runbooks |
| `academic` | papers, reports |
| `marketing` | landing pages, ads, product copy |
| `creative` | fiction, essays, narrative |
| `casual` | emails, posts, informal writing |

## 4. Run the scan

For analysis, ask for JSON on stdout and suppress the terminal summary:

```bash
ai-isms TARGET.md -p technical --json -q > /path/to/scratch/report.json
```

`--json` with no path writes to stdout, and the terminal summary is automatically skipped in that case. Pass a path (`--json report.json`) to write a file instead; with several inputs, files are named `<stem>.ai-isms.json` in the current directory.

Other useful forms:

```bash
ai-isms TARGET.md                          # human-readable summary, top 12 findings
ai-isms TARGET.md -n 40                    # list 40 findings
ai-isms TARGET.md --html /scratch/r.html   # interactive report for the user to open
ai-isms TARGET.md --fail-above 45          # exit status 2 if overall score > 45
ai-isms TARGET.md --disable S01,L04        # skip noisy rules
ai-isms TARGET.md --only R01,R02           # run just these rules
ai-isms TARGET.md --set S01.per250=5       # override a rule threshold
ai-isms --list-rules                       # rule catalog with descriptions
```

Exit codes: `0` success, `1` missing file, `2` score exceeded `--fail-above`. A high score alone does not make the exit status nonzero without `--fail-above`.

## 5. Read the JSON

Top-level keys of interest:

| Key | Meaning |
|---|---|
| `scores.overall` | 0-100, or `null` if the text is too short to grade |
| `scores.band` | `Clean`, `Light`, `Noticeable`, `Heavy`, `Saturated`, or `Ungraded` |
| `scores.dimensions` | per-dimension scores |
| `scores.top_rules` | rule ids ranked by points contributed |
| `scores.confidence` | `low` under 200 words; treat the score as rough |
| `findings` | the list to act on (see below) |
| `metrics` | sentence length variation, lexical diversity, reading ease and similar |
| `sentences` | per-sentence contribution: 2 substantive, 1 supporting, 0 non-contributing |
| `text` | the analysed text; finding offsets index into this string |

Each entry in `findings` has:

- `rule_id`, `category`, `message`
- `spans`: a list of `{start, end}` character offsets into `text`
- `excerpt`: the matched text, so you rarely need to slice it yourself
- `severity` (1-5) and `confidence` (0-1, confidence in the match, not in authorship)
- `suggestion` and `action` (`remove`, `clarify`, `consolidate`, `substantiate`, `vary`, `review`)
- `scope`: `span` or document-level
- `detector`: `pattern`, `lexical`, `statistical`, `semantic-local` or `llm`
- `review`: present only after an LLM pass; may mark a finding as dismissed

Score bands: under 12 Clean, 12-25 Light, 25-45 Noticeable, 45-65 Heavy, 65 and over Saturated. The weights and thresholds are uncalibrated starting points, so use the score to compare drafts, not as a verdict.

Useful one-liners, assuming `jq`:

```bash
jq '.scores | {overall, band, confidence}' report.json
jq -r '.findings | sort_by(-.severity, -.confidence) | .[:10][] | "\(.rule_id) sev\(.severity) \(.excerpt)\n  -> \(.suggestion)"' report.json
jq '.scores.top_rules' report.json
```

Without `jq`, load the file with Python's `json` module.

## 6. Python API (alternative to the CLI)

When you are already running Python, skip the subprocess:

```python
import sys; sys.path.insert(0, "src")   # not needed if installed
from ai_isms import analyze

text = open("TARGET.md", encoding="utf-8").read()
report = analyze(text, profile="technical")
print(report.scores["overall"], report.scores["band"])
for f in sorted(report.findings, key=lambda f: (-f.severity, -f.confidence))[:10]:
    s = f.spans[0]
    print(f.rule_id, f.severity, repr(text[s.start:s.end]), "->", f.suggestion)
```

`analyze` also accepts `disabled=[...]`, `only=[...]` and `thresholds={...}`, matching the CLI flags.

## 7. Optional: the LLM semantic pass

Only with the user's go-ahead:

```bash
export ANTHROPIC_API_KEY=...        # must already be set; never print or log it
pip install -e '.[llm]'             # if the anthropic package is missing
ai-isms TARGET.md --llm --json -q > report.json
```

No API key? Add `--llm-backend claude` to run the pass through `claude -p` with the existing Claude Code login. It needs no `anthropic` package, uses the same `--model` and `--effort` flags, and is slower and a little more expensive per call.

It rates each sentence, adds semantic findings (tautology, circular reasoning, generic advice) and confirms or dismisses local findings. The default model is `claude-haiku-5-5` at `--effort low`; use `--model` and `--effort` to change them. `--no-review` skips the review of local findings. If the API call fails, the local results are still produced and the error is recorded under `llm` in the JSON. A 1,000-word document costs roughly $0.001 with the defaults.

## 8. Reporting back to the user

Give the user a short, useful summary rather than the raw JSON:

1. The score, band and confidence, plus the profile you used and why.
2. The two or three dimensions or rules contributing most (`scores.top_rules`).
3. The highest-impact findings: quote the `excerpt`, name the rule, and give the `suggestion`. Prefer high severity and high confidence.
4. A note on likely false positives, especially if the profile was a rough fit or the text is short.
5. The path to any HTML or JSON report you wrote.

Do not say the text "was written by AI". The tool does not establish that. Say which patterns were found and how they affect readability.

If the user asked you to fix the text, edit it, then rerun the same command and report the before and after scores. Rewrite to remove the underlying problem (add specifics, cut filler) instead of swapping flagged words for synonyms, which tends to trigger other rules.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `command not found: ai-isms` | Not installed or the venv is inactive. Use the `PYTHONPATH=src python -m ai_isms` form. |
| `ModuleNotFoundError: ai_isms` | Running outside the repo root without `PYTHONPATH=src`. |
| `ModuleNotFoundError: anthropic` | Only matters for `--llm`. Run `pip install -e '.[llm]'`. |
| `ai-isms: no such file` | Path is wrong or relative to a different directory. |
| `Reading from stdin (Ctrl-D to finish)…` hangs | No file argument and no piped input. Pass a path or pipe text in. |
| `overall` is `null` | Under 80 words. Read the findings; there is no grade. |
| Many findings on a README or doc | Use `-p technical`, and consider `--disable S05,S06` for heading and list rules. |
| `--set expects KEY=VALUE` | Use the form `RULE.key=number`, for example `S01.per250=5`. |
