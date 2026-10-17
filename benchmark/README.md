# Benchmark

A small, reproducible check of how the scanner scores human writing against AI-style writing. It is meant to catch regressions and to show which rules need tuning. It is not a calibration of the scoring weights: 45 samples is too few for that.

```bash
python benchmark/fetch.py                     # once: download the human baseline into benchmark/data/
python benchmark/run.py                       # table and summary
python benchmark/run.py --truncate 250        # cut every sample to ~250 words, so lengths match
python benchmark/run.py --json out.json       # save per-sample results
python benchmark/run.py --compare benchmark/baseline.json   # show score changes after a rule edit
python benchmark/run.py --ids ai-story-opening ai-cover-letter --llm   # add the semantic pass (see below)
```

Both scripts use only the standard library. `run.py` runs the local rules by default, so it needs no network or API key. Only `fetch.py` touches the network.

## Samples

| Label | Count | Source | License |
|---|---|---|---|
| `human` | 27 | Project Gutenberg (10 passages), Wikipedia (4), the Rust book (2), Stack Exchange answers (6), Reddit comments (5) | Public domain / CC BY-SA / MIT or Apache-2.0 / see below |
| `ai-typical` | 14 | Written by Claude to imitate unedited default output | CC0 |
| `ai-polished` | 4 | Written by Claude to avoid the usual tells | CC0 |

The AI-style set covers blog posts, essays, Reddit comments (advice, tech help, ELI5), a LinkedIn post, a product description, a chat answer, a cover letter, a docs intro and a story opening. Everything is listed in [`manifest.json`](manifest.json) with its genre, scan profile and license.

The Stack Exchange answers (cooking, workplace, travel, money, English) were posted before 2021 and are fetched from the Stack Exchange API. The Reddit comments (2006 to 2016) come from the [Webis-TLDR-17](https://huggingface.co/datasets/webis/tldr-17) corpus through the Hugging Face datasets server, because Reddit itself blocks scripted access. The corpus is CC BY 4.0, but the comments belong to their authors, so these are fetched for local evaluation only and never committed.

Downloaded text goes to `benchmark/data/`, which is gitignored, so nothing third-party is redistributed. Gutenberg and Rust book entries carry a pinned `sha256`; `fetch.py` warns if the source changes. Wikipedia entries are not pinned because articles drift.

## Baseline results (v0.1.0, local rules)

| Label | n | Median | Max | Scoring 12 or more |
|---|---|---|---|---|
| human | 27 | 2.9 | 15.1 | 4% (1 sample) |
| ai-typical | 14 | 21.2 | 38.9 | 79% |
| ai-polished | 4 | 0.0 | 2.9 | 0% |

AUC for human vs ai-typical is 0.90, and 0.90 again when every sample is cut to about 250 words. The full per-sample table is in [`baseline.json`](baseline.json).

What stands out:

- **One human sample reaches Light.** A Stack Exchange workplace answer scores 15.1. It is a real pre-2021 answer made of short reassuring paragraphs ("This is not something you should be scolded over... This is an amazing achievement"), which is the chatbot-validation shape the rules look for. Whether that is a false positive or a fair catch is a judgment call. Every other human sample is under 10.
- **Misses worth looking at.** `ai-story-opening` scores 0.0 even though it is stuffed with fiction clichés ("Little did she know", "ancient wisdom", "a promise of new beginnings"). `ai-product-description` (6.9) and `ai-cover-letter` (8.5) also stay Clean. The local rules have no fiction-cliché or cover-letter vocabulary, and R01 misses "more than just".
- **Polished AI text passes.** All four polished samples score under 3. That is by design: the scanner finds writing problems, not authorship, so specific, varied prose is not flagged. Treat this as a feature, and don't read a low score as evidence a human wrote the text.
- **Quiet human-side rules.** Most rules never fire on the human set. The ones that fire most (S07 repeated sentence starters, D04, R01, L05) fire under one match per 1,000 words.

## With the LLM pass

`--llm` runs the semantic pass through `claude -p` (Claude Code's headless mode, so no API key is needed), using Claude Haiku 5.5 at medium effort. It is slow and costs money, so it is off by default and takes `--ids` to pick samples. Results for eight samples are in [`baseline-llm.json`](baseline-llm.json):

| Sample | Label | Local | + LLM |
|---|---|---|---|
| ai-story-opening | ai-typical | 0.0 | 19.6 |
| ai-product-description | ai-typical | 6.9 | 27.9 |
| ai-cover-letter | ai-typical | 8.5 | 26.8 |
| ai-blog-remote-work | ai-typical | 38.9 | 45.8 |
| polished-blog-sourdough | ai-polished | 0.0 | 3.3 |
| pg-pride-and-prejudice | human | 3.1 | 1.5 |
| reddit-explainlikeimfive | human | 7.3 | 8.4 |
| se-workplace-124717 | human | 15.1 | 22.2 |

All eight calls cost $0.02 in total, about $0.0025 each.

- **It rescues the three local misses.** The story opening moves from Clean to Light, and the product description and cover letter to Noticeable. The model flags "nothing would ever be the same again", "a proven track record", "Crafted with premium, high-quality materials" and similar, and its summaries name the missing specifics.
- **It dismisses false positives on good prose.** Five of the local findings on Pride and Prejudice were dismissed as dialogue conventions, which lowers its score.
- **It does not clear the human Stack Exchange answer.** It raises it from 15.1 to 22.2, because it agrees the reassurance is repeated. That is a real risk for genre-specific human writing, so judge it against more human samples before trusting the LLM score.
- **It is a small sample.** Eight texts, one run each, and LLM output varies between runs.

## Caveats

- The `ai-*` samples are one author's imitation of AI writing, not a random draw from deployed models. They are written to be recognizable, so the AUC is optimistic. Swap in real model output for a harder test.
- Half the human texts are pre-1930 prose. The Stack Exchange and Reddit samples add modern informal writing, but there are only 11 of them, from a handful of topics. Wikipedia and the Rust book are still edited today and may contain some LLM-assisted changes.
- The AI Reddit samples use `**Comment N**` headings and the human ones do not, which may nudge the formatting rules.
- AI samples are 145 to 275 words and human samples about 850, so use `--truncate` for any length-sensitive comparison.
- The LLM numbers cover eight samples only.

## Adding a sample

Add an entry to [`manifest.json`](manifest.json).

- **Human text:** give it a `source` (`gutenberg` with an `id`, `wikipedia` with a `title`, or `url`), a `start_at` phrase, and a word target. Then run `python benchmark/fetch.py --pin`.
- **AI-style text:** put a file under `benchmark/ai/`, point `file` at it, and set the label. Only add text you wrote or generated yourself and are willing to release under CC0.

Use only text published before about 2022 as human baseline, so it can't be unlabeled LLM output. Refresh the saved baseline with `python benchmark/run.py --json benchmark/baseline.json` when rule changes are intentional, and include the new file in the PR.
