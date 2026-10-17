"""E. Repetition and redundancy (D01-D05).

Local detection uses TF-IDF cosine similarity over stemmed content words. That
catches near-verbatim restatement but not true paraphrase with different
vocabulary; the LLM pass covers that and circular reasoning (D05).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..lexicon import WORDY, compile_phrases
from ..models import Finding
from ..text import Document, content_stems
from . import Context, llm_rule, make, rule

CAT = "repetition"


def _tfidf(docs: list[list[str]]) -> list[dict[str, float]]:
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    vecs = []
    for d in docs:
        tf = Counter(d)
        v = {t: (c / len(d)) * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in tf.items()} if d else {}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


@rule(
    "D01", "Near-duplicate sentence", CAT, 4,
    "Two sentences that make the same point in nearly the same words.",
    "Keep the stronger sentence and delete the other.",
    "consolidate", "lexical",
)
def d01(doc: Document, ctx: Context) -> list[Finding]:
    sents = [s for s in doc.sentences if doc.paragraphs[s.para].kind in ("prose", "quote", "list_item")]
    stems = [content_stems(s.words) for s in sents]
    vecs = _tfidf(stems)
    thresh = ctx.t("D01.cosine", 0.55)
    window = int(ctx.t("D01.window", 15))
    out = []
    used: set[int] = set()
    for i in range(len(sents)):
        if len(set(stems[i])) < 4 or i in used:
            continue
        for j in range(i + 1, min(len(sents), i + 1 + window)):
            if len(set(stems[j])) < 4 or j in used:
                continue
            c = _cos(vecs[i], vecs[j])
            if c >= thresh:
                shared = sorted(set(stems[i]) & set(stems[j]))
                out.append(make(
                    "D01", [(sents[i].start, sents[i].end), (sents[j].start, sents[j].end)], scope="sentence",
                    severity=4 if c >= 0.75 else 3, confidence=min(0.9, 0.45 + (c - thresh) * 1.5),
                    message=f"Sentences {i + 1} and {j + 1} repeat each other (similarity {c:.2f})",
                    evidence="Shared terms: " + ", ".join(shared[:10]),
                ))
                used.add(j)
                break
    return out


@rule(
    "D02", "Paragraph adds little new information", CAT, 4,
    "A paragraph whose content words almost all appeared earlier: it re-describes rather than develops.",
    "Merge it into the earlier paragraph or replace it with something new (an example, a consequence, a caveat).",
    "consolidate", "lexical",
)
def d02(doc: Document, ctx: Context) -> list[Finding]:
    seen: set[str] = set()
    out = []
    limit = ctx.t("D02.novelty", 0.35)
    for p in doc.paragraphs:
        if p.kind not in ("prose", "quote", "list_item"):
            continue
        words = [w for i in p.sentences for w in doc.sentences[i].words]
        st = set(content_stems(words))
        if len(words) >= 25 and seen and st:
            novelty = len(st - seen) / len(st)
            if novelty < limit:
                out.append(make(
                    "D02", [(p.start, p.end)], scope="paragraph",
                    severity=4 if novelty < limit / 2 else 3, confidence=min(0.85, 0.5 + (limit - novelty)),
                    message=f"Paragraph is {1 - novelty:.0%} recycled vocabulary",
                    evidence=f"Only {len(st - seen)} of {len(st)} content words are new.",
                ))
        seen |= st
    return out


_RECAP_START = re.compile(
    r"^\s*(in conclusion|in summary|to sum up|to summarize|to recap|overall|ultimately|in short|all in all|in the end|"
    r"so,? to wrap up|wrapping up|in closing|the bottom line)\b", re.I,
)
_RECAP_HEAD = re.compile(r"\b(conclusion|summary|final thoughts|wrapping up|wrap-up|tl;?dr|key takeaways|recap|in closing)\b", re.I)


@rule(
    "D03", "Unnecessary recap", CAT, 3,
    "A closing summary that repeats what was just said, without serving a structural need (e.g. in a short text).",
    "Cut the recap, or end on a new implication, next step, or open question.",
    "remove", "lexical",
)
def d03(doc: Document, ctx: Context) -> list[Finding]:
    paras = [p for p in doc.paragraphs if p.kind in ("prose", "list_item", "heading")]
    if len(paras) < 3:
        return []
    # Find the recap block: last paragraph opening with a recap marker, or everything after a recap heading.
    block: list = []
    for k in range(len(paras) - 1, -1, -1):
        p = paras[k]
        if p.kind == "heading" and _RECAP_HEAD.search(doc.text[p.start:p.end]):
            block = [q for q in paras[k + 1:] if q.kind != "heading"]
            break
    if not block:
        last = [p for p in paras if p.kind == "prose"]
        if last and _RECAP_START.match(doc.masked[last[-1].start:last[-1].end]):
            block = [last[-1]]
    if not block:
        return []
    first = block[0].start
    earlier = set()
    for s in doc.sentences:
        if s.end <= first:
            earlier |= set(content_stems(s.words))
    st = set(content_stems([w for p in block for i in p.sentences for w in doc.sentences[i].words]))
    if not st or not earlier:
        return []
    novelty = len(st - earlier) / len(st)
    short = doc.word_count < ctx.t("D03.short_doc", 300)
    if novelty < 0.5 or short:
        return [make(
            "D03", [(block[0].start, block[-1].end)], scope="section",
            severity=3 if short else 2, confidence=0.75 if novelty < 0.35 else 0.55,
            message="Closing recap repeats earlier content" + (" in a short text" if short else ""),
            evidence=f"{1 - novelty:.0%} of the recap's content words already appeared.",
        )]
    return []


_WORDY_KEYS = list(WORDY)
_WORDY = compile_phrases(_WORDY_KEYS)
_WORDY_EACH = [(compile_phrases([k]), v) for k, v in WORDY.items()]


@rule(
    "D04", "Verbose restatement", CAT, 1,
    "Wordy phrasing with a shorter equivalent (\"in order to\" → \"to\", \"due to the fact that\" → \"because\").",
    "Use the shorter form.",
    "remove", "lexical",
)
def d04(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for m in doc.finditer(_WORDY, prose_only=True):
        rep = next((v for rx_, v in _WORDY_EACH if rx_.fullmatch(m.group(0))), "")
        out.append(make("D04", [(m.start(), m.end())], confidence=0.85,
                        message=f"Wordy: “{m.group(0)}”", suggestion=f"Try: {rep}" if rep else None))
    return out


llm_rule(
    "D05", "Circular reasoning", CAT, 4,
    "An argument that keeps returning to an unsupported premise instead of building on evidence.",
    "Identify the premise and support it, or drop the argument.",
    "substantiate",
)
