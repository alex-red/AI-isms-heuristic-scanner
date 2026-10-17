"""A. Formulaic rhetoric (R01-R06)."""

from __future__ import annotations

import re

from ..lexicon import DRAMATIC_CLOSERS, REVELATION, compile_phrases
from ..models import Finding
from ..text import Document
from . import Context, make, rule
from ._util import merge_spans, per_n_words

CAT = "formulaic_rhetoric"

_A = r"['’]"
_NEG = (
    rf"(?:(?:{_A}s|\s+is|\s+was|{_A}re|\s+are|\s+were)\s+not"
    rf"|\s+(?:isn{_A}t|aren{_A}t|wasn{_A}t|weren{_A}t|doesn{_A}t|don{_A}t))"
)
_PRO_IS = rf"(?:it|this|that|they|we|he|she|you)(?:{_A}s|\s+is|{_A}re|\s+are|\s+was|\s+were)"

CONTRAST_PATTERNS = [
    # "It's not X, it's Y" / "AI isn't just X — it's Y" / "This is not about X. It's about Y."
    re.compile(
        rf"\b[\w\-]+{_NEG}\s+(?:(?:just|only|merely|simply|about|really)\s+)*[^.!?;\n—–]{{1,80}}?"
        rf"(?:\s*[.;:—–]\s*|\s*,\s*|\s+-{{1,2}}\s+)(?:but\s+)?{_PRO_IS}\b",
        re.I,
    ),
    # "not just X, but (also) Y"
    re.compile(r"\bnot\s+(?:just|only|merely|simply)\s+[^.!?\n]{1,80}?,?\s+but\s+(?:also\s+|rather\s+)?\w+", re.I),
    # "less about X and more about Y"
    re.compile(r"\b(?:less|not so much)\s+(?:about\s+)?[^.!?\n]{1,50}?\s+(?:and|but|than)\s+more\s+(?:about\s+)?\w+", re.I),
    # "X isn't the problem. Y is."
    re.compile(
        rf"\b[^.!?\n]{{1,60}}?\s+(?:isn{_A}t|is not|aren{_A}t|are not)\s+the\s+"
        r"(?:problem|point|issue|goal|answer|question|enemy|bottleneck|story|hard part)[.!]\s+[^.!?\n]{1,60}?\s+(?:is|are)[.!]",
        re.I,
    ),
    # "No X. No Y. Just Z."
    re.compile(r"\bNo\s+[^.!?\n]{1,30}\.\s+No\s+[^.!?\n]{1,30}\.\s+(?:Just|Only)\b"),
    # "not because X, but because Y"
    re.compile(r"\bnot\s+because\s+[^.!?\n]{1,80}?,?\s+but\s+because\b", re.I),
]


def _contrast_spans(doc: Document) -> list[tuple[int, int]]:
    spans = []
    for pat in CONTRAST_PATTERNS:
        spans.extend((m.start(), m.end()) for m in doc.finditer(pat, prose_only=True))
    return merge_spans(spans)


@rule(
    "R01", "Artificial negation/contrast", CAT, 3,
    "Framing a point as a correction of something nobody claimed (\"It's not X, it's Y\"). "
    "Occasional use is fine; repetition makes it conspicuous.",
    "State the substantive claim directly instead of contrasting it with a strawman.",
    "clarify",
)
def r01(doc: Document, ctx: Context) -> list[Finding]:
    spans = _contrast_spans(doc)
    rate = per_n_words(len(spans), doc.word_count)
    repeated = len(spans) >= 2 and rate >= ctx.t("R01.per250", 2.0)
    out = []
    for a, b in spans:
        out.append(make(
            "R01", [(a, b)],
            severity=3 if repeated else 2,
            confidence=0.85 if repeated else 0.6,
            evidence=f"{len(spans)} contrast construction(s), {rate:.1f} per 250 words"
            + (" — repeated use is a strong signal." if repeated else "."),
        ))
    return out


_QA_HOOK = re.compile(
    r"^(so what|why does (this|it|that) matter|what does (this|that) mean|why|how|the (result|answer|catch|kicker|twist|"
    r"problem|secret|reason|best part|worst part|real question|real opportunity|bottom line|takeaway|truth|difference|"
    r"upshot|outcome|verdict|solution|fix)|the real \w+|and the (result|best part)|sound familiar|what's the catch|"
    r"what changed|what happened)\b",
    re.I,
)


@rule(
    "R02", "Rhetorical question immediately answered", CAT, 3,
    "A short question posed only to be answered in the next sentence (\"The result? ...\").",
    "Make the statement directly; keep questions for genuine uncertainty.",
    "remove",
)
def r02(doc: Document, ctx: Context) -> list[Finding]:
    hits = []
    for p in doc.paragraphs:
        if p.kind not in ("prose", "quote", "list_item"):
            continue
        ids = p.sentences
        for i, j in zip(ids, ids[1:]):
            q, a = doc.sentences[i], doc.sentences[j]
            qt = q.text.strip()
            if not qt.endswith("?") or a.text.strip().endswith("?") or q.n_words > 10:
                continue
            if qt[:1] in "\"“'":  # quoted dialogue
                continue
            bare = qt.strip("*_ ").rstrip("?")
            hook = bool(_QA_HOOK.match(bare)) or q.n_words <= 4
            hits.append((q, a, hook))
    out = []
    repeated = len(hits) >= 2
    for q, a, hook in hits:
        if not hook and not repeated:
            continue  # a lone genuine question is fine; only hooks or a habit are conspicuous
        conf = 0.85 if hook else 0.5
        if repeated:
            conf = min(0.95, conf + 0.1)
        out.append(make(
            "R02", [(q.start, q.end)],
            severity=3 if (hook or repeated) else 2,
            confidence=conf,
            message="Manufactured question-answer hook" if hook else "Rhetorical question immediately answered",
            evidence=f"Answered by the next sentence: “{a.text.strip()[:80]}”",
        ))
    return out


_TRIPLET = re.compile(
    r"\b([A-Za-z][\w'’\-]*(?:\s+[A-Za-z][\w'’\-]*)?),\s+([A-Za-z][\w'’\-]*(?:\s+[A-Za-z][\w'’\-]*)?),?\s+(?:and|or)\s+"
    r"([A-Za-z][\w'’\-]*(?:\s+[A-Za-z][\w'’\-]*)?)\b"
)


@rule(
    "R03", "Forced triplets / rule-of-three", CAT, 2,
    "Habitual three-part lists and three-fragment slogans (\"Faster. Smarter. Better.\").",
    "Keep the items that carry information; vary list length.",
    "vary",
)
def r03(doc: Document, ctx: Context) -> list[Finding]:
    out: list[Finding] = []
    # Fragment triplets: 3+ consecutive sentences of <= 3 words.
    sents = doc.prose_sentences
    i = 0
    while i < len(sents):
        j = i
        while j < len(sents) and sents[j].n_words <= 3 and sents[j].para == sents[i].para:
            j += 1
        if j - i >= 3:
            out.append(make(
                "R03", [(sents[i].start, sents[j - 1].end)], severity=2, confidence=0.85,
                message="Three-fragment slogan", evidence=f"{j - i} consecutive fragments of ≤3 words",
            ))
            i = j
        else:
            i += 1
    # Inline triplets: flag only when habitual.
    trips = [(m.start(), m.end()) for m in doc.finditer(_TRIPLET, prose_only=True)]
    min_count = ctx.t("R03.min", 3)
    if len(trips) >= min_count and per_n_words(len(trips), doc.word_count, 150) >= ctx.t("R03.per150", 1.0):
        out.append(make(
            "R03", trips, severity=2, confidence=0.55, scope="document",
            message=f"Habitual three-item lists ({len(trips)} found)",
            evidence=f"{per_n_words(len(trips), doc.word_count, 150):.1f} triplets per 150 words",
        ))
    return out


_REVELATION = compile_phrases(REVELATION)
_WEAK_REVELATION = {"it turns out", "turns out", "but wait", "the truth is", "the reality is"}


@rule(
    "R04", "Manufactured revelation", CAT, 3,
    "Teasing an insight before delivering it (\"Here's what most people miss\").",
    "Delete the teaser and lead with the insight itself.",
    "remove",
)
def r04(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for m in doc.finditer(_REVELATION, prose_only=True):
        weak = m.group(0).lower().replace("’", "'") in _WEAK_REVELATION
        out.append(make("R04", [(m.start(), m.end())], severity=2 if weak else 3, confidence=0.5 if weak else 0.8))
    return out


_CLOSER = compile_phrases(DRAMATIC_CLOSERS)
_CLOSER_START = re.compile(r"^(and that|and this|that['’]s|this is|that is|it['’]s|which is|and it|so yes|period)\b", re.I)


@rule(
    "R05", "Dramatic one-line conclusion", CAT, 3,
    "A short, portentous sentence that closes a paragraph without adding meaning (\"And that changes everything.\").",
    "Cut it, or replace it with the concrete consequence you mean.",
    "remove",
)
def r05(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    prose = [p for p in doc.paragraphs if p.kind in ("prose", "quote") and p.sentences]
    for k, p in enumerate(prose):
        last = doc.sentences[p.sentences[-1]]
        if last.n_words > 9:
            continue
        txt = last.text.strip().strip("*_")
        if _CLOSER.search(txt):
            out.append(make("R05", [(last.start, last.end)], confidence=0.85))
        elif len(p.sentences) == 1 and last.n_words <= 6 and k > 0 and _CLOSER_START.match(txt):
            out.append(make("R05", [(last.start, last.end)], severity=2, confidence=0.55,
                            evidence="Standalone one-line paragraph used as a punchline."))
    return out


_VS = re.compile(r"\b(?:vs\.?|versus)\s|\bless\s+\w+(?:\s+\w+)?,\s+more\s+\w+|\bfrom\s+\w+\s+to\s+\w+,\s+from\b", re.I)


@rule(
    "R06", "Repeated binary framing", CAT, 3,
    "Multiple X-versus-Y or not-X-but-Y constructions across paragraphs; everything becomes a dichotomy.",
    "Keep the one contrast that matters; describe the rest on its own terms.",
    "vary",
)
def r06(doc: Document, ctx: Context) -> list[Finding]:
    spans = _contrast_spans(doc) + [(m.start(), m.end()) for m in doc.finditer(_VS, prose_only=True)]
    spans = merge_spans(spans)
    paras = {doc.sentence_at(a).para for a, _ in spans if doc.sentence_at(a)}
    if len(spans) >= ctx.t("R06.min", 3) and len(paras) >= 2:
        return [make(
            "R06", spans, scope="document", confidence=min(0.9, 0.5 + 0.1 * len(spans)),
            message=f"Binary framing used {len(spans)}× across {len(paras)} paragraphs",
        )]
    return []
