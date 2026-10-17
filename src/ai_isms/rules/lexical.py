"""B. Generic language and lexical clichés (L01-L06)."""

from __future__ import annotations

import re

from ..lexicon import (
    AI_VOCAB, BUZZ_ADJECTIVES, CLICHE_METAPHORS, INTENSIFIERS, SCENE_SETTING,
    SENTENCE_INITIAL_TRANSITIONS, STOCK_TRANSITIONS, compile_phrases,
)
from ..models import Finding
from ..text import Document
from . import Context, make, rule
from ._util import is_sentence_start, per_n_words

CAT = "generic_vocabulary"

_AI_VOCAB = compile_phrases(AI_VOCAB)


@rule(
    "L01", "AI-associated vocabulary cluster", CAT, 2,
    "Several words that are over-represented in LLM output (delve, tapestry, pivotal, seamless…) "
    "concentrated in a short window. Single words are never flagged.",
    "Swap the cluster for plain, specific words; ask what each one actually means here.",
    "vary", "lexical",
)
def l01(doc: Document, ctx: Context) -> list[Finding]:
    hits = [(doc.word_index_at(m.start()), m.group(0).lower(), m.start(), m.end())
            for m in doc.finditer(_AI_VOCAB, prose_only=True)]
    window = int(ctx.t("L01.window", 150))
    need = int(ctx.t("L01.distinct", 3))
    out = []
    i = 0
    while i < len(hits):
        j = i
        while j < len(hits) and hits[j][0] - hits[i][0] < window:
            j += 1
        group = hits[i:j]
        distinct = {h[1] for h in group}
        if len(distinct) >= need:
            out.append(make(
                "L01", [(h[2], h[3]) for h in group],
                severity=3 if len(distinct) >= need + 3 else 2,
                confidence=min(0.9, 0.55 + 0.07 * (len(distinct) - need)),
                message=f"AI-associated vocabulary cluster ({len(distinct)} distinct terms within ~{window} words)",
                evidence="Terms: " + ", ".join(sorted(distinct)),
            ))
            i = j
        else:
            i += 1
    return out


_BUZZ = compile_phrases(BUZZ_ADJECTIVES)


@rule(
    "L02", "Corporate buzzword stacking", CAT, 3,
    "Several vague evaluative adjectives in a row (\"seamless, scalable, and innovative\") with no concrete qualification.",
    "Replace adjectives with the measurable property they gesture at (latency, price, steps saved).",
    "substantiate", "lexical",
)
def l02(doc: Document, ctx: Context) -> list[Finding]:
    hits = [(doc.word_index_at(m.start()), m.start(), m.end()) for m in doc.finditer(_BUZZ, prose_only=True)]
    out = []
    group: list[tuple[int, int, int]] = []

    def flush() -> None:
        if len(group) >= 2:
            s = doc.sentence_at(group[0][1])
            has_number = bool(s and re.search(r"\d", s.text))
            size = len(group)
            if size >= 3 or (group[-1][0] - group[0][0] <= 2):
                out.append(make(
                    "L02", [(group[0][1], group[-1][2])],
                    severity=3 if size >= 3 else 2,
                    confidence=(0.8 if size >= 3 else 0.55) * (0.7 if has_number else 1.0),
                    message=f"Buzzword stack ({size} vague adjectives)",
                ))

    for h in hits:
        same_sentence = group and doc.sentence_at(group[-1][1]) is doc.sentence_at(h[1])
        if group and same_sentence and h[0] - group[-1][0] <= 3:
            group.append(h)
        else:
            flush()
            group = [h]
    flush()
    return out


_SCENE = compile_phrases(SCENE_SETTING)


@rule(
    "L03", "Generic scene-setting introduction", CAT, 3,
    "Throat-clearing openers (\"In today's rapidly evolving landscape…\") that delay the point.",
    "Delete the opener and start with your first concrete claim.",
    "remove",
)
def l03(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for m in doc.finditer(_SCENE):
        s = doc.sentence_at(m.start())
        opening = s is not None and (s.index == 0 or doc.paragraphs[s.para].sentences[0] == s.index)
        out.append(make("L03", [(m.start(), m.end())], severity=3 if opening else 2,
                        confidence=0.9 if opening else 0.7))
    return out


_INTENS = compile_phrases(INTENSIFIERS)
_WEAK_INTENS = {"really", "very", "simply", "super", "totally", "literally", "entirely", "significantly"}


@rule(
    "L04", "Empty intensifiers", CAT, 1,
    "Adverbs like truly, fundamentally, incredibly that add emphasis but no meaning, at high density.",
    "Delete them; if the sentence feels weak without them, strengthen the claim with evidence.",
    "remove", "lexical",
)
def l04(doc: Document, ctx: Context) -> list[Finding]:
    hits = list(doc.finditer(_INTENS, prose_only=True))
    score = sum(0.5 if m.group(0).lower() in _WEAK_INTENS else 1.0 for m in hits)
    density = per_n_words(score, doc.word_count, 100)
    if score >= 3 and density >= ctx.t("L04.per100", 1.2):
        return [make(
            "L04", [(m.start(), m.end()) for m in hits],
            severity=2 if density >= 2.5 else 1, confidence=min(0.85, 0.5 + density / 10),
            scope="document", message=f"Empty intensifiers ({len(hits)}×, {density:.1f} per 100 words)",
        )]
    return []


_STOCK = compile_phrases(STOCK_TRANSITIONS)
_INITIAL = re.compile(r"\b(" + "|".join(SENTENCE_INITIAL_TRANSITIONS) + r"),", re.I)


@rule(
    "L05", "Stock transitional phrase", CAT, 1,
    "Filler connectives (\"It's worth noting\", \"That said\", \"Moreover,\") that signal structure without adding any.",
    "Cut the transition; let the logical relationship come from the content.",
    "remove",
)
def l05(doc: Document, ctx: Context) -> list[Finding]:
    spans = [(m.start(), m.end(), m.group(0).lower()) for m in doc.finditer(_STOCK, prose_only=True)]
    spans += [(m.start(), m.end() - 1, m.group(1).lower()) for m in doc.finditer(_INITIAL, prose_only=True)
              if is_sentence_start(doc, m.start())]
    distinct = {s[2] for s in spans}
    heavy = len(distinct) >= 3
    return [make("L05", [(a, b)], severity=2 if heavy else 1, confidence=0.8 if heavy else 0.6,
                 evidence=f"{len(spans)} stock transitions in this text ({len(distinct)} distinct).")
            for a, b, _ in spans]


_METAPHOR = compile_phrases(CLICHE_METAPHORS)


@rule(
    "L06", "Decorative / cliché metaphor", CAT, 2,
    "Stock figurative language (tapestry, north star, double-edged sword) that decorates rather than clarifies.",
    "Say the literal thing. Keep a metaphor only if it explains something a literal phrase can't.",
    "clarify",
)
def l06(doc: Document, ctx: Context) -> list[Finding]:
    hits = list(doc.finditer(_METAPHOR, prose_only=True))
    many = len(hits) >= 3
    return [make("L06", [(m.start(), m.end())], severity=2, confidence=0.75 if many else 0.6) for m in hits]
