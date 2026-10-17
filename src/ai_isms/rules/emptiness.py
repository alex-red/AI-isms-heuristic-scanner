"""D. Semantic emptiness and information density (C01-C07).

These are the most important rules and the hardest to do without a semantic
model. The local versions are *proxies* built on specificity signals (numbers,
names, quotes, code, examples) versus abstract nouns. They are reported with
modest confidence; the optional LLM pass (``--llm``) confirms, dismisses or adds
to them.
"""

from __future__ import annotations

import re

from ..lexicon import GENERIC_ADVICE, SIGNIFICANCE, SWEEPING, VAGUE_REFERENTS, compile_phrases
from ..models import Finding
from ..text import Document, content_stems
from . import Context, make, rule
from ._util import abstract_words, per_n_words, specifics

CAT = "semantic_emptiness"

_VAGUE_VERBS = re.compile(
    r"\b(chang|transform|reshap|redefin|revolutioniz|creat|driv|enabl|unlock|empower|shap|elevat|enhanc|improv|"
    r"foster|harness|leverag|navigat|streamlin|optimiz|boost|maximiz|accelerat)\w*",
    re.I,
)


def _prose(doc: Document):
    return [s for s in doc.sentences if doc.paragraphs[s.para].kind in ("prose", "quote", "list_item")]


@rule(
    "C01", "Says little despite its length", CAT, 4,
    "A long sentence built from abstractions (solutions, opportunities, innovation, growth) with no concrete subject, "
    "mechanism, example or number.",
    "Name who does what, how, and with what measurable result, or cut the sentence.",
    "substantiate", "semantic-local",
)
def c01(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    min_words = int(ctx.t("C01.min_words", 14))
    for s in _prose(doc):
        if s.n_words < min_words:
            continue
        abstract = abstract_words(s)
        if specifics(doc, s) == 0 and len(abstract) >= ctx.t("C01.min_abstract", 3):
            verbs = bool(_VAGUE_VERBS.search(s.text))
            out.append(make(
                "C01", [(s.start, s.end)], scope="sentence",
                severity=4 if len(abstract) >= 5 else 3,
                confidence=0.6 if verbs else 0.45,
                evidence=f"No concrete details; abstract nouns: {', '.join(dict.fromkeys(a.lower() for a in abstract))}",
            ))
    return out


_COPULA_SPLIT = re.compile(r"\s+(?:because|which means|meaning that|so that|so|since|as a result|therefore)\s+", re.I)
_IMPORTANCE = {"important", "matter", "matters", "crucial", "essential", "vital", "key", "significant", "critical", "valuable"}


@rule(
    "C02", "Tautology", CAT, 4,
    "A conclusion that restates its premise (\"X matters because it is important\"; \"it is what it is\").",
    "Supply the actual reason, consequence or evidence.",
    "substantiate", "semantic-local",
)
def c02(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for s in _prose(doc):
        t = s.text
        if re.search(r"\bit is what it is\b", t, re.I):
            out.append(make("C02", [(s.start, s.end)], confidence=0.7, scope="sentence"))
            continue
        parts = _COPULA_SPLIT.split(t, maxsplit=1)
        if len(parts) != 2:
            continue
        left, right = (set(content_stems(re.findall(r"[\w'’]+", p))) for p in parts)
        if not left or not right:
            continue
        jac = len(left & right) / len(left | right)
        both_importance = any(w in _IMPORTANCE for w in left) and any(w in _IMPORTANCE for w in right)
        if (jac >= 0.5 and min(len(left), len(right)) >= 2) or both_importance:
            out.append(make(
                "C02", [(s.start, s.end)], scope="sentence", confidence=0.6 if both_importance else 0.45,
                evidence=f"Premise and conclusion share {jac:.0%} of their content words"
                         + ("; both halves just assert importance" if both_importance else ""),
            ))
    return out


_SWEEP = compile_phrases(SWEEPING)


@rule(
    "C03", "Unsupported sweeping claim", CAT, 4,
    "Grand claims (\"changes everything\", \"unprecedented\", \"the future of work\") with no support in the surrounding text.",
    "Scope the claim and back it with a number, source or example, or soften it to what you can show.",
    "substantiate", "semantic-local",
)
def c03(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for m in doc.finditer(_SWEEP, prose_only=True):
        s = doc.sentence_at(m.start())
        supported = s is not None and specifics(doc, s) > 0
        out.append(make("C03", [(m.start(), m.end())], severity=3 if supported else 4,
                        confidence=0.45 if supported else 0.7,
                        evidence="Sentence contains some concrete detail." if supported else "No supporting detail in the sentence."))
    return out


_VAGUE = compile_phrases(VAGUE_REFERENTS)


@rule(
    "C04", "Vague referents and actors", CAT, 2,
    "Heavy reliance on \"things\", \"people\", \"solutions\", \"various factors\" instead of identifiable subjects.",
    "Name the specific people, objects, or factors.",
    "clarify", "lexical",
)
def c04(doc: Document, ctx: Context) -> list[Finding]:
    hits = list(doc.finditer(_VAGUE, prose_only=True))
    density = per_n_words(len(hits), doc.word_count, 100)
    if len(hits) >= 4 and density >= ctx.t("C04.per100", 2.0):
        return [make(
            "C04", [(m.start(), m.end()) for m in hits], scope="document",
            severity=3 if density >= 4 else 2, confidence=min(0.85, 0.45 + density / 10),
            message=f"Vague referents ({len(hits)}×, {density:.1f} per 100 words)",
        )]
    return []


_SIG = compile_phrases(SIGNIFICANCE)


@rule(
    "C05", "Manufactured significance", CAT, 3,
    "Labelling an ordinary development as historic, groundbreaking or game-changing.",
    "Describe what changed and let the reader judge its importance.",
    "clarify",
)
def c05(doc: Document, ctx: Context) -> list[Finding]:
    hits = list(doc.finditer(_SIG, prose_only=True))
    return [make("C05", [(m.start(), m.end())], confidence=0.7 if len(hits) >= 2 else 0.55) for m in hits]


@rule(
    "C06", "Abstract filler instead of specifics", CAT, 4,
    "Several consecutive sentences with no examples, facts, mechanisms or useful distinctions.",
    "Add one concrete example, number, or mechanism per claim, or compress the run to a single sentence.",
    "substantiate", "semantic-local",
)
def c06(doc: Document, ctx: Context) -> list[Finding]:
    sents = [s for s in doc.sentences if doc.paragraphs[s.para].kind in ("prose", "quote")]
    flags = []
    for s in sents:
        stems = content_stems(s.words)
        abstract = abstract_words(s)
        flags.append(s.n_words >= 6 and specifics(doc, s) == 0 and stems and len(abstract) / max(len(stems), 1) >= 0.12)
    out = []
    need = int(ctx.t("C06.run", 3))
    i = 0
    while i < len(sents):
        j = i
        while j < len(sents) and flags[j]:
            j += 1
        run_words = sum(s.n_words for s in sents[i:j])
        if j - i >= need and run_words >= 40:
            out.append(make(
                "C06", [(sents[i].start, sents[j - 1].end)], scope="paragraph",
                severity=4 if j - i >= need + 2 else 3, confidence=min(0.75, 0.45 + 0.06 * (j - i)),
                message=f"{j - i} consecutive sentences with no concrete detail ({run_words} words)",
            ))
            i = j
        else:
            i += 1
    return out


_ADVICE = compile_phrases(GENERIC_ADVICE)


@rule(
    "C07", "Generic recommendation", CAT, 3,
    "Advice that could be pasted into any context unchanged (\"Stay curious\", \"Communicate clearly\", \"It's essential to…\").",
    "Make it specific to this situation: what exactly to do, when, and how to tell it worked.",
    "substantiate", "semantic-local",
)
def c07(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for m in doc.finditer(_ADVICE, prose_only=True):
        s = doc.sentence_at(m.start())
        concrete = s is not None and specifics(doc, s) > 0
        out.append(make("C07", [(m.start(), m.end())], severity=2 if concrete else 3,
                        confidence=0.35 if concrete else 0.6))
    return out
