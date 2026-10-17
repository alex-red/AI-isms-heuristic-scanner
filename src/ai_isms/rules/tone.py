"""F. Performative tone (T01-T06)."""

from __future__ import annotations

import re

from ..lexicon import EMOTIVE, FALSE_BALANCE, FAMILIARITY, HEDGES, HONESTY, PROFUNDITY, VALIDATION, compile_phrases
from ..models import Finding
from ..text import EMOJI_RE, Document
from . import Context, make, rule
from ._util import per_n_words

CAT = "performative_tone"

_VALID = compile_phrases(VALIDATION)


@rule(
    "T01", "Unsolicited validation", CAT, 3,
    "Praise for the question or the reader (\"Great question!\", \"You're absolutely right\") with no relevant reason.",
    "Delete it and answer directly.",
    "remove",
)
def t01(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for m in doc.finditer(_VALID):
        s = doc.sentence_at(m.start())
        opening = s is not None and s.index <= 1
        out.append(make("T01", [(m.start(), m.end())], severity=3 if opening else 2, confidence=0.9 if opening else 0.6))
    return out


_PROF = compile_phrases(PROFUNDITY)


@rule(
    "T02", "False profundity", CAT, 3,
    "Inspirational or philosophical conclusions not earned by the material (\"It's a journey, not a destination\").",
    "End on the concrete takeaway instead of a maxim.",
    "remove",
)
def t02(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    prose = [p for p in doc.paragraphs if p.kind in ("prose", "quote")]
    last_para = prose[-1].index if prose else -1
    for m in doc.finditer(_PROF, prose_only=True):
        s = doc.sentence_at(m.start())
        closing = s is not None and s.para == last_para
        out.append(make("T02", [(m.start(), m.end())], confidence=0.8 if closing else 0.6,
                        evidence="Appears in the closing paragraph." if closing else ""))
    return out


_EMOTE = compile_phrases(EMOTIVE)


@rule(
    "T03", "Artificial emotional intensity", CAT, 3,
    "Exaggerated enthusiasm or drama unrelated to the task: exclamation marks, hype words, emoji in prose.",
    "Let the content carry the excitement; remove hype words and most exclamation marks.",
    "remove", "lexical",
)
def t03(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    words = doc.word_count
    bangs = [m for m in doc.finditer(re.compile(r"!+"), prose_only=True)]
    if len(bangs) >= 3 and per_n_words(len(bangs), words, 100) > ctx.t("T03.bang_per100", 1.0):
        out.append(make("T03", [(m.start(), m.end()) for m in bangs], scope="document", severity=2, confidence=0.75,
                        message=f"Exclamation marks: {len(bangs)} ({per_n_words(len(bangs), words, 100):.1f} per 100 words)"))
    emo = list(doc.finditer(_EMOTE, prose_only=True))
    if len(emo) >= 2:
        out.append(make("T03", [(m.start(), m.end()) for m in emo], scope="document",
                        severity=3 if len(emo) >= 4 else 2, confidence=min(0.85, 0.45 + 0.1 * len(emo)),
                        message=f"Hype vocabulary ({len(emo)}×)"))
    emoji = [m for m in doc.finditer(EMOJI_RE, prose_only=True)
             if (s := doc.sentence_at(m.start())) and doc.paragraphs[s.para].kind == "prose"]
    if len(emoji) >= 2:
        out.append(make("T03", [(m.start(), m.end()) for m in emoji], scope="document", severity=2, confidence=0.7,
                        message=f"Emoji used for emphasis in prose ({len(emoji)}×)"))
    return out


_HONEST = compile_phrases(HONESTY)


@rule(
    "T04", "Performative honesty", CAT, 2,
    "Announcing candour (\"I'll be completely honest\", \"No sugarcoating\") instead of simply being candid.",
    "Drop the announcement; the statement is either honest or it isn't.",
    "remove",
)
def t04(doc: Document, ctx: Context) -> list[Finding]:
    return [make("T04", [(m.start(), m.end())], confidence=0.8) for m in doc.finditer(_HONEST, prose_only=True)]


_HEDGE = compile_phrases(HEDGES)
_DOUBLE_HEDGE = re.compile(
    r"\b(?:could|may|might|can)\s+(?:potentially|possibly|perhaps|conceivably)\b|\bpotentially\s+(?:could|may|might)\b"
    r"|\bmight\s+possibly\b|\bperhaps\s+(?:may|might|could)\b", re.I,
)
_BALANCE = compile_phrases(FALSE_BALANCE)


@rule(
    "T05", "Excessive hedging or false balance", CAT, 3,
    "Stacked qualifiers, or refusing a warranted conclusion with \"it depends\" / \"there's no one-size-fits-all\".",
    "Commit to the conclusion the evidence supports, and state the specific condition under which it changes.",
    "clarify",
)
def t05(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    for m in doc.finditer(_DOUBLE_HEDGE, prose_only=True):
        out.append(make("T05", [(m.start(), m.end())], severity=2, confidence=0.85, message="Double hedge"))
    for m in doc.finditer(_BALANCE, prose_only=True):
        out.append(make("T05", [(m.start(), m.end())], confidence=0.75, message="False balance / non-answer"))
    min_hedges = int(ctx.t("T05.per_sentence", 3))
    for s in doc.sentences:
        if doc.paragraphs[s.para].kind not in ("prose", "quote", "list_item"):
            continue
        hits = list(_HEDGE.finditer(doc.masked, s.start, s.end))
        if len(hits) >= min_hedges:
            out.append(make("T05", [(h.start(), h.end()) for h in hits], severity=2, confidence=0.6, scope="sentence",
                            message=f"{len(hits)} hedges in one sentence"))
    return out


_FAMILIAR = compile_phrases(FAMILIARITY)


@rule(
    "T06", "Inappropriate conversational familiarity", CAT, 2,
    "Chatty or reassuring asides (\"Let's dive in!\", \"Don't worry\", \"I hope this helps\") in technical, formal or factual writing.",
    "Remove the aside; keep the information.",
    "remove",
)
def t06(doc: Document, ctx: Context) -> list[Finding]:
    return [make("T06", [(m.start(), m.end())], confidence=0.75) for m in doc.finditer(_FAMILIAR, prose_only=True)]
