"""C. Mechanical structure and cadence (S01-S07)."""

from __future__ import annotations

import re
import statistics

from ..models import Finding
from ..text import BOLD_RE, EMOJI_RE, Document
from . import Context, make, rule
from ._util import merge_spans, per_n_words

CAT = "structure_rhythm"

_DASH = re.compile(r"—|\s–\s|\s--\s")


@rule(
    "S01", "Excessive em-dash usage", CAT, 1,
    "Em dashes used far more often than typical prose. One or two are fine; a dash in every other sentence is a tic.",
    "Use commas, colons, parentheses or a full stop for most of them.",
    "vary", "statistical",
)
def s01(doc: Document, ctx: Context) -> list[Finding]:
    dashes = list(doc.finditer(_DASH, prose_only=True))
    rate = per_n_words(len(dashes), doc.word_count)
    limit = ctx.t("S01.per250", 3.0)
    if len(dashes) >= 3 and rate > limit:
        sev = 1 if rate < limit * 1.7 else 2 if rate < limit * 2.7 else 3
        return [make(
            "S01", [(m.start(), m.end()) for m in dashes], severity=sev, confidence=0.9, scope="document",
            message=f"Em-dash density {rate:.1f} per 250 words (threshold {limit:g})",
        )]
    return []


@rule(
    "S02", "Staccato rhythm", CAT, 3,
    "Runs of very short sentences or one-line paragraphs used for drama (\"It works. It scales. It ships.\").",
    "Combine related fragments into fuller sentences; save short ones for real emphasis.",
    "vary", "statistical",
)
def s02(doc: Document, ctx: Context) -> list[Finding]:
    sents = doc.prose_sentences
    max_words = int(ctx.t("S02.words", 6))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(sents):
        j = i
        while j < len(sents) and sents[j].n_words <= max_words and (
            j == i or sents[j].para - sents[j - 1].para <= 1  # don't bridge tables/headings
        ):
            j += 1
        if j - i >= 3:
            runs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    if len(runs) >= 2 or any(b - a >= 4 for a, b in runs):
        for a, b in runs:
            out.append(make(
                "S02", [(sents[a].start, sents[b - 1].end)],
                severity=3 if b - a >= 4 else 2, confidence=0.8, scope="paragraph",
                message=f"Staccato run: {b - a} consecutive sentences of ≤{max_words} words",
            ))
    # "Broetry": most paragraphs are a single sentence.
    prose = [p for p in doc.paragraphs if p.kind == "prose" and p.sentences]
    singles = [p for p in prose if len(p.sentences) == 1]
    if len(prose) >= 6 and len(singles) / len(prose) >= ctx.t("S02.single_para_share", 0.6):
        out.append(make(
            "S02", [(p.start, p.end) for p in singles], severity=3, confidence=0.75, scope="document",
            message=f"One-line paragraphs: {len(singles)} of {len(prose)} paragraphs are a single sentence",
        ))
    return out


def _cv(xs: list[int]) -> float:
    m = statistics.fmean(xs)
    return statistics.pstdev(xs) / m if m else 0.0


@rule(
    "S03", "Uniform sentence length", CAT, 3,
    "Sentence lengths with unusually low variation: a metronomic rhythm readers notice even when they can't name it.",
    "Mix long and short sentences deliberately; let complex ideas take longer sentences.",
    "vary", "statistical",
)
def s03(doc: Document, ctx: Context) -> list[Finding]:
    sents = [s for s in doc.prose_sentences if s.n_words >= 3]
    min_n = int(ctx.t("S03.min_sentences", 10))
    if len(sents) < min_n:
        return []
    lengths = [s.n_words for s in sents]
    cv = _cv(lengths)
    limit = ctx.t("S03.cv", 0.35)
    if cv < limit:
        return [make(
            "S03", [(sents[0].start, sents[-1].end)], scope="document",
            severity=4 if cv < limit - 0.12 else 3, confidence=min(0.9, 0.55 + (limit - cv) * 2),
            message=f"Uniform sentence lengths (coefficient of variation {cv:.2f}, threshold {limit:g})",
            evidence=f"{len(sents)} sentences, mean {statistics.fmean(lengths):.1f} words, "
                     f"stdev {statistics.pstdev(lengths):.1f}",
        )]
    # Local monotone stretches.
    win, local_limit = 8, ctx.t("S03.window_cv", 0.2)
    spans = [(sents[i].start, sents[i + win - 1].end) for i in range(len(sents) - win + 1)
             if _cv(lengths[i:i + win]) < local_limit]
    return [make("S03", [sp], severity=2, confidence=0.6, scope="paragraph",
                 message=f"Monotone stretch: {win}+ sentences of near-identical length") for sp in merge_spans(spans)]


@rule(
    "S04", "Identical paragraph architecture", CAT, 4,
    "Paragraphs built from the same template: same number of sentences, same opener, same short punchline at the end.",
    "Let paragraph shape follow content; not every paragraph needs a closing zinger.",
    "vary", "statistical",
)
def s04(doc: Document, ctx: Context) -> list[Finding]:
    paras = [p for p in doc.paragraphs if p.kind == "prose" and len(p.sentences) >= 2]
    out: list[Finding] = []
    if len(paras) < 3:
        return out

    def ends_short(p) -> bool:
        last = doc.sentences[p.sentences[-1]]
        body = [doc.sentences[i].n_words for i in p.sentences[:-1]]
        return last.n_words <= 8 and statistics.fmean(body) >= 1.6 * last.n_words

    # Pattern 1: most paragraphs end with a short punchline.
    punchy = [p for p in paras if ends_short(p)]
    if len(paras) >= 4 and len(punchy) / len(paras) >= ctx.t("S04.punch_share", 0.6):
        out.append(make(
            "S04", [(doc.sentences[p.sentences[-1]].start, doc.sentences[p.sentences[-1]].end) for p in punchy],
            scope="document", confidence=0.75,
            message=f"{len(punchy)} of {len(paras)} paragraphs end on a short punchline",
        ))
    # Pattern 2: runs of 3+ consecutive paragraphs sharing a signature.
    sig = [(len(p.sentences), ends_short(p), doc.sentences[p.sentences[0]].first_word) for p in paras]
    i = 0
    while i < len(paras):
        j = i + 1
        while j < len(paras) and sig[j][0] == sig[i][0] and sig[j][1] == sig[i][1] and (
            sig[j][2] == sig[i][2] or sig[i][1]
        ):
            j += 1
        if j - i >= 3:
            same_opener = len({s[2] for s in sig[i:j]}) == 1
            out.append(make(
                "S04", [(paras[k].start, paras[k].end) for k in range(i, j)], scope="section",
                confidence=0.8 if same_opener else 0.6,
                message=f"{j - i} consecutive paragraphs share one template ({sig[i][0]} sentences"
                        + (", same opener" if same_opener else "") + (", punchline ending" if sig[i][1] else "") + ")",
            ))
            i = j
        else:
            i += 1
    return out


@rule(
    "S05", "Unnecessary bolding, headings or emoji", CAT, 2,
    "Formatting density out of proportion to the text: bold every few words, headings over two-line sections, emoji bullets.",
    "Reserve bold for the one thing a skimmer must not miss; merge thin sections.",
    "remove", "statistical",
)
def s05(doc: Document, ctx: Context) -> list[Finding]:
    out = []
    words = max(doc.word_count, 1)
    bolds = list(BOLD_RE.finditer(doc.masked))
    if len(bolds) >= 4 and per_n_words(len(bolds), words, 100) > ctx.t("S05.bold_per100", 2.5):
        out.append(make(
            "S05", [(m.start(), m.end()) for m in bolds], scope="document", confidence=0.75,
            message=f"Heavy bolding: {len(bolds)} bold spans ({per_n_words(len(bolds), words, 100):.1f} per 100 words)",
        ))
    heads = [p for p in doc.paragraphs if p.kind == "heading"]
    if len(heads) >= 3 and words < ctx.t("S05.short_doc", 400) and words / len(heads) < ctx.t("S05.words_per_heading", 60):
        out.append(make(
            "S05", [(p.start, p.end) for p in heads], scope="document", confidence=0.65,
            message=f"{len(heads)} headings for {words} words (~{words // len(heads)} words per section)",
        ))
    emoji_lines = []
    for p in doc.paragraphs:
        if p.kind in ("heading", "list_item") and EMOJI_RE.match(doc.text[p.start:p.end].lstrip("*_ ")):
            emoji_lines.append((p.start, p.end))
    if len(emoji_lines) >= ctx.t("S05.emoji_lines", 3):
        out.append(make("S05", emoji_lines, scope="document", confidence=0.8,
                        message=f"Emoji-led headings/bullets ({len(emoji_lines)})"))
    return out


@rule(
    "S06", "Excessive listification", CAT, 2,
    "Content chopped into bullets where connected prose would carry the reasoning better.",
    "Turn related bullets into a paragraph that states how the points connect.",
    "consolidate", "statistical",
)
def s06(doc: Document, ctx: Context) -> list[Finding]:
    items = [p for p in doc.paragraphs if p.kind == "list_item"]
    out = []
    if len(items) >= ctx.t("S06.min_items", 6):
        item_words = sum(doc.sentences[i].n_words for p in items for i in p.sentences)
        share = item_words / max(doc.word_count, 1)
        avg = item_words / len(items)
        if share > ctx.t("S06.share", 0.6) and avg < 14:
            out.append(make(
                "S06", [(p.start, p.end) for p in items], scope="document", confidence=0.6,
                message=f"{share:.0%} of the text is in {len(items)} short bullets (avg {avg:.0f} words)",
            ))
    lead = doc.markdown.get("bold_lead_items", 0)
    if lead >= 4 and lead / max(len(items), 1) >= 0.5:
        spans = [(p.start, p.end) for p in items
                 if re.match(r"\s*(\*\*|__)[^*_\n]{1,60}?(:\1|\1\s*[:—–-])", doc.text[p.start:p.end])]
        out.append(make("S06", spans, scope="document", confidence=0.65,
                        message=f"“**Label:** explanation” bullet pattern ({lead} items)"))
    return out


@rule(
    "S07", "Repeated sentence starters", CAT, 1,
    "Three or more nearby sentences opening the same way (\"This… This… This…\").",
    "Vary openings, or merge the sentences so the shared subject appears once.",
    "vary", "statistical",
)
def s07(doc: Document, ctx: Context) -> list[Finding]:
    sents = [s for s in doc.sentences if doc.paragraphs[s.para].kind in ("prose", "quote", "list_item") and s.words]

    def key(s) -> str:
        w = [x.lower() for x in s.words[:2]]
        return " ".join(w) if w[0] in ("the", "a", "an") and len(w) > 1 else w[0]

    keys = [key(s) for s in sents]
    win = int(ctx.t("S07.window", 5))
    need = int(ctx.t("S07.min", 3))
    flagged: dict[str, set[int]] = {}
    for i in range(len(sents)):
        window = range(i, min(i + win, len(sents)))
        same = [j for j in window if keys[j] == keys[i] and doc.paragraphs[sents[j].para].section == doc.paragraphs[sents[i].para].section]
        if len(same) >= need:
            flagged.setdefault(keys[i], set()).update(same)
    out = []
    for k, idx in flagged.items():
        idx_sorted = sorted(idx)
        in_list = all(doc.paragraphs[sents[j].para].kind == "list_item" for j in idx_sorted)
        spans = []
        for j in idx_sorted:
            s = sents[j]
            m = re.search(r"[^\W_][\w'’\-]*(?:\s+[\w'’\-]+)?" if " " in k else r"[^\W_][\w'’\-]*", doc.masked[s.start:s.end])
            if m:
                spans.append((s.start + m.start(), s.start + m.end()))
        out.append(make(
            "S07", spans, severity=1 if in_list else 2, confidence=0.4 if in_list else 0.75,
            message=f"{len(idx_sorted)} nearby sentences start with “{k.capitalize()}”",
        ))
    return out
