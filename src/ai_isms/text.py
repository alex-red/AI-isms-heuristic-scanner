"""Preprocessing: protected spans, Markdown-aware paragraphs, sentences, words.

Everything here is offset-preserving so findings can point at exact character
ranges in the original text. Code blocks, inline code, URLs and HTML tags are
"masked" (replaced with spaces of equal length) so pattern rules never fire on
them, while offsets stay aligned with the original.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Iterator

# ---------------------------------------------------------------------------
# Regexes

FENCE_RE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.M | re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"\bhttps?://[^\s)>\]]+|\bwww\.[^\s)>\]]+")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]{0,200}>")
MD_LINK_TARGET_RE = re.compile(r"\]\(([^)\s]+)\)")

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+")
LIST_RE = re.compile(r"^([ \t]*)([-*+•]|\d{1,3}[.)])[ \t]+")
QUOTE_RE = re.compile(r"^[ \t]*>[ \t]?")
TABLE_RE = re.compile(r"^[ \t]*\|")
HR_RE = re.compile(r"^[ \t]*([-*_])([ \t]*\1){2,}[ \t]*$")

WORD_RE = re.compile(r"[^\W_](?:[\w’'\-]*[^\W_])?", re.UNICODE)
QUOTED_RE = re.compile(r"\"[^\"\n]{2,200}\"|“[^”\n]{2,200}”")
# A paragraph that is only a bold label, e.g. "**Setup** (optional)" -> treated like a heading.
BOLD_LABEL_RE = re.compile(r"^\s*(\*\*|__)[^*_\n]{1,80}\1\s*(\([^)\n]{0,60}\))?\s*:?\s*$")
BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF⭐⬆↔-↪✅❌✨]"
)

_BOUNDARY_RE = re.compile(r"([.!?…]+)([\"”’')\]]*)(?=\s|$)")

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "eg", "ie", "inc", "ltd", "co", "corp", "no", "fig", "approx", "est", "dept",
    "u.s", "u.k", "a.m", "p.m", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "cf", "al", "vol", "pp", "ca", "mt",
}

STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for from
    further had hadn't has hasn't have haven't having he he'd he'll he's her here
    here's hers herself him himself his how how's i i'd i'll i'm i've if in into
    is isn't it it's its itself let's me more most mustn't my myself no nor not of
    off on once only or other ought our ours ourselves out over own same shan't
    she she'd she'll she's should shouldn't so some such than that that's the
    their theirs them themselves then there there's these they they'd they'll
    they're they've this those through to too under until up very was wasn't we
    we'd we'll we're we've were weren't what what's when when's where where's which
    while who who's whom why why's will with won't would wouldn't you you'd you'll
    you're you've your yours yourself yourselves just also really can will may
    might must shall us one get got make made way""".split()
)


def normalize_apostrophes(s: str) -> str:
    return s.replace("’", "'").replace("‘", "'")


def stem(word: str) -> str:
    """Tiny suffix stripper. Good enough for overlap measures, not linguistics."""
    w = normalize_apostrophes(word.lower())
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 4:
        return w
    for suf, rep in (
        ("ational", "ate"), ("ization", "ize"), ("fulness", "ful"), ("iveness", "ive"),
        ("ousness", "ous"), ("ically", "ic"), ("ations", "ate"), ("ation", "ate"),
        ("ments", ""), ("ment", ""), ("ness", ""), ("ities", ""), ("ity", ""),
        ("ingly", ""), ("edly", ""), ("ing", ""), ("ies", "y"), ("ied", "y"),
        ("ed", ""), ("ly", ""), ("es", ""), ("s", ""),
    ):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: len(w) - len(suf)] + rep
            break
    return w


def content_stems(words: list[str]) -> list[str]:
    return [stem(w) for w in words if normalize_apostrophes(w.lower()) not in STOPWORDS and len(w) > 2]


# ---------------------------------------------------------------------------
# Data structures


@dataclass
class Sentence:
    index: int
    start: int
    end: int
    text: str
    para: int
    words: list[str]

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def first_word(self) -> str:
        return normalize_apostrophes(self.words[0].lower()) if self.words else ""


@dataclass
class Paragraph:
    index: int
    start: int
    end: int
    kind: str  # prose | heading | list_item | quote | code | table
    section: int
    sentences: list[int] = field(default_factory=list)
    level: int = 0  # heading level / list indent


@dataclass
class Document:
    text: str
    masked: str
    protected: list[tuple[int, int]]
    paragraphs: list[Paragraph]
    sentences: list[Sentence]
    word_starts: list[int]  # char offset of every prose word, sorted
    markdown: dict[str, int]
    quoted: list[tuple[int, int]] = field(default_factory=list)  # "quoted" text: mentions, not uses

    # -- convenience -------------------------------------------------------
    @property
    def word_count(self) -> int:
        return len(self.word_starts)

    @property
    def prose_sentences(self) -> list[Sentence]:
        return [s for s in self.sentences if self.paragraphs[s.para].kind in ("prose", "quote")]

    def sentence_at(self, offset: int) -> Sentence | None:
        starts = self._sentence_starts
        i = bisect_right(starts, offset) - 1
        if 0 <= i < len(self.sentences) and self.sentences[i].start <= offset < self.sentences[i].end:
            return self.sentences[i]
        return None

    def word_index_at(self, offset: int) -> int:
        return bisect_right(self.word_starts, offset)

    def finditer(self, pattern: re.Pattern[str], *, prose_only: bool = False) -> Iterator[re.Match[str]]:
        """Iterate regex matches over the masked text (never inside code/URLs)."""
        for m in pattern.finditer(self.masked):
            if m.start() == m.end():
                continue
            if self._in_quotes(m.start(), m.end()):
                continue
            if prose_only:
                s = self.sentence_at(m.start())
                if s is None or self.paragraphs[s.para].kind not in ("prose", "quote", "list_item"):
                    continue
            yield m

    def _in_quotes(self, a: int, b: int) -> bool:
        i = bisect_right(self._quote_starts, a) - 1
        return i >= 0 and self.quoted[i][0] <= a and b <= self.quoted[i][1]

    def __post_init__(self) -> None:
        self._sentence_starts = [s.start for s in self.sentences]
        self._quote_starts = [q[0] for q in self.quoted]


# ---------------------------------------------------------------------------
# Masking


def _mask(text: str) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]]]:
    chars = list(text)
    protected: list[tuple[int, int]] = []
    fences: list[tuple[int, int]] = []

    def blank(a: int, b: int) -> None:
        for i in range(a, b):
            if chars[i] != "\n":
                chars[i] = " "
        protected.append((a, b))

    for m in FENCE_RE.finditer(text):
        fences.append((m.start(), m.end()))
        blank(m.start(), m.end())
    interim = "".join(chars)
    for rx in (INLINE_CODE_RE, URL_RE, HTML_TAG_RE):
        for m in rx.finditer(interim):
            blank(m.start(), m.end())
        interim = "".join(chars)
    for m in MD_LINK_TARGET_RE.finditer(interim):
        blank(m.start(1), m.end(1))
    protected.sort()
    return "".join(chars), protected, fences


# ---------------------------------------------------------------------------
# Sentence splitting


def split_sentences(masked: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split masked[start:end] into sentence (start, end) offsets."""
    out: list[tuple[int, int]] = []
    seg = masked[start:end]
    cur = 0
    for m in _BOUNDARY_RE.finditer(seg):
        punct = m.group(1)
        stop = m.end()
        # Look at the token preceding the punctuation.
        before = seg[cur : m.start()]
        tok_m = re.search(r"(\S+)$", before)
        tok = tok_m.group(1).lower().strip("(\"'“‘") if tok_m else ""
        if punct == ".":
            if tok in ABBREVIATIONS or tok.rstrip(".") in ABBREVIATIONS:
                continue
            if len(tok) == 1 and tok.isalpha():  # initials: "J. Smith"
                continue
        # Do not split if the next non-space char is lowercase (e.g. "e.g. foo").
        nxt = re.match(r"\s*(\S)", seg[stop:])
        if nxt and nxt.group(1).islower() and punct != "…":
            continue
        a, b = cur, stop
        # Trim whitespace.
        while a < b and seg[a].isspace():
            a += 1
        if b > a:
            out.append((start + a, start + b))
        cur = stop
    a = cur
    while a < len(seg) and seg[a].isspace():
        a += 1
    b = len(seg)
    while b > a and seg[b - 1].isspace():
        b -= 1
    if b > a:
        out.append((start + a, start + b))
    return out


# ---------------------------------------------------------------------------
# Paragraphs


def _iter_blocks(masked: str) -> Iterator[tuple[int, int]]:
    pos = 0
    for m in re.finditer(r"\n[ \t]*\n+", masked):
        if m.start() > pos:
            yield pos, m.start()
        pos = m.end()
    if pos < len(masked):
        yield pos, len(masked)


def _line_spans(masked: str, a: int, b: int) -> list[tuple[int, int]]:
    spans = []
    pos = a
    while pos < b:
        nl = masked.find("\n", pos, b)
        e = b if nl == -1 else nl
        spans.append((pos, e))
        pos = e + 1
    return spans


def parse(text: str) -> Document:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    masked, protected, fences = _mask(text)

    paragraphs: list[Paragraph] = []
    section = 0
    md = {"headings": 0, "bold": 0, "list_items": 0, "emoji": 0, "tables": 0, "code_blocks": len(fences),
          "quotes": 0, "bold_lead_items": 0, "emoji_lead_lines": 0}

    def add(kind: str, a: int, b: int, level: int = 0) -> None:
        # trim whitespace
        while a < b and masked[a].isspace():
            a += 1
        while b > a and masked[b - 1].isspace():
            b -= 1
        if b > a:
            paragraphs.append(Paragraph(len(paragraphs), a, b, kind, section, level=level))

    for ba, bb in _iter_blocks(masked):
        block = masked[ba:bb]
        if not block.strip():
            # Entirely masked: a code block (or only URLs).
            if any(fa <= ba < fb or ba <= fa < bb for fa, fb in fences):
                paragraphs.append(Paragraph(len(paragraphs), ba, bb, "code", section))
            continue
        prose_start: int | None = None
        prose_end = ba
        for la, lb in _line_spans(masked, ba, bb):
            line = masked[la:lb]
            hm, lm, qm = HEADING_RE.match(line), LIST_RE.match(line), QUOTE_RE.match(line)
            special = hm or lm or TABLE_RE.match(line) or HR_RE.match(line)
            if special and prose_start is not None:
                add("prose", prose_start, prose_end)
                prose_start = None
            if hm:
                section += 1
                md["headings"] += 1
                add("heading", la + hm.end(), lb, level=len(hm.group(1)))
            elif lm:
                md["list_items"] += 1
                body = line[lm.end():]
                # "**Term:** explanation" or "**Term** — explanation"
                if re.match(r"\s*(\*\*|__)[^*_\n]{1,60}?(:\1|\1\s*[:—–-])", body):
                    md["bold_lead_items"] += 1
                add("list_item", la + lm.end(), lb, level=len(lm.group(1).expandtabs(4)) // 2)
            elif TABLE_RE.match(line):
                md["tables"] += 1
                add("table", la, lb)
            elif HR_RE.match(line):
                continue
            elif qm:
                md["quotes"] += 1
                if prose_start is not None:
                    add("prose", prose_start, prose_end)
                    prose_start = None
                add("quote", la + qm.end(), lb)
            else:
                # Continuation of a list item (indented line directly after it)?
                if paragraphs and paragraphs[-1].kind == "list_item" and prose_start is None and line[:1] in (" ", "\t") and paragraphs[-1].end >= la - 1:
                    paragraphs[-1].end = lb
                    continue
                if prose_start is None:
                    prose_start = la
                prose_end = lb
        if prose_start is not None:
            add("prose", prose_start, prose_end)

    for p in paragraphs:
        if p.kind == "prose" and BOLD_LABEL_RE.match(masked[p.start:p.end]):
            p.kind = "heading"

    md["bold"] = sum(1 for _ in BOLD_RE.finditer(masked))
    md["emoji"] = len(EMOJI_RE.findall(masked))
    md["emoji_lead_lines"] = sum(
        1 for la, lb in _line_spans(masked, 0, len(masked))
        if EMOJI_RE.match(re.sub(r"^\s*(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)?", "", masked[la:lb]))
    )

    sentences: list[Sentence] = []
    word_starts: list[int] = []
    for p in paragraphs:
        if p.kind in ("code", "table"):
            continue
        spans = [(p.start, p.end)] if p.kind == "heading" else split_sentences(masked, p.start, p.end)
        for a, b in spans:
            seg = masked[a:b]
            words = [w.group(0) for w in WORD_RE.finditer(seg)]
            if not words:
                continue
            if p.kind != "heading":
                word_starts.extend(a + w.start() for w in WORD_RE.finditer(seg))
            s = Sentence(len(sentences), a, b, text[a:b], p.index, words)
            p.sentences.append(s.index)
            sentences.append(s)

    quoted = [(m.start(), m.end()) for m in QUOTED_RE.finditer(masked)]
    return Document(text, masked, protected, paragraphs, sentences, sorted(word_starts), md, quoted)
