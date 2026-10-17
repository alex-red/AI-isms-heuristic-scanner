"""Shared helpers for rule implementations."""

from __future__ import annotations

import re
from functools import lru_cache

from ..lexicon import ABSTRACT_NOUNS
from ..text import WORD_RE, Document, Sentence, normalize_apostrophes

NUM_RE = re.compile(r"\d")
QUOTE_RE = re.compile(r"[\"“][^\"”\n]{3,}[\"”]")
EXAMPLE_RE = re.compile(r"\b(for example|for instance|e\.g\.|such as|i\.e\.|specifically|namely)\b", re.I)
# Acronyms of 3+ letters (API, HTTP), dotted/slashed identifiers, camelCase.
TECH_TOKEN_RE = re.compile(r"\b(?:[A-Z]{3,}s?|[A-Za-z]+[._/][A-Za-z]\w*|[a-z]+[A-Z]\w*)\b")


def per_n_words(count: float, words: int, n: int = 250) -> float:
    return count / max(words, 1) * n


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def specifics(doc: Document, s: Sentence) -> int:
    """Count concrete-detail signals in a sentence: numbers, names, quotes, code, examples."""
    seg = doc.masked[s.start : s.end]
    n = 0
    n += min(2, len(NUM_RE.findall(seg)))
    n += len(QUOTE_RE.findall(doc.text[s.start : s.end]))
    n += len(EXAMPLE_RE.findall(seg))
    n += len(TECH_TOKEN_RE.findall(seg))
    # Masked (code / URL) regions inside the sentence count as concrete.
    n += sum(1 for a, b in doc.protected if s.start <= a < s.end)
    # Proper nouns: capitalised words mid-clause (not after a label colon, bullet bold, or quote).
    for k, m in enumerate(WORD_RE.finditer(seg)):
        w = m.group(0)
        if k == 0 or not w[:1].isupper() or w.isupper() or w in ("I", "I'm", "I've", "I'll", "I'd"):
            continue
        prev = seg[: m.start()].rstrip()[-1:]
        if prev.isalnum() or prev in ",(":
            n += 1
    return n


def abstract_words(s: Sentence) -> list[str]:
    return [w for w in s.words if normalize_apostrophes(w.lower()) in ABSTRACT_NOUNS]


@lru_cache(maxsize=None)
def _cached_regex(pattern: str, flags: int) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def rx(pattern: str, flags: int = re.I) -> re.Pattern[str]:
    return _cached_regex(pattern, flags)


def is_sentence_start(doc: Document, offset: int) -> bool:
    s = doc.sentence_at(offset)
    if s is None:
        return False
    lead = doc.masked[s.start : offset]
    return not re.search(r"[\w]", lead)
