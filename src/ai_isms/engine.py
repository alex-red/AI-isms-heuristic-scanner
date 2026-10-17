"""Analysis pipeline: preprocess -> deterministic rules -> (optional LLM) -> merge -> score."""

from __future__ import annotations

import re
import statistics
import time
from typing import TYPE_CHECKING, Any, Iterable

from . import __version__
from .lexicon import compile_phrases, AI_VOCAB
from .models import Finding, Report, SentenceNote
from .profiles import get_profile
from .rules import CHECKS, RULES, Context
from .rules._util import abstract_words, specifics
from .scoring import score
from .text import Document, content_stems, parse

if TYPE_CHECKING:
    from .llm import LLMOptions

# Findings that, when covering most of a sentence, mark it as non-contributing.
_ZERO_RULES = {"R02", "R04", "R05", "T01", "T02", "T04", "T06", "L03"}
_DEDUP_SCOPES = {"span", "sentence", "paragraph"}


def _coverage(f: Finding) -> int:
    return sum(s.end - s.start for s in f.spans)


def consolidate(findings: list[Finding]) -> None:
    """Down-weight findings that describe the same defect on the same text.

    One phrase should not collect full penalties for buzzwords, abstraction and
    low density at once. The strongest finding keeps weight 1; overlapping
    findings from other rules keep their highlight but count for 35%.
    """
    order = sorted(findings, key=lambda f: -(f.severity * f.confidence))
    kept: list[Finding] = []
    for f in order:
        if f.review and f.review.get("verdict") == "dismiss":
            continue
        if f.scope not in _DEDUP_SCOPES or _coverage(f) > 1500:
            kept.append(f)
            continue
        cov = max(_coverage(f), 1)
        for g in kept:
            if g.rule_id == f.rule_id or g.scope not in _DEDUP_SCOPES:
                continue
            ov = sum(a.overlaps(b) for a in f.spans for b in g.spans)
            if ov / cov >= 0.6:
                f.weight = min(f.weight, 0.35)
                f.related.append(g.id)
                g.related.append(f.id)
                break
        kept.append(f)


def estimate_contributions(doc: Document, findings: list[Finding]) -> list[SentenceNote]:
    """Local estimate of each sentence's semantic contribution (0/1/2)."""
    zero_cover: dict[int, int] = {}
    dup_later: set[int] = set()
    for f in findings:
        if f.rule_id == "D01" and len(f.spans) == 2:
            s = doc.sentence_at(f.spans[1].start)
            if s:
                dup_later.add(s.index)
        if f.rule_id in _ZERO_RULES:
            for sp in f.spans:
                s = doc.sentence_at(sp.start)
                if s:
                    zero_cover[s.index] = zero_cover.get(s.index, 0) + (sp.end - sp.start)

    notes = []
    seen: set[str] = set()
    for s in doc.sentences:
        if doc.paragraphs[s.para].kind == "heading":
            continue
        st = set(content_stems(s.words))
        novelty = len(st - seen) / len(st) if st else 0.0
        spec = specifics(doc, s)
        covered = zero_cover.get(s.index, 0) / max(s.end - s.start, 1)
        if s.index in dup_later:
            c, why = 0, "Repeats an earlier sentence"
        elif covered >= 0.5 or (s.n_words <= 9 and covered >= 0.3):
            c, why = 0, "Mostly rhetorical framing"
        elif not st or (novelty < 0.25 and spec == 0):
            c, why = 0, "No new content words or concrete detail"
        elif (spec >= 1 and novelty >= 0.3) or (novelty >= 0.6 and len(st) >= 5 and not abstract_words(s)):
            c, why = 2, "Adds new terms" + (" with concrete detail" if spec else "")
        else:
            c, why = 1, "Supporting / partly new"
        notes.append(SentenceNote(s.index, s.start, s.end, c, "local", why, s.n_words))
        seen |= st
    return notes


def _flesch(doc: Document) -> float | None:
    sents = doc.prose_sentences
    words = [w for s in sents for w in s.words if w.isalpha()]
    if len(sents) < 3 or len(words) < 30:
        return None

    def syl(w: str) -> int:
        w = w.lower()
        groups = re.findall(r"[aeiouy]+", w)
        n = len(groups)
        if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
            n -= 1
        return max(1, n)

    syllables = sum(syl(w) for w in words)
    return round(206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syllables / len(words)), 1)


def _mattr(words: list[str], window: int = 50) -> float | None:
    words = [w.lower() for w in words]
    if len(words) < window:
        return round(len(set(words)) / len(words), 3) if words else None
    ratios = [len(set(words[i:i + window])) / window for i in range(0, len(words) - window + 1, 5)]
    return round(statistics.fmean(ratios), 3)


_AI_VOCAB = compile_phrases(AI_VOCAB)


def compute_metrics(doc: Document, notes: list[SentenceNote]) -> dict[str, Any]:
    prose = doc.prose_sentences
    lengths = [s.n_words for s in prose]
    all_words = [w for s in prose for w in s.words]
    mean = statistics.fmean(lengths) if lengths else 0.0
    sd = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    dashes = len(re.findall(r"—|\s–\s|\s--\s", doc.masked))
    vocab_hits = sum(1 for _ in doc.finditer(_AI_VOCAB, prose_only=True))
    spec = [specifics(doc, s) for s in prose]
    abstract = sum(len(abstract_words(s)) for s in prose)
    contrib = {k: sum(1 for n in notes if n.contribution == k) for k in (0, 1, 2)}
    return {
        "words": doc.word_count,
        "sentences": len(doc.sentences),
        "prose_sentences": len(prose),
        "paragraphs": sum(1 for p in doc.paragraphs if p.kind in ("prose", "quote")),
        "mean_sentence_words": round(mean, 1),
        "sentence_words_stdev": round(sd, 1),
        "sentence_length_cv": round(sd / mean, 3) if mean else None,
        "sentence_lengths": [{"i": s.index, "n": s.n_words} for s in prose],
        "em_dashes": dashes,
        "em_dashes_per_250": round(dashes / max(doc.word_count, 1) * 250, 2),
        "ai_vocab_hits": vocab_hits,
        "ai_vocab_per_1000": round(vocab_hits / max(doc.word_count, 1) * 1000, 1),
        "specifics_per_sentence": round(statistics.fmean(spec), 2) if spec else 0.0,
        "abstract_noun_share": round(abstract / max(len(all_words), 1), 3),
        "lexical_diversity_mattr": _mattr(all_words),
        "flesch_reading_ease": _flesch(doc),
        "contribution": contrib,
        "markdown": doc.markdown,
    }


def analyze(
    text: str,
    *,
    profile: str = "general",
    disabled: Iterable[str] = (),
    only: Iterable[str] | None = None,
    thresholds: dict[str, float] | None = None,
    llm: "LLMOptions | None" = None,
    source: str | None = None,
) -> Report:
    t0 = time.perf_counter()
    prof = get_profile(profile)
    disabled = {d.upper() for d in disabled}
    only_set = {o.upper() for o in only} if only else None
    ctx = Context(
        profile=profile,
        severity_scale=dict(prof["scale"]),
        thresholds={**prof["thresholds"], **(thresholds or {})},
        disabled=disabled,
    )
    doc = parse(text)

    findings: list[Finding] = []
    for rid, check in CHECKS.items():
        if rid in disabled or (only_set and rid not in only_set):
            continue
        scale = ctx.severity_scale.get(rid, 1.0)
        if scale <= 0:
            continue
        for f in check(doc, ctx):
            if scale != 1.0:
                f.severity = max(1, min(5, round(f.severity * scale)))
                if scale < 0.6:
                    f.confidence = round(f.confidence * 0.8, 2)
            findings.append(f)

    findings.sort(key=lambda f: (f.spans[0].start if f.spans else 0, f.rule_id))
    for i, f in enumerate(findings, 1):
        f.id = f"F{i:03d}"

    notes = estimate_contributions(doc, findings)
    llm_info = None
    if llm is not None:
        from .llm import run_llm

        result = run_llm(doc, findings, llm, profile=profile)
        llm_info = result.info
        for k, f in enumerate(result.findings, 1):
            f.id = f"L{k:03d}"
        findings.extend(result.findings)
        if result.contributions:
            by_idx = {n.index: n for n in notes}
            for idx, (c, why) in result.contributions.items():
                if idx in by_idx:
                    by_idx[idx].contribution, by_idx[idx].reason, by_idx[idx].source = c, why, "llm"
        findings.sort(key=lambda f: (f.spans[0].start if f.spans else 0, f.rule_id))

    for f in findings:
        if f.review and f.review.get("verdict") == "dismiss":
            f.weight = 0.0
        elif f.review and f.review.get("verdict") == "confirm":
            f.confidence = max(f.confidence, 0.85)
    consolidate(findings)

    metrics = compute_metrics(doc, notes)
    scores = score(findings, notes, doc.word_count, llm=bool(llm_info and llm_info.get("ok")))
    rules_used = {k: v for k, v in RULES.items()}
    metrics["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return Report(
        text=doc.text,
        findings=findings,
        sentences=notes,
        metrics=metrics,
        scores=scores,
        rules=rules_used,
        config={
            "profile": profile,
            "profile_description": prof["description"],
            "disabled": sorted(disabled),
            "only": sorted(only_set) if only_set else None,
            "source": source,
            "version": __version__,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        llm=llm_info,
        paragraphs=[{"start": p.start, "end": p.end, "kind": p.kind, "level": p.level} for p in doc.paragraphs],
    )
