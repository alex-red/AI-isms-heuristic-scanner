"""Scan every benchmark sample and summarize how the scanner separates human from AI-style text.

    python benchmark/fetch.py                     # once: download the human baseline
    python benchmark/run.py                       # table + summary
    python benchmark/run.py --json results.json   # also save per-sample results
    python benchmark/run.py --compare benchmark/baseline.json   # show score changes vs a saved run
    python benchmark/run.py --label human         # only one label
    python benchmark/run.py --truncate 250        # cut all samples to about 250 words (length-matched comparison)
    python benchmark/run.py --ids ai-story-opening ai-cover-letter --llm   # add the semantic pass via `claude -p`

Standard library only. Runs the local rules; no network, no API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ai_isms import __version__, analyze  # noqa: E402
from ai_isms.llm import LLMOptions  # noqa: E402

THRESHOLDS = (12, 25, 45)  # Clean/Light, Light/Noticeable, Noticeable/Heavy
LABELS = ("human", "ai-typical", "ai-polished")


def load(entry: dict, truncate: int | None = None) -> str | None:
    path = HERE / entry["file"] if "file" in entry else HERE / "data" / f"{entry['id']}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if truncate:
        # Keep whole paragraphs until about `truncate` words, so long and short samples are comparable.
        kept, count = [], 0
        for para in text.split("\n\n"):
            kept.append(para)
            count += len(para.split())
            if count >= truncate:
                break
        text = "\n\n".join(kept)
    return text


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability that a random `pos` score exceeds a random `neg` score (ties count half)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def scan(manifest: dict, only_label: str | None, truncate: int | None, ids: list[str] | None = None,
         llm: LLMOptions | None = None) -> tuple[list[dict], list[str]]:
    todo, missing = [], []
    for entry in manifest["samples"]:
        if (only_label and entry["label"] != only_label) or (ids and entry["id"] not in ids):
            continue
        text = load(entry, truncate)
        if text is None:
            missing.append(entry["id"])
        else:
            todo.append((entry, text))

    def one(item: tuple[dict, str]) -> dict:
        entry, text = item
        report = analyze(text, profile=entry["profile"], source=entry["id"])
        row = {
            "id": entry["id"], "label": entry["label"], "genre": entry["genre"], "profile": entry["profile"],
            "words": report.metrics.get("words") or len(text.split()),
            "score": report.scores["overall"], "band": report.scores["band"],
            "dimensions": {k: v.get("score") if isinstance(v, dict) else v for k, v in report.scores["dimensions"].items()},
            "findings": Counter(f.rule_id for f in report.findings),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        }
        if llm:
            full = analyze(text, profile=entry["profile"], source=entry["id"], llm=llm)
            info = full.llm or {}
            row.update({
                "score_local": row["score"], "score": full.scores["overall"], "band": full.scores["band"],
                "llm_ok": info.get("ok", False), "llm_errors": info.get("errors", []), "llm_cost_usd": info.get("cost_usd"),
                "llm_summary": info.get("summary", ""), "llm_dismissed": info.get("dismissed", 0),
                "llm_findings": [{"rule_id": f.rule_id, "severity": f.severity, "confidence": f.confidence,
                                  "quote": full.text[f.spans[0].start:f.spans[0].end][:120]}
                                 for f in full.findings if f.detector == "llm"],
            })
        return row

    if llm:  # model calls dominate the time; overlap a few of them
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(one, todo))
    else:
        rows = [one(t) for t in todo]
    return rows, missing


def print_table(rows: list[dict], base: dict[str, float] | None) -> None:
    llm = any("score_local" in r for r in rows)
    print(f"{'sample':36}{'label':13}{'genre':11}{'profile':11}{'words':>6}" + (f"{'local':>7}{'+llm':>7}" if llm else f"{'score':>7}") + f"  {'band':11}" + ("  vs base" if base else ""))
    for label in LABELS:
        for r in sorted((r for r in rows if r["label"] == label), key=lambda r: r["score"] if r["score"] is not None else -1):
            score = "-" if r["score"] is None else f"{r['score']:.1f}"
            delta = ""
            if base and r["id"] in base and r["score"] is not None:
                delta = f"  {r['score'] - base[r['id']]:+.1f}"
            cols = f"{r['score_local']:>7.1f}{score:>7}" if "score_local" in r and r["score_local"] is not None else f"{score:>7}"
            print(f"{r['id']:36}{r['label']:13}{r['genre']:11}{r['profile']:11}{r['words']:>6}{cols}  {r['band']:11}{delta}")
        print()


def print_llm(rows: list[dict]) -> None:
    total = sum(r.get("llm_cost_usd") or 0 for r in rows)
    print(f"LLM pass: {sum(1 for r in rows if r.get('llm_ok'))}/{len(rows)} succeeded, ${total:.4f} total\n")
    for r in rows:
        print(f"{r['id']}  local {r['score_local']} -> {r['score']}  ({len(r.get('llm_findings', []))} LLM findings, "
              f"{r.get('llm_dismissed', 0)} local dismissed)")
        for err in r.get("llm_errors", []):
            print(f"  ERROR {err}")
        if r.get("llm_summary"):
            print(f"  summary: {r['llm_summary']}")
        for f in sorted(r.get("llm_findings", []), key=lambda f: -f["severity"] * f["confidence"])[:6]:
            print(f"  {f['rule_id']} sev{f['severity']} conf{f['confidence']:.2f}  “{f['quote']}”")
        print()


def print_summary(rows: list[dict]) -> None:
    graded = [r for r in rows if r["score"] is not None]
    by_label = {lab: [r["score"] for r in graded if r["label"] == lab] for lab in LABELS}

    print("Score by label")
    print(f"  {'label':13}{'n':>3}{'min':>7}{'median':>8}{'p90':>7}{'max':>7}   share >= " + " / ".join(map(str, THRESHOLDS)))
    for lab, s in by_label.items():
        if not s:
            continue
        s_sorted = sorted(s)
        p90 = s_sorted[min(len(s_sorted) - 1, int(0.9 * len(s_sorted)))]
        shares = " / ".join(pct(sum(x >= t for x in s), len(s)) for t in THRESHOLDS)
        print(f"  {lab:13}{len(s):>3}{min(s):>7.1f}{statistics.median(s):>8.1f}{p90:>7.1f}{max(s):>7.1f}   {shares}")

    print("\nSeparation (AUC: chance a random AI sample outscores a random human one; 0.5 = no signal)")
    human = by_label["human"]
    a = auc(by_label["ai-typical"], human)
    print(f"  human vs ai-typical   " + ("n/a" if a is None else f"{a:.3f}"))
    if by_label["ai-polished"] and human:
        above = sum(x > max(human) for x in by_label["ai-polished"])
        print(f"  ai-polished above the highest human sample: {above}/{len(by_label['ai-polished'])} (they are written to look human, so low is expected)")

    print("\nScore by genre (median)")
    genres = sorted({r["genre"] for r in graded})
    for g in genres:
        parts = []
        for lab in LABELS:
            s = [r["score"] for r in graded if r["genre"] == g and r["label"] == lab]
            parts.append(f"{lab} {statistics.median(s):5.1f}" if s else f"{lab}     -")
        print(f"  {g:11}" + "   ".join(parts))

    human_rows = [r for r in graded if r["label"] == "human"]
    words = sum(r["words"] for r in human_rows)
    if words:
        counts: Counter = Counter()
        docs: defaultdict = defaultdict(int)
        for r in human_rows:
            counts.update(r["findings"])
            for rid in r["findings"]:
                docs[rid] += 1
        print(f"\nRules firing most on human text (per 1,000 words; {len(human_rows)} samples, {words} words)")
        for rid, n in counts.most_common(10):
            print(f"  {rid}  {1000 * n / words:5.2f}/1k   in {docs[rid]}/{len(human_rows)} samples")

    ai_rows = [r for r in graded if r["label"] == "ai-typical"]
    quiet = [rid for rid in {x for r in ai_rows for x in r["findings"]} if rid not in {x for r in human_rows for x in r["findings"]}]
    if quiet:
        print("\nRules that fire on ai-typical text but never on human text: " + ", ".join(sorted(quiet)))

    worst_h = sorted(human_rows, key=lambda r: -r["score"])[:3]
    best_a = sorted((r for r in graded if r["label"].startswith("ai")), key=lambda r: r["score"])[:3]
    print("\nHighest-scoring human samples (false-positive candidates): " + ", ".join(f"{r['id']} {r['score']:.1f}" for r in worst_h))
    print("Lowest-scoring AI samples (misses): " + ", ".join(f"{r['id']} {r['score']:.1f}" for r in best_a))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PATH", help="write per-sample results to PATH")
    ap.add_argument("--compare", metavar="PATH", help="show score changes against a results file written by --json")
    ap.add_argument("--label", choices=LABELS, help="only scan one label")
    ap.add_argument("--ids", nargs="+", metavar="ID", help="only scan these sample ids")
    ap.add_argument("--llm", action="store_true", help="also run the semantic pass through `claude -p` (costs money, slow)")
    ap.add_argument("--model", default="claude-haiku-5-5", help="model for --llm (default: claude-haiku-5-5)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"], help="effort for --llm (default: medium)")
    ap.add_argument("--truncate", type=int, metavar="WORDS", help="cut every sample to about WORDS words (whole paragraphs) before scanning")
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    llm = LLMOptions(model=args.model, effort=args.effort, backend="claude-cli") if args.llm else None
    rows, missing = scan(manifest, args.label, args.truncate, args.ids, llm)
    if missing:
        print(f"skipped {len(missing)} samples with no text on disk (run benchmark/fetch.py): {', '.join(missing)}\n", file=sys.stderr)
    if not rows:
        print("nothing to scan", file=sys.stderr)
        return 1

    base = None
    if args.compare:
        saved = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        base = {r["id"]: r["score"] for r in saved["samples"] if r["score"] is not None}

    print(f"ai-isms {__version__} · {len(rows)} samples\n")
    print_table(rows, base)
    if llm:
        print_llm(rows)
    else:
        print_summary(rows)

    if args.json:
        payload = {"version": __version__, "samples": [{**r, "findings": dict(r["findings"])} for r in rows]}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nresults → {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
