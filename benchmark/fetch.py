"""Download and clean the human-written baseline texts listed in manifest.json.

    python benchmark/fetch.py            # fetch anything missing
    python benchmark/fetch.py --force    # refetch everything
    python benchmark/fetch.py --pin      # record sha256 of fetched texts in the manifest

Texts are written to benchmark/data/<id>.txt, which is gitignored: nothing
downloaded here is redistributed with the repository. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MANIFEST = HERE / "manifest.json"
UA = "ai-isms-benchmark/0.1 (https://github.com/alex-red/AI-isms-heuristic-scanner)"


def http_get(url: str, attempts: int = 5) -> str:
    """GET with backoff on 429/5xx, honoring Retry-After."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == attempts - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 15 * (i + 1)
            print(f"  HTTP {e.code}, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def gutenberg(src: dict) -> str:
    raw = http_get(f"https://www.gutenberg.org/cache/epub/{src['id']}/pg{src['id']}.txt").replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    return body


def wikipedia(src: dict) -> str:
    qs = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
        "titles": src["title"], "redirects": 1, "format": "json",
    })
    data = json.loads(http_get(f"https://en.wikipedia.org/w/api.php?{qs}"))
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    # "== Heading ==" -> "## Heading", and drop empty sections / reference tails.
    text = re.sub(r"^(={2,6})\s*(.+?)\s*\1\s*$", lambda m: "#" * len(m.group(1)) + " " + m.group(2), text, flags=re.M)
    text = re.split(r"^#{2,6} (?:See also|References|External links|Notes|Further reading|Bibliography)\s*$", text, flags=re.M)[0]
    return text


def url(src: dict) -> str:
    return http_get(src["url"])


class _Html2Text(HTMLParser):
    """Stack Exchange answer HTML -> light Markdown (paragraphs, lists, bold/italic, code)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "ul", "ol", "blockquote", "h1", "h2", "h3", "h4"):
            self.out.append("\n\n" + ("> " if tag == "blockquote" else "") + ("## " if tag.startswith("h") else ""))
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "br":
            self.out.append("\n")
        elif tag in ("strong", "b"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag == "pre":
            self.in_pre = True
            self.out.append("\n\n```\n")
        elif tag == "code" and not self.in_pre:
            self.out.append("`")

    def handle_endtag(self, tag):
        if tag in ("strong", "b"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag == "pre":
            self.in_pre = False
            self.out.append("\n```\n\n")
        elif tag == "code" and not self.in_pre:
            self.out.append("`")

    def handle_data(self, data):
        self.out.append(data if self.in_pre else re.sub(r"\s+", " ", data))


def stackexchange(src: dict) -> str:
    qs = urllib.parse.urlencode({"site": src["site"], "filter": "withbody"})
    data = json.loads(http_get(f"https://api.stackexchange.com/2.3/answers/{src['answer_id']}?{qs}"))
    if not data.get("items"):
        raise ValueError("answer not found")
    parser = _Html2Text()
    parser.feed(data["items"][0]["body"])
    time.sleep(1)  # be polite to the API
    return re.sub(r"\n{3,}", "\n\n", "".join(parser.out)).strip()


def reddit_tldr(src: dict) -> str:
    """One comment from the Webis-TLDR-17 Reddit corpus (2006-2016) via the Hugging Face datasets server."""
    qs = urllib.parse.urlencode({"dataset": "webis/tldr-17", "config": "default", "split": "train",
                                 "offset": src["row"], "length": 1})
    row = json.loads(http_get(f"https://datasets-server.huggingface.co/rows?{qs}"))["rows"][0]["row"]
    if row["id"] != src["id"]:
        raise ValueError(f"row {src['row']} is {row['id']}, expected {src['id']}")
    # The corpus flattens line breaks into runs of spaces; turn them back into paragraph breaks.
    time.sleep(2)
    return re.sub(r" {3,}", "\n\n", row["content"])


FETCHERS = {"gutenberg": gutenberg, "wikipedia": wikipedia, "url": url,
            "stackexchange": stackexchange, "reddit-tldr": reddit_tldr}


def paragraphs(text: str, reflow: bool) -> list[str]:
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        block = block.strip("\n")
        if not block.strip():
            continue
        out.append(re.sub(r"\s*\n\s*", " ", block).strip() if reflow else block.strip())
    return out


def clean(src: dict, raw: str) -> str:
    """Cut to a passage starting at `start_at` and ending after about `words` words."""
    if src.get("start_at"):
        i = raw.find(src["start_at"])
        if i < 0:
            raise ValueError(f"start_at not found: {src['start_at']!r}")
        raw = raw[i:]
    if src["type"] == "wikipedia":
        raw = re.sub(r"\n+", "\n\n", raw)  # the API separates paragraphs with single newlines
    # Gutenberg hard-wraps lines; reflow them. Markdown sources are already one paragraph per block.
    paras = paragraphs(raw, reflow=src["type"] == "gutenberg")
    target = src.get("words", 700)
    kept, count = [], 0
    for p in paras:
        # Drop Gutenberg illustration markers, bare chapter numbers, and other short non-prose lines.
        if re.fullmatch(r"\[(?:Illustration|Footnote)[^\]]*\]", p) or re.fullmatch(r"[IVXLC\d. ]+", p):
            continue
        kept.append(p)
        count += len(p.split())
        if count >= target:
            break
    text = "\n\n".join(kept)
    # Gutenberg italics markers (_word_) and footnote numbers ([2]) are not prose; leave other sources untouched.
    if src["type"] == "gutenberg":
        text = re.sub(r"_([^_\n]+)_", r"\1", text)
        text = re.sub(r"\s?\[\d+\]", "", text)
    return text + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="refetch even if the file exists")
    ap.add_argument("--pin", action="store_true", help="write sha256 of fetched texts into the manifest")
    ap.add_argument("ids", nargs="*", help="only these sample ids")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    DATA.mkdir(exist_ok=True)
    failures = 0
    for entry in manifest["samples"]:
        src = entry.get("source")
        if not src or (args.ids and entry["id"] not in args.ids):
            continue
        path = DATA / f"{entry['id']}.txt"
        if path.exists() and not args.force:
            text = path.read_text(encoding="utf-8")
            status = "cached"
        else:
            try:
                text = clean(src, FETCHERS[src["type"]](src))
            except Exception as e:  # noqa: BLE001 - report and continue with the other sources
                print(f"FAIL  {entry['id']}: {e}", file=sys.stderr)
                failures += 1
                continue
            path.write_text(text, encoding="utf-8")
            status = "fetched"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if entry.get("sha256") and entry["sha256"] != digest:
            print(f"WARN  {entry['id']}: content differs from pinned sha256 (source changed?)", file=sys.stderr)
        if args.pin:
            entry["sha256"] = digest
        print(f"{status:8}{entry['id']:34}{len(text.split()):>6} words")

    if args.pin:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("pinned hashes written to manifest.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
