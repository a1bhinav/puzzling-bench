#!/usr/bin/env python3
"""
Build puzzling-bench v1 from the outputs of parse_dump.py.

Usage:
    python3 scripts/build_v1.py --proc data/processed_202607 --out data/v1
    python3 scripts/build_v1.py --proc data/processed_202607 --out data/v1 --images data/images --cutoff 2026-05-01

Reads:   <proc>/questions.jsonl, <proc>/answers.jsonl
         <images>/manifest.jsonl   (optional; maps image URLs to local files)
Writes:  <out>/puzzles.jsonl       one record per kept puzzle (fields below)
         <out>/stats.txt           filter funnel + distributions
         <out>/dropped.jsonl       {"post_id", "reason"} for every dropped question

v1 filter (docs/answer_matching.md): keep a question iff it is not closed,
has an accepted answer, and that answer is present in the dump.

RECORD FIELDS
  post_id, url, title, tags, score, view_count, answer_count, comment_count,
  creation_date, year, owner_user_id, owner_display_name, content_license, snapshot
  question_html            raw body
  question_text            plain text, spoiler blocks INCLUDED   -> with_hints condition
  question_text_no_hints   plain text, spoiler blocks REMOVED    -> no_hints condition
  question_hints           list of spoiler-block texts found in the question
  has_hint                 bool
  question_image_urls, question_image_paths, has_image
  accepted_answer_id, accepted_answer_date, accepted_answer_score, answer_owner_user_id,
  self_answered, hours_to_accepted
  owner_profile_url, answer_owner_display_name, answer_owner_profile_url, answer_url,
  answer_content_license   attribution; profile URLs are null for deleted users
  short_answer             first spoiler block of the accepted answer, else its first paragraph
  short_answer_source      "spoiler" | "first_paragraph"
  n_answer_spoilers        spoiler blocks in the accepted answer (>1 is a multi-part signal)
  full_answer              accepted answer as plain text
  full_answer_html         raw body
  answer_length            len(full_answer)
  answer_image_urls, answer_image_paths, answer_has_image
  n_question_marks         '?' count in question_text (rough multi-part signal)

Standard library only.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser

SNAPSHOT = "stackexchange_20260630_sakura"
SITE = "https://puzzling.stackexchange.com"
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "blockquote", "pre", "h1", "h2", "h3",
              "h4", "h5", "h6", "tr", "table", "hr", "dd", "dt"}


# ----------------------------------------------------------------------------- HTML -> text
class HTMLText(HTMLParser):
    """Convert post HTML to plain text; capture <blockquote class="spoiler"> blocks separately."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.full, self.nospoil, self.spoilers = [], [], []
        self._bq = []      # stack of bools: is each open <blockquote> a spoiler?
        self._cur = []     # text buffer for the outermost open spoiler
        self._pre = 0

    def _in_spoiler(self):
        return any(self._bq)

    def _emit(self, s):
        self.full.append(s)
        (self._cur if self._in_spoiler() else self.nospoil).append(s)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "blockquote":
            is_sp = "spoiler" in (a.get("class") or "").split()
            if is_sp and not self._in_spoiler():
                self._cur = []
            self._bq.append(is_sp)
        if tag == "pre":
            self._pre += 1
        if tag in BLOCK_TAGS:
            self._emit("\n")
        if tag == "img":
            alt = (a.get("alt") or "").strip()
            self._emit(f"[image: {alt}]" if alt and alt != "enter image description here" else "[image]")

    def handle_endtag(self, tag):
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
        if tag in BLOCK_TAGS and tag != "br":
            self._emit("\n")
        if tag == "blockquote" and self._bq:
            was_sp = self._bq.pop()
            if was_sp and not self._in_spoiler():
                self.spoilers.append(clean("".join(self._cur)))
                self._cur = []

    def handle_data(self, data):
        self._emit(data if self._pre else re.sub(r"\s+", " ", data))


def clean(s):
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


HINT_LABEL = re.compile(r"^\W*(?:hints?|clues?|edit(?:ed)?)(?:\s*\d+)?\s*[:.\-–—]?\W*$", re.I)


def strip_hint_labels(text):
    """Remove label-only lines ('Hint:', 'Hint 2:', 'Clue:') left dangling once spoilers are gone."""
    return clean("\n".join(ln for ln in text.split("\n") if not HINT_LABEL.match(ln)))


def html_to_text(html):
    """Return (full_text, text_without_spoilers, [spoiler texts])."""
    p = HTMLText()
    p.feed(html or "")
    p.close()
    full, nospoil = clean("".join(p.full)), clean("".join(p.nospoil))
    if p.spoilers:
        nospoil = strip_hint_labels(nospoil)
    return full, nospoil, p.spoilers


def first_paragraph(text):
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para and not para.startswith("[image"):
            return para
    return text.strip()


def parse_dt(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ----------------------------------------------------------------------------- images
def load_manifest(images_dir):
    """url -> local path for images fetch_images.py marked ok. Tolerant to key names."""
    path = os.path.join(images_dir, "manifest.jsonl")
    m = {}
    if not os.path.exists(path):
        print(f"[warn] no manifest at {path}; image paths left empty", file=sys.stderr)
        return m
    first_keys = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if first_keys is None:
                first_keys = sorted(rec)
            url = rec.get("url")
            local = rec.get("path") or rec.get("local_path") or rec.get("file") or rec.get("local")
            status = str(rec.get("status", "")).lower()
            if url and local and status in ("ok", "200", "success", "true"):
                m[url] = local
    if not m:
        print(f"[warn] manifest read but no usable records; keys seen: {first_keys}", file=sys.stderr)
    return m


def _variants(u):
    yield u
    if u.startswith("//"):
        yield "https:" + u
        yield "http:" + u
    elif u.startswith("https://"):
        yield u[6:]
    elif u.startswith("http://"):
        yield u[5:]


def local_paths(urls, manifest):
    out = []
    for u in urls:
        for v in _variants(u):
            if v in manifest:
                out.append(manifest[v])
                break
    return out


# ----------------------------------------------------------------------------- records
def profile_url(user_id):
    """Author profile link for attribution; None for deleted users (the dump keeps only their name)."""
    return f"{SITE}/users/{user_id}" if user_id is not None else None


def build_record(q, a, manifest, snapshot):
    q_text, q_nohint, q_hints = html_to_text(q.get("body_html", ""))
    a_text, _, a_spoilers = html_to_text(a.get("body_html", ""))
    if a_spoilers:
        short, src = a_spoilers[0], "spoiler"
    else:
        short, src = first_paragraph(a_text), "first_paragraph"
    qd, ad = parse_dt(q.get("creation_date", "")), parse_dt(a.get("creation_date", ""))
    hours = round((ad - qd).total_seconds() / 3600, 3) if qd and ad else None
    q_imgs = q.get("image_urls") or []
    a_imgs = a.get("image_urls") or []
    q_owner, a_owner = q.get("owner_user_id"), a.get("owner_user_id")
    return {
        "post_id": q["id"],
        "url": f"{SITE}/q/{q['id']}",
        "title": q.get("title", ""),
        "tags": q.get("tags") or [],
        "score": q.get("score"),
        "view_count": q.get("view_count"),
        "answer_count": q.get("answer_count"),
        "comment_count": q.get("comment_count"),
        "creation_date": q.get("creation_date"),
        "year": int(str(q.get("creation_date", "0000"))[:4]),
        "owner_user_id": q_owner,
        "owner_display_name": q.get("owner_display_name"),
        "owner_profile_url": profile_url(q_owner),
        "content_license": q.get("content_license"),
        "snapshot": snapshot,
        "question_html": q.get("body_html", ""),
        "question_text": q_text,
        "question_text_no_hints": q_nohint,
        "question_hints": q_hints,
        "has_hint": bool(q_hints),
        "question_image_urls": q_imgs,
        "question_image_paths": local_paths(q_imgs, manifest),
        "has_image": bool(q_imgs),
        "accepted_answer_id": a["id"],
        "accepted_answer_date": a.get("creation_date"),
        "accepted_answer_score": a.get("score"),
        "answer_owner_user_id": a_owner,
        "answer_owner_display_name": a.get("owner_display_name"),
        "answer_owner_profile_url": profile_url(a_owner),
        "answer_url": f"{SITE}/a/{a['id']}",
        "answer_content_license": a.get("content_license"),
        "self_answered": a_owner is not None and a_owner == q_owner,
        "hours_to_accepted": hours,
        "short_answer": short,
        "short_answer_source": src,
        "n_answer_spoilers": len(a_spoilers),
        "full_answer": a_text,
        "full_answer_html": a.get("body_html", ""),
        "answer_length": len(a_text),
        "answer_image_urls": a_imgs,
        "answer_image_paths": local_paths(a_imgs, manifest),
        "answer_has_image": bool(a_imgs),
        "n_question_marks": q_text.count("?"),
    }


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------------------------------------------------------- stats
def quant(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def dist(xs, unit=""):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return (f"median {statistics.median(xs):,.1f}{unit}  (Q1 {quant(xs, .25):,.1f}{unit}, "
            f"Q3 {quant(xs, .75):,.1f}{unit}, n={len(xs):,})")


def write_stats(kept, dropped, n_total, args, path):
    L = []
    n = len(kept)

    def pct(a, d):
        return f"{a:,} ({100 * a / d:.1f}%)" if d else f"{a:,}"

    L.append("=" * 72)
    L.append(f"PUZZLING-BENCH v1  —  {args.proc} -> {args.out}   snapshot: {args.snapshot}")
    L.append("=" * 72)
    L.append("FUNNEL")
    L.append(f"  questions in dump:                 {n_total:,}")
    for reason in ("closed", "no_accepted_answer", "accepted_answer_missing"):
        L.append(f"  dropped: {reason:<26} {pct(dropped.get(reason, 0), n_total)}")
    L.append(f"  kept (v1):                         {pct(n, n_total)}")
    L.append("")
    L.append("KEPT PUZZLES")
    text_only = [r for r in kept if not r["has_image"] and not r["answer_has_image"]]
    L.append(f"  text-only (no images in Q or A):   {pct(len(text_only), n)}")
    L.append(f"  question has image:                {pct(sum(r['has_image'] for r in kept), n)}")
    L.append(f"  accepted answer has image:         {pct(sum(r['answer_has_image'] for r in kept), n)}")
    L.append(f"  has hint (spoiler in question):    {pct(sum(r['has_hint'] for r in kept), n)}")
    L.append(f"  short answer from spoiler:         {pct(sum(r['short_answer_source'] == 'spoiler' for r in kept), n)}")
    L.append(f"  short answer from first paragraph: {pct(sum(r['short_answer_source'] == 'first_paragraph' for r in kept), n)}")
    L.append(f"  >1 spoiler in accepted answer:     {pct(sum(r['n_answer_spoilers'] > 1 for r in kept), n)}")
    L.append(f"  self-answered:                     {pct(sum(r['self_answered'] for r in kept), n)}")
    L.append(f"  hours to accepted:                 {dist([r['hours_to_accepted'] for r in kept], 'h')}")
    L.append(f"  answer length (chars):             {dist([r['answer_length'] for r in kept])}")
    L.append(f"  short answer length (chars):       {dist([len(r['short_answer']) for r in kept])}")
    L.append(f"  question length (chars):           {dist([len(r['question_text']) for r in kept])}")
    L.append("")
    L.append("BY YEAR")
    for y, c in sorted(Counter(r["year"] for r in kept).items()):
        L.append(f"    {y}   {c:,}")
    L.append("")
    post = [r for r in kept if str(r["creation_date"]) >= args.cutoff]
    L.append(f"POST-CUTOFF (creation_date >= {args.cutoff})")
    L.append(f"  puzzles: {len(post):,}   text-only: {sum(1 for r in post if not r['has_image'] and not r['answer_has_image']):,}"
             f"   with hint: {sum(r['has_hint'] for r in post):,}")
    months = Counter(str(r["creation_date"])[:7] for r in post)
    L.append("  by month: " + "  ".join(f"{m}: {c:,}" for m, c in sorted(months.items())))
    L.append("")
    L.append("TOP TAGS (30)")
    tags = Counter(t for r in kept for t in r["tags"])
    for t, c in tags.most_common(30):
        L.append(f"    {t:<28} {c:>6,}  ({100 * c / n:.1f}%)")
    L.append("")
    L.append(f"outputs written to {args.out}/")
    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed_202607", help="parse_dump.py output folder")
    ap.add_argument("--out", default="data/v1")
    ap.add_argument("--images", default="data/images", help="folder with manifest.jsonl from fetch_images.py")
    ap.add_argument("--cutoff", default="2026-05-01", help="post-cutoff date for stats (ISO)")
    ap.add_argument("--snapshot", default=SNAPSHOT)
    args = ap.parse_args()

    print("reading questions ...", file=sys.stderr)
    questions = list(read_jsonl(os.path.join(args.proc, "questions.jsonl")))
    print("reading answers ...", file=sys.stderr)
    answers = {a["id"]: a for a in read_jsonl(os.path.join(args.proc, "answers.jsonl"))}
    manifest = load_manifest(args.images)
    os.makedirs(args.out, exist_ok=True)

    kept, dropped = [], Counter()
    print("building records ...", file=sys.stderr)
    with open(os.path.join(args.out, "dropped.jsonl"), "w", encoding="utf-8") as fd:
        for q in questions:
            reason = None
            if q.get("closed_date"):
                reason = "closed"
            elif not q.get("accepted_answer_id"):
                reason = "no_accepted_answer"
            elif q["accepted_answer_id"] not in answers:
                reason = "accepted_answer_missing"
            if reason:
                dropped[reason] += 1
                fd.write(json.dumps({"post_id": q["id"], "reason": reason}) + "\n")
                continue
            kept.append(build_record(q, answers[q["accepted_answer_id"]], manifest, args.snapshot))

    kept.sort(key=lambda r: r["post_id"])
    with open(os.path.join(args.out, "puzzles.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_stats(kept, dropped, len(questions), args, os.path.join(args.out, "stats.txt"))


if __name__ == "__main__":
    main()
