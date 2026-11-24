#!/usr/bin/env python3
"""
Parse a Stack Exchange data dump (Puzzling SE) into JSONL files and print
first-pass statistics for the benchmark.

Usage:
    python3 scripts/parse_dump.py --dump data/raw/puzzling --out data/processed

Outputs (in --out):
    questions.jsonl   one question per line, with tags, accepted answer id, images, etc.
    answers.jsonl     one answer per line, with parent question id, is_accepted flag
    comments.jsonl    one comment per line, with post id
    tags.json         {tag: count}
    image_urls.txt    every unique image URL found in question + answer bodies
    stats.txt         the report printed to stdout

Standard library only — no pip installs needed.
"""

import argparse
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

IMG_SRC_RE = re.compile(r'<img[^>]+src="([^"]+)"', re.I)
IMG_HREF_RE = re.compile(
    r'href="(https?://(?:i\.stack\.imgur\.com|i\.sstatic\.net)/[^"]+)"', re.I
)
SPOILER_RE = re.compile(r'<blockquote class="spoiler">', re.I)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def iter_rows(path: Path):
    """Stream <row .../> elements from a dump XML file without loading it all."""
    for _event, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag == "row":
            yield elem.attrib
            elem.clear()


def parse_tags(raw: str) -> list[str]:
    """Handle both dump formats: '<a><b>' (old) and '|a|b|' (new, 2024+)."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("<"):
        return re.findall(r"<([^>]+)>", raw)
    return [t for t in raw.split("|") if t]


def to_int(s, default=0) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def to_dt(s):
    return datetime.fromisoformat(s) if s else None


def extract_images(body_html: str) -> list[str]:
    urls = IMG_SRC_RE.findall(body_html or "")
    urls += IMG_HREF_RE.findall(body_html or "")
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def pct(n, d) -> str:
    return f"{n:,} ({100 * n / d:.1f}%)" if d else "0"


def quantiles(xs, label, unit=""):
    if not xs:
        return f"  {label}: n/a"
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], xs[len(xs) // 2], xs[-1]]
    return (
        f"  {label}: median {q[1]:.1f}{unit}  "
        f"(Q1 {q[0]:.1f}{unit}, Q3 {q[2]:.1f}{unit}, n={len(xs):,})"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="data/raw/puzzling", help="folder containing Posts.xml etc.")
    ap.add_argument("--out", default="data/processed", help="output folder")
    args = ap.parse_args()

    dump = Path(args.dump)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in ("Posts.xml", "Comments.xml", "Users.xml", "Tags.xml"):
        if not (dump / f).exists():
            sys.exit(f"missing {dump / f}")

    # ---- Users (for attribution: CC BY-SA requires author names) ---------
    print("parsing Users.xml ...", file=sys.stderr)
    users = {}
    for r in iter_rows(dump / "Users.xml"):
        users[to_int(r.get("Id"))] = r.get("DisplayName", "")

    # ---- Tags -------------------------------------------------------------
    print("parsing Tags.xml ...", file=sys.stderr)
    tag_table = {r["TagName"]: to_int(r.get("Count")) for r in iter_rows(dump / "Tags.xml")}

    # ---- Posts ------------------------------------------------------------
    print("parsing Posts.xml ...", file=sys.stderr)
    questions: dict[int, dict] = {}
    answers: dict[int, dict] = {}
    other_post_types = Counter()

    for r in iter_rows(dump / "Posts.xml"):
        ptype = to_int(r.get("PostTypeId"))
        pid = to_int(r.get("Id"))
        body = r.get("Body", "")
        owner_id = to_int(r.get("OwnerUserId"), default=None)

        if ptype == 1:
            questions[pid] = {
                "id": pid,
                "title": r.get("Title", ""),
                "body_html": body,
                "tags": parse_tags(r.get("Tags", "")),
                "score": to_int(r.get("Score")),
                "view_count": to_int(r.get("ViewCount")),
                "answer_count": to_int(r.get("AnswerCount")),
                "comment_count": to_int(r.get("CommentCount")),
                "favorite_count": to_int(r.get("FavoriteCount")),
                "creation_date": r.get("CreationDate"),
                "last_activity_date": r.get("LastActivityDate"),
                "closed_date": r.get("ClosedDate"),
                "accepted_answer_id": to_int(r.get("AcceptedAnswerId"), default=None),
                "owner_user_id": owner_id,
                "owner_display_name": users.get(owner_id, r.get("OwnerDisplayName", "")),
                "image_urls": extract_images(body),
                "has_spoiler": bool(SPOILER_RE.search(body)),
                "content_license": r.get("ContentLicense", ""),
            }
        elif ptype == 2:
            answers[pid] = {
                "id": pid,
                "parent_id": to_int(r.get("ParentId")),
                "body_html": body,
                "score": to_int(r.get("Score")),
                "comment_count": to_int(r.get("CommentCount")),
                "creation_date": r.get("CreationDate"),
                "last_activity_date": r.get("LastActivityDate"),
                "owner_user_id": owner_id,
                "owner_display_name": users.get(owner_id, r.get("OwnerDisplayName", "")),
                "image_urls": extract_images(body),
                "has_spoiler": bool(SPOILER_RE.search(body)),
                "is_accepted": False,  # filled in below
                "content_license": r.get("ContentLicense", ""),
            }
        else:
            other_post_types[ptype] += 1

    # mark accepted answers
    for q in questions.values():
        aid = q["accepted_answer_id"]
        if aid and aid in answers:
            answers[aid]["is_accepted"] = True

    # ---- Comments ---------------------------------------------------------
    print("parsing Comments.xml ...", file=sys.stderr)
    comments = []
    comments_per_post = Counter()
    for r in iter_rows(dump / "Comments.xml"):
        uid = to_int(r.get("UserId"), default=None)
        post_id = to_int(r.get("PostId"))
        comments.append({
            "id": to_int(r.get("Id")),
            "post_id": post_id,
            "text": r.get("Text", ""),
            "score": to_int(r.get("Score")),
            "creation_date": r.get("CreationDate"),
            "user_id": uid,
            "user_display_name": users.get(uid, r.get("UserDisplayName", "")),
        })
        comments_per_post[post_id] += 1

    # ---- Write outputs ----------------------------------------------------
    print("writing outputs ...", file=sys.stderr)

    def dump_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump_jsonl(out / "questions.jsonl", sorted(questions.values(), key=lambda x: x["id"]))
    dump_jsonl(out / "answers.jsonl", sorted(answers.values(), key=lambda x: x["id"]))
    dump_jsonl(out / "comments.jsonl", sorted(comments, key=lambda x: x["id"]))

    tag_counts = Counter(t for q in questions.values() for t in q["tags"])
    with open(out / "tags.json", "w", encoding="utf-8") as fh:
        json.dump(dict(tag_counts.most_common()), fh, indent=2, ensure_ascii=False)

    all_imgs = set()
    for q in questions.values():
        all_imgs.update(q["image_urls"])
    for a in answers.values():
        all_imgs.update(a["image_urls"])
    with open(out / "image_urls.txt", "w") as fh:
        fh.write("\n".join(sorted(all_imgs)) + "\n")

    # ---- Stats ------------------------------------------------------------
    nq, na, nc = len(questions), len(answers), len(comments)
    q_vals = list(questions.values())

    q_accepted = [q for q in q_vals if q["accepted_answer_id"] in answers]
    q_no_answers = [q for q in q_vals if q["answer_count"] == 0]
    q_answered_not_accepted = [q for q in q_vals if q["answer_count"] > 0 and q["accepted_answer_id"] not in answers]
    q_closed = [q for q in q_vals if q["closed_date"]]
    q_img = [q for q in q_vals if q["image_urls"]]
    q_spoiler = [q for q in q_vals if q["has_spoiler"]]
    q_no_tags = [q for q in q_vals if not q["tags"]]
    a_img = [a for a in answers.values() if a["image_urls"]]
    a_spoiler = [a for a in answers.values() if a["has_spoiler"]]

    # accepted answer: time to acceptance (hours) & who answered
    hours_to_accept, self_answered = [], 0
    for q in q_accepted:
        a = answers[q["accepted_answer_id"]]
        dq, da = to_dt(q["creation_date"]), to_dt(a["creation_date"])
        if dq and da:
            hours_to_accept.append((da - dq).total_seconds() / 3600)
        if a["owner_user_id"] is not None and a["owner_user_id"] == q["owner_user_id"]:
            self_answered += 1

    # image hosts
    host_counts = Counter()
    for u in all_imgs:
        m = re.match(r"https?://([^/]+)/", u)
        host_counts[m.group(1) if m else "?"] += 1

    years = Counter(q["creation_date"][:4] for q in q_vals if q["creation_date"])
    tags_per_q = Counter(len(q["tags"]) for q in q_vals)
    answers_per_q = [q["answer_count"] for q in q_vals]
    comments_on_q = [comments_per_post[q["id"]] for q in q_vals]

    lines = []
    L = lines.append
    L("=" * 72)
    L(f"PUZZLING SE DUMP — {dump}")
    L("=" * 72)
    L(f"questions: {nq:,}    answers: {na:,}    comments: {nc:,}    users: {len(users):,}")
    if other_post_types:
        L(f"other post types skipped: {dict(other_post_types)}")
    L("")
    L("QUESTIONS")
    L(f"  with accepted answer:            {pct(len(q_accepted), nq)}")
    L(f"  answered but nothing accepted:   {pct(len(q_answered_not_accepted), nq)}")
    L(f"  zero answers:                    {pct(len(q_no_answers), nq)}")
    L(f"  closed:                          {pct(len(q_closed), nq)}")
    L(f"  with >=1 image:                  {pct(len(q_img), nq)}")
    L(f"  with spoiler block (hint proxy): {pct(len(q_spoiler), nq)}")
    L(f"  with no tags:                    {pct(len(q_no_tags), nq)}")
    L(quantiles([q["score"] for q in q_vals], "score"))
    L(quantiles([q["view_count"] for q in q_vals], "views"))
    L(quantiles(answers_per_q, "answers per question"))
    L(quantiles(comments_on_q, "comments per question"))
    L("")
    L("ACCEPTED ANSWERS")
    L(f"  self-answered by question owner: {pct(self_answered, len(q_accepted))}")
    L(quantiles(hours_to_accept, "hours to accepted answer", "h"))
    L(f"  accepted answers with image:     {pct(sum(1 for q in q_accepted if answers[q['accepted_answer_id']]['image_urls']), len(q_accepted))}")
    L(f"  accepted answers with spoiler:   {pct(sum(1 for q in q_accepted if answers[q['accepted_answer_id']]['has_spoiler']), len(q_accepted))}")
    L("")
    L("ANSWERS (all)")
    L(f"  with >=1 image:                  {pct(len(a_img), na)}")
    L(f"  with spoiler block:              {pct(len(a_spoiler), na)}")
    L("")
    L("IMAGES")
    L(f"  unique image URLs: {len(all_imgs):,}")
    for host, n in host_counts.most_common(8):
        L(f"    {host:<28} {n:,}")
    L("")
    L("TAGS")
    L(f"  distinct tags used: {len(tag_counts):,}   (Tags.xml lists {len(tag_table):,})")
    L("  tags per question: " + ", ".join(f"{k}:{v:,}" for k, v in sorted(tags_per_q.items())))
    L("  top 40:")
    for t, n in tag_counts.most_common(40):
        L(f"    {t:<28} {n:>6,}  ({100 * n / nq:.1f}%)")
    L("")
    L("QUESTIONS BY YEAR")
    for y in sorted(years):
        L(f"    {y}  {years[y]:>6,}")
    L("")
    L(f"outputs written to {out}/")

    report = "\n".join(lines)
    print(report)
    (out / "stats.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
