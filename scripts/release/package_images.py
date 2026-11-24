#!/usr/bin/env python3
"""Pack the image files referenced by puzzles.jsonl into deterministic tar shards plus an index.

    python3 scripts/release/package_images.py data/v1_release/puzzles.jsonl data/images data/v1_release/images

Writes <out>/images-XXXXX-of-YYYYY.tar (at most --shard-bytes each) and <out>/index.jsonl with one row per
image reference: post_id, role (question|answer), answer_id, position, url, host, file, shard, bytes, sha256,
content_type. file/shard/bytes/sha256 are null when the URL was never fetched. Tar members carry no owner
names, ids or real modification times. Source files are only read. Standard library only.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile


def variants(u):  # same URL normalisation as scripts/build_v1.py
    yield u
    if u.startswith("//"):
        yield "https:" + u
        yield "http:" + u
    elif u.startswith("https://"):
        yield u[6:]
    elif u.startswith("http://"):
        yield u[5:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("puzzles")
    ap.add_argument("images_dir", help="folder with the fetched images and manifest.jsonl (scripts/fetch_images.py)")
    ap.add_argument("out_dir")
    ap.add_argument("--shard-bytes", type=int, default=500_000_000)
    a = ap.parse_args()

    manifest = {}
    with open(os.path.join(a.images_dir, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r.get("status", "")).lower() in ("ok", "200", "success", "true") and r.get("path"):
                manifest[r["url"]] = (r["path"], r.get("content_type"))

    refs, files = [], {}
    with open(a.puzzles, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for role, urls, aid in (("question", r["question_image_urls"], None),
                                    ("answer", r["answer_image_urls"], r["accepted_answer_id"])):
                for pos, u in enumerate(urls):
                    hit = next((manifest[v] for v in variants(u) if v in manifest), None)
                    name = os.path.basename(hit[0]) if hit else None
                    if name:
                        files[name] = hit[1]
                    host = re.sub(r"^(?:https?:)?//([^/]+).*", r"\1", u).lower()
                    refs.append({"post_id": r["post_id"], "role": role, "answer_id": aid, "position": pos,
                                 "url": u, "host": host, "file": name})

    os.makedirs(a.out_dir, exist_ok=True)
    names = sorted(files)
    sizes = {}
    for n in names:
        p = os.path.join(a.images_dir, n)
        if os.path.islink(p) or not os.path.isfile(p):
            sys.exit(f"not a regular file: {n}")
        sizes[n] = os.path.getsize(p)
    shards, cur, cur_bytes = [], [], 0
    for n in names:
        if cur and cur_bytes + sizes[n] > a.shard_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(n)
        cur_bytes += sizes[n]
    if cur:
        shards.append(cur)

    where, digest = {}, {}
    for i, members in enumerate(shards):
        shard = f"images-{i:05d}-of-{len(shards):05d}.tar"
        with tarfile.open(os.path.join(a.out_dir, shard), "w", format=tarfile.USTAR_FORMAT) as tar:
            for n in members:
                with open(os.path.join(a.images_dir, n), "rb") as fh:
                    data = fh.read()
                digest[n] = hashlib.sha256(data).hexdigest()
                ti = tarfile.TarInfo(n)
                ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
                tar.addfile(ti, io.BytesIO(data))
                where[n] = shard
        print(f"{shard}: {len(members)} files, {sum(sizes[n] for n in members):,} bytes", file=sys.stderr)

    with open(os.path.join(a.out_dir, "index.jsonl"), "w", encoding="utf-8") as fo:
        for ref in refs:
            n = ref["file"]
            ref.update({"shard": where.get(n), "bytes": sizes.get(n), "sha256": digest.get(n),
                        "content_type": files.get(n)})
            fo.write(json.dumps(ref, ensure_ascii=False) + "\n")
    print(json.dumps({"references": len(refs), "unfetched_references": sum(r["file"] is None for r in refs),
                      "files": len(names), "bytes": sum(sizes.values()), "shards": len(shards)}))


if __name__ == "__main__":
    main()
