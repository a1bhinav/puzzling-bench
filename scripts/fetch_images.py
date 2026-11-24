#!/usr/bin/env python3
"""
Download every image referenced in the Puzzling dump.

Usage:
    python3 scripts/fetch_images.py                      # defaults below
    python3 scripts/fetch_images.py --rps 20 --workers 16
    python3 scripts/fetch_images.py --limit 200          # smoke test

Reads:   data/processed/image_urls.txt   (one URL per line, from parse_dump.py)
Writes:  data/images/<file>              the images
         data/images/manifest.jsonl      one record per URL: status, local path, bytes, content-type
         data/images/missing.txt         URLs that came back 404/410/403 (input for the archive fallback)
         data/images/failed.txt          URLs that errored after all retries (rerun to retry these)

Resumable: rerunning skips anything already marked ok in the manifest.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ssl
import html
import json
import mimetypes
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

# i.sstatic.net returns 403 to unfamiliar User-Agents, so identify as a browser.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Referer": "https://puzzling.stackexchange.com/",
    "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5",
}
PERMANENT_MISS = {403, 404, 410}
MAX_TRIES = 3


class RateLimiter:
    """Simple global limiter: at most `rps` requests per second across all threads."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps if rps > 0 else 0
        self.lock = threading.Lock()
        self.next_ok = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            wait_for = self.next_ok - now
            self.next_ok = max(now, self.next_ok) + self.interval
        if wait_for > 0:
            time.sleep(wait_for)


def normalize(url: str) -> str | None:
    url = html.unescape(url.strip())
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    if not url.startswith("https://"):
        return None
    parts = urlsplit(url)
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def ext_from(url: str, content_type: str | None) -> str:
    path = urlsplit(url).path
    ext = Path(path).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}:
        return ext
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ext or ".bin"


def choose_filename(url: str, dup_basenames: set[str], content_type: str | None) -> str:
    """
    Keep the original filename (e.g. AbCd1.png) when it's unique case-insensitively
    across the whole URL set — macOS is case-insensitive, so AbCd1.png and abcd1.png
    would clobber each other. On any collision, or weird path, fall back to a hash.
    """
    stem = Path(urlsplit(url).path).stem
    ext = ext_from(url, content_type)
    if stem and (stem.lower() not in dup_basenames) and all(c.isalnum() or c in "-_" for c in stem):
        return stem + ext
    return hashlib.sha1(url.encode()).hexdigest()[:16] + ext


# One SSL context for the whole process (building one per request is the slow part),
# and one persistent HTTPS connection per worker thread per host.
SSL_CTX = ssl.create_default_context()
_tls = threading.local()
REDIRECTS = {301, 302, 303, 307, 308}


def _get_conn(netloc: str) -> http.client.HTTPSConnection:
    conns = getattr(_tls, "conns", None)
    if conns is None:
        conns = _tls.conns = {}
    conn = conns.get(netloc)
    if conn is None:
        conn = http.client.HTTPSConnection(netloc, timeout=30, context=SSL_CTX)
        conns[netloc] = conn
    return conn


def _drop_conn(netloc: str) -> None:
    conn = getattr(_tls, "conns", {}).pop(netloc, None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def http_get(url: str, max_redirects: int = 3):
    """GET with keep-alive reuse; follows redirects. Returns (status, headers, body)."""
    for _ in range(max_redirects + 1):
        parts = urlsplit(url)
        netloc = parts.netloc
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        conn = _get_conn(netloc)
        try:
            conn.request("GET", path, headers=HEADERS)
            resp = conn.getresponse()
            body = resp.read()
        except Exception:
            _drop_conn(netloc)  # connection is in an unknown state; start fresh next time
            raise
        if resp.getheader("Connection", "").lower() == "close":
            _drop_conn(netloc)
        if resp.status in REDIRECTS:
            loc = resp.getheader("Location")
            if not loc:
                return resp.status, resp.headers, body
            url = urljoin(url, loc)
            continue
        return resp.status, resp.headers, body
    return 599, None, b""  # too many redirects


def fetch_one(url: str, out_dir: Path, limiter: RateLimiter, dup_basenames: set[str]) -> dict:
    rec = {"url": url, "status": None, "http": None, "path": None, "bytes": 0, "content_type": None, "error": None}
    for attempt in range(1, MAX_TRIES + 1):
        limiter.wait()
        status = None
        try:
            status, headers, data = http_get(url)
        except Exception as e:  # timeouts, resets, TLS errors, bad hosts
            rec["error"] = f"{type(e).__name__}: {e}"[:200]

        if status == 200:
            ctype = headers.get("Content-Type", "") or ""
            rec["http"], rec["content_type"] = 200, ctype
            if not data or (ctype and not ctype.startswith("image/") and "octet-stream" not in ctype):
                rec["status"] = "not_image"
                rec["error"] = f"content-type {ctype!r}, {len(data)} bytes"
                return rec
            name = choose_filename(url, dup_basenames, ctype)
            (out_dir / name).write_bytes(data)
            rec.update(status="ok", path=name, bytes=len(data), error=None)
            return rec

        if status in PERMANENT_MISS:
            rec["http"], rec["status"] = status, "missing"
            return rec

        if status is not None:
            rec["http"], rec["error"] = status, f"HTTP {status}"
            if status == 429:
                time.sleep(5 * attempt)

        if attempt < MAX_TRIES:
            time.sleep(1.5 * attempt)

    rec["status"] = "failed"
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", default="data/processed/image_urls.txt")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--rps", type=float, default=30, help="global requests/sec (default 30)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only fetch the first N URLs (smoke test)")
    ap.add_argument("--retry-missing", action="store_true", help="re-request URLs previously marked missing/not_image")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # --- load URL list ---------------------------------------------------
    raw = [l for l in Path(args.urls).read_text().splitlines() if l.strip()]
    urls, invalid = [], []
    for u in raw:
        n = normalize(u)
        (urls if n else invalid).append(n or u)
    urls = list(dict.fromkeys(urls))  # dedupe, keep order
    if args.limit:
        urls = urls[: args.limit]

    # --- case-insensitive collision check on basenames ---------------------
    base_counts = Counter(Path(urlsplit(u).path).stem.lower() for u in urls)
    dup_basenames = {b for b, n in base_counts.items() if n > 1}

    # --- resume: skip anything already ok ----------------------------------
    skip_statuses = {"ok"} if args.retry_missing else {"ok", "missing", "not_image"}
    done_ok = set()
    if manifest_path.exists():
        with open(manifest_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("status") in skip_statuses:
                        done_ok.add(r["url"])
                except json.JSONDecodeError:
                    pass
    todo = [u for u in urls if u not in done_ok]

    print(f"urls: {len(urls):,} valid, {len(invalid)} invalid, {len(done_ok):,} already done, {len(todo):,} to fetch", file=sys.stderr)
    if dup_basenames:
        print(f"note: {len(dup_basenames)} basenames collide case-insensitively -> those get hashed filenames", file=sys.stderr)
    if not todo:
        print("nothing to do", file=sys.stderr)
        return

    limiter = RateLimiter(args.rps)
    counts = Counter()
    t0 = time.monotonic()

    ex = ThreadPoolExecutor(max_workers=args.workers)
    interrupted = False
    with open(manifest_path, "a") as mf:
        futures = [ex.submit(fetch_one, u, out_dir, limiter, dup_basenames) for u in todo]
        written = set()
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                written.add(fut)
                counts[rec["status"]] += 1
                mf.write(json.dumps(rec) + "\n")
                if i % 50 == 0:
                    mf.flush()
                if i % 500 == 0 or i == len(todo):
                    elapsed = time.monotonic() - t0
                    rate = i / elapsed if elapsed else 0
                    eta = (len(todo) - i) / rate if rate else 0
                    print(
                        f"  {i:,}/{len(todo):,}  ok={counts['ok']:,} missing={counts['missing']:,} "
                        f"not_image={counts['not_image']:,} failed={counts['failed']:,}  "
                        f"{rate:.1f}/s  eta {eta / 60:.0f} min",
                        file=sys.stderr,
                    )
        except KeyboardInterrupt:
            interrupted = True
            print("\ninterrupted — cancelling queued downloads, letting in-flight ones finish ...", file=sys.stderr)
            ex.shutdown(wait=True, cancel_futures=True)
            for fut in futures:  # record in-flight downloads that finished during shutdown
                if fut.done() and not fut.cancelled() and fut not in written:
                    try:
                        rec = fut.result()
                        mf.write(json.dumps(rec) + "\n")
                    except Exception:
                        pass
        finally:
            mf.flush()
    if not interrupted:
        ex.shutdown(wait=True)

    # --- rebuild missing.txt / failed.txt from the full manifest -------------
    latest: dict[str, dict] = {}
    with open(manifest_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                latest[r["url"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
    missing = sorted(u for u, r in latest.items() if r.get("status") in ("missing", "not_image"))
    failed = sorted(u for u, r in latest.items() if r.get("status") == "failed")
    (out_dir / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    (out_dir / "failed.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
    if invalid:
        (out_dir / "invalid_urls.txt").write_text("\n".join(invalid) + "\n")

    total_ok = sum(1 for r in latest.values() if r.get("status") == "ok")
    total_bytes = sum(r.get("bytes", 0) for r in latest.values() if r.get("status") == "ok")
    print("\nSUMMARY (whole manifest)", file=sys.stderr)
    print(f"  ok:        {total_ok:,}  ({total_bytes / 1e9:.2f} GB)", file=sys.stderr)
    print(f"  missing:   {len(missing):,}   -> {out_dir}/missing.txt  (archive fallback)", file=sys.stderr)
    print(f"  failed:    {len(failed):,}   -> {out_dir}/failed.txt   (rerun to retry)", file=sys.stderr)
    if invalid:
        print(f"  invalid:   {len(invalid):,}   -> {out_dir}/invalid_urls.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
