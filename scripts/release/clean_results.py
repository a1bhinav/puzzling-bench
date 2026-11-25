#!/usr/bin/env python3
"""Copy the per-row evaluation outputs listed in release_manifest.py into the dataset's results/ folder, cleaned.

    python3 scripts/release/clean_results.py <private data dir> <out dir> --terms TERMS.json

Applied to every string field, model text included:
    local file-system paths (/Users/..., /var/folders/...)   -> "[local path removed]"
    provider request and message ids (req_..., msg_...)      -> "req_[id removed]" / "msg_[id removed]"
Applied to metadata only (model text is never rewritten beyond the rules above and TERMS "text"):
    session_id keys are dropped
Added to result rows logged without run_meta (the vLLM-served rows in weak_judges/ are left as they are):
    {"harness": "B"} for the earlier Claude runs (HARNESS_B_MODELS); a non-errored row with more than
        B_MAX_PROMPT_TOKENS prompt tokens does not fit harness B's logging and stops the run
    {"harness": "A"} for any other model; such a row must be errored, otherwise the run stops
Read from TERMS (a local JSON file, never published):
    "labels":    {field: {substring: label}}   e.g. map a harness field's value to the paper's label
    "prefixes":  [prefix, ...]                  stripped from model-name fields
    "errors":    [[regex, replacement], ...]    applied to `error` strings
    "anywhere":  [[regex, replacement], ...]    applied to every string (e.g. endpoint hosts, tool paths, and
                                                error texts that quote CLI or account messages, which are
                                                replaced with short neutral descriptions)
    "rename":    {old_key: new_key}             key renames at any depth
    "text":      [[post_id, model, regex, replacement], ...]   approved edits to individual model replies
Docs are copied with HTML comments removed. Source files are only read. Prints change counts, never values.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release_manifest import DOCS, FILES, REFERENCES  # noqa: E402

STOP = r"[^\s\"'<>()\[\]`,;]"
PATH_RX = re.compile(rf"(?:/private)?/var/folders/{STOP}+|/Users/{STOP}+")
ID_RX = re.compile(r"\b(req|msg)_[A-Za-z0-9]{8,}")
MODEL_KEYS = {"model", "served_model", "model_requested", "model_served", "judge_model", "judge_served_model"}
TEXT_KEYS = {"raw_output", "forced_output"}
HARNESS_B_MODELS = {"claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"}
B_MAX_PROMPT_TOKENS = 40  # harness B logged only a few prompt tokens; harness A rows record >= 1,906


class Cleaner:
    def __init__(self, terms):
        self.labels = terms.get("labels", {})
        self.prefixes = tuple(p.lower() for p in terms.get("prefixes", []))
        self.errors = [(re.compile(a), b) for a, b in terms.get("errors", [])]
        self.anywhere = [(re.compile(a, re.I), b) for a, b in terms.get("anywhere", [])]
        self.rename = terms.get("rename", {})
        self.text = [(pid, model, re.compile(a), b) for pid, model, a, b in terms.get("text", [])]
        self.stats = collections.Counter()

    def row(self, r, dst):
        out = self.value(r, None)
        for pid, model, rx, rep in self.text:
            if out.get("post_id") == pid and out.get("model") == model:
                for k in TEXT_KEYS & out.keys():
                    if isinstance(out[k], str):
                        out[k], n = rx.subn(rep, out[k])
                        self.stats["approved reply edit"] += n
        if "model" in out and not isinstance(out.get("run_meta"), dict) and not dst.startswith("weak_judges/"):
            label = "B" if out["model"] in HARNESS_B_MODELS else "A"
            if label == "B" and not out.get("error") and (out.get("prompt_tokens") or 0) > B_MAX_PROMPT_TOKENS:
                sys.exit(f"{dst}: post {out.get('post_id')} has no run_meta but {out.get('prompt_tokens')} prompt tokens")
            if label == "A" and not out.get("error"):
                sys.exit(f"{dst}: post {out.get('post_id')} has no run_meta, no error and is not a harness B model")
            out["run_meta"] = {"harness": label}
            self.stats[f"run_meta added ({label})"] += 1
        return out

    def value(self, o, key):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k == "session_id":
                    self.stats["session_id dropped"] += 1
                    continue
                nk = self.rename.get(k, k)
                self.stats["key renamed"] += nk != k
                out[nk] = self.value(v, k)
            return out
        if isinstance(o, list):
            return [self.value(v, key) for v in o]
        if not isinstance(o, str):
            return o
        s = o
        for rx, rep, name in ((PATH_RX, "[local path removed]", "local path"), (ID_RX, r"\1_[id removed]", "request/message id")):
            s, n = rx.subn(rep, s)
            self.stats[name] += n
        for rx, rep in self.anywhere:
            s, n = rx.subn(rep, s)
            self.stats["term replaced"] += n
        if key in self.labels:
            new = next((lab for sub, lab in self.labels[key].items() if sub.lower() in s.lower()), s)
            self.stats["label"] += new != s
            s = new
        elif key in MODEL_KEYS and s.lower().startswith(self.prefixes):
            s = s.split(":", 1)[1]
            self.stats["model prefix"] += 1
        elif key == "error":
            for rx, rep in self.errors:
                s, n = rx.subn(rep, s)
                self.stats["error wording"] += n
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--terms", required=True, help="local JSON file with the label/term rules (not published)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; remove it first")
    with open(a.terms, encoding="utf-8") as f:
        cleaner = Cleaner(json.load(f))
    total = collections.Counter()
    for src, dst, _ in FILES + REFERENCES:
        cleaner.stats = collections.Counter()
        rows = 0
        os.makedirs(os.path.dirname(os.path.join(a.out, dst)), exist_ok=True)
        with open(os.path.join(a.data, src), encoding="utf-8") as fi, \
                open(os.path.join(a.out, dst), "w", encoding="utf-8") as fo:
            for line in fi:
                if line.strip():
                    fo.write(json.dumps(cleaner.row(json.loads(line), dst), ensure_ascii=False) + "\n")
                    rows += 1
        total.update(cleaner.stats)
        print(f"{dst}\t{rows}\t" + ", ".join(f"{k}: {v}" for k, v in sorted(cleaner.stats.items()) if v))
    for src, dst, _ in DOCS:
        os.makedirs(os.path.dirname(os.path.join(a.out, dst)), exist_ok=True)
        with open(os.path.join(a.data, src), encoding="utf-8") as fi:
            text, n = re.subn(r"<!--.*?-->\s*", "", fi.read(), flags=re.S)
        with open(os.path.join(a.out, dst), "w", encoding="utf-8") as fo:
            fo.write(text)
        print(f"{dst}\tcopied, HTML comments removed: {n}")
    print("TOTAL\t" + ", ".join(f"{k}: {v}" for k, v in sorted(total.items())))


if __name__ == "__main__":
    main()
