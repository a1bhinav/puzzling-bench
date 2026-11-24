#!/usr/bin/env python3
"""
Answer matching for puzzling-bench, implementing docs/answer_matching.md (protocol v1).

Tier 1   tier1_match(candidate, truth)                 normalized string / numeric match, no API
Tier 2   judge(question, short, full, candidate)       LLM judge over any OpenAI-compatible
                                                       /chat/completions endpoint (OpenAI, Gemini's
                                                       OpenAI-compatible endpoint, OpenRouter, vLLM, ...)
Both     score_answer(record, candidate)               tier 1, then tier 2 only if tier 1 fails

Environment for the judge (or pass model=/base_url=/api_key= explicitly). All three are REQUIRED whenever
the judge is used -- deliberately no fallback to the model-under-test's LLM_* / OPENAI_* settings:
    JUDGE_BASE_URL   e.g. https://api.openai.com/v1
    JUDGE_API_KEY
    JUDGE_MODEL      a cheap model

CLI:
    python3 scripts/score.py --selftest        # tier-1 unit tests, no network
    python3 scripts/score.py --judge-smoke     # one judge call, prints the raw reply

Standard library only.
"""
import argparse
import html
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from fractions import Fraction

# ----------------------------------------------------------------------------- tier 1
NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
ONES = "one|two|three|four|five|six|seven|eight|nine"
SYMBOLS = {"+": " plus ", "×": " times ", "*": " times ", "÷": " divided by ", "=": " equals ",
           "&": " and ", "%": " percent ", "°": " degrees "}
ARTICLES = {"a", "an", "the"}
LEADIN = re.compile(r"^(?:the\s+)?(?:final\s+)?(?:answer|solution)\s*(?:is|:|=)\s*", re.I)
WRAP = " \t\n\"'`*_()[]{}."


def normalize(s):
    """Lowercase, strip lead-ins/wrapping/markdown, map symbols and number words, drop articles."""
    s = html.unescape(str(s or "")).strip().lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-"), ("−", "-")):
        s = s.replace(a, b)
    s = LEADIN.sub("", s)
    s = s.strip(WRAP)
    if s == "-":
        return "minus"
    for k, v in SYMBOLS.items():
        s = s.replace(k, v)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)                                  # 3,000 -> 3000
    s = re.sub(rf"\b({TENS})[-\s]({ONES})\b",
               lambda m: str(NUM_WORDS[m.group(1)] + NUM_WORDS[m.group(2)]), s)  # forty-two -> 42
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", s)                                 # self-driving -> self driving
    s = re.sub(r"[^\w\s./-]", " ", s)                                           # drop punctuation; keep . / - for numbers
    toks = []
    for t in s.split():
        t = t.strip(".,;:!?").rstrip("-")
        if not t or t in ARTICLES:
            continue
        toks.append(str(NUM_WORDS.get(t, t)))
    return " ".join(toks)


def _as_number(s):
    try:
        return Fraction(s.replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


def tier1_match(candidate, truth):
    c, t = normalize(candidate), normalize(truth)
    if not c or not t:
        return False
    if c == t:
        return True
    nc, nt = _as_number(c), _as_number(t)
    return nc is not None and nt is not None and nc == nt


# ----------------------------------------------------------------------------- chat client
def chat(messages, **kw):
    """Minimal OpenAI-compatible chat call. Returns the assistant text (see chat_full for metadata)."""
    return chat_full(messages, **kw)["content"]


def chat_full(messages, model=None, base_url=None, api_key=None, temperature=0.0, max_tokens=1024,
              timeout=180, retries=4, extra_body=None):
    """OpenAI-compatible chat call returning {"content", "finish_reason", "usage", "reasoning"}.
    Retries on 429/5xx/timeouts; on a 400 that rejects max_tokens/temperature (newer reasoning
    models) it adapts and retries once."""
    base_url = (base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1").rstrip("/")
    api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("no model: pass model= or set LLM_MODEL")
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature
    if extra_body:
        payload.update(extra_body)      # e.g. {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            return {"content": msg.get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage"),
                    "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                    "model": data.get("model"),                  # the model the server says it served
                    "run_meta": data.get("x_run_meta")}          # optional endpoint metadata: harness, effort, served model, cost
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            last = f"HTTP {e.code}: {body[:300]}"
            if e.code == 400:
                if "max_tokens" in body and "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    continue
                if "temperature" in body and "temperature" in payload:
                    payload.pop("temperature")
                    continue
                raise RuntimeError(last)
            if e.code in (408, 409, 425, 429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            last = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"chat failed after {retries} attempts: {last}")


def judge_config(model=None, base_url=None, api_key=None, extra_body=None):
    """Judge settings: explicit args, else JUDGE_*. Never the model-under-test's LLM_*/OPENAI_* (a run against a
    local vLLM server must not silently judge with the same server). Raises if anything is missing.
    extra_body (or JUDGE_EXTRA_BODY, a JSON object) is merged into every judge request, e.g.
    {"reasoning_effort": "low"} for an endpoint that accepts it."""
    E = os.environ.get
    cfg = {"model": model or E("JUDGE_MODEL"), "base_url": base_url or E("JUDGE_BASE_URL"),
           "api_key": api_key or E("JUDGE_API_KEY")}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError("judge not configured: set JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY "
                           f"(or --judge-base-url / --judge-model / --judge-api-key); missing: {', '.join(missing)}")
    if extra_body is None and E("JUDGE_EXTRA_BODY"):
        extra_body = json.loads(E("JUDGE_EXTRA_BODY"))
    if extra_body:
        cfg["extra_body"] = extra_body
    return cfg


# ----------------------------------------------------------------------------- tier 2
JUDGE_SYSTEM = """You grade answers to puzzles. You will be shown a puzzle, the reference answer written by a human (a short answer plus the full accepted answer for context), and a candidate answer produced by an AI.

Decide only whether the candidate names the SAME solution as the reference, for what the puzzle asks. Do not solve the puzzle yourself and do not grade the reasoning. Different wording, formatting, units, or extra explanation are fine if the solution is the same. If the candidate gives several different final answers, hedges between options, or answers a different question, it is NO_MATCH. If the puzzle has several parts, MATCH only if every part matches.

Reply with exactly two lines:
VERDICT: MATCH or NO_MATCH
REASON: one sentence"""


def judge(question, short_answer, full_answer, candidate, model=None, base_url=None, api_key=None, extra_body=None):
    user = (f"PUZZLE:\n{(question or '')[:6000]}\n\n"
            f"REFERENCE SHORT ANSWER:\n{short_answer}\n\n"
            f"REFERENCE FULL ANSWER (context):\n{(full_answer or '')[:4000]}\n\n"
            f"CANDIDATE ANSWER:\n{candidate}")
    cfg = judge_config(model, base_url, api_key, extra_body)
    res = chat_full([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=200, **cfg)
    raw = res["content"]
    m = re.search(r"VERDICT\s*:\s*\**\s*(MATCH|NO_MATCH)", raw, re.I)
    r = re.search(r"REASON\s*:\s*(.+)", raw, re.I | re.S)
    return {"verdict": m.group(1).upper() if m else "NO_MATCH",
            "reason": (r.group(1).strip() if r else raw.strip())[:500],
            "raw": raw, "parsed": bool(m),
            "judge_model": cfg["model"],                 # what was asked for ...
            "judge_served_model": res.get("model"),      # ... and what the server says it served
            "judge_meta": res.get("run_meta"),           # harness / effort / cost (when the endpoint reports x_run_meta)
            "judge_usage": res.get("usage")}


def score_answer(rec, candidate, use_judge=True, **judge_kw):
    """rec is a puzzles.jsonl record. Returns tier1/verdict/correct plus judge details if used."""
    t1 = tier1_match(candidate, rec["short_answer"])
    out = {"tier1": t1, "judge_used": False, "verdict": "MATCH" if t1 else None,
           "reason": None, "correct": t1}
    if not t1 and use_judge and str(candidate or "").strip():
        j = judge(rec["question_text"], rec["short_answer"], rec["full_answer"], candidate, **judge_kw)
        out.update(judge_used=True, verdict=j["verdict"], reason=j["reason"],
                   judge_raw=j["raw"], correct=(j["verdict"] == "MATCH"),
                   judge_model=j["judge_model"], judge_served_model=j["judge_served_model"],
                   judge_meta=j["judge_meta"], judge_usage=j["judge_usage"])
    return out


# ----------------------------------------------------------------------------- CLI
SELFTEST = [
    ("42", "forty-two", True),
    ("The answer is 12.", "12", True),
    ("Red herring", "a red herring", True),
    ("3,000", "3000", True),
    ("1/2", "0.5", True),
    ("+", "plus", True),
    ("**Paris**", "paris", True),
    ("\"Time\"", "time", True),
    ("twelve", "12", True),
    ("-", "minus", True),
    ("-5", "minus five", False),       # goes to the judge; tier 1 is deliberately strict
    ("Final answer: 7", "seven", True),
    ("A self-driving car", "A self driving car?", True),
    ("forty-two", "42", True),
    ("-5", "-5", True),
    ("42", "43", False),
    ("cat", "the cat and the dog", False),
    ("12", "12 and 13", False),
    ("", "x", False),
    ("yes", "no", False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--judge-smoke", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        bad = 0
        for c, t, want in SELFTEST:
            got = tier1_match(c, t)
            flag = "ok " if got == want else "BAD"
            bad += got != want
            print(f"{flag}  {c!r:28} vs {t!r:24} -> {got}   [{normalize(c)!r} | {normalize(t)!r}]")
        print(f"\n{len(SELFTEST) - bad}/{len(SELFTEST)} passed")
        sys.exit(1 if bad else 0)
    if args.judge_smoke:
        j = judge("What is 2 + 2?", "4", "It is four, because 2+2=4.", "The total is four.")
        print(json.dumps(j, indent=2))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
