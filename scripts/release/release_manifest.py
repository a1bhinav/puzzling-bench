"""Which evaluation files go into the released dataset's results/ folder, where, and which part of the paper
each one backs. Paths on the left are relative to the private data/ folder of the evaluation runs; paths on
the right are relative to results/. Imported by clean_results.py."""

M9 = ["gpt-6.1-sol", "claude-opus-5.5", "claude-opus-4-8", "kimi-k3", "gpt-5.6-sol", "grok-4.7",
      "claude-sonnet-5", "claude-sonnet-5.5", "claude-haiku-4-5"]
JUDGES = ["opus48", "opus55", "gpt56"]

FILES = []  # (source, destination, backs)
for m in M9:
    for j in JUDGES:
        FILES.append((f"eval/{m}.pilot255.judged.canon2.{j}.jsonl", f"pilot255/{m}.judged.locked_ref.{j}.jsonl",
                      "Table 4, Figures 1-3, Tables 6-7, Sec. 5.1-5.2, 6.1, 6.3, App. C-F"))
for m in M9:
    FILES.append((f"eval/{m}.pilot255.judged.opus48.jsonl", f"pilot255_reference_ablation/{m}.judged.spoiler_ref.opus48.jsonl",
                  "Sec. 6.2 (spoiler-based reference); Sec. 6.4 (after re-running timed-out rows)"))
    FILES.append((f"eval/{m}.pilot255.judged.canon.opus48.jsonl", f"pilot255_reference_ablation/{m}.judged.extracted_ref_round1.opus48.jsonl",
                  "Sec. 6.2 (first LLM-extraction round)"))
    for j in ["opus55", "gpt56"]:
        FILES.append((f"eval/{m}.pilot255.judged.{j}.jsonl", f"pilot255_reference_ablation/{m}.judged.spoiler_ref.{j}.jsonl",
                      "Sec. 6.1 own-family bonus under the spoiler-based reference"))
for m in ["gpt-6.1-sol", "claude-opus-5.5"]:
    FILES.append((f"eval/{m}.pilot255.nohints.judged.canon2.opus48.jsonl", f"pilot255_no_hints/{m}.judged.locked_ref.opus48.jsonl",
                  "Table 3 (no-hints run; the with-hints run is in pilot255/)"))
    FILES.append((f"eval/{m}.imgpilot100.judged.canon2.opus48.jsonl", f"imgpilot100/{m}.image.judged.locked_ref.opus48.jsonl",
                  "Table 5, Sec. 5.4 (image attached)"))
    FILES.append((f"eval/{m}.imgpilot100.noimg.judged.canon2.opus48.jsonl", f"imgpilot100/{m}.no_image.judged.locked_ref.opus48.jsonl",
                  "Table 5, Sec. 5.4 (image withheld)"))
for m in ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"]:
    FILES.append((f"eval/{m}.judged.jsonl", f"full_text_only/{m}.judged.spoiler_ref.jsonl",
                  "Sec. 5.3 (full text-only runs); Sec. 5.2 (Sonnet 5 by posting period)"))
for m in ["kimi-k3", "grok-4.7"]:
    FILES.append((f"eval/{m}.pilot255.judged.opus48.stale.bak", f"pilot255_errored_rows/{m}.before_resume.judged.spoiler_ref.opus48.jsonl",
                  "Sec. 6.4 (before re-running timed-out rows)"))
for s, judges in [("Qwen_Qwen3-8B", ["opus48", "llama", "qwen14b"]), ("Qwen_Qwen3-14B", ["opus48", "qwen14b"]),
                  ("meta-llama_Llama-3.1-8B-Instruct", ["opus48", "llama", "qwen14b"])]:
    for j in judges:
        name = {"llama": "llama-3.1-8b", "qwen14b": "qwen3-14b", "opus48": "opus48"}[j]
        FILES.append((f"eval/{s}.judged.{j}.jsonl", f"weak_judges/{s}.judged.spoiler_ref.{name}.jsonl",
                      "Sec. 6.1 weak-judge paragraph"))

REFERENCES = [
    ("v1/canonical.v2.pilot255.jsonl", "references/pilot255.locked_ref.jsonl", "Sec. 3.3 (locked LLM-extracted reference)"),
    ("v1/canonical.v2.imgpilot100.jsonl", "references/imgpilot100.locked_ref.jsonl", "Sec. 3.3, 5.4"),
    ("v1/canonical.pilot255.jsonl", "references/pilot255.extracted_ref_round1.jsonl", "Sec. 6.2"),
    ("v1/canonical.imgpilot100.jsonl", "references/imgpilot100.extracted_ref_round1.jsonl", "Sec. 6.2"),
    ("eval/imgpilot100.ids.jsonl", "references/imgpilot100.ids.jsonl", "Sec. 4.5 (frozen image-pilot ids)"),
]
DOCS = [
    ("eval/canon_spotcheck50.md", "references/human_review_50.md", "Sec. 3.3 (50-sample human review)"),
    ("eval/canon_edits.md", "references/manual_corrections.md", "Sec. 3.3 (two manual corrections)"),
]
