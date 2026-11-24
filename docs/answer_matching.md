# Answer-matching protocol (v1)

Snapshot: `stackexchange_20260630_sakura` (content through 2026-06-30 23:58 UTC).

## Ground truth
- Source: the accepted answer (any author) on a non-closed question.
- short_answer: first spoiler block of the accepted answer (90.3% have one); else first paragraph.
- full_answer: the entire accepted answer, given to the judge as context only.
- Multi-part questions: short_answer split into parts (LLM-assisted, once, at build time; flagged for review).

## Model output
- Prompt: reason freely, then end with a single line `ANSWER: <...>`. Only that line is scored.

## Scoring (two tiers)
1. Normalized string match: lowercase, strip punctuation/whitespace/articles, numbers compared as values. Match -> correct.
2. Else LLM judge (cheap model, temperature 0). Input: question, short_answer, full_answer, model ANSWER line.
   Output: MATCH / NO_MATCH + one-sentence reason. The judge decides only whether the two answers name the same
   solution for what the question asks; it is never asked to solve the puzzle.
- Multi-part: correct only if every part matches; partial credit reported separately.

## Audit / reporting
- Log every judge call. Hand-check 100 random judge decisions (50 MATCH, 50 NO_MATCH); report agreement.
- Report tier-1-only and tier-1+2 accuracy side by side so it is clear how much rides on the judge.

## Hint condition
- with_hints: original question text. no_hints: hint/clue sentences removed (LLM-extracted, flagged for review).
