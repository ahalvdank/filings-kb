# Gemini Neon manual judge run: provenance

- Judge: Gemini Neon, used by hand by Andrew in one chat on 2026-11-26. Unmetered; `cost_usd` is "0".
- Instructions: `PROMPT.md` (the shared rubric `judge_prompt.md` plus batch line-format instructions).
- Units: Gemini's input limit cut `units_for_manual_judge.md` off inside unit 112
  (`nvda-export-control-risk-fy22-fy26:p2u4`). Units 112-126 were then sent as
  `units_for_manual_judge_part2.md`. Gemini therefore saw a truncated part 1 plus part 2, not the
  single file whose hash is in `packet_units_sha256`.
- Reply: `gemini_reply_raw.jsonl`, the two pasted replies concatenated (111 + 15 lines).
  One repair was made in `gemini_reply_repaired.jsonl`: line 116 had unescaped inner quotes around
  `could` in `unsupported_span`; they were escaped. No verdict or other text was changed.
- Import: `python -m evals.v2.faithfulness.manual import --units evals/v2/faithfulness/units.jsonl
  --reply evals/v2/faithfulness/manual/gemini_reply_repaired.jsonl
  --out evals/v2/faithfulness/judge_gemini_manual.jsonl --run-label gemini-neon-manual --model gemini-neon`
- All 126 units are judged in one conversation, so this is a second-opinion judge, not an
  independent replicate of the API runs.

# Spot-check supplement

`spot_sheet.csv` (40 rows) was frozen from Opus run 1 before the GPT and Gemini runs existed. After
them, 12 units flagged by some judge (11 GPT-only, 1 Gemini-only, or flagged only by Opus run 2) were
not on it. `spot_sheet_supplement.csv` adds exactly those 12 units, shuffled with seed 20261126,
same columns, no judge verdict shown. Built by the controller with `units.format_passages`.
