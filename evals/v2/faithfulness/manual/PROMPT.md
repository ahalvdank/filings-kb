You are checking one sentence from an answer about SEC 10-K filings against the passages it cites.

Judge only whether the cited passages, together, state everything the sentence claims. Read every passage before you decide. A claim may be supported by one passage or by several read together.

Rules:

- Use only the cited passages. Do not use outside knowledge, even if you are sure a claim is true or false.
- Numbers must match the text, including scale (thousands, millions, billions), units, signs, periods and fiscal years. A number that appears in a passage but for a different period, company or line item does not support the claim.
- Company names, years, comparisons ("increased", "more than", "the largest") and causal links ("because", "driven by") are claims too. Each one must be stated or directly implied by the passages.
- Wording may differ. A faithful paraphrase of what a passage says is supported.

Verdicts:

- `supported`: the passages state everything the sentence claims. Set `unsupported_span` to null.
- `partial`: some of the sentence is supported, but at least one claim is not. Set `unsupported_span` to the unsupported part, quoted exactly from the sentence.
- `unsupported`: the passages do not support the sentence's main claim. Set `unsupported_span` to the unsupported part quoted exactly from the sentence, or the whole sentence.

Reply with the JSON object only.

---

## How to answer this batch

The file `units_for_manual_judge.md` holds 126 units. Each unit starts with a `### <unit_id>`
heading, then the sentence, then the passages it cites. Judge every unit independently,
by the rules above: use only that unit's own passages, and let no other unit's sentence or
passages affect its verdict.

For this packet, the line format below overrides "Reply with the JSON object only." above.
Reply with one JSON object per line, in this exact form, and nothing else:

{"unit_id": "...", "verdict": "supported|partial|unsupported", "unsupported_span": "..."|null}

- `unit_id` is the unit's heading, copied exactly.
- `unsupported_span` is null for `supported`, and an exact quote from the sentence otherwise.
- Every unit_id appears exactly once: 126 lines in all. No commentary, headings or
  explanations.

The units file is long, so your reply may come in several parts. If you have to stop
before the end, stop after a complete line; when asked to continue, carry on from the
next unit. Keep going until all 126 units are done.
