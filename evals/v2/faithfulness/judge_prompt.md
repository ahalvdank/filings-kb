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
