# Claim diff prompt (v1)

You compare two lists of structured claims extracted from the same section of
one company's annual reports (10-K), for consecutive fiscal years. The lists
are the only evidence you have. Do not use outside knowledge.

Each claim is one line: `[Type] name | key=value; key=value`. The same
underlying fact may be worded differently in the two years; treat reworded
claims about the same thing as unchanged, not as one removed and one added.

Produce a single JSON object and nothing else (no prose, no Markdown fences):

{"added": [string], "removed": [string], "changed": [{"before": string, "after": string}]}

- `added`: claims present only in the current year. Copy the claim's name.
- `removed`: claims present only in the prior year. Copy the claim's name.
- `changed`: the same underlying claim whose substance changed between the years
  (for example a different amount, status, scope or target). `before` and
  `after` are short statements of the prior and current version.
- Leave a list empty when nothing applies. Do not list unchanged claims.
