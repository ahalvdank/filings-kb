"""v2 Eval 3: the 50-claim manual extraction audit.

Worksheet (CSV, header required): ``claim_id,ticker,claim_type,statement,source_quote,label``.
``label`` is ``TP`` or one failure mode: hallucinated, wrong_entity_link, wrong_fiscal_year,
truncated_context, schema_misclassification. A blank or unknown label is an error, never a
silent false positive.
"""

import csv
from collections import Counter
from pathlib import Path

FAILURE_MODES = frozenset({
    "hallucinated", "wrong_entity_link", "wrong_fiscal_year", "truncated_context",
    "schema_misclassification",
})
VALID_LABELS = frozenset({"TP"}) | FAILURE_MODES
COLUMNS = ("claim_id", "ticker", "claim_type", "statement", "source_quote", "label")


def score_claim_audit(worksheet_csv: Path) -> dict:
    """Precision, false-positive rate and failure breakdown of a labelled worksheet.

    Raises:
        ValueError: if a column is missing, or any row has a blank or unknown label (the
            message lists the claim ids).
    """
    with worksheet_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"worksheet is missing columns: {', '.join(missing)}")
        rows = list(reader)
    bad = [r["claim_id"] for r in rows if (r["label"] or "").strip() not in VALID_LABELS]
    if bad:
        raise ValueError(f"blank or unknown label for claims: {', '.join(bad)}")
    counts = Counter(r["label"].strip() for r in rows)
    total = len(rows)
    tp = counts["TP"]
    return {
        "total_claims": total,
        "precision": round(tp / total, 3) if total else None,
        "false_positive_rate": round((total - tp) / total, 3) if total else None,
        "failure_breakdown": {mode: counts[mode] for mode in sorted(FAILURE_MODES)},
    }


if __name__ == "__main__":
    print("Usage: score_claim_audit(Path('claim_audit.csv'))")
