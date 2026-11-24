"""v2 Eval 2: citation faithfulness, LLM judge against a human spot check.

Input (JSONL, one cited sentence per line)::

    {"trace": "slug", "unit": "p1u2", "sentence": "...", "judge_supported": true,
     "human_supported": null}

``judge_supported`` is required on every row; ``human_supported`` is ``true``/``false`` on
the stratified spot-check rows and ``null`` or absent elsewhere. Booleans must be real
booleans (or the strings ``"true"``/``"false"``); anything else raises ``ValueError``
instead of being read as true.
"""

import json
from pathlib import Path


def parse_bool(value: object, field: str) -> bool:
    """Strict boolean: ``True``/``False`` or the strings "true"/"false" (any case).

    Raises:
        ValueError: for anything else, so ``"false"`` is never read as true.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{field}: expected a boolean, got {value!r}")


def cohens_kappa(y_human: list[int], y_judge: list[int]) -> tuple[float | None, float | None]:
    """``(kappa, raw_agreement)`` for two binary raters.

    Kappa is None when it is undefined: no ratings, or chance agreement is 1.0 (both
    raters used one label throughout). Raw agreement is returned in that case; it is
    None only when there are no ratings.
    """
    if len(y_human) != len(y_judge):
        raise ValueError("raters must label the same rows")
    n = len(y_human)
    if n == 0:
        return None, None
    po = sum(h == j for h, j in zip(y_human, y_judge, strict=True)) / n
    ph1, pj1 = sum(y_human) / n, sum(y_judge) / n
    pe = ph1 * pj1 + (1 - ph1) * (1 - pj1)
    if pe == 1.0:
        return None, po
    return (po - pe) / (1 - pe), po


def evaluate_faithfulness(judge_results_path: Path) -> dict:
    """Judge faithfulness over all rows, and judge-human kappa over the spot-checked rows."""
    rows = [json.loads(line) for line in
            judge_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    judged = [parse_bool(r["judge_supported"], "judge_supported") for r in rows]
    spot = [(parse_bool(r["human_supported"], "human_supported"), j)
            for r, j in zip(rows, judged, strict=True) if r.get("human_supported") is not None]
    kappa, agreement = cohens_kappa([int(h) for h, _ in spot], [int(j) for _, j in spot])
    return {
        "total_sentences": len(rows),
        "judge_faithfulness": sum(judged) / len(rows) if rows else None,
        "human_spot_check_n": len(spot),
        "cohens_kappa": None if kappa is None else round(kappa, 3),
        "raw_agreement": None if agreement is None else round(agreement, 3),
    }


if __name__ == "__main__":
    print("Usage: evaluate_faithfulness(Path('faithfulness_audit.jsonl'))")
