"""v2 Eval 2: citation faithfulness, LLM judge against a human spot check.

Input (JSONL, one cited sentence per line)::

    {"trace": "slug", "unit": "p1u2", "sentence": "...", "judge_supported": true,
     "human_supported": null}

``judge_supported`` is required on every row; ``human_supported`` is ``true``/``false`` on
the stratified spot-check rows and ``null`` or absent elsewhere. Booleans must be real
booleans (or the strings ``"true"``/``"false"``); anything else raises ``ValueError``
instead of being read as true.
"""

import csv
import json
from pathlib import Path

JUDGE_VERDICTS = ("supported", "partial", "unsupported")
HARNESS_VERDICTS = ("parse_error", "refusal")
HUMAN_VERDICTS = JUDGE_VERDICTS


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


def _judge_verdicts(path: Path) -> dict[str, str]:
    """``unit_id -> verdict`` from a judge JSONL file.

    Raises:
        ValueError: on a duplicate unit_id or a verdict that is not a known value.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        unit_id, verdict = row["unit_id"], row["verdict"]
        if unit_id in out:
            raise ValueError(f"{path}: duplicate unit_id {unit_id}")
        if verdict not in JUDGE_VERDICTS + HARNESS_VERDICTS:
            raise ValueError(f"{path}: {unit_id} has unknown verdict {verdict!r}")
        out[unit_id] = verdict
    return out


def _human_verdicts(spot_csv: Path, judged: dict[str, str]) -> dict[str, str]:
    """Labelled rows of the spot sheet; blank ``human_verdict`` rows are skipped.

    Raises:
        ValueError: on a label outside ``HUMAN_VERDICTS`` or a unit the judge file lacks.
    """
    out: dict[str, str] = {}
    with spot_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["human_verdict"].strip().lower()
            if not label:
                continue
            if label not in HUMAN_VERDICTS:
                raise ValueError(f"{spot_csv}: {row['unit_id']}: human_verdict must be one of "
                                 f"{HUMAN_VERDICTS}, got {row['human_verdict']!r}")
            if row["unit_id"] not in judged:
                raise ValueError(f"{spot_csv}: {row['unit_id']} is not in the judge file")
            out[row["unit_id"]] = label
    return out


def _binary(verdict: str) -> int:
    return int(verdict == "supported")


def _expected_count(expected_units: int | Path, judged: dict[str, str]) -> int:
    """Number of units the run should cover.

    Raises:
        ValueError: if ``expected_units`` is a units.jsonl path and the judge file names a
            unit that is not in it.
    """
    if isinstance(expected_units, int):
        return expected_units
    ids = {json.loads(line)["unit_id"] for line in
           expected_units.read_text(encoding="utf-8").splitlines() if line.strip()}
    unknown = sorted(set(judged) - ids)
    if unknown:
        raise ValueError(f"judge file names units not in {expected_units}: {unknown[:5]}")
    return len(ids)


def evaluate_faithfulness_v2(judge_jsonl: Path, spot_csv: Path | None = None,
                             second_run: Path | None = None,
                             expected_units: int | Path | None = None) -> dict:
    """Three-way judge verdict counts, judge-human 2x2 and kappa, and judge self-agreement.

    ``faithfulness`` is ``supported / (units - parse_errors - refusals)``: a unit the judge
    gave no verdict on is excluded, and the rate is None if none remain.
    ``faithfulness_lower_bound`` is ``supported / units``, counting those excluded units as
    not supported. With ``expected_units`` (a count, or the units.jsonl path) ``coverage``
    is ``units / expected``. The 2x2 and kappa are binary (``supported`` vs
    ``not_supported``) over labelled spot rows that have a judge verdict.
    ``self_agreement`` compares ``judge_jsonl`` with ``second_run`` over units with a
    verdict in both.
    """
    judged = _judge_verdicts(judge_jsonl)
    counts = {v: sum(1 for x in judged.values() if x == v)
              for v in JUDGE_VERDICTS + HARNESS_VERDICTS}
    scored = len(judged) - counts["parse_error"] - counts["refusal"]
    result: dict = {
        "units": len(judged), "supported": counts["supported"], "partial": counts["partial"],
        "unsupported": counts["unsupported"], "parse_errors": counts["parse_error"],
        "refusals": counts["refusal"],
        "faithfulness": counts["supported"] / scored if scored else None,
        "faithfulness_lower_bound": counts["supported"] / len(judged) if judged else None,
    }
    if expected_units is not None:
        expected = _expected_count(expected_units, judged)
        result |= {"expected_units": expected,
                   "coverage": len(judged) / expected if expected else None}
    if spot_csv is not None:
        humans = _human_verdicts(spot_csv, judged)
        pairs = [(_binary(h), _binary(judged[u])) for u, h in humans.items()
                 if judged[u] in JUDGE_VERDICTS]
        name = {1: "supported", 0: "not_supported"}
        table = {f"judge_{name[j]}&human_{name[h]}": 0 for j in (1, 0) for h in (1, 0)}
        for h, j in pairs:
            table[f"judge_{name[j]}&human_{name[h]}"] += 1
        kappa, agreement = cohens_kappa([h for h, _ in pairs], [j for _, j in pairs])
        result |= {"spot_n": len(pairs), "table": table, "raw_agreement": agreement,
                   "cohens_kappa": kappa}
    if second_run is not None:
        other = _judge_verdicts(second_run)
        both = [u for u in judged if judged[u] in JUDGE_VERDICTS
                and other.get(u) in JUDGE_VERDICTS]
        first = [_binary(judged[u]) for u in both]
        again = [_binary(other[u]) for u in both]
        kappa, binary_agreement = cohens_kappa(first, again)
        result["self_agreement"] = {
            "n": len(both),
            "three_way_agreement": (sum(judged[u] == other[u] for u in both) / len(both)
                                    if both else None),
            "binary_agreement": binary_agreement,
            "binary_kappa": kappa,
        }
    return result


if __name__ == "__main__":
    print("Usage: evaluate_faithfulness(Path('faithfulness_audit.jsonl'))")
