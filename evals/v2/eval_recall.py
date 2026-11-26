"""v2 Eval 1: retrieval Recall@k over a golden question set.

Golden file (JSON)::

    {"questions": [{"slug": "nvda-export-control-risk-fy22-fy26",
                    "expected_targets": [{"ticker": "NVDA", "fiscal_year": 2025,
                                          "section": "Item 1A",
                                          "must_contain_phrase": "export control"}]}]}

A target is hit at ``k`` when one of the ``k`` highest-scoring passages of the trace has
that ticker and fiscal year, a section name containing ``section``, and the phrase
somewhere in ``before + match + after``. All comparisons ignore case.

Passages are taken from the exported trace, which holds the passages the answer saw plus
a capped number of rejected candidates, not the whole top-20 pool, so recall measured on
exports is a lower bound.
"""

import csv
import json
from pathlib import Path

DEFAULT_KS = (5, 20)
POOLED_KS = (5, 10, 20)


def _full_context(passage: dict) -> str:
    ctx = passage.get("context") or {}
    return " ".join(str(ctx.get(key, "")) for key in ("before", "match", "after")).lower()


def _hit(passage: dict, target: dict) -> bool:
    return (
        passage["ticker"] == target["ticker"]
        and passage["fiscal_year"] == target["fiscal_year"]
        and target["section"].lower() in passage["section"].lower()
        and target["must_contain_phrase"].lower() in _full_context(passage)
    )


def evaluate_recall(golden_path: Path, traces_dir: Path, ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """Recall@k over every target of the golden set.

    A question whose trace file is missing counts all its targets as misses and is listed
    under ``missing_traces``.

    Returns:
        ``{"recall@5": float | None, ..., "targets": int, "missing_traces": [slug, ...]}``;
        recall is None when the golden set has no targets.
    """
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    hits = dict.fromkeys(ks, 0)
    total = 0
    missing: list[str] = []
    for item in golden["questions"]:
        trace_file = traces_dir / f"{item['slug']}.json"
        if not trace_file.is_file():
            missing.append(item["slug"])
            total += len(item["expected_targets"])
            continue
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
        ranked = sorted(trace["passages"], key=lambda p: p["score"], reverse=True)
        for target in item["expected_targets"]:
            total += 1
            for k in ks:
                if any(_hit(p, target) for p in ranked[:k]):
                    hits[k] += 1
    result: dict = {f"recall@{k}": (hits[k] / total if total else None) for k in ks}
    result["targets"] = total
    result["missing_traces"] = missing
    return result


def _read_labels(labels_csv: Path) -> dict[tuple[str, str], bool]:
    """Read the labelled sheet into {(qid, chunk_id): relevant}; each label must be y or n."""
    labels: dict[tuple[str, str], bool] = {}
    with labels_csv.open(encoding="utf-8", newline="") as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            key = (row["qid"], row["chunk_id"])
            label = (row["relevant"] or "").strip().lower()
            if label not in ("y", "n"):
                raise ValueError(f"{labels_csv} line {line}: qid={key[0]} chunk_id={key[1]} "
                                 f"has relevant={row['relevant']!r}; expected y or n")
            if key in labels:
                raise ValueError(f"{labels_csv} line {line}: duplicate label row for "
                                 f"qid={key[0]} chunk_id={key[1]}")
            labels[key] = label == "y"
    return labels


def evaluate_pooled_recall(pool_json: Path, labels_csv: Path,
                           ks: tuple[int, ...] = POOLED_KS) -> dict:
    """Recall@k against pooled relevance labels.

    The relevant set of a question is its pooled passages labelled ``y``. Recall@k of a
    question is ``|relevant & dense_ranking[:k]| / |relevant|``, so a relevant passage found
    only by BM25 is a miss at every k. Questions with no relevant passage are excluded.
    Micro sums numerators and denominators over questions; macro averages per-question recall.

    Returns:
        ``{"recall@5": {"micro": float | None, "macro": float | None}, ...,
        "relevant_passages": int, "questions_scored": int,
        "questions_excluded_no_relevant": [qid, ...], "labelled_rows": int}``;
        recall values are None when no question has a relevant passage.

    Raises:
        ValueError: a label is not y or n, a row is duplicated, a labelled row is not in the
            pool, or a pooled passage has no labelled row.
    """
    questions = json.loads(pool_json.read_text(encoding="utf-8"))["questions"]
    labels = _read_labels(labels_csv)
    pooled = {(qid, cid) for qid, entry in questions.items() for cid in entry["pooled"]}
    extra = sorted(set(labels) - pooled)
    missing = sorted(pooled - set(labels))
    if extra or missing:
        raise ValueError(f"labels do not match pool; rows not in pool: {extra}; "
                         f"pool rows without a label: {missing}")

    hits = dict.fromkeys(ks, 0)
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    total_relevant = 0
    excluded: list[str] = []
    for qid, entry in questions.items():
        relevant = {cid for cid in entry["pooled"] if labels[(qid, cid)]}
        if not relevant:
            excluded.append(qid)
            continue
        total_relevant += len(relevant)
        for k in ks:
            found = len(relevant & set(entry["dense_ranking"][:k]))
            hits[k] += found
            recalls[k].append(found / len(relevant))
    scored = len(questions) - len(excluded)
    result: dict = {
        f"recall@{k}": {"micro": hits[k] / total_relevant if total_relevant else None,
                        "macro": sum(recalls[k]) / scored if scored else None}
        for k in ks}
    result["relevant_passages"] = total_relevant
    result["questions_scored"] = scored
    result["questions_excluded_no_relevant"] = excluded
    result["labelled_rows"] = len(labels)
    return result


if __name__ == "__main__":
    print("Usage: evaluate_recall(Path('golden_recall.json'), Path('exports/traces'))")
