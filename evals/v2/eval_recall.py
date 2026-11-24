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

import json
from pathlib import Path

DEFAULT_KS = (5, 20)


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


if __name__ == "__main__":
    print("Usage: evaluate_recall(Path('golden_recall.json'), Path('exports/traces'))")
