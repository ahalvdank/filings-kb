"""v2 Eval 4: cost and wall time per filing ingested.

Input (JSONL, one record per stage run)::

    {"accession": "0001045810-25-000023", "ticker": "NVDA", "fiscal_year": 2025,
     "stage": "extract_claims", "model": "claude-sonnet-4-6", "input_tokens": 1200,
     "output_tokens": 300, "wall_seconds": 4.2, "usd_cost": "0.0081"}

Stages: parse_chunk, embed, extract_claims, entity_link, signals. Filings are keyed by
accession (a ticker can have two filings in one fiscal year). ``usd_cost`` is parsed as a
decimal so sums are exact; give it as a string. Percentiles are nearest-rank, no
interpolation. No cost is reported for the committed sample: its extractions were imported
from a private cache and the ingest report records ``cost_usd`` of ``"0"``.
"""

import json
import math
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

STAGES = frozenset({"parse_chunk", "embed", "extract_claims", "entity_link", "signals"})


def percentile(values: list[Decimal] | list[float], pct: int) -> Decimal | float | None:
    """Nearest-rank percentile (``pct`` in 1..100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(pct / 100 * len(ordered)), 1) - 1]


def summarize_ingest_costs(log_jsonl: Path) -> dict:
    """Median and p90 cost and time per filing, and totals per stage.

    Raises:
        ValueError: for an unknown stage.
    """
    per_filing: dict[str, dict] = defaultdict(
        lambda: {"usd": Decimal(0), "seconds": 0.0, "tokens": 0})
    per_stage: dict[str, dict] = defaultdict(
        lambda: {"usd": Decimal(0), "seconds": 0.0, "tokens": 0})
    for line in log_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["stage"] not in STAGES:
            raise ValueError(f"unknown stage {rec['stage']!r}")
        usd = Decimal(str(rec["usd_cost"]))
        tokens = int(rec["input_tokens"]) + int(rec["output_tokens"])
        for bucket in (per_filing[rec["accession"]], per_stage[rec["stage"]]):
            bucket["usd"] += usd
            bucket["seconds"] += float(rec["wall_seconds"])
            bucket["tokens"] += tokens
    costs = [v["usd"] for v in per_filing.values()]
    times = [v["seconds"] for v in per_filing.values()]
    return {
        "filings_measured": len(per_filing),
        "median_usd": str(statistics.median(costs)) if costs else None,
        "p90_usd": str(percentile(costs, 90)) if costs else None,
        "median_seconds": round(statistics.median(times), 1) if times else None,
        "p90_seconds": round(percentile(times, 90), 1) if times else None,
        "total_usd": str(sum(costs, Decimal(0))),
        "per_stage": {s: {"usd": str(v["usd"]), "seconds": round(v["seconds"], 1),
                          "tokens": v["tokens"]} for s, v in sorted(per_stage.items())},
    }


if __name__ == "__main__":
    print("Usage: summarize_ingest_costs(Path('ingest_cost_log.jsonl'))")
