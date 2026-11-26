"""Cost and time per 10-K from the extraction run's ``replay_<TICKER>.log`` files.

Feeds ``evals.v2.eval_ingest_cost.summarize_ingest_costs`` (stage ``extract_claims``).

Rules:

* Only ``extract ok`` lines count; ``call ok`` and other lines are ignored. A filing's cost,
  seconds and tokens are the sums over every ``extract ok`` line for its accession (one per
  section or sub-chunk).
* Excluded and counted, in this order:
  ``excluded_malformed`` (an ``extract ok`` line missing ``acc``, ``cost_usd`` or
  ``wallclock``, or with a non-numeric value);
  ``excluded_mode_unconfirmed`` (ticker absent from ``replay_costs.tsv``, or with no
  ``default(lean=false)`` row and no lean row; the ticker is listed in
  ``tickers_unconfirmed`` and the CLI exits 3 unless ``--allow-unconfirmed``);
  ``excluded_lean_mode`` (ticker whose only rows are lean, or a call logged before the
  earliest default row's ``start`` for a ticker that has both lean and default rows);
  ``excluded_missing_tokens`` (no ``input_tokens``/``output_tokens``).
  ``excluded_lean_or_missing_tokens`` is the sum of the last two.
* Log timestamps are local New York time (``America/New_York``, DST-aware); TSV times are UTC.
* Seconds per filing is the sum of ``wallclock`` over its calls, so it includes retries and
  waits.
* ``input_tokens`` in the log is only the uncached part. Prompt caching puts nearly all input
  into ``cache_creation``/``cache_read``, so the four token kinds are reported separately.

The output holds aggregates only: no accessions, and no tickers beyond the run's list.
"""

import argparse
import csv
import json
import re
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from evals.v2.eval_ingest_cost import percentile, summarize_ingest_costs

DEFAULT_MODE = "default(lean=false)"
LOG_TZ = ZoneInfo("America/New_York")
PROBE_SUFFIX = ".lean_probe"
_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) .*? extract ok (.*)$")
_KV_RE = re.compile(r"(\w+)=(\S+)")
_TOKEN_KINDS = ("input_tokens_uncached", "cache_creation_tokens", "cache_read_tokens",
                "total_input_tokens", "output_tokens")


def parse_extract_ok(line: str) -> dict | None:
    """Parse one ``extract ok`` log line.

    Returns:
        A dict, or None when the line is not an ``extract ok`` line. Token fields are None
        when the line lacks them. ``usd_cost`` stays the logged string.

    Raises:
        ValueError: for an ``extract ok`` line missing ``acc``, ``cost_usd`` or ``wallclock``,
            or with a non-numeric value.
    """
    match = _LINE_RE.match(line.strip())
    if match is None:
        return None
    stamp, rest = match.groups()
    kv = dict(_KV_RE.findall(rest))
    missing = [key for key in ("acc", "cost_usd", "wallclock") if key not in kv]
    if missing:
        raise ValueError(f"extract ok line missing {missing}")
    Decimal(kv["cost_usd"])

    def count(key: str) -> int | None:
        return int(kv[key]) if key in kv else None

    return {
        "logged_at": datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f"),
        "accession": kv["acc"],
        "section": kv.get("section"),
        "usd_cost": kv["cost_usd"],
        "wall_seconds": float(kv["wallclock"].rstrip("s")),
        "input_tokens_uncached": count("input_tokens"),
        "output_tokens": count("output_tokens"),
        "cache_creation_tokens": count("cache_creation"),
        "cache_read_tokens": count("cache_read"),
    }


def _ticker_of(log_path: Path) -> str:
    name = log_path.stem.removeprefix("replay_")
    return name.removesuffix(PROBE_SUFFIX)


def _read_tsv(tsv: Path) -> list[dict]:
    with tsv.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def to_stage_records(log_paths: list[Path], tsv: Path) -> tuple[list[dict], dict]:
    """Turn replay logs into ``extract_claims`` stage records, keeping only default-mode calls.

    Returns:
        ``(records, stats)``. ``stats`` has the exclusion counters named in the module
        docstring, ``partial`` and ``timeouts`` (summed over default-mode rows), ``tickers``
        (those with a default-mode row) and ``tickers_unconfirmed``.
    """
    rows = _read_tsv(tsv)
    default_starts: dict[str, datetime] = {}
    partial = timeouts = 0
    for row in rows:
        if row["mode"] != DEFAULT_MODE:
            continue
        start = _utc(row["start"])
        ticker = row["ticker"]
        default_starts[ticker] = min(start, default_starts.get(ticker, start))
        partial += int(row["partial"])
        timeouts += int(row["timeouts"])
    has_lean_row = {row["ticker"] for row in rows if row["mode"].startswith("lean")}

    records: list[dict] = []
    excluded_lean = excluded_missing = excluded_unconfirmed = excluded_malformed = 0
    unconfirmed: set[str] = set()
    for log_path in sorted(log_paths):
        ticker = _ticker_of(log_path)
        cutoff = default_starts.get(ticker)
        for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = parse_extract_ok(raw)
            except ValueError:
                excluded_malformed += 1
                continue
            if rec is None:
                continue
            if cutoff is None and ticker not in has_lean_row:
                excluded_unconfirmed += 1
                unconfirmed.add(ticker)
                continue
            logged_utc = rec["logged_at"].replace(tzinfo=LOG_TZ)
            if cutoff is None or logged_utc < cutoff:
                excluded_lean += 1
                continue
            if rec["input_tokens_uncached"] is None or rec["output_tokens"] is None:
                excluded_missing += 1
                continue
            # A missing cache field means the call reported none.
            rec["cache_creation_tokens"] = rec["cache_creation_tokens"] or 0
            rec["cache_read_tokens"] = rec["cache_read_tokens"] or 0
            rec["total_input_tokens"] = (rec["input_tokens_uncached"]
                                         + rec["cache_creation_tokens"]
                                         + rec["cache_read_tokens"])
            rec["ticker"] = ticker
            rec["stage"] = "extract_claims"
            records.append(rec)
    stats = {
        "excluded_lean_or_missing_tokens": excluded_lean + excluded_missing,
        "excluded_lean_mode": excluded_lean,
        "excluded_missing_tokens": excluded_missing,
        "excluded_mode_unconfirmed": excluded_unconfirmed,
        "excluded_malformed": excluded_malformed,
        "partial": partial,
        "timeouts": timeouts,
        "tickers": sorted(default_starts),
        "tickers_unconfirmed": sorted(unconfirmed),
    }
    return records, stats


def _spread(values: list[int]) -> dict:
    return {"median": statistics.median(values) if values else None,
            "p90": percentile(values, 90), "total": sum(values)}


def summarize_replay(records: list[dict], stats: dict) -> dict:
    """Per-filing median, p90 and totals for cost, seconds and each token kind."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "ingest_cost_log.jsonl"
        with jsonl.open("w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps({
                    "accession": rec["accession"], "stage": "extract_claims",
                    "usd_cost": rec["usd_cost"], "wall_seconds": rec["wall_seconds"],
                    "input_tokens": rec["total_input_tokens"],
                    "output_tokens": rec["output_tokens"]}) + "\n")
        summary = summarize_ingest_costs(jsonl)

    per_filing: dict[str, dict[str, int]] = {}
    for rec in records:
        bucket = per_filing.setdefault(rec["accession"], dict.fromkeys(_TOKEN_KINDS, 0))
        for kind in _TOKEN_KINDS:
            bucket[kind] += rec[kind]
    summary["tokens_per_filing"] = {
        kind: _spread([b[kind] for b in per_filing.values()]) for kind in _TOKEN_KINDS}
    summary["total_seconds"] = round(sum(r["wall_seconds"] for r in records), 1)
    summary["seconds_basis"] = ("sum of wallclock over a filing's extract calls; includes "
                                "retries and waits")
    summary["excluded_lean_or_missing_tokens"] = stats["excluded_lean_or_missing_tokens"]
    summary["excluded_lean_mode"] = stats["excluded_lean_mode"]
    summary["excluded_missing_tokens"] = stats["excluded_missing_tokens"]
    summary["excluded_mode_unconfirmed"] = stats["excluded_mode_unconfirmed"]
    summary["excluded_malformed"] = stats["excluded_malformed"]
    summary["tickers_unconfirmed"] = stats["tickers_unconfirmed"]
    summary["partial"] = stats["partial"]
    summary["timeouts"] = stats["timeouts"]
    summary["tickers"] = stats["tickers"]
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI: parse a replay-log directory and write aggregate numbers as JSON.

    Raises:
        FileNotFoundError: when ``replay_costs.tsv`` is not in the logs directory.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, required=True,
                        help="directory with replay_<TICKER>.log and replay_costs.tsv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unconfirmed", action="store_true",
                        help="write the result even if some tickers have no confirmed mode")
    args = parser.parse_args(argv)
    tsv = args.logs / "replay_costs.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(tsv)
    records, stats = to_stage_records(sorted(args.logs.glob("replay_*.log")), tsv)
    if stats["excluded_mode_unconfirmed"] and not args.allow_unconfirmed:
        print(f"error: {stats['excluded_mode_unconfirmed']} calls from tickers with no "
              f"confirmed default-mode row in replay_costs.tsv: "
              f"{', '.join(stats['tickers_unconfirmed'])}. Wait for the run to finish or "
              f"pass --allow-unconfirmed.", file=sys.stderr)
        raise SystemExit(3)
    summary = summarize_replay(records, stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {summary['filings_measured']} filings")


if __name__ == "__main__":
    main()
