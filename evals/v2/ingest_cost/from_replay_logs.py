"""Cost and time per 10-K from the extraction run's ``replay_<TICKER>.log`` files.

Feeds ``evals.v2.eval_ingest_cost.summarize_ingest_costs`` (stage ``extract_claims``).

Rules:

* Only ``extract ok`` lines count; ``call ok`` and other lines are ignored. A filing's cost,
  seconds and tokens are the sums over every ``extract ok`` line for its accession (one per
  section or sub-chunk).
* Excluded and counted, in this order:
  ``excluded_malformed`` (an ``extract ok`` line missing ``acc``, ``cost_usd`` or
  ``wallclock``, or with a non-numeric value);
  ``excluded_oauth_notional`` (ticker with a ``default(lean=false) oauth`` row and no plain
  default row: Claude-subscription calls, not billed per call; listed in
  ``tickers_oauth_notional``; never triggers the exit 3 below);
  ``excluded_mode_unconfirmed`` (ticker absent from ``replay_costs.tsv``, or whose rows are
  none of ``default(lean=false)``, ``... oauth`` or ``lean...``; the ticker is listed in
  ``tickers_unconfirmed`` and the CLI exits 3 unless ``--allow-unconfirmed``);
  ``excluded_lean_mode`` (ticker whose only rows are lean, or, for a ticker with both lean
  and default rows, a call logged before the earliest default row's ``start``; a ticker with
  only default rows keeps every call, since a killed first run writes no TSV row);
  ``excluded_missing_tokens`` (no ``input_tokens``/``output_tokens``).
  ``excluded_lean_or_missing_tokens`` is the sum of the last two.
* Logs for one ticker merge by accession: ``replay_<T>.log``, ``replay_<T>.lean_probe.log``
  and the renamed ``replay_<T>.<YYYYMMDDHHMMSS>.prev.log``.
* ``--exclude-accession ACC=REASON`` drops a filing before aggregation; the output carries
  only the reason and call count.
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
import math
import re
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from evals.v2.eval_ingest_cost import percentile, summarize_ingest_costs

DEFAULT_MODE = "default(lean=false)"
OAUTH_MODE = "default(lean=false) oauth"
LOG_TZ = ZoneInfo("America/New_York")
PROBE_SUFFIX = ".lean_probe"
_PREV_RE = re.compile(r"\.\d{14}\.prev$")
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
    try:
        cost = Decimal(kv["cost_usd"])
    except InvalidOperation as exc:
        raise ValueError(f"extract ok line has non-numeric cost_usd {kv['cost_usd']!r}") from exc
    wall = float(kv["wallclock"].rstrip("s"))
    if not cost.is_finite() or not math.isfinite(wall):
        raise ValueError("extract ok line has a non-finite cost_usd or wallclock")

    def count(key: str) -> int | None:
        return int(kv[key]) if key in kv else None

    return {
        "logged_at": datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f"),
        "accession": kv["acc"],
        "section": kv.get("section"),
        "usd_cost": kv["cost_usd"],
        "wall_seconds": wall,
        "input_tokens_uncached": count("input_tokens"),
        "output_tokens": count("output_tokens"),
        "cache_creation_tokens": count("cache_creation"),
        "cache_read_tokens": count("cache_read"),
    }


def _ticker_of(log_path: Path) -> str:
    name = log_path.stem.removeprefix("replay_")
    return _PREV_RE.sub("", name).removesuffix(PROBE_SUFFIX)


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
    oauth_tickers = {row["ticker"] for row in rows if row["mode"] == OAUTH_MODE}

    records: list[dict] = []
    excluded_lean = excluded_missing = excluded_unconfirmed = excluded_malformed = 0
    excluded_oauth = 0
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
            if cutoff is None:
                if ticker in oauth_tickers:
                    excluded_oauth += 1
                elif ticker in has_lean_row:
                    excluded_lean += 1
                else:
                    excluded_unconfirmed += 1
                    unconfirmed.add(ticker)
                continue
            if ticker in has_lean_row and rec["logged_at"].replace(tzinfo=LOG_TZ) < cutoff:
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
        "excluded_oauth_notional": excluded_oauth,
        "excluded_malformed": excluded_malformed,
        "partial": partial,
        "timeouts": timeouts,
        "tickers": sorted(default_starts),
        "tickers_unconfirmed": sorted(unconfirmed),
        "tickers_oauth_notional": sorted(oauth_tickers - set(default_starts)),
    }
    return records, stats


def exclude_accessions(records: list[dict],
                       reasons: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Drop the records of the given accessions.

    Args:
        records: stage records.
        reasons: accession -> reason.

    Returns:
        ``(kept, excluded_filings)`` where each excluded entry is ``{"reason", "calls"}``
        (never the accession).

    Raises:
        ValueError: when an excluded accession has no records.
    """
    excluded = []
    for acc, reason in reasons.items():
        calls = sum(1 for r in records if r["accession"] == acc)
        if calls == 0:
            raise ValueError(f"--exclude-accession given for an accession with no records "
                             f"(reason: {reason!r})")
        excluded.append({"reason": reason, "calls": calls})
    return [r for r in records if r["accession"] not in reasons], excluded


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
    summary["excluded_oauth_notional"] = stats["excluded_oauth_notional"]
    summary["tickers_oauth_notional"] = stats["tickers_oauth_notional"]
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
    parser.add_argument("--exclude-accession", action="append", default=[],
                        metavar="ACC=REASON",
                        help="drop a filing before aggregation (repeatable); only the "
                             "reason and call count are written")
    args = parser.parse_args(argv)
    reasons: dict[str, str] = {}
    for item in args.exclude_accession:
        acc, sep, reason = item.partition("=")
        if not sep or not acc or not reason:
            parser.error(f"--exclude-accession expects ACC=REASON, got {item!r}")
        reasons[acc] = reason
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
    records, excluded_filings = exclude_accessions(records, reasons)
    summary = summarize_replay(records, stats)
    summary["excluded_filings"] = excluded_filings
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {summary['filings_measured']} filings")


if __name__ == "__main__":
    main()
