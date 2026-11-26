"""Tests for the replay-log cost parser. All lines are synthetic; tickers, accessions,
dates and numbers are invented."""

import json
from decimal import Decimal

import pytest

from evals.v2.ingest_cost.from_replay_logs import (
    main,
    parse_extract_ok,
    summarize_replay,
    to_stage_records,
)

TSV_HEADER = "ticker\tstart\tend\trc\tlogged_cost_usd\tcalls_ok\tpartial\ttimeouts\targs\tmode\n"
DEFAULT = "default(lean=false)"
# Winter date: New York is UTC-5, so 12:00 local == 17:00Z.
DAY = "2030-01-15"
START = "2030-01-14T00:00:00Z"


def line(acc="0000000001-00-000001", section="Item_1", ts=f"{DAY} 12:00:00,000",
         wall="100.0", inp="2", out="1000", cr="0", cc="5000", cost="0.100000",
         with_tokens=True):
    tokens = (f"input_tokens={inp} output_tokens={out} " if with_tokens else "")
    return (f"{ts} INFO src.knowledge_base.llm_extractor extract ok acc={acc} "
            f"section={section} backend=subprocess attempt=1 attempt_wallclock={wall}s "
            f"wallclock={wall}s entities=1 relationships=1 chunks=1 {tokens}"
            f"cache_read={cr} cache_creation={cc} cache_creation_5m={cc} "
            f"cache_creation_1h=0 cost_usd={cost}")


def tsv_row(ticker, start, mode=DEFAULT, partial=0, timeouts=0):
    return (f"{ticker}\t{start}\t2030-01-20T09:00:00Z\t0\t1.0\t3\t{partial}\t{timeouts}\t"
            f"--limit 1\t{mode}\n")


def write(tmp_path, ticker, lines, rows, name=None):
    log = tmp_path / (name or f"replay_{ticker}.log")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tsv = tmp_path / "replay_costs.tsv"
    tsv.write_text(TSV_HEADER + "".join(rows), encoding="utf-8")
    return log, tsv


def test_parse_extract_ok_reads_all_fields_as_exact_strings_for_cost():
    rec = parse_extract_ok(line(cost="0.1000000001"))
    assert rec["accession"] == "0000000001-00-000001"
    assert rec["section"] == "Item_1"
    assert rec["usd_cost"] == "0.1000000001"
    assert Decimal(rec["usd_cost"]) == Decimal("0.1000000001")
    assert rec["input_tokens_uncached"] == 2
    assert rec["output_tokens"] == 1000
    assert rec["cache_creation_tokens"] == 5000
    assert rec["cache_read_tokens"] == 0
    assert rec["wall_seconds"] == pytest.approx(100.0)
    assert rec["logged_at"].isoformat() == f"{DAY}T12:00:00"


def test_non_extract_lines_ignored():
    assert parse_extract_ok(f"call ok AAA {DAY} section=Item_1 wallclock=100.0s") is None
    assert parse_extract_ok(f"{DAY} 12:00:00,000 WARNING x claude stdout not valid JSON "
                            "acc=0000000001-00-000001") is None
    assert parse_extract_ok("") is None


@pytest.mark.parametrize("drop", ["acc=", "cost_usd=", "wallclock="])
def test_extract_ok_missing_required_field_raises_value_error(drop):
    mangled = " ".join(p for p in line().split(" ")
                       if not p.startswith(drop) and not p.startswith("attempt_" + drop))
    with pytest.raises(ValueError):
        parse_extract_ok(mangled)


def test_malformed_extract_ok_lines_are_excluded_and_counted(tmp_path):
    no_acc = " ".join(p for p in line(section="Item_2").split(" ") if not p.startswith("acc="))
    no_cost = " ".join(p for p in line(section="Item_3").split(" ")
                       if not p.startswith("cost_usd="))
    log, tsv = write(tmp_path, "AAA", [line(), no_acc, no_cost],
                     [tsv_row("AAA", START)])
    records, stats = to_stage_records([log], tsv)
    assert len(records) == 1
    assert stats["excluded_malformed"] == 2


def test_line_without_token_fields_is_excluded_and_counted(tmp_path):
    log, tsv = write(tmp_path, "AAA",
                     [line(), line(section="Item_2", with_tokens=False)],
                     [tsv_row("AAA", START)])
    records, stats = to_stage_records([log], tsv)
    assert len(records) == 1
    assert stats["excluded_missing_tokens"] == 1
    assert stats["excluded_lean_mode"] == 0


def test_rows_marked_lean_in_tsv_are_excluded(tmp_path):
    log, tsv = write(tmp_path, "AAA", [line(), line(section="Item_2")],
                     [tsv_row("AAA", START, mode="lean(probe; x)")])
    records, stats = to_stage_records([log], tsv)
    assert records == []
    assert stats["excluded_lean_mode"] == 2
    assert stats["excluded_mode_unconfirmed"] == 0
    assert stats["tickers_unconfirmed"] == []


def test_ticker_missing_from_tsv_is_unconfirmed_not_lean(tmp_path):
    log, tsv = write(tmp_path, "AAA", [line()], [tsv_row("BBB", START)])
    records, stats = to_stage_records([log], tsv)
    assert records == []
    assert stats["excluded_mode_unconfirmed"] == 1
    assert stats["excluded_lean_mode"] == 0
    assert stats["tickers_unconfirmed"] == ["AAA"]


def test_ticker_with_only_non_default_non_lean_row_is_unconfirmed(tmp_path):
    log, tsv = write(tmp_path, "AAA", [line()], [tsv_row("AAA", START, mode="other")])
    _, stats = to_stage_records([log], tsv)
    assert stats["excluded_mode_unconfirmed"] == 1
    assert stats["tickers_unconfirmed"] == ["AAA"]


def test_lean_then_default_keeps_only_calls_after_default_start(tmp_path):
    # default row starts 2030-01-16T01:00:00Z == 2030-01-15 20:00:00 local (EST, UTC-5)
    before = line(section="Item_1", ts=f"{DAY} 19:59:59,000")
    after = line(section="Item_1A", ts=f"{DAY} 20:00:01,000")
    log, tsv = write(tmp_path, "AAA", [before, after],
                     [tsv_row("AAA", "2030-01-14T01:00:00Z", mode="lean(probe)"),
                      tsv_row("AAA", "2030-01-16T01:00:00Z")])
    records, stats = to_stage_records([log], tsv)
    assert [r["section"] for r in records] == ["Item_1A"]
    assert stats["excluded_lean_mode"] == 1


def test_cutoff_uses_new_york_dst_not_a_fixed_offset(tmp_path):
    # US DST began 2030-03-10. Default start 12:00Z == 08:00 local EDT (UTC-4).
    # A 07:30 local line is 11:30Z (before start); a fixed UTC-5 would read it as 12:30Z.
    early = line(section="Item_1", ts="2030-03-11 07:30:00,000")
    late = line(section="Item_1A", ts="2030-03-11 08:30:00,000")
    log, tsv = write(tmp_path, "AAA", [early, late],
                     [tsv_row("AAA", "2030-03-01T00:00:00Z", mode="lean(probe)"),
                      tsv_row("AAA", "2030-03-11T12:00:00Z")])
    records, _ = to_stage_records([log], tsv)
    assert [r["section"] for r in records] == ["Item_1A"]


def test_probe_log_file_is_attributed_to_its_ticker_and_filtered_by_time(tmp_path):
    probe = tmp_path / "replay_AAA.lean_probe.log"
    probe.write_text(line(ts=f"{DAY} 10:00:00,000", with_tokens=False) + "\n")
    main_log, tsv = write(tmp_path, "AAA", [line(ts=f"{DAY} 21:00:00,000")],
                          [tsv_row("AAA", "2030-01-14T01:00:00Z", mode="lean(probe)"),
                           tsv_row("AAA", "2030-01-15T20:00:00Z")])
    records, stats = to_stage_records([probe, main_log], tsv)
    assert len(records) == 1
    assert stats["excluded_lean_mode"] == 1


def test_stats_report_tickers_and_partial_timeouts_for_default_rows_only(tmp_path):
    log_a, tsv = write(tmp_path, "AAA", [line()],
                       [tsv_row("AAA", START, partial=2, timeouts=1),
                        tsv_row("BBB", START, partial=5, timeouts=5, mode="lean(probe)")])
    _, stats = to_stage_records([log_a], tsv)
    assert stats["partial"] == 2
    assert stats["timeouts"] == 1
    assert stats["tickers"] == ["AAA"]


def test_per_filing_aggregation_sums_sections_and_subchunks(tmp_path):
    a, b = "0000000001-00-000001", "0000000002-00-000002"
    lines = [
        line(acc=a, section="Item_1", wall="10.0", inp="2", out="100", cr="5", cc="7",
             cost="0.10"),
        line(acc=a, section="Item_1", wall="20.5", inp="3", out="200", cr="11", cc="13",
             cost="0.25"),  # sub-chunk of the same section
        line(acc=a, section="Item_1A", wall="1.0", inp="1", out="50", cr="0", cc="1",
             cost="0.05"),
        line(acc=b, section="Item_1", wall="5.0", inp="2", out="10", cr="0", cc="2",
             cost="0.01"),
    ]
    log, tsv = write(tmp_path, "AAA", lines, [tsv_row("AAA", START)])
    records, stats = to_stage_records([log], tsv)
    summary = summarize_replay(records, stats)

    assert summary["filings_measured"] == 2
    assert summary["total_usd"] == "0.41"          # exact decimal, 0.10+0.25+0.05+0.01
    assert summary["median_usd"] == "0.205"         # median of 0.40 and 0.01
    assert summary["seconds_basis"].startswith("sum of wallclock")
    assert summary["total_seconds"] == 36.5
    tok = summary["tokens_per_filing"]
    assert tok["input_tokens_uncached"]["total"] == 8
    assert tok["cache_creation_tokens"]["total"] == 23
    assert tok["cache_read_tokens"]["total"] == 16
    assert tok["total_input_tokens"]["total"] == 8 + 23 + 16
    assert tok["output_tokens"]["total"] == 360
    # filing a: uncached 6, filing b: 2 -> median 4
    assert tok["input_tokens_uncached"]["median"] == 4
    assert tok["output_tokens"]["median"] == 180  # (350 + 10) / 2
    assert "input_tokens" not in tok
    for key in ("excluded_lean_mode", "excluded_mode_unconfirmed",
                "excluded_missing_tokens", "excluded_malformed",
                "excluded_lean_or_missing_tokens"):
        assert summary[key] == 0
    assert summary["tickers_unconfirmed"] == []


def test_main_writes_only_aggregates_no_accessions(tmp_path):
    acc = "0000000009-00-000009"
    write(tmp_path, "AAA", [line(acc=acc)], [tsv_row("AAA", START)])
    out = tmp_path / "out" / "result.json"
    main(["--logs", str(tmp_path), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert acc not in text
    data = json.loads(text)
    assert data["filings_measured"] == 1
    assert data["tickers"] == ["AAA"]


def test_main_exits_3_when_a_ticker_is_unconfirmed(tmp_path, capsys):
    write(tmp_path, "AAA", [line()], [tsv_row("BBB", START)])
    out = tmp_path / "result.json"
    with pytest.raises(SystemExit) as exc:
        main(["--logs", str(tmp_path), "--out", str(out)])
    assert exc.value.code == 3
    assert "AAA" in capsys.readouterr().err
    assert not out.exists()


def test_main_allow_unconfirmed_writes_result_with_unconfirmed_listed(tmp_path):
    write(tmp_path, "AAA", [line()], [tsv_row("BBB", START)])
    out = tmp_path / "result.json"
    main(["--logs", str(tmp_path), "--out", str(out), "--allow-unconfirmed"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tickers_unconfirmed"] == ["AAA"]
    assert data["excluded_mode_unconfirmed"] == 1


def test_main_missing_tsv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--logs", str(tmp_path), "--out", str(tmp_path / "r.json")])
