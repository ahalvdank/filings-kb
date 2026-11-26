"""Tests for the v2 faithfulness units, the LLM-judge harness and the v2 scoring."""

import csv
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.v2 import eval_faithfulness
from evals.v2.faithfulness import judge, units

SLUG = "q-one"


# --- fixtures ----------------------------------------------------------------------


def _db(tmp_path: Path, chunks: dict[str, str]) -> Path:
    path = tmp_path / "kb.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, accession TEXT, item TEXT, "
                     "ticker TEXT, fiscal_year INTEGER, start_offset INTEGER, end_offset INTEGER, "
                     "text TEXT)")
        conn.executemany("INSERT INTO chunks (chunk_id, text) VALUES (?, ?)", chunks.items())
    return path


def _answer(paragraphs: list[list[dict]], passages: list[dict]) -> dict:
    return {"paragraphs": paragraphs, "passages": passages}


HEADER = {"ticker": "NVDA", "fiscal_year": 2026, "form": "10-K", "item": "Item_1A"}


def _passage(cite: int | None, chunk_id: str, included: bool = True, **header) -> dict:
    return {"cite": cite, "chunk_id": chunk_id, "included": included, **(HEADER | header),
            "context": {"before": "", "match": "IGNORED", "after": ""}}


def _write_case(tmp_path: Path, answer: dict, *, with_trace: bool = True,
                extra_answer_slug: str | None = None) -> tuple[Path, Path]:
    answers, traces = tmp_path / "answers", tmp_path / "traces"
    answers.mkdir()
    traces.mkdir()
    (answers / f"{SLUG}.json").write_text(json.dumps(answer))
    if with_trace:
        (traces / f"{SLUG}.json").write_text(json.dumps({"slug": SLUG}))
    if extra_answer_slug:  # an answer with no exported trace must be ignored
        (answers / f"{extra_answer_slug}.json").write_text(json.dumps(answer))
    return answers, traces


def _unit(unit_id: str, sentence: str = "Revenue was $5.5 billion.") -> dict:
    return {"unit_id": unit_id, "slug": unit_id.split(":")[0], "sentence": sentence,
            "cites": [1], "passages": [{"cite": 1, "chunk_id": "c1", **HEADER,
                                        "text": "passage text"}]}


class FakeMessages:
    """Records every create() call and replays scripted responses in order."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(responses)


def _response(text: str | None, stop_reason: str = "end_turn", input_tokens: int = 1000,
              output_tokens: int = 200) -> SimpleNamespace:
    content = [SimpleNamespace(type="thinking", thinking="...")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            cache_read_input_tokens=None, cache_creation_input_tokens=None)
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=usage)


def _verdict(verdict: str, span: str | None = None) -> str:
    return json.dumps({"verdict": verdict, "unsupported_span": span})


def _six_dollar_response() -> SimpleNamespace:
    # 300,000 output tokens at $20/MTok is exactly $6.
    return _response(_verdict("supported"), input_tokens=0, output_tokens=300_000)


# --- units -------------------------------------------------------------------------


def test_build_units_maps_cites_to_chunk_text_and_counts(tmp_path) -> None:
    answer = _answer(
        paragraphs=[
            [{"text": "Sentence one.", "cites": [2, 1]}, {"text": "Sentence two.", "cites": [1]}],
            [{"text": "Sentence three.", "cites": [3]}],
        ],
        passages=[_passage(1, "c-a"), _passage(2, "c-b", ticker="AMD", fiscal_year=2025,
                                              item="Item_7_MDA"), _passage(3, "c-c"),
                  _passage(None, "c-rejected", included=False)],
    )
    answers, traces = _write_case(tmp_path, answer, extra_answer_slug="no-trace")
    db = _db(tmp_path, {"c-a": "Text A.", "c-b": "Text B.", "c-c": "Text C.", "c-rejected": "R."})

    built = units.build_units(answers, traces, db)

    assert [u["unit_id"] for u in built] == [f"{SLUG}:p1u1", f"{SLUG}:p1u2", f"{SLUG}:p2u1"]
    assert built[0] == {
        "unit_id": f"{SLUG}:p1u1", "slug": SLUG, "sentence": "Sentence one.", "cites": [2, 1],
        "passages": [{"cite": 2, "chunk_id": "c-b", "ticker": "AMD", "fiscal_year": 2025,
                      "form": "10-K", "item": "Item_7_MDA", "text": "Text B."},
                     {"cite": 1, "chunk_id": "c-a", **HEADER, "text": "Text A."}],
    }
    assert sum(len(u["cites"]) for u in built) == 4  # (unit, cite) pairs
    assert {u["slug"] for u in built} == {SLUG}  # the answer without a trace is skipped


def test_build_units_raises_on_missing_cite(tmp_path) -> None:
    answer = _answer([[{"text": "S.", "cites": [1, 9]}]], [_passage(1, "c-a")])
    answers, traces = _write_case(tmp_path, answer)
    db = _db(tmp_path, {"c-a": "Text A."})
    with pytest.raises(ValueError, match="cite 9"):
        units.build_units(answers, traces, db)


def test_build_units_raises_on_cite_to_excluded_passage(tmp_path) -> None:
    answer = _answer([[{"text": "S.", "cites": [1]}]], [_passage(1, "c-a", included=False)])
    answers, traces = _write_case(tmp_path, answer)
    with pytest.raises(ValueError, match="not included"):
        units.build_units(answers, traces, _db(tmp_path, {"c-a": "Text A."}))


def test_build_units_raises_on_chunk_missing_from_db(tmp_path) -> None:
    answer = _answer([[{"text": "S.", "cites": [1]}]], [_passage(1, "c-a")])
    answers, traces = _write_case(tmp_path, answer)
    with pytest.raises(ValueError, match="c-a"):
        units.build_units(answers, traces, _db(tmp_path, {}))


def test_build_units_raises_on_trace_without_answer(tmp_path) -> None:
    answers, traces = _write_case(tmp_path, _answer([], []))
    (traces / "orphan.json").write_text("{}")
    with pytest.raises(ValueError, match="orphan"):
        units.build_units(answers, traces, _db(tmp_path, {}))


def test_units_jsonl_round_trips(tmp_path) -> None:
    rows = [_unit("a:p1u1"), _unit("a:p1u2")]
    path = tmp_path / "units.jsonl"
    units.write_units(rows, path)
    assert units.read_units(path) == rows


# --- judge_unit --------------------------------------------------------------------


def test_judge_unit_parses_structured_verdict() -> None:
    client = FakeClient([_response(_verdict("partial", "by $5.5 billion"))])
    result = judge.judge_unit(client, _unit("a:p1u1"), "PROMPT")

    assert result == {
        "unit_id": "a:p1u1", "verdict": "partial", "unsupported_span": "by $5.5 billion",
        "raw": _verdict("partial", "by $5.5 billion"), "stop_reason": "end_turn",
        "usage": {"input_tokens": 1000, "output_tokens": 200,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }
    (call,) = client.messages.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "PROMPT"
    assert call["max_tokens"] == judge.MAX_TOKENS
    assert call["output_config"]["effort"] == "high"
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["verdict"]["enum"] == ["supported", "partial", "unsupported"]
    assert "thinking" not in call and "tool_choice" not in call
    user = call["messages"][0]["content"]
    assert "Revenue was $5.5 billion." in user and "passage text" in user
    assert "[1] NVDA FY2026 10-K, Item 1A\npassage text" in user


def test_judge_message_uses_the_answer_models_passage_header(tmp_path) -> None:
    # Built from a fixture answer, the judge must see the header the answer model saw
    # (filings_kb.answer.build_user_message), not the chunk_id.
    answer = _answer([[{"text": "In FY2025 AMD said X.", "cites": [1]}]],
                     [_passage(1, "acc-1:Item_7_MDA:0003", ticker="AMD", fiscal_year=2025,
                               item="Item_7_MDA")])
    answers, traces = _write_case(tmp_path, answer)
    (built,) = units.build_units(answers, traces, _db(tmp_path, {"acc-1:Item_7_MDA:0003": "X."}))
    message = judge.user_message(built)
    assert "[1] AMD FY2025 10-K, Item 7\nX." in message
    assert "acc-1:Item_7_MDA:0003" not in message


@pytest.mark.parametrize("response", [
    _response("not json at all"),
    _response(json.dumps({"verdict": "mostly", "unsupported_span": None})),
    _response(json.dumps({"verdict": "supported"})),
    _response(json.dumps({"verdict": "supported", "unsupported_span": None, "extra": 1})),
    _response(json.dumps({"verdict": "partial", "unsupported_span": 3})),
    _response(None),
    _response(_verdict("supported"), stop_reason="max_tokens"),
    _response(_verdict("supported"), stop_reason="model_context_window_exceeded"),
], ids=["junk", "bad-enum", "missing-span", "extra-key", "span-not-string", "no-text-block",
        "max-tokens", "context-window"])
def test_judge_unit_records_parse_error_not_supported(response) -> None:
    result = judge.judge_unit(FakeClient([response]), _unit("a:p1u1"), "PROMPT")
    assert result["verdict"] == "parse_error"
    assert result["unsupported_span"] is None
    assert result["usage"]["input_tokens"] == 1000  # a failed call still costs money


def test_judge_unit_records_refusal() -> None:
    result = judge.judge_unit(FakeClient([_response(None, stop_reason="refusal")]),
                              _unit("a:p1u1"), "PROMPT")
    assert result["verdict"] == "refusal"
    assert result["unsupported_span"] is None


def test_call_cost_is_exact_decimal() -> None:
    usage = {"input_tokens": 1_234, "output_tokens": 567, "cache_read_input_tokens": 1_000,
             "cache_creation_input_tokens": 2_000}
    # 1234*4 + 567*20 + 1000*0.2 + 2000*5 = 4936 + 11340 + 200 + 10000 = 26476 micro-dollars
    assert judge.call_cost(usage) == Decimal("0.026476")
    assert isinstance(judge.call_cost(usage), Decimal)
    with pytest.raises(KeyError):
        judge.call_cost(usage, model="unknown-model")


# --- run ---------------------------------------------------------------------------


def _prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "judge_prompt.md"
    path.write_text("Judge prompt.")
    return path


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_run_stops_before_exceeding_budget(tmp_path) -> None:
    client = FakeClient([_six_dollar_response() for _ in range(5)])
    out = tmp_path / "judge_run1.jsonl"
    summary = judge.run([_unit(f"a:p1u{i}") for i in range(1, 6)], out, "run1",
                        max_usd=Decimal("15"), client=client, prompt_path=_prompt_file(tmp_path))

    assert len(client.messages.calls) == 2  # 6 + 6 = 12; a third call would reach 18 > 15
    assert summary["called"] == 2
    assert summary["stopped_on_budget"] is True
    assert summary["spent_usd"] == Decimal("12")
    records = _records(out)
    assert [r["unit_id"] for r in records] == ["a:p1u1", "a:p1u2"]
    first = records[0]
    assert first["model"] == "claude-opus-5-5" and first["effort"] == "high"
    assert first["run_label"] == "run1"
    assert first["prompt_sha256"] == judge.sha256_file(_prompt_file(tmp_path))
    assert Decimal(first["cost_usd"]) == Decimal("6")
    assert first["usage"]["output_tokens"] == 300_000


def test_run_projects_a_larger_unit_than_any_seen(tmp_path) -> None:
    # The first call costs about $0.004; the second unit's own worst case (about 1M input
    # tokens = $4, plus full output) is far above that, so it is not sent under a $1 cap.
    cheap = _response(_verdict("supported"), input_tokens=1000, output_tokens=0)
    client = FakeClient([cheap, cheap])
    big = _unit("a:p1u2", sentence="x" * 3_000_000)
    summary = judge.run([_unit("a:p1u1"), big], tmp_path / "o.jsonl", "run1",
                        max_usd=Decimal("1"), client=client, prompt_path=_prompt_file(tmp_path))
    assert len(client.messages.calls) == 1
    assert summary["stopped_on_budget"] is True
    assert judge.projected_call_cost("p", big) > Decimal("4")


def test_run_first_call_projection_blocks_tiny_budget(tmp_path) -> None:
    client = FakeClient([_six_dollar_response()])
    summary = judge.run([_unit("a:p1u1")], tmp_path / "out.jsonl", "run1",
                        max_usd=Decimal("0.01"), client=client, prompt_path=_prompt_file(tmp_path))
    assert client.messages.calls == []
    assert summary["stopped_on_budget"] is True


def test_run_resumes_skipping_done_units(tmp_path) -> None:
    out = tmp_path / "judge_run1.jsonl"
    prompt = _prompt_file(tmp_path)
    all_units = [_unit(f"a:p1u{i}") for i in range(1, 4)]
    judge.run(all_units[:1], out, "run1", client=FakeClient([_six_dollar_response()]),
              prompt_path=prompt)

    client = FakeClient([_six_dollar_response() for _ in range(3)])
    summary = judge.run(all_units, out, "run1", max_usd=Decimal("15"), client=client,
                        prompt_path=prompt)

    sent = [c["messages"][0]["content"] for c in client.messages.calls]
    assert len(sent) == 1  # unit 1 is done; prior $6 counts, so 6 + 6 = 12 and 18 > 15 stops
    assert summary["skipped"] == 1 and summary["called"] == 1
    assert summary["spent_usd"] == Decimal("12")
    assert [r["unit_id"] for r in _records(out)] == ["a:p1u1", "a:p1u2"]


def test_run_respects_limit(tmp_path) -> None:
    client = FakeClient([_response(_verdict("supported")) for _ in range(3)])
    summary = judge.run([_unit(f"a:p1u{i}") for i in range(1, 4)], tmp_path / "o.jsonl", "r",
                        client=client, prompt_path=_prompt_file(tmp_path), limit=2)
    assert summary["called"] == 2 and summary["stopped_on_budget"] is False


def test_main_exits_2_without_api_key(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "units.jsonl"
    units.write_units([_unit("a:p1u1")], path)
    with pytest.raises(SystemExit) as exc:
        judge.main(["--units", str(path), "--out", str(tmp_path / "o.jsonl"),
                    "--run-label", "run1", "--max-usd", "15"])
    assert exc.value.code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# --- spot sheet --------------------------------------------------------------------


def _judge_file(path: Path, verdicts: dict[str, str]) -> Path:
    path.write_text("".join(json.dumps({"unit_id": u, "verdict": v}) + "\n"
                            for u, v in verdicts.items()))
    return path


def test_spot_sheet_includes_all_flagged_and_hides_judge(tmp_path) -> None:
    all_units = [_unit(f"a:p1u{i}") for i in range(1, 21)]
    verdicts = {u["unit_id"]: "supported" for u in all_units}
    verdicts.update({"a:p1u3": "partial", "a:p1u7": "unsupported", "a:p1u9": "parse_error",
                     "a:p1u11": "refusal"})
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", verdicts)
    out = tmp_path / "spot.csv"

    written = units.write_spot_sheet(all_units, judge_jsonl, out, n=8)

    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == ["unit_id", "sentence", "passages", "human_verdict", "note"]
    ids = [r["unit_id"] for r in rows]
    flagged = {"a:p1u3", "a:p1u7", "a:p1u9", "a:p1u11"}
    assert written == 8 and len(ids) == 8 and len(set(ids)) == 8
    assert flagged <= set(ids)  # every non-supported unit, harness verdicts included
    assert all(verdicts[i] == "supported" for i in set(ids) - flagged)
    assert all(r["human_verdict"] == "" and r["note"] == "" for r in rows)
    assert rows[0]["passages"] == "[1] NVDA FY2026 10-K, Item 1A\npassage text"
    # Seeded: the same call gives the same sheet.
    again = tmp_path / "again.csv"
    units.write_spot_sheet(all_units, judge_jsonl, again, n=8)
    assert again.read_text() == out.read_text()


def test_spot_sheet_raises_when_judge_run_is_incomplete(tmp_path) -> None:
    all_units = [_unit("a:p1u1"), _unit("a:p1u2")]
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", {"a:p1u1": "supported"})
    with pytest.raises(ValueError, match="does not cover"):
        units.write_spot_sheet(all_units, judge_jsonl, tmp_path / "s.csv")


def test_spot_sheet_takes_all_flagged_when_over_n(tmp_path) -> None:
    all_units = [_unit(f"a:p1u{i}") for i in range(1, 6)]
    verdicts = {u["unit_id"]: "unsupported" for u in all_units}
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", verdicts)
    assert units.write_spot_sheet(all_units, judge_jsonl, tmp_path / "s.csv", n=3) == 5


# --- v2 scoring --------------------------------------------------------------------


def _spot_csv(path: Path, labels: dict[str, str]) -> Path:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_id", "sentence", "passages", "human_verdict", "note"])
        for unit_id, label in labels.items():
            writer.writerow([unit_id, "s", "p", label, ""])
    return path


def test_v2_scoring_excludes_parse_errors_and_builds_2x2(tmp_path) -> None:
    verdicts = {"u1": "supported", "u2": "supported", "u3": "supported", "u4": "partial",
                "u5": "unsupported", "u6": "parse_error", "u7": "refusal"}
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", verdicts)
    spot = _spot_csv(tmp_path / "spot.csv", {
        "u1": "supported", "u2": "supported", "u3": "partial",  # judge S: human S, S, not-S
        "u4": "unsupported", "u5": "supported",               # judge not-S: human not-S, S
        "u6": "supported",                                     # parse error: left out
    })

    result = eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, spot)

    assert result["units"] == 7
    assert (result["supported"], result["partial"], result["unsupported"]) == (3, 1, 1)
    assert (result["parse_errors"], result["refusals"]) == (1, 1)
    assert result["faithfulness"] == pytest.approx(3 / 5)
    assert result["faithfulness_lower_bound"] == pytest.approx(3 / 7)
    assert result["spot_n"] == 5
    assert result["table"] == {
        "judge_supported&human_supported": 2, "judge_supported&human_not_supported": 1,
        "judge_not_supported&human_supported": 1, "judge_not_supported&human_not_supported": 1,
    }
    assert result["raw_agreement"] == pytest.approx(3 / 5)
    # po = 0.6; pe = 0.6*0.6 + 0.4*0.4 = 0.52; kappa = 0.08 / 0.48
    assert result["cohens_kappa"] == pytest.approx(0.08 / 0.48)


def test_v2_scoring_without_spot_sheet_has_no_table(tmp_path) -> None:
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", {"u1": "supported", "u2": "parse_error"})
    result = eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, None)
    assert result["faithfulness"] == 1.0
    assert "table" not in result and "self_agreement" not in result


def test_v2_scoring_blank_human_labels_are_skipped(tmp_path) -> None:
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", {"u1": "supported", "u2": "partial"})
    spot = _spot_csv(tmp_path / "spot.csv", {"u1": "supported", "u2": " "})
    assert eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, spot)["spot_n"] == 1


def test_v2_scoring_all_parse_errors_gives_none(tmp_path) -> None:
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", {"u1": "parse_error"})
    assert eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, None)["faithfulness"] is None


@pytest.mark.parametrize("labels, match", [
    ({"u1": "yes"}, "human_verdict"),
    ({"u9": "supported"}, "u9"),
])
def test_v2_scoring_rejects_bad_spot_rows(tmp_path, labels, match) -> None:
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", {"u1": "supported"})
    with pytest.raises(ValueError, match=match):
        eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, _spot_csv(tmp_path / "s.csv", labels))


def test_v2_scoring_rejects_duplicate_and_unknown_verdicts(tmp_path) -> None:
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"unit_id": "u1", "verdict": "supported"}\n'
                   '{"unit_id": "u1", "verdict": "partial"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        eval_faithfulness.evaluate_faithfulness_v2(dup, None)
    bad = _judge_file(tmp_path / "bad.jsonl", {"u1": "maybe"})
    with pytest.raises(ValueError, match="verdict"):
        eval_faithfulness.evaluate_faithfulness_v2(bad, None)


def test_v2_scoring_self_agreement_across_two_runs(tmp_path) -> None:
    run1 = _judge_file(tmp_path / "r1.jsonl", {"u1": "supported", "u2": "partial",
                                               "u3": "unsupported", "u4": "supported",
                                               "u5": "parse_error"})
    run2 = _judge_file(tmp_path / "r2.jsonl", {"u1": "supported", "u2": "unsupported",
                                               "u3": "supported", "u4": "supported",
                                               "u5": "supported"})
    result = eval_faithfulness.evaluate_faithfulness_v2(run1, None, second_run=run2)
    agreement = result["self_agreement"]
    assert agreement["n"] == 4  # u5 has no verdict in run 1
    assert agreement["three_way_agreement"] == pytest.approx(2 / 4)  # u1, u4
    assert agreement["binary_agreement"] == pytest.approx(3 / 4)     # u1, u2, u4


def test_run_refuses_to_mix_configurations_in_one_file(tmp_path) -> None:
    out = tmp_path / "judge_run1.jsonl"
    prompt = _prompt_file(tmp_path)
    judge.run([_unit("a:p1u1")], out, "run1", client=FakeClient([_six_dollar_response()]),
              prompt_path=prompt)
    prompt.write_text("A changed judge prompt.")
    with pytest.raises(ValueError, match="prompt_sha256"):
        judge.run([_unit("a:p1u2")], out, "run1", client=FakeClient([]), prompt_path=prompt)


def test_v2_scoring_reports_coverage(tmp_path) -> None:
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", {"a:p1u1": "supported",
                                                          "a:p1u2": "partial"})
    by_count = eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, expected_units=4)
    assert (by_count["expected_units"], by_count["coverage"]) == (4, 0.5)
    units_path = tmp_path / "units.jsonl"
    units.write_units([_unit(f"a:p1u{i}") for i in range(1, 6)], units_path)
    by_path = eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, expected_units=units_path)
    assert by_path["coverage"] == pytest.approx(2 / 5)
    assert "coverage" not in eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl)


def test_v2_scoring_raises_on_judged_unit_not_in_units(tmp_path) -> None:
    judge_jsonl = _judge_file(tmp_path / "judge.jsonl", {"a:p1u1": "supported", "z:p9u9": "partial"})
    units_path = tmp_path / "units.jsonl"
    units.write_units([_unit("a:p1u1")], units_path)
    with pytest.raises(ValueError, match="z:p9u9"):
        eval_faithfulness.evaluate_faithfulness_v2(judge_jsonl, expected_units=units_path)
