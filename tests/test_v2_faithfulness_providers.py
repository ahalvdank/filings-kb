"""Tests for the OpenAI judge provider and the manual (pasted-reply) judge path."""

import json
import os
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.v2 import eval_faithfulness
from evals.v2.faithfulness import judge, manual, units

HEADER = {"ticker": "NVDA", "fiscal_year": 2026, "form": "10-K", "item": "Item_1A"}


def _unit(unit_id: str, sentence: str = "Revenue was $5.5 billion.") -> dict:
    return {"unit_id": unit_id, "slug": unit_id.split(":")[0], "sentence": sentence,
            "cites": [1, 2],
            "passages": [{"cite": 1, "chunk_id": "c1", **HEADER, "text": "passage one"},
                         {"cite": 2, "chunk_id": "c2", **HEADER, "item": "Item_7_MDA",
                          "text": "passage two"}]}


def _verdict(verdict: str, span: str | None = None) -> str:
    return json.dumps({"verdict": verdict, "unsupported_span": span})


# --- fake OpenAI client --------------------------------------------------------------


class FakeResponses:
    """Records every create() call and replays scripted responses in order."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeOpenAI:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = FakeResponses(responses)


def _oa_usage(input_tokens: int = 1000, output_tokens: int = 200, cached: int = 0,
              reasoning: int = 150, cache_write: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=cache_write),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning))


def _oa_response(text: str | None, status: str = "completed", refusal: str | None = None,
                 incomplete_reason: str | None = None,
                 usage: SimpleNamespace | None = None) -> SimpleNamespace:
    """Mirrors openai.types.responses.Response: ``output_text`` is a property there."""
    content = []
    if text is not None:
        content.append(SimpleNamespace(type="output_text", text=text, annotations=[]))
    if refusal is not None:
        content.append(SimpleNamespace(type="refusal", refusal=refusal))
    output = [SimpleNamespace(type="reasoning", summary=[]),
              SimpleNamespace(type="message", role="assistant", status=status, content=content)]
    details = (SimpleNamespace(reason=incomplete_reason) if incomplete_reason is not None
               else None)
    return SimpleNamespace(status=status, incomplete_details=details, output=output,
                           output_text=text or "", error=None,
                           usage=usage if usage is not None else _oa_usage())


# --- OpenAI adapter --------------------------------------------------------------------


def test_openai_judge_unit_parses_structured_verdict() -> None:
    client = FakeOpenAI([_oa_response(_verdict("partial", "by $5.5 billion"))])
    result = judge.judge_unit_openai(client, _unit("a:p1u1"), "PROMPT", model="gpt-6.2-sol")

    assert result == {
        "unit_id": "a:p1u1", "verdict": "partial", "unsupported_span": "by $5.5 billion",
        "raw": _verdict("partial", "by $5.5 billion"), "stop_reason": "completed",
        "usage": {"input_tokens": 1000, "output_tokens": 200, "cached_tokens": 0,
                  "cache_write_tokens": 0, "reasoning_tokens": 150},
    }
    (call,) = client.responses.calls
    assert call["model"] == "gpt-6.2-sol"
    assert call["instructions"] == "PROMPT"
    assert call["input"] == judge.user_message(_unit("a:p1u1"))
    assert call["reasoning"] == {"effort": "high"}
    assert call["max_output_tokens"] == judge.MAX_TOKENS  # the budget projection relies on it
    assert call["text"] == {"format": {"type": "json_schema", "name": "faithfulness_verdict",
                                       "schema": judge.VERDICT_SCHEMA, "strict": True}}
    assert "temperature" not in call and "tools" not in call


def test_openai_judge_unit_records_refusal() -> None:
    response = _oa_response(None, refusal="I can't help with that.")
    result = judge.judge_unit_openai(FakeOpenAI([response]), _unit("a:p1u1"), "PROMPT",
                                     model="gpt-6.2-sol")
    assert result["verdict"] == "refusal"
    assert result["unsupported_span"] is None
    assert result["raw"] == "I can't help with that."


def test_openai_refusal_wins_over_parseable_text() -> None:
    response = _oa_response(_verdict("supported"), refusal="No.")
    result = judge.judge_unit_openai(FakeOpenAI([response]), _unit("a:p1u1"), "PROMPT",
                                     model="gpt-6.2-sol")
    assert result["verdict"] == "refusal"


@pytest.mark.parametrize("response, stop_reason", [
    (_oa_response(_verdict("supported"), status="incomplete",
                  incomplete_reason="max_output_tokens"), "incomplete:max_output_tokens"),
    (_oa_response('{"verdict": "supp', status="incomplete",
                  incomplete_reason="content_filter"), "incomplete:content_filter"),
    (_oa_response(_verdict("supported"), status="failed"), "failed"),
    (_oa_response("not json"), "completed"),
    (_oa_response(json.dumps({"verdict": "mostly", "unsupported_span": None})), "completed"),
    (_oa_response(None), "completed"),
], ids=["incomplete-max-tokens", "incomplete-filter", "failed", "junk", "bad-enum", "no-text"])
def test_openai_judge_unit_records_parse_error(response, stop_reason) -> None:
    result = judge.judge_unit_openai(FakeOpenAI([response]), _unit("a:p1u1"), "PROMPT",
                                     model="gpt-6.2-sol")
    assert result["verdict"] == "parse_error"
    assert result["unsupported_span"] is None
    assert result["stop_reason"] == stop_reason
    assert result["usage"]["input_tokens"] == 1000  # a failed call still costs money


def test_openai_call_cost_bills_reasoning_once_and_cached_at_cached_rate() -> None:
    usage = {"input_tokens": 10_000, "output_tokens": 3_000, "cached_tokens": 4_000,
             "cache_write_tokens": 0, "reasoning_tokens": 2_500}
    # (10000 - 4000) * 2.00 + 4000 * 0.10 + 3000 * 10.00 = 12000 + 400 + 30000 = 42400 micro-$.
    # Adding cached on top of input would give 0.0504; adding reasoning on top of output 0.0674.
    assert judge.call_cost(usage, model="gpt-6.2-sol") == Decimal("0.0424")
    assert isinstance(judge.call_cost(usage, model="gpt-6.2-sol"), Decimal)


def test_openai_cost_from_a_fake_response_round_trips_through_usage() -> None:
    response = _oa_response(_verdict("supported"),
                            usage=_oa_usage(input_tokens=10_000, output_tokens=3_000,
                                            cached=4_000, reasoning=2_500))
    result = judge.judge_unit_openai(FakeOpenAI([response]), _unit("a:p1u1"), "PROMPT",
                                     model="gpt-6.2-sol")
    assert judge.call_cost(result["usage"], model="gpt-6.2-sol") == Decimal("0.0424")


def test_openai_missing_usage_stops_the_run(tmp_path) -> None:
    # Zeros would leave spend uncounted and the cap unable to fire; raising stops the run.
    response = _oa_response(_verdict("supported"))
    response.usage = None
    with pytest.raises(RuntimeError, match="usage"):
        judge.judge_unit_openai(FakeOpenAI([response]), _unit("a:p1u1"), "PROMPT",
                                model="gpt-6.2-sol")
    out = tmp_path / "o.jsonl"
    first = _oa_response(_verdict("supported"))
    with pytest.raises(RuntimeError):
        judge.run([_unit("a:p1u1"), _unit("a:p1u2")], out, "g",
                  client=FakeOpenAI([first, response]), prompt_path=_prompt_file(tmp_path),
                  provider="openai", model="gpt-6.2-sol")
    assert [r["unit_id"] for r in _records(out)] == ["a:p1u1"]  # earlier records stay flushed


def test_openai_projection_uses_openai_prices() -> None:
    unit = _unit("a:p1u1")
    est_input = -(-(len("p") + len(judge.user_message(unit))) // judge.CHARS_PER_TOKEN_ESTIMATE)
    expected = (Decimal(est_input) * 2 + Decimal(judge.MAX_TOKENS) * 10) / Decimal(1_000_000)
    assert judge.projected_call_cost("p", unit, "gpt-6.2-sol") == expected


# --- run() with the OpenAI provider ----------------------------------------------------


def _prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "judge_prompt.md"
    path.write_text("Judge prompt.")
    return path


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _five_dollar_response() -> SimpleNamespace:
    # 500,000 output tokens at $10/MTok is exactly $5.
    return _oa_response(_verdict("supported"),
                        usage=_oa_usage(input_tokens=0, output_tokens=500_000, reasoning=499_000))


def test_run_openai_pins_config_stops_on_budget_and_resumes(tmp_path) -> None:
    out = tmp_path / "judge_gpt.jsonl"
    prompt = _prompt_file(tmp_path)
    all_units = [_unit(f"a:p1u{i}") for i in range(1, 6)]
    judge.run(all_units[:1], out, "gpt1", client=FakeOpenAI([_five_dollar_response()]),
              prompt_path=prompt, provider="openai", model="gpt-6.2-sol")

    client = FakeOpenAI([_five_dollar_response() for _ in range(4)])
    summary = judge.run(all_units, out, "gpt1", max_usd=Decimal("12"), client=client,
                        prompt_path=prompt, provider="openai", model="gpt-6.2-sol")

    assert summary["skipped"] == 1 and summary["called"] == 1  # 5 + 5 = 10; 15 > 12 stops
    assert summary["spent_usd"] == Decimal("10")
    assert summary["stopped_on_budget"] is True
    records = _records(out)
    assert [r["unit_id"] for r in records] == ["a:p1u1", "a:p1u2"]
    assert all(r["provider"] == "openai" and r["model"] == "gpt-6.2-sol"
               and r["effort"] == "high" and r["run_label"] == "gpt1"
               and r["prompt_sha256"] == judge.sha256_file(prompt)
               and Decimal(r["cost_usd"]) == Decimal("5") for r in records)


def test_run_anthropic_records_have_no_provider_field(tmp_path) -> None:
    # Anthropic records stay byte-identical to judge runs 1-2: no provider key.
    response = SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text=_verdict("supported"))],
        usage=SimpleNamespace(input_tokens=10, output_tokens=10, cache_read_input_tokens=0,
                              cache_creation_input_tokens=0))
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))
    out = tmp_path / "o.jsonl"
    judge.run([_unit("a:p1u1")], out, "run1", client=client, prompt_path=_prompt_file(tmp_path))
    assert "provider" not in _records(out)[0]


def test_run_refuses_openai_on_a_file_with_old_anthropic_records(tmp_path) -> None:
    out = tmp_path / "o.jsonl"
    prompt = _prompt_file(tmp_path)
    old = {"unit_id": "a:p1u1", "verdict": "supported", "usage": {}, "model": "gpt-6.2-sol",
           "effort": "high", "prompt_sha256": judge.sha256_file(prompt), "run_label": "r"}
    out.write_text(json.dumps(old) + "\n")  # no provider field: counts as anthropic
    with pytest.raises(ValueError, match="provider"):
        judge.run([_unit("a:p1u2")], out, "r", client=FakeOpenAI([]), prompt_path=prompt,
                  provider="openai", model="gpt-6.2-sol")


@pytest.mark.parametrize("provider, model", [("openai", "claude-opus-5-5"),
                                             ("anthropic", "gpt-6.2-sol"),
                                             ("gemini", "gpt-6.2-sol")])
def test_run_rejects_a_model_the_provider_has_no_price_for(tmp_path, provider, model) -> None:
    client = FakeOpenAI([])
    with pytest.raises(ValueError, match=model):
        judge.run([_unit("a:p1u1")], tmp_path / "o.jsonl", "r", client=client,
                  prompt_path=_prompt_file(tmp_path), provider=provider, model=model)
    assert client.responses.calls == []


# --- key loading and CLI ---------------------------------------------------------------


def _units_file(tmp_path: Path) -> Path:
    path = tmp_path / "units.jsonl"
    units.write_units([_unit("a:p1u1")], path)
    return path


def test_main_openai_exits_2_without_openai_key(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")  # must not stand in for OpenAI's
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=file-key\n")
    with pytest.raises(SystemExit) as exc:
        judge.main(["--provider", "openai", "--units", str(_units_file(tmp_path)),
                    "--out", str(tmp_path / "o.jsonl"), "--run-label", "gpt1",
                    "--max-usd", "15", "--env-file", str(env)])
    assert exc.value.code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "o.jsonl").exists()


def test_api_key_openai_reads_only_that_variable_from_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=oa-key\nANTHROPIC_API_KEY=an-key\n")
    assert judge.api_key(env, provider="openai") == "oa-key"
    assert judge.api_key(env) == "an-key"
    assert "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert judge.api_key(env, provider="openai") == "env-key"


def test_main_openai_wires_provider_default_model_and_key(tmp_path, monkeypatch) -> None:
    import openai

    created: list[FakeOpenAI] = []

    def fake_openai(api_key: str) -> FakeOpenAI:
        assert api_key == "test-key"
        created.append(FakeOpenAI([_oa_response(_verdict("supported"))]))
        return created[-1]

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = tmp_path / "o.jsonl"
    judge.main(["--provider", "openai", "--units", str(_units_file(tmp_path)), "--out", str(out),
                "--run-label", "gpt1", "--max-usd", "1"])
    (call,) = created[0].responses.calls
    assert call["model"] == "gpt-6.2-sol"
    (record,) = _records(out)
    assert (record["provider"], record["model"]) == ("openai", "gpt-6.2-sol")


# --- manual judge: export --------------------------------------------------------------


def test_manual_export_renders_prompt_and_exact_passages(tmp_path) -> None:
    rows = [_unit("a:p1u1"), _unit("a:p1u2", sentence="Margins rose.")]
    out_dir = tmp_path / "manual"
    manual.export(rows, out_dir)

    prompt = (out_dir / "PROMPT.md").read_text(encoding="utf-8")
    judge_prompt = judge.PROMPT_PATH.read_text(encoding="utf-8")
    assert judge_prompt in prompt
    assert "2 units" in prompt and "exactly once" in prompt
    assert '{"unit_id": "...", "verdict": "supported|partial|unsupported", ' \
           '"unsupported_span": "..."|null}' in prompt

    body = (out_dir / "units_for_manual_judge.md").read_text(encoding="utf-8")
    expected = "\n\n".join(f"### {u['unit_id']}\n\nSentence: {u['sentence']}\n\n"
                           f"{units.format_passages(u)}" for u in rows) + "\n"
    assert body == expected
    assert "[2] NVDA FY2026 10-K, Item 7\npassage two" in body


# --- manual judge: import --------------------------------------------------------------


def _reply_line(unit_id: str, verdict: str, span: str | None = None) -> str:
    return json.dumps({"unit_id": unit_id, "verdict": verdict, "unsupported_span": span})


def test_manual_import_round_trips_with_fences_and_blank_lines() -> None:
    rows = [_unit("a:p1u1"), _unit("a:p1u2"), _unit("a:p1u3")]
    reply = "\n".join(["```json", _reply_line("a:p1u2", "partial", "$5.5 billion"), "",
                       _reply_line("a:p1u1", "supported"), "```", "", "```",
                       "  " + _reply_line("a:p1u3", "unsupported", "Revenue") + "  ", "```"])
    records = manual.import_reply(rows, reply, run_label="gemini1", model="gemini-x")

    assert [r["unit_id"] for r in records] == ["a:p1u1", "a:p1u2", "a:p1u3"]  # units order
    assert [r["verdict"] for r in records] == ["supported", "partial", "unsupported"]
    assert records[1]["unsupported_span"] == "$5.5 billion"
    first = records[0]
    assert first == {"unit_id": "a:p1u1", "verdict": "supported", "unsupported_span": None,
                     "raw": _reply_line("a:p1u1", "supported"), "stop_reason": None,
                     "usage": None, "model": "gemini-x", "effort": None,
                     "prompt_sha256": judge.sha256_file(judge.PROMPT_PATH),
                     "run_label": "gemini1", "cost_usd": "0", "provider": "manual"}


@pytest.mark.parametrize("line", [
    json.dumps({"unit_id": "a:p1u1", "verdict": "mostly", "unsupported_span": None}),
    json.dumps({"unit_id": "a:p1u1", "verdict": "supported"}),
    json.dumps({"unit_id": "a:p1u1", "verdict": "partial", "unsupported_span": 3}),
    json.dumps({"unit_id": "a:p1u1", "verdict": "supported", "unsupported_span": None, "x": 1}),
], ids=["bad-enum", "missing-span", "span-not-string", "extra-key"])
def test_manual_import_bad_verdict_becomes_parse_error(line) -> None:
    (record,) = manual.import_reply([_unit("a:p1u1")], line, run_label="g", model="m")
    assert record["verdict"] == "parse_error"
    assert record["unsupported_span"] is None
    assert record["raw"] == line


def test_manual_import_raises_listing_missing_duplicate_and_unknown_ids() -> None:
    rows = [_unit("a:p1u1"), _unit("a:p1u2"), _unit("a:p1u3")]
    reply = "\n".join([_reply_line("a:p1u1", "supported"), _reply_line("a:p1u1", "partial", "x"),
                       _reply_line("z:p9u9", "supported")])
    with pytest.raises(ValueError) as exc:
        manual.import_reply(rows, reply, run_label="g", model="m")
    message = str(exc.value)
    assert "missing" in message and "a:p1u2" in message and "a:p1u3" in message
    assert "duplicate" in message and "a:p1u1" in message
    assert "unknown" in message and "z:p9u9" in message


@pytest.mark.parametrize("reply", ["Here are the verdicts:\n" + _reply_line("a:p1u1", "supported"),
                                   '{"verdict": "supported", "unsupported_span": null}',
                                   '["a:p1u1", "supported"]'],
                         ids=["prose", "no-unit-id", "not-an-object"])
def test_manual_import_raises_on_a_line_it_cannot_attribute(reply) -> None:
    with pytest.raises(ValueError, match="line 1"):
        manual.import_reply([_unit("a:p1u1")], reply, run_label="g", model="m")


def test_manual_cli_import_scores_through_evaluate_faithfulness_v2(tmp_path) -> None:
    rows = [_unit("a:p1u1"), _unit("a:p1u2"), _unit("a:p1u3"), _unit("a:p1u4")]
    units_path = tmp_path / "units.jsonl"
    units.write_units(rows, units_path)
    reply = tmp_path / "reply.txt"
    reply.write_text("\n".join([
        "```", _reply_line("a:p1u1", "supported"), _reply_line("a:p1u2", "partial", "x"),
        _reply_line("a:p1u3", "supported"),
        json.dumps({"unit_id": "a:p1u4", "verdict": "maybe", "unsupported_span": None}), "```"]))
    out = tmp_path / "judge_gemini1.jsonl"
    manual.main(["import", "--units", str(units_path), "--reply", str(reply), "--out", str(out),
                 "--run-label", "gemini1", "--model", "gemini-x"])

    api_run = tmp_path / "judge_run1.jsonl"
    api_run.write_text("".join(json.dumps({"unit_id": u, "verdict": v}) + "\n" for u, v in
                               {"a:p1u1": "supported", "a:p1u2": "supported",
                                "a:p1u3": "supported", "a:p1u4": "supported"}.items()))
    alone = eval_faithfulness.evaluate_faithfulness_v2(out, expected_units=units_path)
    assert (alone["units"], alone["supported"], alone["partial"], alone["parse_errors"]) == \
        (4, 2, 1, 1)
    assert alone["coverage"] == 1.0
    paired = eval_faithfulness.evaluate_faithfulness_v2(api_run, second_run=out)
    assert paired["self_agreement"]["n"] == 3  # a:p1u4 is a parse error in the manual run
    assert paired["self_agreement"]["three_way_agreement"] == pytest.approx(2 / 3)


def test_manual_cli_import_refuses_to_overwrite(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    units.write_units([_unit("a:p1u1")], units_path)
    reply = tmp_path / "reply.txt"
    reply.write_text(_reply_line("a:p1u1", "supported"))
    out = tmp_path / "judge_g.jsonl"
    out.write_text("existing\n")
    with pytest.raises(SystemExit) as exc:
        manual.main(["import", "--units", str(units_path), "--reply", str(reply),
                     "--out", str(out), "--run-label", "g", "--model", "m"])
    assert exc.value.code == 2
    assert out.read_text() == "existing\n"


def test_manual_cli_export_writes_both_files(tmp_path) -> None:
    units_path = tmp_path / "units.jsonl"
    units.write_units([_unit("a:p1u1")], units_path)
    out_dir = tmp_path / "m"
    manual.main(["export", "--units", str(units_path), "--out-dir", str(out_dir)])
    assert sorted(p.name for p in out_dir.iterdir()) == ["PROMPT.md", "units_for_manual_judge.md"]
