"""Tests for extraction: validation, retry, cache, cost logging, backends."""

import json
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from filings_kb import llm
from filings_kb.extract import extract_section, load_price, read_envelope
from filings_kb.llm import AnthropicBackend, ClaudeCliBackend, LLMResult
from filings_kb.schema import ExtractionPayload
from filings_kb.sections import Section, section_text_sha256
from filings_kb.store import Store

MODEL = "claude-sonnet-4-6"
ACC = "0001045810-25-000023"

GOOD = {
    "source_file": f"filings/NVDA/{ACC}.json",
    "source_type": "sec_filing",
    "entities": [
        {"type": "Risk", "name": "Export Controls", "aliases": [], "identifiers": {},
         "attributes": {"category": "regulatory"}},
    ],
    "relationships": [],
    "chunks": [
        {"entity_name": "Export Controls", "entity_type_entity": "Risk",
         "content": "Export controls could harm us.", "chunk_type": "risk_factor",
         "valid_from": None, "valid_until": None, "attributes": {}},
    ],
}


class FakeBackend:
    """Scripted backend: returns queued texts, records every call."""

    def __init__(self, texts: list[str], tokens_in: int = 1000, tokens_out: int = 500):
        self.texts = list(texts)
        self.calls: list[tuple[str, str, str]] = []
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        self.calls.append((system, user, model))
        return LLMResult(self.texts.pop(0), self.tokens_in, self.tokens_out)


@pytest.fixture
def section() -> Section:
    return Section(accession=ACC, ticker="NVDA", fiscal_year=2025, item="Item_1A",
                   title="Risk Factors", text="Export controls could harm us. " * 5)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "kb.sqlite")


def test_valid_json_validated_and_cached(section, store, tmp_path):
    backend = FakeBackend([json.dumps(GOOD)])
    result = extract_section(section, backend, MODEL, tmp_path / "cache", store)

    assert isinstance(result, ExtractionPayload)
    assert result.entities[0].name == "Export Controls"
    cached = json.loads((tmp_path / "cache" / ACC / "Item_1A.json").read_text())
    assert cached["prompt_version"] == "v3"
    assert cached["model"] == MODEL
    assert cached["origin"] == "live"
    assert cached["parts"][0]["text_sha256"] == section_text_sha256(section.text)
    assert cached["parts"][0]["payload"] == GOOD
    assert [r.status for r in store.runs()] == ["ok"]


def test_cache_hit_makes_no_backend_call(section, store, tmp_path):
    cache = tmp_path / "cache"
    extract_section(section, FakeBackend([json.dumps(GOOD)]), MODEL, cache, store)

    second = FakeBackend([])
    result = extract_section(section, second, MODEL, cache, store)

    assert second.calls == []
    assert result is not None and result.entities[0].name == "Export Controls"
    assert store.runs()[-1].status == "cache_hit"
    assert store.runs()[-1].cost_usd == Decimal("0")


def test_cache_hit_concatenates_imported_parts_despite_hash_mismatch(
    section, store, tmp_path
):
    other = json.loads(json.dumps(GOOD))
    other["entities"][0]["name"] = "Second Part Risk"
    envelope = {
        "accession": ACC, "item": "Item_1A", "prompt_version": "v3", "model": MODEL,
        "origin": "imported",
        "parts": [
            {"text_sha256": "a" * 64, "extracted_at": "2026-01-01T00:00:00Z",
             "payload": GOOD},
            {"text_sha256": "b" * 64, "extracted_at": "2026-01-02T00:00:00Z",
             "payload": other},
        ],
    }
    path = tmp_path / "cache" / ACC / "Item_1A.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(envelope))

    backend = FakeBackend([])
    result = extract_section(section, backend, MODEL, tmp_path / "cache", store)

    assert backend.calls == []
    assert [e.name for e in result.entities] == ["Export Controls", "Second Part Risk"]
    assert len(result.chunks) == 1  # identical chunk in both parts is deduplicated


def test_cache_for_other_model_is_a_miss(section, store, tmp_path):
    path = tmp_path / "cache" / ACC / "Item_1A.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "accession": ACC, "item": "Item_1A", "prompt_version": "v3",
        "model": "claude-sonnet-5", "origin": "imported",
        "parts": [{"text_sha256": "a" * 64, "extracted_at": "t", "payload": GOOD}],
    }))
    backend = FakeBackend([json.dumps(GOOD)])
    extract_section(section, backend, MODEL, tmp_path / "cache", store)
    assert len(backend.calls) == 1


def test_invalid_json_retried_once_with_error_text_in_prompt(section, store, tmp_path):
    backend = FakeBackend(["not json at all", json.dumps(GOOD)])
    result = extract_section(section, backend, MODEL, tmp_path / "cache", store)

    assert result is not None
    assert len(backend.calls) == 2
    first_user, retry_user = backend.calls[0][1], backend.calls[1][1]
    assert "validation error" not in first_user.lower()
    assert first_user in retry_user
    assert "validation" in retry_user.lower() and "not json at all" in retry_user


def test_schema_violation_error_text_reaches_retry_prompt(section, store, tmp_path):
    bad = json.loads(json.dumps(GOOD))
    bad["entities"][0]["type"] = "NotAType"
    backend = FakeBackend([json.dumps(bad), json.dumps(GOOD)])
    extract_section(section, backend, MODEL, tmp_path / "cache", store)
    assert "NotAType" in backend.calls[1][1]


def test_two_invalid_responses_logs_failed_run_and_returns_none(section, store, tmp_path):
    backend = FakeBackend(["nope", "still nope"])
    result = extract_section(section, backend, MODEL, tmp_path / "cache", store)

    assert result is None
    assert len(backend.calls) == 2
    assert not (tmp_path / "cache" / ACC / "Item_1A.json").exists()
    runs = store.runs()
    assert [r.status for r in runs] == ["failed"]
    assert runs[0].key == f"{ACC}/Item_1A"
    assert runs[0].tokens_in == 2000 and runs[0].tokens_out == 1000


def test_cost_logged_from_token_counts(section, store, tmp_path):
    backend = FakeBackend([json.dumps(GOOD)], tokens_in=1000, tokens_out=500)
    extract_section(section, backend, MODEL, tmp_path / "cache", store)

    run = store.runs()[0]
    # 1000 * $3/M + 500 * $15/M, exact
    assert run.cost_usd == Decimal("0.0105")
    assert (run.tokens_in, run.tokens_out) == (1000, 500)


def test_load_price_is_decimal_and_unknown_model_raises():
    inp, out = load_price(MODEL)
    assert (inp, out) == (Decimal("3"), Decimal("15"))
    with pytest.raises(ValueError, match="no price"):
        load_price("no-such-model")


# --- backends (no real calls) -------------------------------------------------


def test_cli_backend_builds_command_and_parses_json(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        out = json.dumps({"result": "{}", "is_error": False,
                          "usage": {"input_tokens": 7, "output_tokens": 3}})
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    res = ClaudeCliBackend().complete("SYS", "USER BODY", MODEL)

    assert res == LLMResult("{}", 7, 3)
    cmd = seen["cmd"]
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--model") + 1] == MODEL
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert "USER BODY" not in cmd  # large bodies go on stdin, not argv
    assert seen["kw"]["input"] == "USER BODY"


def test_cli_backend_strips_claude_code_context(monkeypatch):
    """Each call must not carry the user's MCP servers, skills, settings or CLAUDE.md."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = json.dumps({"result": "x", "is_error": False})
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    ClaudeCliBackend().complete("SYS", "U", MODEL)
    cmd = seen["cmd"]
    assert "--strict-mcp-config" in cmd
    assert json.loads(cmd[cmd.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert "--disable-slash-commands" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--no-session-persistence" in cmd


def test_cli_backend_missing_usage_reports_zero(monkeypatch):
    out = json.dumps({"result": "x", "is_error": False})
    monkeypatch.setattr(
        llm.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=""),
    )
    assert ClaudeCliBackend().complete("s", "u", MODEL) == LLMResult("x", 0, 0)


def test_cli_backend_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(
        llm.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        ClaudeCliBackend().complete("s", "u", MODEL)


def test_cli_backend_error_result_raises(monkeypatch):
    out = json.dumps({"result": "bad", "is_error": True})
    monkeypatch.setattr(
        llm.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=""),
    )
    with pytest.raises(RuntimeError, match="bad"):
        ClaudeCliBackend().complete("s", "u", MODEL)


def test_anthropic_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()


# --- fix round 1 ----------------------------------------------------------------


def _write_envelope(tmp_path, origin, parts, model=MODEL):
    path = tmp_path / "cache" / ACC / "Item_1A.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "accession": ACC, "item": "Item_1A", "prompt_version": "v3", "model": model,
        "origin": origin, "parts": parts,
    }))
    return path


def test_live_cache_miss_when_section_text_changed(section, store, tmp_path):
    cache = tmp_path / "cache"
    extract_section(section, FakeBackend([json.dumps(GOOD)]), MODEL, cache, store)
    changed = Section(**{**section.__dict__, "text": section.text + " New sentence."})

    backend = FakeBackend([json.dumps(GOOD)])
    extract_section(changed, backend, MODEL, cache, store)

    assert len(backend.calls) == 1


def test_imported_cache_hit_ignores_text_hash(section, store, tmp_path):
    _write_envelope(tmp_path, "imported", [
        {"text_sha256": "f" * 64, "extracted_at": "t", "payload": GOOD}])
    backend = FakeBackend([])
    result = extract_section(section, backend, MODEL, tmp_path / "cache", store)
    assert backend.calls == [] and result is not None


def test_merge_drops_exact_duplicates_across_parts(section, store, tmp_path):
    second = json.loads(json.dumps(GOOD))
    second["entities"].append({"type": "Risk", "name": "Other", "aliases": [],
                               "identifiers": {}, "attributes": {}})
    # same chunk content, different key order, must still count as a duplicate
    second["chunks"][0] = dict(reversed(list(second["chunks"][0].items())))
    _write_envelope(tmp_path, "imported", [
        {"text_sha256": "a" * 64, "extracted_at": "1", "payload": GOOD},
        {"text_sha256": "b" * 64, "extracted_at": "2", "payload": second}])

    result = extract_section(section, FakeBackend([]), MODEL, tmp_path / "cache", store)

    assert [e.name for e in result.entities] == ["Export Controls", "Other"]
    assert len(result.chunks) == 1


def test_read_envelope_preserves_per_part_provenance(tmp_path):
    second = json.loads(json.dumps(GOOD))
    second["source_file"] = "filings/NVDA/part2.json"
    path = _write_envelope(tmp_path, "imported", [
        {"text_sha256": "a" * 64, "extracted_at": "2026-01-01", "payload": GOOD},
        {"text_sha256": "b" * 64, "extracted_at": "2026-01-02", "payload": second}])

    env = read_envelope(path)

    assert (env.origin, env.prompt_version, env.model) == ("imported", "v3", MODEL)
    assert [(p.text_sha256, p.extracted_at, p.source_file) for p in env.parts] == [
        ("a" * 64, "2026-01-01", GOOD["source_file"]),
        ("b" * 64, "2026-01-02", "filings/NVDA/part2.json"),
    ]
    assert isinstance(env.parts[0].payload, ExtractionPayload)


def test_backend_error_logs_failed_run_with_spent_tokens_and_reraises(
    section, store, tmp_path
):
    class Flaky(FakeBackend):
        def complete(self, system, user, model):
            if self.calls:
                self.calls.append((system, user, model))
                raise RuntimeError("cli died")
            return super().complete(system, user, model)

    backend = Flaky(["not json"], tokens_in=1000, tokens_out=500)
    with pytest.raises(RuntimeError, match="cli died"):
        extract_section(section, backend, MODEL, tmp_path / "cache", store)

    run = store.runs()[0]
    assert run.status == "failed"
    assert (run.tokens_in, run.tokens_out) == (1000, 500)
    assert run.cost_usd == Decimal("0.0105")
    assert not (tmp_path / "cache" / ACC / "Item_1A.json").exists()


# --- final review I9: paid-path errors and fenced JSON ------------------------------------


@pytest.mark.parametrize("wrap", [
    "```json\n{}\n```", "Here is the extraction:\n```json\n{}\n```\nDone.", "{} trailing note",
])
def test_fenced_or_wrapped_json_is_accepted_and_cached_as_the_object(section, store, tmp_path,
                                                                     wrap):
    backend = FakeBackend([wrap.replace("{}", json.dumps(GOOD))])
    result = extract_section(section, backend, MODEL, tmp_path / "cache", store)
    assert result is not None and len(backend.calls) == 1
    cached = json.loads((tmp_path / "cache" / ACC / "Item_1A.json").read_text())
    assert cached["parts"][0]["payload"] == GOOD
    assert store.runs()[0].status == "ok"


def _raises_after_first_call(exc: BaseException) -> FakeBackend:
    class Flaky(FakeBackend):
        def complete(self, system, user, model):
            if self.calls:
                self.calls.append((system, user, model))
                raise exc
            return super().complete(system, user, model)

    return Flaky(["not json"], tokens_in=1000, tokens_out=500)


def _assert_failed_run_with_cost(store: Store, tmp_path: Path, detail: str) -> None:
    [run] = store.runs()
    assert run.status == "failed" and (run.tokens_in, run.tokens_out) == (1000, 500)
    assert run.cost_usd == Decimal("0.0105") and detail in run.detail
    assert not (tmp_path / "cache" / ACC / "Item_1A.json").exists()


def test_cli_timeout_logs_failed_run_with_cost_and_reraises(section, store, tmp_path):
    backend = _raises_after_first_call(subprocess.TimeoutExpired(["claude"], 900))
    with pytest.raises(subprocess.TimeoutExpired):
        extract_section(section, backend, MODEL, tmp_path / "cache", store)
    _assert_failed_run_with_cost(store, tmp_path, "TimeoutExpired")


def test_anthropic_api_error_logs_failed_run_with_cost_and_reraises(
        section, store, tmp_path, monkeypatch):
    import sys
    import types

    class APIError(Exception):
        """Stands in for anthropic.APIError (the optional package may not be installed)."""

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(APIError=APIError))
    backend = _raises_after_first_call(APIError("overloaded"))
    with pytest.raises(APIError):
        extract_section(section, backend, MODEL, tmp_path / "cache", store)
    _assert_failed_run_with_cost(store, tmp_path, "overloaded")


def test_unrelated_exception_is_not_logged_as_a_backend_failure(section, store, tmp_path):
    backend = _raises_after_first_call(KeyError("bug"))
    with pytest.raises(KeyError):
        extract_section(section, backend, MODEL, tmp_path / "cache", store)
    assert store.runs() == []
