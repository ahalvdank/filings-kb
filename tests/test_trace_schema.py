"""schemas/*.json mirror the D2 TypeScript contract field for field."""

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from filings_kb.export import build_trace, validate_against
from tests.test_export import COMMIT, TICKERS, make_run, sig

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"


def schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


# Field names copied from the D2 interfaces (neon-build-request.md section 4).
TRACE_REQUIRED = {"slug", "title", "question", "category", "tickers", "meta", "answer_md",
                  "passages", "claims"}
TRACE_OPTIONAL = {"signals", "audit_note_md"}
PASSAGE_FIELDS = {"cite", "included", "score", "ticker", "company", "form", "fiscal_year", "filed",
                  "section", "accession", "sec_url", "context"}
CLAIM_REQUIRED = {"claim_type", "entity_type", "entity", "ticker", "statement", "raw_extraction"}
CLAIM_OPTIONAL = {"relationship", "cite", "superseded"}  # superseded: additive (Andrew)
SIGNAL_REQUIRED = {"ticker", "section", "prior_fy", "current_fy", "tone_shift", "sentiment_shift"}
MANIFEST_FIELDS = {"pipeline_version", "build_commit", "export_script_url", "run_timestamp",
                   "sample", "full_corpus", "traces"}


def test_schemas_are_valid_draft_2020_12():
    for name in ("trace", "manifest"):
        jsonschema.Draft202012Validator.check_schema(schema(name))


def test_trace_schema_fields_match_the_contract():
    t = schema("trace")
    assert set(t["required"]) == TRACE_REQUIRED
    assert set(t["properties"]) == TRACE_REQUIRED | TRACE_OPTIONAL
    p = t["properties"]["passages"]["items"]
    assert set(p["required"]) == set(p["properties"]) == PASSAGE_FIELDS
    c = t["properties"]["claims"]["items"]
    assert set(c["properties"]) == CLAIM_REQUIRED | CLAIM_OPTIONAL
    assert CLAIM_REQUIRED <= set(c["required"])
    s = t["properties"]["signals"]["items"]
    assert set(s["required"]) == SIGNAL_REQUIRED
    assert set(s["properties"]) == SIGNAL_REQUIRED | {"claim_diff"}
    assert set(t["properties"]["meta"]["required"]) == {
        "run_timestamp", "build_commit", "models", "top_k", "max_passages", "max_rejected",
        "display_cutoff"}
    # I6: every trace records all four models (global constraint), sentiment included.
    assert set(t["properties"]["meta"]["properties"]["models"]["required"]) == {
        "embedding", "extraction", "synthesis", "sentiment"}
    assert t["properties"]["category"]["enum"] == [
        "risk_shift", "contradiction", "capital_structure", "cross_company"]


def test_manifest_schema_fields_match_the_contract():
    m = schema("manifest")
    assert set(m["required"]) == set(m["properties"]) == MANIFEST_FIELDS
    assert set(m["properties"]["traces"]["items"]["required"]) == {
        "slug", "title", "question", "category", "tickers"}


def test_build_trace_validates_against_json_schema(store):
    trace = build_trace(make_run(store), [sig()], COMMIT, TICKERS)
    validate_against(trace, SCHEMAS, "trace", "t")


@pytest.mark.parametrize("mutate, reason", [
    (lambda t: t.update(extra=1), "unknown top-level key"),
    (lambda t: t.pop("answer_md"), "missing answer_md"),
    (lambda t: t["meta"].update(build_commit="abc"), "short commit"),
    (lambda t: t["passages"][0].update(form="8-K"), "bad form"),
    (lambda t: t["passages"][0].pop("context"), "missing context"),
    (lambda t: t["passages"][0]["context"].pop("after"), "missing context.after"),
    (lambda t: t["passages"][0].update(filed="Feb 26"), "bad date"),
    (lambda t: t.update(category="other"), "bad category"),
    (lambda t: t["meta"]["models"].pop("synthesis"), "missing model"),
    (lambda t: t["meta"]["models"].pop("sentiment"), "missing sentiment model"),
    (lambda t: t["meta"].pop("max_passages"), "missing max_passages"),
])
def test_trace_schema_rejects_contract_violations(store, mutate, reason):
    trace = copy.deepcopy(build_trace(make_run(store), [], COMMIT, TICKERS))
    mutate(trace)
    with pytest.raises(ValueError, match="trace schema violation"):
        validate_against(trace, SCHEMAS, "trace", "t")


def test_claim_superseded_flag_is_optional_and_boolean(store):
    trace = copy.deepcopy(build_trace(make_run(store), [], COMMIT, TICKERS))
    claim = {"claim_type": "Risk", "entity_type": "Company", "entity": "E", "ticker": "NVDA",
             "statement": "s", "raw_extraction": {}, "cite": None}
    trace["claims"] = [claim, {**claim, "superseded": True}]
    validate_against(trace, SCHEMAS, "trace", "t")
    trace["claims"] = [{**claim, "superseded": "yes"}]
    with pytest.raises(ValueError):
        validate_against(trace, SCHEMAS, "trace", "t")
