"""Tests for ingest orchestration (fakes only: no network, no model, no LLM)."""

import json
import sqlite3
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from filings_kb.extract import write_envelope
from filings_kb.ingest import CacheMissError, IngestReport, ingest_sample
from filings_kb.sections import Section, section_text_sha256
from filings_kb.store import Store
from filings_kb.types import FilingRef, SampleRow
from tests.conftest import FakeBackend

ACC = "0001045810-25-000023"
ACC_A = "0001045810-25-000099"  # the 10-K/A, filed 2025-03-15
ITEM_TEXT = "Our business is risky. We depend on TSMC. Export rules may change."
MODEL = "claude-sonnet-4-6"


def _payload(extra_entities: list[dict] | None = None) -> dict:
    return {
        "source_file": f"filings/NVDA/{ACC}.json",
        "source_type": "sec_filing",
        "entities": [
            {"type": "Company", "name": "NVIDIA Corporation", "aliases": [],
             "identifiers": {"ticker": "NVDA"}, "attributes": {}},
            {"type": "Risk", "name": "NVDA Export Control Risk", "aliases": [],
             "identifiers": {}, "attributes": {"category": "regulatory"}},
            {"type": "Risk", "name": "NVDA Supplier Risk", "aliases": [],
             "identifiers": {}, "attributes": {}},
            *(extra_entities or []),
        ],
        "relationships": [
            {"source_name": "NVDA 10-K FY2025", "source_type_entity": "Filing",
             "target_name": "NVDA Export Control Risk", "target_type_entity": "Risk",
             "type": "discloses_risk", "attributes": {}},
            {"source_name": "Ghost Co", "source_type_entity": "Company",
             "target_name": "NVDA Supplier Risk", "target_type_entity": "Risk",
             "type": "has_risk", "attributes": {}},
        ],
        "chunks": [],
    }


def _write_env(cache: Path, item: str, text_sha: str, payload: dict,
               parts: int = 1, acc: str = ACC) -> None:
    write_envelope(cache / acc / f"{item}.json", {
        "accession": acc, "item": item, "prompt_version": "v3", "model": MODEL,
        "origin": "imported",
        "parts": [{"text_sha256": text_sha, "extracted_at": "2026-01-01T00:00:00Z",
                   "payload": payload} for _ in range(parts)],
    })


class Env:
    """Temp sample, data dir and fake pipeline pieces."""

    def __init__(self, tmp: Path, items: list[str]) -> None:
        self.data = tmp / "data"
        self.cache = self.data / "committed" / "extractions"
        self.sample = tmp / "sample.yaml"
        self.sample.write_text(
            "rows:\n  - {ticker: NVDA, fiscal_year: 2025, accession: " + f'"{ACC}"' + "}\n"
            "extract_items:\n" + "".join(f"  - {i}\n" for i in items),
            encoding="utf-8",
        )
        self.detected: list[Section] = []
        self.embedded: list[str] = []

    def fetch(self, row: SampleRow, raw_dir: Path) -> FilingRef:
        filed = date(2025, 3, 15) if row.form == "10-K/A" else date(2025, 2, 26)
        return FilingRef(row.ticker, "0001045810", "NVIDIA CORP", row.accession,
                         row.form, filed, row.fiscal_year,
                         date(2025, 1, 26), "nvda.htm", raw_dir / "nvda.htm")

    def detect(self, ref: FilingRef) -> list[Section]:
        return [s for s in self.detected if s.accession == ref.accession]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def run(self, live: bool = False, *, backend: FakeBackend | None = None,
            refresh: bool = False) -> IngestReport:
        from filings_kb.embed import EmbeddingIndex
        index = EmbeddingIndex(self.data / "chroma", "fake", max_tokens=512,
                               embed_fn=self.embed, count_tokens=lambda t: len(t.split()))
        return ingest_sample(self.sample, live, data_dir=self.data, fetch=self.fetch,
                             detect=self.detect, index=index,
                             count_tokens=lambda t: len(t.split()), backend=backend,
                             refresh=refresh, write_committed=refresh)


def _section(item: str, text: str = ITEM_TEXT, acc: str = ACC) -> Section:
    return Section(acc, "NVDA", 2025, item, "title", text)


def _with_amendment(e: Env) -> Env:
    e.sample.write_text(e.sample.read_text().replace(
        "extract_items:",
        f'  - {{ticker: NVDA, fiscal_year: 2025, accession: "{ACC_A}", form: 10-K/A, '
        f'amends: "{ACC}"}}\nextract_items:'))
    e.detected = [_section("Item_1A"), _section("Item_7A"),
                  _section("Item_7A", acc=ACC_A)]
    for item in ("Item_1A", "Item_7A"):
        _write_env(e.cache, item, section_text_sha256(ITEM_TEXT), _payload())
    _write_env(e.cache, "Item_7A", section_text_sha256(ITEM_TEXT), _payload(), acc=ACC_A)
    return e


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path, ["Item_1A", "Item_7A"])


def test_ingest_without_live_fails_on_cache_miss_naming_the_section(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    with pytest.raises(CacheMissError, match=f"{ACC}/Item_1A"):
        env.run(live=False)


def test_ingest_report_counts_match_store(env: Env) -> None:
    env.detected = [_section("Item_1A"), _section("Item_7A"), _section("Item_9")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    _write_env(env.cache, "Item_7A", section_text_sha256(ITEM_TEXT), _payload())
    report = env.run()
    store = Store(env.data / "kb.sqlite")
    with sqlite3.connect(store.path) as conn:
        n_sections = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert report.filings == 1
    assert report.sections == n_sections == 3
    assert report.chunks == n_chunks and n_chunks >= 3
    assert report.claims == len(store.claims_for(ACC)) == 4
    assert report.failed_sections == []
    assert report.cost_usd == Decimal("0")


def test_claims_are_subject_of_the_filer_company_by_identifiers(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    env.run()
    store = Store(env.data / "kb.sqlite")
    claims = store.claims_for(ACC)
    assert {c.subject_name for c in claims} == {"NVIDIA CORP"}
    companies = store.entities_of_type("Company")
    # payload "NVIDIA Corporation" (ticker NVDA) links to the filer, not a duplicate
    assert len(companies) == 1
    filer = companies[0]
    assert filer.identifiers["cik"] == "0001045810"
    assert all(c.item == "Item_1A" for c in claims)


def test_payload_relationships_keep_source_and_unresolved_are_counted(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    report = env.run()
    with sqlite3.connect(env.data / "kb.sqlite") as conn:
        rows = conn.execute(
            "SELECT source_accession, source_item FROM relationships "
            "WHERE type = 'discloses_risk'").fetchall()
    assert rows == [(ACC, "Item_1A")]
    assert report.dropped_relationships == 1  # "Ghost Co" is not in the payload


def test_rerun_is_idempotent_for_rows_and_vectors(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    first = env.run()
    with sqlite3.connect(env.data / "kb.sqlite") as conn:
        before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("filings", "sections", "chunks", "entities", "relationships")}
    embedded_before = len(env.embedded)
    second = env.run()
    with sqlite3.connect(env.data / "kb.sqlite") as conn:
        after = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                 for t in before}
    assert before == after
    assert len(env.embedded) == embedded_before  # nothing re-embedded
    assert (first.sections, first.chunks, first.claims) == (
        second.sections, second.chunks, second.claims)


def test_extract_item_not_detected_is_reported_missing_not_failed(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    report = env.run()
    assert report.failed_sections == []
    assert report.missing_sections == {ACC: ["Item_7A"]}


def test_cached_envelope_for_undetected_section_still_ingests_claims(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    _write_env(env.cache, "Item_7A", "deadbeef", _payload())
    report = env.run()
    assert report.claims == 4
    assert report.imported_not_detected == {ACC: ["Item_7A"]}
    assert report.missing_sections == {ACC: ["Item_7A"]}  # still not detected


def test_fidelity_counts_match_mismatch_and_split(env: Env) -> None:
    env.detected = [_section("Item_1A"), _section("Item_7A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    _write_env(env.cache, "Item_7A", "0" * 64, _payload(), parts=2)
    report = env.run()
    fid = report.fidelity
    assert fid["overall"]["match"] == 1
    assert fid["overall"]["not_comparable_split"] == 1
    assert fid["overall"]["mismatch"] == 0
    assert fid["overall"]["match_rate"] == 1.0
    assert fid["by_section_type"]["Item_1A"]["match"] == 1


def test_fidelity_lists_mismatch_with_both_hashes(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", "f" * 64, _payload())
    report = env.run()
    (m,) = report.fidelity["mismatches"]
    assert m["accession"] == ACC and m["section_type"] == "Item_1A"
    assert m["envelope_text_sha256"] == "f" * 64
    assert m["detected_text_sha256"] == section_text_sha256(ITEM_TEXT)
    assert report.fidelity["overall"]["match_rate"] == 0.0


def test_report_serialises_to_json(env: Env, tmp_path: Path) -> None:
    from filings_kb.ingest import write_report
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    report = env.run()
    out = tmp_path / "r.json"
    write_report(report, out)
    data = json.loads(out.read_text())
    assert data["filings"] == 1 and data["cost_usd"] == "0"
    assert "duration_s" not in data and "fidelity" in data  # durations go to data/run_log.json


def test_skip_item_without_envelope_is_recorded_not_failed(tmp_path: Path) -> None:
    e = Env(tmp_path, ["Item_1A", "Item_4"])
    e.sample.write_text(e.sample.read_text() + "skip_items:\n  - Item_4\n")
    e.detected = [_section("Item_1A"), _section("Item_4")]
    _write_env(e.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    report = e.run()
    assert report.failed_sections == []
    assert report.skipped_by_policy == {ACC: ["Item_4"]}


def test_non_skip_item_without_envelope_still_raises(tmp_path: Path) -> None:
    e = Env(tmp_path, ["Item_1A", "Item_4"])
    e.sample.write_text(e.sample.read_text() + "skip_items:\n  - Item_4\n")
    e.detected = [_section("Item_1A"), _section("Item_4")]
    with pytest.raises(CacheMissError, match="Item_1A"):
        e.run()


def test_crash_mid_section_rolls_back_and_rerun_has_no_duplicates(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    original = Store.add_relationship
    calls = {"n": 0}

    def flaky(self: Store, *args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # after the first claim has been written
            raise RuntimeError("simulated crash")
        original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Store, "add_relationship", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        env.run()
    store = Store(env.data / "kb.sqlite")
    assert store.claims_for(ACC) == []
    assert store.entities_of_type("Risk") == []
    assert [r for r in store.runs() if r.stage == "link"] == []

    monkeypatch.setattr(Store, "add_relationship", original)
    report = env.run()
    assert report.claims == 2
    assert len(store.entities_of_type("Risk")) == 2


def test_amendment_closes_out_original_section_claims_not_others(env: Env) -> None:
    _with_amendment(env)
    report = env.run()
    store = Store(env.data / "kb.sqlite")
    assert [c.item for c in store.claims_for(ACC)] == ["Item_1A", "Item_1A"]
    assert [c.item for c in store.claims_for(ACC_A)] == ["Item_7A", "Item_7A"]
    assert store.closed_claim_count(ACC, ["Item_7A"]) == 2  # closed, never deleted
    assert report.amendments == [{
        "accession": ACC_A, "amends": ACC, "superseded_sections": ["Item_7A"],
        "claims_closed_out": 2}]
    assert report.filings == 2 and report.failed_sections == []


def test_claims_as_of_before_amendment_returns_original(env: Env) -> None:
    from datetime import UTC, datetime
    _with_amendment(env)
    env.run()
    store = Store(env.data / "kb.sqlite")
    before = datetime(2025, 3, 14, 23, 59, tzinfo=UTC)
    at = datetime(2025, 3, 15, tzinfo=UTC)
    assert len(store.claims_as_of(ACC, before)) == 4  # Item_1A and Item_7A
    assert store.claims_as_of(ACC_A, before) == []  # amendment not yet filed
    assert [c.item for c in store.claims_as_of(ACC, at)] == ["Item_1A", "Item_1A"]
    assert len(store.claims_as_of(ACC_A, at)) == 2


def test_amendment_rerun_is_idempotent(env: Env) -> None:
    _with_amendment(env)
    first = env.run()
    with sqlite3.connect(env.data / "kb.sqlite") as conn:
        before = conn.execute(
            "SELECT COUNT(*), COUNT(valid_until) FROM entities").fetchone()
    second = env.run()
    with sqlite3.connect(env.data / "kb.sqlite") as conn:
        after = conn.execute(
            "SELECT COUNT(*), COUNT(valid_until) FROM entities").fetchone()
    assert before == after
    assert second.amendments == first.amendments


def test_amendment_section_without_claims_supersedes_nothing(env: Env) -> None:
    _with_amendment(env)
    empty = _payload()
    empty["entities"] = [e for e in empty["entities"] if e["type"] == "Company"]
    empty["relationships"] = []
    _write_env(env.cache, "Item_7A", section_text_sha256(ITEM_TEXT), empty, acc=ACC_A)
    report = env.run()
    store = Store(env.data / "kb.sqlite")
    assert len(store.claims_for(ACC)) == 4
    assert report.amendments[0]["superseded_sections"] == []


def test_amendment_closes_out_original_linked_after_amendment(env: Env) -> None:
    _with_amendment(env)
    # Run 1: the original has no Item_7A yet, so only the amendment's Item_7A links.
    env.detected = [_section("Item_1A"), _section("Item_7A", acc=ACC_A)]
    (env.cache / ACC / "Item_7A.json").unlink()
    env.run()
    store = Store(env.data / "kb.sqlite")
    assert [c.item for c in store.claims_for(ACC_A)] == ["Item_7A", "Item_7A"]
    # Run 2: the original's Item_7A now links; the amendment's key is already linked.
    env.detected = [_section("Item_1A"), _section("Item_7A"), _section("Item_7A", acc=ACC_A)]
    _write_env(env.cache, "Item_7A", section_text_sha256(ITEM_TEXT), _payload())
    report = env.run()
    assert [c.item for c in store.claims_for(ACC)] == ["Item_1A", "Item_1A"]
    assert report.amendments[0]["claims_closed_out"] == 2


def test_claim_entities_are_not_shared_across_accessions(env: Env) -> None:
    _with_amendment(env)
    env.run()
    with sqlite3.connect(env.data / "kb.sqlite") as conn:
        shared = conn.execute(
            "SELECT src FROM relationships WHERE type = 'subject_of' "
            "GROUP BY src HAVING COUNT(DISTINCT source_accession || source_item) > 1"
        ).fetchall()
    assert shared == []


# --- --refresh (T10 r1 ruling 3): bypass the committed cache, extract live ------------


def _refreshed_payload() -> dict:
    payload = _payload()
    payload["entities"][1]["name"] = "NVDA Refreshed Export Risk"
    payload["relationships"][0]["target_name"] = "NVDA Refreshed Export Risk"
    return payload


def test_refresh_bypasses_the_cache_and_ingests_the_live_extraction(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    backend = FakeBackend(json.dumps(_refreshed_payload()))
    env.run(live=True, backend=backend, refresh=True)
    assert len(backend.calls) == 1 and ITEM_TEXT in backend.calls[0][1]
    envelope = json.loads((env.cache / ACC / "Item_1A.json").read_text())
    assert envelope["origin"] == "live"
    assert envelope["parts"][0]["payload"]["entities"][1]["name"] == "NVDA Refreshed Export Risk"
    names = {c.name for c in Store(env.data / "kb.sqlite").claims_for(ACC)}
    assert "NVDA Refreshed Export Risk" in names and "NVDA Export Control Risk" not in names


def test_without_refresh_a_live_run_uses_the_cache(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    backend = FakeBackend()
    env.run(live=True, backend=backend)
    assert backend.calls == []


def test_refresh_needs_live(env: Env) -> None:
    with pytest.raises(ValueError, match="live"):
        env.run(live=False, refresh=True)


def test_refresh_refuses_a_store_that_already_holds_links(env: Env) -> None:
    # Ingest skips sections it has linked, so a refresh into an existing store would keep
    # the old claims beside the new envelope. It must start from a fresh kb.sqlite.
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    env.run()
    backend = FakeBackend(json.dumps(_refreshed_payload()))
    with pytest.raises(ValueError, match="kb.sqlite"):
        env.run(live=True, backend=backend, refresh=True)
    assert backend.calls == []


def test_refresh_keeps_cached_claims_of_a_section_the_detector_missed(env: Env) -> None:
    # No detected text to send the model, so the imported envelope is used as before.
    env.detected = []
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    backend = FakeBackend()
    report = env.run(live=True, backend=backend, refresh=True)
    assert backend.calls == [] and report.imported_not_detected == {ACC: ["Item_1A"]}
    assert json.loads((env.cache / ACC / "Item_1A.json").read_text())["origin"] == "imported"


# --- lazy extraction backend (T10 r1b): a full cache needs no key ---------------------


class _CountingBackend:
    """Stands in for AnthropicBackend: counts constructions, replies with one payload."""

    built = 0

    def __init__(self) -> None:
        type(self).built += 1
        self.inner = FakeBackend(json.dumps(_payload()))

    def complete(self, system: str, user: str, model: str):
        return self.inner.complete(system, user, model)


@pytest.fixture
def counting(monkeypatch: pytest.MonkeyPatch) -> type[_CountingBackend]:
    import filings_kb.llm
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(filings_kb.llm, "AnthropicBackend", _CountingBackend)
    _CountingBackend.built = 0
    return _CountingBackend


def test_cache_only_ingest_never_constructs_a_backend(env: Env, counting) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    env.run(live=False)
    assert counting.built == 0


def test_live_ingest_with_a_full_cache_never_constructs_a_backend(env: Env, counting) -> None:
    # `make sample` on a clean clone: every section is cached, so no key is needed.
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    env.run(live=True)
    assert counting.built == 0


def test_live_ingest_constructs_the_backend_once_on_the_first_cache_miss(env: Env,
                                                                         counting) -> None:
    env.detected = [_section("Item_1A"), _section("Item_7A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())
    env.run(live=True)  # Item_7A has no envelope
    assert counting.built == 1
    local = env.data / "local" / "extractions"  # not committed: no --write-committed
    assert json.loads((local / ACC / "Item_7A.json").read_text())["origin"] == "live"


# --- final review I5: each claim's extraction chunk content is stored at ingest -----------


def _payload_with_chunks() -> dict:
    payload = _payload()
    payload["chunks"] = [
        {"entity_name": "NVDA Export Control Risk", "entity_type_entity": "Risk",
         "content": "Export rules may change.", "chunk_type": "risk_factor", "attributes": {}},
        {"entity_name": "NVDA Export Control Risk", "entity_type_entity": "Risk",
         "content": "We depend on TSMC.", "chunk_type": "risk_factor", "attributes": {}},
        {"entity_name": "NVIDIA Corporation", "entity_type_entity": "Company",
         "content": "Not a claim.", "chunk_type": "risk_factor", "attributes": {}},
        {"entity_name": None, "entity_type_entity": None,
         "content": "Unattached.", "chunk_type": "risk_factor", "attributes": {}},
    ]
    return payload


def test_ingest_stores_extraction_chunk_content_per_claim(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload_with_chunks())
    env.run()
    assert Store(env.data / "kb.sqlite").claim_evidence(ACC, "Item_1A") == {
        ("Risk", "NVDA Export Control Risk"): ["Export rules may change.", "We depend on TSMC."]}


def test_evidence_is_backfilled_for_a_section_linked_before_it_existed(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload())  # no chunks
    env.run()
    store = Store(env.data / "kb.sqlite")
    ids = sorted(c.entity_id for c in store.claims_for(ACC))
    assert store.claim_evidence(ACC, "Item_1A") == {}
    _write_env(env.cache, "Item_1A", section_text_sha256(ITEM_TEXT), _payload_with_chunks())
    env.run()
    env.run()  # idempotent
    assert sorted(c.entity_id for c in store.claims_for(ACC)) == ids  # never relinked
    assert store.claim_evidence(ACC, "Item_1A") == {
        ("Risk", "NVDA Export Control Risk"): ["Export rules may change.", "We depend on TSMC."]}


# --- final review C2: without write_committed nothing under data/committed is written ----


def test_live_cache_miss_without_write_committed_goes_to_the_local_dir(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    backend = FakeBackend(json.dumps(_payload()))
    env.run(live=True, backend=backend)
    assert not (env.cache / ACC / "Item_1A.json").exists()
    assert (env.data / "local" / "extractions" / ACC / "Item_1A.json").exists()
    env.run(live=True, backend=FakeBackend())  # the local envelope is a cache hit: no call
    env.run(live=False)  # and serves a cache-only run too


def test_live_cache_miss_with_write_committed_goes_to_the_committed_dir(env: Env) -> None:
    env.detected = [_section("Item_1A")]
    from filings_kb.ingest import ingest_sample as real
    from filings_kb.embed import EmbeddingIndex
    index = EmbeddingIndex(env.data / "chroma", "fake", max_tokens=512, embed_fn=env.embed,
                           count_tokens=lambda t: len(t.split()))
    real(env.sample, True, data_dir=env.data, fetch=env.fetch, detect=env.detect, index=index,
         count_tokens=lambda t: len(t.split()), backend=FakeBackend(json.dumps(_payload())),
         write_committed=True)
    assert (env.cache / ACC / "Item_1A.json").exists()
    assert not (env.data / "local").exists()


def test_refresh_needs_write_committed(env: Env) -> None:
    with pytest.raises(ValueError, match="write_committed"):
        ingest_sample(env.sample, True, data_dir=env.data, refresh=True, index=object(),
                      count_tokens=len)
