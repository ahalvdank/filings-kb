"""Tests for the bitemporal SQLite store."""

import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from filings_kb.chunking import Chunk
from filings_kb.sections import Section
from filings_kb.store import CLAIM_TYPES, ClaimRow, Store
from filings_kb.types import FilingRef

ACC = "0001045810-25-000023"


def _filing() -> FilingRef:
    return FilingRef(
        ticker="NVDA", cik="1045810", company="NVIDIA CORP", accession=ACC,
        form="10-K", filed=date(2025, 2, 26), fiscal_year=2025,
        fiscal_year_end=date(2025, 1, 26), primary_doc="nvda-20250126.htm",
        html_path=Path("data/raw/nvda.htm"),
    )


def _section() -> Section:
    return Section(accession=ACC, ticker="NVDA", fiscal_year=2025,
                   item="Item_1A", title="Risk Factors", text="Risk text. More.")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "kb.sqlite")
    s.put_filing(_filing())
    s.put_section(_section())
    return s


def _rows(store: Store, sql: str, args: tuple = ()) -> list[tuple]:
    with sqlite3.connect(store.path) as conn:
        return conn.execute(sql, args).fetchall()


def test_claim_types_are_the_nine_from_the_spec() -> None:
    assert CLAIM_TYPES == frozenset({
        "Risk", "MaterialEvent", "CapitalStructureEvent", "LegalProceeding",
        "CapitalReturn", "GuidanceUpdate", "RevenueSegment", "ESGCommitment",
        "InsiderTransaction",
    })


def test_schema_creation_is_idempotent(tmp_path: Path) -> None:
    Store(tmp_path / "kb.sqlite")
    Store(tmp_path / "kb.sqlite")  # must not raise


def test_put_filing_and_section_are_idempotent(store: Store) -> None:
    store.put_filing(_filing())
    store.put_section(_section())
    assert len(_rows(store, "SELECT * FROM filings")) == 1
    assert len(_rows(store, "SELECT * FROM sections")) == 1


def test_put_chunks_roundtrip_and_idempotent(store: Store) -> None:
    c = Chunk("c1", ACC, "NVDA", 2025, "Item_1A", 0, 10, "Risk text.")
    store.put_chunks([c])
    store.put_chunks([c])
    rows = _rows(store, "SELECT chunk_id, start_offset, end_offset, text FROM chunks")
    assert rows == [("c1", 0, 10, "Risk text.")]


def test_put_chunks_requires_existing_section(store: Store) -> None:
    c = Chunk("c1", ACC, "NVDA", 2025, "Item_9", 0, 3, "Abc")
    with pytest.raises(sqlite3.IntegrityError):
        store.put_chunks([c])


def test_put_entity_returns_uuid_and_stores_json(store: Store) -> None:
    eid = store.put_entity("Risk", "Export controls", {"severity": "high"}, {"k": "v"})
    row = _rows(store, "SELECT id, type, name, attributes, identifiers, valid_until "
                       "FROM entities")[0]
    assert row[0] == eid and len(eid) == 36
    assert row[1:3] == ("Risk", "Export controls")
    assert '"severity"' in row[3] and '"k"' in row[4]
    assert row[5] is None


def test_close_out_sets_valid_until_and_row_remains(store: Store) -> None:
    eid = store.put_entity("Risk", "R", {}, {})
    at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    store.close_out(eid, at)
    rows = _rows(store, "SELECT valid_until FROM entities WHERE id=?", (eid,))
    assert rows == [("2026-01-02T03:04:05+00:00",)]


def test_close_out_unknown_entity_raises(store: Store) -> None:
    with pytest.raises(KeyError):
        store.close_out("nope", datetime(2026, 1, 1, tzinfo=UTC))


def test_close_out_rejects_naive_datetime(store: Store) -> None:
    eid = store.put_entity("Risk", "R", {}, {})
    with pytest.raises(ValueError):
        store.close_out(eid, datetime(2026, 1, 1))


def _claim(store: Store, ctype: str, name: str, subject: str) -> str:
    cid = store.put_entity(ctype, name, {"a": 1}, {})
    store.add_relationship(cid, subject, "subject_of", ACC, "Item_1A", {})
    return cid


def test_claims_for_returns_only_claim_types_with_subject(store: Store) -> None:
    company = store.put_entity("Company", "NVIDIA", {}, {"ticker": "NVDA"})
    risk = _claim(store, "Risk", "Export controls", company)
    # A non-claim entity linked the same way must not appear.
    other = store.put_entity("Person", "Jensen", {}, {})
    store.add_relationship(other, company, "subject_of", ACC, "Item_1A", {})
    # A claim from another filing must not appear.
    other_claim = store.put_entity("Risk", "Elsewhere", {}, {})
    store.add_relationship(other_claim, company, "subject_of", "0000000000-00-000000",
                           "Item_1A", {})
    rows = store.claims_for(ACC)
    assert rows == [ClaimRow(
        entity_id=risk, claim_type="Risk", name="Export controls",
        attributes={"a": 1}, subject_id=company, subject_type="Company",
        subject_name="NVIDIA", accession=ACC, item="Item_1A",
    )]


def test_claims_for_excludes_closed_out_claims_and_relationships(store: Store) -> None:
    company = store.put_entity("Company", "NVIDIA", {}, {})
    closed = _claim(store, "Risk", "Old", company)
    kept = _claim(store, "MaterialEvent", "New", company)
    store.close_out(closed, datetime(2026, 1, 1, tzinfo=UTC))
    assert [r.entity_id for r in store.claims_for(ACC)] == [kept]


def test_claims_for_excludes_closed_relationship(store: Store) -> None:
    company = store.put_entity("Company", "NVIDIA", {}, {})
    cid = _claim(store, "Risk", "R", company)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE relationships SET valid_until='2026-01-01T00:00:00+00:00' "
                     "WHERE src=?", (cid,))
    assert store.claims_for(ACC) == []


def test_add_relationship_requires_existing_entities(store: Store) -> None:
    cid = store.put_entity("Risk", "R", {}, {})
    with pytest.raises(sqlite3.IntegrityError):
        store.add_relationship(cid, "missing", "subject_of", ACC, "Item_1A", {})


def test_log_run_stores_cost_as_exact_decimal(store: Store) -> None:
    store.log_run("extract", "k1", "ok", 10, 20, Decimal("0.300"), "d")
    cost_type, cost = _rows(store, "SELECT typeof(cost_usd), cost_usd FROM runs")[0]
    assert cost_type == "text" and cost == "0.300"
    assert store.runs()[0].cost_usd == Decimal("0.300")
    assert isinstance(store.runs()[0].cost_usd, Decimal)


def test_log_run_rejects_float_cost(store: Store) -> None:
    with pytest.raises(TypeError):
        store.log_run("extract", "k1", "ok", 1, 2, 0.3, "d")  # type: ignore[arg-type]


def test_duplicate_relationship_is_not_inserted_twice(store: Store) -> None:
    a = store.put_entity("Risk", "r", {}, {})
    b = store.put_entity("Company", "c", {}, {})
    for _ in range(3):
        store.add_relationship(a, b, "subject_of", ACC, "Item_1A", {})
    store.add_relationship(a, b, "subject_of", ACC, "Item_1A", {"k": 1})  # differs
    assert len(_rows(store, "SELECT * FROM relationships")) == 2


def test_transaction_rolls_back_all_writes_on_error(store: Store) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction():
            store.put_entity("Risk", "r", {}, {})
            store.log_run("link", "k", "ok", 0, 0, Decimal("0"), "")
            raise RuntimeError("boom")
    assert _rows(store, "SELECT * FROM entities") == []
    assert store.runs() == []


def test_transaction_commits_together_and_reads_own_writes(store: Store) -> None:
    with store.transaction():
        store.put_entity("Risk", "r", {}, {})
        assert len(store.entities_of_type("Risk")) == 1
    assert len(store.entities_of_type("Risk")) == 1


def test_transactions_do_not_nest(store: Store) -> None:
    with store.transaction():
        with pytest.raises(RuntimeError, match="nest"):
            with store.transaction():
                pass


def _open_claim(store: Store, accession: str, item: str, name: str) -> str:
    subject = store.put_entity("Company", "NVIDIA", {}, {"ticker": "NVDA"})
    cid = store.put_entity("Risk", name, {}, {})
    store.add_relationship(cid, subject, "subject_of", accession, item, {})
    return cid


def test_close_out_claims_closes_only_that_accession_and_item(store: Store) -> None:
    _open_claim(store, ACC, "Item_1A", "a")
    _open_claim(store, ACC, "Item_7A", "b")
    at = datetime(2025, 3, 1, tzinfo=UTC)
    assert store.close_out_claims(ACC, "Item_7A", at) == 1
    assert store.close_out_claims(ACC, "Item_7A", at) == 0  # already closed
    assert [c.name for c in store.claims_for(ACC)] == ["a"]
    assert store.closed_claim_count(ACC, ["Item_7A"]) == 1


def test_close_out_claims_requires_timezone_aware_time(store: Store) -> None:
    with pytest.raises(ValueError, match="timezone"):
        store.close_out_claims(ACC, "Item_7A", datetime(2025, 3, 1))


def test_claims_as_of_respects_filing_date_and_close_out(store: Store) -> None:
    _open_claim(store, ACC, "Item_7A", "old")  # NVDA filing filed 2025-02-26
    store.close_out_claims(ACC, "Item_7A", datetime(2025, 4, 1, tzinfo=UTC))
    before_filing = datetime(2025, 2, 25, tzinfo=UTC)
    known = datetime(2025, 3, 15, tzinfo=UTC)
    at_close = datetime(2025, 4, 1, tzinfo=UTC)
    assert store.claims_as_of(ACC, before_filing) == []
    assert [c.name for c in store.claims_as_of(ACC, known)] == ["old"]
    assert store.claims_as_of(ACC, at_close) == []
    assert store.claims_for(ACC) == []


def test_claims_as_of_requires_timezone_aware_time(store: Store) -> None:
    with pytest.raises(ValueError, match="timezone"):
        store.claims_as_of(ACC, datetime(2025, 3, 15))


def test_claims_as_of_on_original_filed_date_includes_its_claims(store: Store) -> None:
    _open_claim(store, ACC, "Item_1A", "x")  # filed 2025-02-26
    assert [c.name for c in store.claims_as_of(ACC, datetime(2025, 2, 26, tzinfo=UTC))] == ["x"]
    assert store.claims_as_of(ACC, datetime(2025, 2, 25, 23, 59, 59, tzinfo=UTC)) == []


def _amendment_store(store: Store) -> Store:
    """Add an amended filing (10-K/A) carrying only Item_7_MDA for the same fiscal year."""
    amend = "0001045810-25-000099"
    store.put_filing(FilingRef(
        ticker="NVDA", cik="1045810", company="NVIDIA CORP", accession=amend,
        form="10-K/A", filed=date(2025, 6, 1), fiscal_year=2025,
        fiscal_year_end=date(2025, 1, 26), primary_doc="a.htm",
        html_path=Path("data/raw/a.htm"),
    ))
    for acc, item in ((ACC, "Item_7_MDA"), (amend, "Item_7_MDA")):
        store.put_section(Section(accession=acc, ticker="NVDA", fiscal_year=2025,
                                  item=item, title="MD&A", text="mdna"))
    return store


def test_section_accession_prefers_amendment_that_carries_the_item(store: Store) -> None:
    _amendment_store(store)
    assert store.section_accession("NVDA", 2025, "Item_7_MDA") == "0001045810-25-000099"
    assert store.section_accession("NVDA", 2025, "Item_1A") == ACC


def test_section_accession_none_when_absent(store: Store) -> None:
    assert store.section_accession("NVDA", 2025, "Item_7_MDA") is None
    assert store.section_accession("XOM", 2025, "Item_1A") is None


def test_chunk_texts_in_offset_order(store: Store) -> None:
    store.put_chunks([
        Chunk("c2", ACC, "NVDA", 2025, "Item_1A", 10, 20, "second"),
        Chunk("c1", ACC, "NVDA", 2025, "Item_1A", 0, 10, "first"),
    ])
    assert store.chunk_texts(ACC, "Item_1A") == ["first", "second"]
    assert store.chunk_texts(ACC, "Item_7_MDA") == []


def test_fiscal_years_by_ticker_sorted(store: Store) -> None:
    _amendment_store(store)
    assert store.fiscal_years() == {"NVDA": [2025]}


# --- read helpers for the trace export ----------------------------------------


def test_filing_roundtrips_and_filings_for_lists_original_before_amendment(store: Store) -> None:
    assert store.filing(ACC) == _filing()
    assert store.filing("nope") is None
    amendment = FilingRef(
        ticker="NVDA", cik="1045810", company="NVIDIA CORP", accession="0001045810-25-000099",
        form="10-K/A", filed=date(2025, 3, 1), fiscal_year=2025,
        fiscal_year_end=date(2025, 1, 26), primary_doc="a.htm", html_path=Path("a.htm"),
    )
    store.put_filing(amendment)
    assert [f.accession for f in store.filings_for("NVDA", 2025)] == [ACC, amendment.accession]
    assert store.filings_for("NVDA", 2030) == []


def test_section_and_chunks_by_id(store: Store) -> None:
    assert store.section(ACC, "Item_1A") == _section()
    assert store.section(ACC, "Item_9") is None
    chunk = Chunk(f"{ACC}:Item_1A:0000", ACC, "NVDA", 2025, "Item_1A", 0, 11, "Risk text.")
    store.put_chunks([chunk])
    assert store.chunks_by_id([chunk.chunk_id, "missing"]) == {chunk.chunk_id: chunk}
    assert store.chunks_by_id([]) == {}


def test_closed_claims_for_returns_only_closed_claims_in_the_requested_items(
    store: Store,
) -> None:
    company = store.put_entity("Company", "NVIDIA", {}, {})
    old = _claim(store, "Risk", "Old", company)
    _claim(store, "Risk", "Still open", company)
    store.close_out(old, datetime(2026, 1, 1, tzinfo=UTC))
    assert [r.name for r in store.closed_claims_for(ACC, ["Item_1A"])] == ["Old"]
    assert store.closed_claims_for(ACC, ["Item_7_MDA"]) == []
    assert store.closed_claims_for(ACC, []) == []


def test_accessions_lists_every_filing_sorted(store: Store) -> None:
    other = FilingRef(
        ticker="AMD", cik="2488", company="AMD", accession="0000002488-26-000018",
        form="10-K", filed=date(2026, 2, 4), fiscal_year=2025,
        fiscal_year_end=date(2025, 12, 27), primary_doc="a.htm", html_path=Path("a.htm"),
    )
    store.put_filing(other)
    assert store.accessions() == [other.accession, ACC]


# --- final review minor 2: close_out never moves an existing valid_until ---------------


def test_close_out_twice_keeps_the_first_valid_until(tmp_path):
    import sqlite3
    from datetime import UTC, datetime

    store = Store(tmp_path / "kb.sqlite")
    eid = store.put_entity("Risk", "Export controls", {}, {})
    store.close_out(eid, datetime(2025, 1, 1, tzinfo=UTC))
    store.close_out(eid, datetime(2026, 6, 1, tzinfo=UTC))  # a no-op, not an error
    with sqlite3.connect(store.path) as conn:
        [(until,)] = conn.execute("SELECT valid_until FROM entities WHERE id=?", (eid,))
    assert until == "2025-01-01T00:00:00+00:00"


def test_claim_evidence_roundtrips_in_order_and_ignores_duplicates(tmp_path):
    store = Store(tmp_path / "kb.sqlite")
    rows = [("Risk", "A", "first text"), ("Risk", "A", "second text"), ("Risk", "B", "b text")]
    store.put_claim_evidence("acc", "Item_1A", rows)
    store.put_claim_evidence("acc", "Item_1A", rows)
    assert store.claim_evidence("acc", "Item_1A") == {
        ("Risk", "A"): ["first text", "second text"], ("Risk", "B"): ["b text"]}
    assert store.claim_evidence("acc", "Item_7A") == {}
