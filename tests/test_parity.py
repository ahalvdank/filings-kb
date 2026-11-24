"""Tests for evals/parity.py: fidelity, claim counts against the private KB, gaps."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals import parity
from filings_kb.link import load_link_thresholds
from filings_kb.store import Store
from filings_kb.types import SampleRow

ACC = "0001045810-25-000023"
ROWS = [SampleRow("NVDA", 2025, ACC)]
THRESHOLD = load_link_thresholds()[0]  # link_auto, 90
PRIVATE_SCHEMA = """
CREATE TABLE entities (id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
    aliases JSON NOT NULL DEFAULT '[]', identifiers JSON NOT NULL DEFAULT '{}', attributes JSON NOT NULL DEFAULT '{}');
CREATE TABLE relationships (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
    type TEXT NOT NULL, valid_from TEXT, valid_until TEXT, source_document TEXT NOT NULL);
"""
DOC = "filings/NVDA/2025-02_10-K_000104581025000023.md"
MANIFEST = "filings/NVDA/2025-02_10-K.md"  # accession-less duplicate


def _private(path: Path, claims: list[tuple],
             entity_valid_until: bool = False) -> Path:
    """Private-schema DB. ``claims``: ``(id, type, source_document, rel_valid_until[, name[, aliases]])``."""
    conn = sqlite3.connect(path)
    conn.executescript(PRIVATE_SCHEMA)
    if entity_valid_until:
        conn.execute("ALTER TABLE entities ADD COLUMN valid_until TEXT")
    conn.execute("INSERT INTO entities (id, type, name) VALUES ('filing', 'Filing', 'NVDA 10-K')")
    for i, (cid, ctype, doc, until, *name) in enumerate(claims):
        aliases = json.dumps(name[1]) if len(name) > 1 else "[]"
        conn.execute("INSERT OR IGNORE INTO entities (id, type, name, aliases) VALUES (?,?,?,?)",
                     (cid, ctype, name[0] if name else f"claim {cid}", aliases))
        conn.execute("INSERT INTO relationships VALUES (?,?,?,?,?,?,?)",
                     (f"r{i}", "filing", cid, "discloses_risk", None, until, doc))
    conn.commit()
    conn.close()
    return path


def test_parity_opens_private_db_read_only(tmp_path: Path) -> None:
    db = _private(tmp_path / "kg.db", [("a", "Risk", DOC, None)])
    before = db.read_bytes()
    conn = parity.open_readonly(db)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM entities")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("CREATE TABLE x (a)")
    conn.close()
    assert db.read_bytes() == before


def test_parity_refuses_a_missing_private_db(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parity.open_readonly(tmp_path / "nope.db")


def _public(path: Path) -> Store:
    store = Store(path)
    company = store.put_entity("Company", "NVDA", {}, {"ticker": "NVDA"})
    for item, ctype in (("Item_1A", "Risk"), ("Item_1A", "Risk"), ("Item_7_MDA", "Risk"),
                        ("Item_7_MDA", "CapitalReturn")):
        cid = store.put_entity(ctype, "claim", {}, {})
        store.add_relationship(cid, company, "subject_of", ACC, item, {})
    closed = store.put_entity("Risk", "old claim", {}, {})
    store.add_relationship(closed, company, "subject_of", ACC, "Item_1A", {})
    store.close_out(closed, datetime(2026, 1, 1, tzinfo=UTC))
    return store


def _entity(kind: str, name: str) -> dict:
    return {"type": kind, "name": name, "attributes": {}, "identifiers": {}}


def _envelope(folder: Path, item: str, *entities: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{item}.json").write_text(json.dumps({
        "accession": ACC, "item": item, "model": "m", "origin": "imported",
        "prompt_version": "v3", "parts": [{
            "text_sha256": "a" * 64, "extracted_at": "2026-06-17T00:00:00Z",
            "payload": {"source_file": "x", "source_type": "sec_filing",
                        "entities": list(entities), "relationships": [], "chunks": []}}]}))


def _matching_cache(root: Path) -> Path:
    """Cache holding what ``_public`` ingested: 4 Risk (one closed out later) + 1 CapitalReturn."""
    _envelope(root / ACC, "Item_1A", _entity("Risk", "a"), _entity("Risk", "b"),
              _entity("Risk", "old"))
    _envelope(root / ACC, "Item_7_MDA", _entity("Risk", "d"), _entity("CapitalReturn", "e"))
    return root


def test_public_counts_are_open_claims_per_type(tmp_path: Path) -> None:
    _public(tmp_path / "kb.sqlite")
    counts = parity.public_claim_counts(parity.open_readonly(tmp_path / "kb.sqlite"), ROWS)
    assert counts == {"Risk": 3, "CapitalReturn": 1}


def test_fidelity_summary_quotes_coverage_alongside_the_rate() -> None:
    report = {"fidelity": {"overall": {"comparable": 682, "match": 681, "mismatch": 1,
                                       "not_comparable_split": 103, "not_detected": 39,
                                       "match_rate": 0.9985}}}
    f = parity.fidelity_summary(report)
    assert (f["match"], f["comparable"], f["sections"]) == (681, 682, 824)
    assert f["match_rate_pct"] == 99.9 and f["coverage_pct"] == 82.8


def test_fidelity_summary_raises_when_nothing_is_comparable() -> None:
    with pytest.raises(ValueError, match="comparable"):
        parity.fidelity_summary({"fidelity": {"overall": {
            "comparable": 0, "match": 0, "mismatch": 0, "not_comparable_split": 3,
            "not_detected": 0}}})


def test_cache_counts_merge_parts_like_ingest(tmp_path: Path) -> None:
    risk = {"type": "Risk", "name": "NVDA export risk", "attributes": {}, "identifiers": {}}
    other = {"type": "Risk", "name": "NVDA supply risk", "attributes": {}, "identifiers": {}}
    person = {"type": "Person", "name": "Jensen", "attributes": {}, "identifiers": {}}

    def part(*entities: dict) -> dict:
        return {"text_sha256": "a" * 64, "extracted_at": "2026-06-17T00:00:00Z",
                "payload": {"source_file": "x", "source_type": "sec_filing",
                            "entities": list(entities), "relationships": [], "chunks": []}}

    folder = tmp_path / ACC
    folder.mkdir()
    (folder / "Item_1A.json").write_text(json.dumps({
        "accession": ACC, "item": "Item_1A", "model": "m", "origin": "imported",
        "prompt_version": "v3",
        "parts": [part(risk, person), part(risk, other)]}))  # `risk` repeats across parts
    assert parity.cache_claim_counts(tmp_path, ROWS) == {"Risk": 2}


def test_private_unlinked_counts_only_entities_no_relationship_touches(tmp_path: Path) -> None:
    db = _private(tmp_path / "kg.db", [("a", "Risk", DOC, None)])
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO entities (id, type, name) VALUES ('orphan', 'InsiderTransaction', 'x')")
    assert parity.private_unlinked_counts(parity.open_readonly(db)) == {"InsiderTransaction": 1}


def test_main_exits_1_when_the_public_store_disagrees_with_the_cache(tmp_path: Path) -> None:
    private = _private(tmp_path / "kg.db", [("a", "Risk", DOC, None)])
    _public(tmp_path / "kb.sqlite")  # 4 open claims + 1 closed; cache dir below holds none
    sample = tmp_path / "sample.yaml"
    sample.write_text(f"rows:\n  - {{ticker: NVDA, fiscal_year: 2025, accession: \"{ACC}\"}}\n"
                      "extract_items: [Item_1A]\n")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"fidelity": {"overall": {
        "comparable": 10, "match": 10, "mismatch": 0, "not_comparable_split": 0,
        "not_detected": 0}}}))
    (tmp_path / "cache").mkdir()
    args = ["--private-kb", str(private), "--public-kb", str(tmp_path / "kb.sqlite"),
            "--sample", str(sample), "--report", str(report), "--out", str(tmp_path / "p.json"),
            "--extractions", str(tmp_path / "cache")]
    assert parity.main(args) == 1


# --- private claims of a filing: accession document plus manifest -------------------


def test_private_claims_are_open_rows_of_the_accession_doc_and_its_manifest(tmp_path) -> None:
    db = _private(tmp_path / "kg.db", [
        ("a", "Risk", DOC, None),                      # counted
        ("a", "Risk", DOC, None),                      # same claim linked twice: still one
        ("b", "Risk", DOC, "2026-01-01"),              # closed-out re-ingest row: not counted
        ("c", "Risk", MANIFEST, None),                 # manifest-only claim: counted
        ("d", "CapitalReturn", DOC, None),
        ("e", "Person", DOC, None),                    # not a claim type
        ("f", "Risk", "filings/AMD/2025-02_10-K_000104581025000023.md", None),  # other ticker
        ("g", "Risk", "filings/NVDA/2024-02_10-K.md", None),  # another filing's manifest
    ])
    conn = parity.open_readonly(db)
    claims = parity.private_claims(conn, ROWS[0])
    assert set(claims["Risk"]) == {"a", "c"} and set(claims["CapitalReturn"]) == {"d"}
    assert "Person" not in claims
    assert set(parity.private_claims(conn, ROWS[0], include_manifests=False)["Risk"]) == {"a"}


def test_amendment_manifest_keeps_the_slash_in_its_form() -> None:
    row = SampleRow("AMD", 2025, "0000002488-26-000021", form="10-K/A")
    docs = ["filings/AMD/2026-02_10-K-A_000000248826000021.md"]
    assert parity.manifest_paths(docs, row) == ["filings/AMD/2026-02_10-K/A.md"]
    assert parity.manifest_paths([DOC], ROWS[0]) == [MANIFEST]


def test_private_entities_filtered_on_valid_until_when_the_column_exists(tmp_path) -> None:
    db = _private(tmp_path / "kg.db", [("a", "Risk", DOC, None), ("b", "Risk", DOC, None)],
                  entity_valid_until=True)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE entities SET valid_until='2026-01-01' WHERE id='b'")
    assert set(parity.private_claims(parity.open_readonly(db), ROWS[0])["Risk"]) == {"a"}


# --- clustering and matched recall --------------------------------------------------


def test_cluster_names_merges_near_duplicates_at_link_auto() -> None:
    clusters = parity.cluster_names(
        ["NVDA Supplier Risk", "NVDA Export Control Risk", "NVDA export control risks",
         "NVDA Export Control Risk"], THRESHOLD)
    assert sorted(len(c) for c in clusters) == [1, 3]
    assert parity.cluster_names([], THRESHOLD) == []


def test_cluster_names_is_independent_of_input_order() -> None:
    names = ["NVDA Gaming Segment", "NVDA Data Center Segment", "NVDA gaming segment",
             "NVDA Automotive Segment"]
    assert parity.cluster_names(names, THRESHOLD) == parity.cluster_names(names[::-1], THRESHOLD)


def test_matched_clusters_counts_clusters_with_a_name_on_the_other_side() -> None:
    clusters = parity.cluster_names(["NVDA Export Control Risk", "NVDA Supplier Risk"], THRESHOLD)
    assert parity.matched_clusters(clusters, ["NVDA export control risks"], THRESHOLD) == 1
    assert parity.matched_clusters(clusters, [], THRESHOLD) == 0


def _public_named(path: Path, claims: list[tuple[str, str, str]], acc: str = ACC) -> Store:
    """Public store with open ``(type, name, item)`` claims of ``acc``."""
    store = Store(path)
    company = store.put_entity("Company", "NVDA", {}, {"ticker": "NVDA"})
    for ctype, name, item in claims:
        cid = store.put_entity(ctype, name, {}, {})
        store.add_relationship(cid, company, "subject_of", acc, item, {})
    return store


def test_claim_repeated_across_three_sections_counts_once(tmp_path: Path) -> None:
    _public_named(tmp_path / "kb.sqlite", [("RevenueSegment", "NVDA Graphics Segment", item)
                                           for item in ("Item_1", "Item_7_MDA", "Item_15")])
    db = _private(tmp_path / "kg.db", [("a", "RevenueSegment", DOC, None, "NVDA Graphics Segment")])
    [r] = parity.compare_claims(parity.open_readonly(tmp_path / "kb.sqlite"),
                                parity.open_readonly(db), ROWS, THRESHOLD).recall
    assert (r.public_clusters, r.private_clusters) == (1, 1)
    assert r.public_in_private_pct == 100.0 and r.private_in_public_pct == 100.0


def test_manifest_only_claim_is_counted_and_matched(tmp_path: Path) -> None:
    _public_named(tmp_path / "kb.sqlite", [("Risk", "NVDA Export Control Risk", "Item_1A"),
                                           ("Risk", "NVDA Supplier Concentration Risk", "Item_1A")])
    db = _private(tmp_path / "kg.db", [
        ("a", "Risk", DOC, None, "NVDA Export Control Risk"),
        ("c", "Risk", MANIFEST, None, "NVDA Supplier Concentration Risk")])
    comparison = parity.compare_claims(parity.open_readonly(tmp_path / "kb.sqlite"),
                                       parity.open_readonly(db), ROWS, THRESHOLD)
    [r] = comparison.recall
    assert (r.private_clusters, r.private_matched, r.public_matched) == (2, 2, 2)
    assert comparison.private_accession_doc == {"Risk": 1}
    assert comparison.private_with_manifest == {"Risk": 2}


def test_recall_is_computed_both_ways_and_per_filing(tmp_path: Path) -> None:
    other = "0001045810-24-000029"
    rows = [*ROWS, SampleRow("NVDA", 2024, other)]
    _public_named(tmp_path / "kb.sqlite", [("Risk", "NVDA Export Control Risk", "Item_1A"),
                                           ("Risk", "NVDA Cyber Risk", "Item_1A"),
                                           ("Risk", "NVDA Tax Risk", "Item_1A")])
    db = _private(tmp_path / "kg.db", [
        ("a", "Risk", DOC, None, "NVDA export control risks"),
        ("b", "Risk", DOC, None, "NVDA Climate Risk"),
        # same name as a public claim, but under another filing: no match for ACC
        ("c", "Risk", "filings/NVDA/2024-02_10-K_000104581024000029.md", None, "NVDA Tax Risk")])
    [r] = parity.compare_claims(parity.open_readonly(tmp_path / "kb.sqlite"),
                                parity.open_readonly(db), rows, THRESHOLD).recall
    assert (r.public_clusters, r.public_matched) == (3, 1)    # public -> private 1 of 3
    assert (r.private_clusters, r.private_matched) == (3, 1)  # private -> public 1 of 3
    assert r.public_in_private_pct == 33.3 and r.private_in_public_pct == 33.3


@pytest.mark.parametrize("pub, priv, pub_hit, priv_hit, failing", [
    (25, 25, 20, 20, False),   # exactly 80% both ways
    (25, 25, 19, 25, True),    # public -> private below 80%: gated
    (25, 25, 25, 19, True),    # private -> public below 80%: gated
    (19, 19, 1, 1, False),     # fewer than 20 private clusters: never failing
    (5, 20, 1, 20, True),      # n is the private cluster count; 1 of 5 public matched
    (30, 0, 0, 0, False),      # nothing private: not comparable, never failing
])
def test_gate_threshold_is_80_percent_in_both_directions_with_20_private_clusters(pub, priv, pub_hit, priv_hit,
                                                                failing) -> None:
    r = parity.TypeRecall("Risk", pub, priv, pub_hit, priv_hit)
    assert r.failing is failing
    if priv == 0:
        assert r.comparable is False and r.public_in_private_pct is None


def _main_args(tmp_path: Path, private: Path, match: int = 95) -> list[str]:
    sample = tmp_path / "sample.yaml"
    sample.write_text(f"rows:\n  - {{ticker: NVDA, fiscal_year: 2025, accession: \"{ACC}\"}}\n"
                      "extract_items: [Item_1A]\n")
    report = tmp_path / f"report{match}.json"
    report.write_text(json.dumps({"fidelity": {"overall": {
        "comparable": 100, "match": match, "mismatch": 100 - match,
        "not_comparable_split": 0, "not_detected": 0}}}))
    return ["--private-kb", str(private), "--public-kb", str(tmp_path / "kb.sqlite"),
            "--sample", str(sample), "--report", str(report), "--out", str(tmp_path / "p.json"),
            "--extractions", str(_matching_cache(tmp_path / "cache"))]


def test_main_writes_counts_only_json_and_fails_below_90_percent_fidelity(tmp_path) -> None:
    private = _private(tmp_path / "kg.db", [("a", "Risk", DOC, None, "claim")])
    _public(tmp_path / "kb.sqlite")
    assert parity.main(_main_args(tmp_path, private)) == 0
    text = (tmp_path / "p.json").read_text()
    data = json.loads(text)
    assert set(data) == {"filings", "fidelity", "claims", "ingest_vs_cache", "rule"}
    risk = data["claims"]["Risk"]
    assert (risk["public_clusters"], risk["private_clusters"]) == (1, 1)  # 3 "claim"s cluster
    assert risk["diagnostic_raw_counts"]["public_claims"] == 3
    assert risk["recall_public_in_private_pct"] == 100.0
    assert data["rule"]["cluster_threshold_wratio"] == THRESHOLD
    assert "claim" not in text.replace("claims", "")  # counts only, no claim names
    assert parity.main(_main_args(tmp_path, private, match=89)) == 1


WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
         "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
         "sierra", "tango", "uniform", "victor"]  # 22 names that never cluster together


def _gate_run(tmp_path: Path, public_words: list[str], private_words: list[str]) -> int:
    store = Store(tmp_path / "kb.sqlite")
    company = store.put_entity("Company", "NVDA", {}, {"ticker": "NVDA"})
    for w in public_words:
        cid = store.put_entity("Risk", f"{w} exposure", {}, {})
        store.add_relationship(cid, company, "subject_of", ACC, "Item_1A", {})
    private = _private(tmp_path / "kg.db", [(w, "Risk", DOC, None, f"{w} exposure")
                                            for w in private_words])
    args = _main_args(tmp_path, private)
    # The committed cache must agree with the store, so write one that does.
    cache = tmp_path / "cache2"
    _envelope(cache / ACC, "Item_1A", *(_entity("Risk", f"{w} exposure") for w in public_words))
    args[args.index("--extractions") + 1] = str(cache)
    return parity.main(args)


EXTRA = ["whiskey", "xray", "yankee", "zulu", "apple", "banana", "cherry", "damson", "elder",
         "fig"]


@pytest.mark.parametrize("public_words, private_words, low_direction", [
    (WORDS + EXTRA, WORDS[:20], "public_in_private"),   # 20 of 32 public claims have a private match
    (WORDS[:20], WORDS + EXTRA, "private_in_public"),   # 20 of 32 private claims have a public match
])
def test_gate_fails_on_either_direction_below_80(tmp_path, public_words, private_words,
                                                 low_direction) -> None:
    assert _gate_run(tmp_path, public_words, private_words) == 1
    risk = json.loads((tmp_path / "p.json").read_text())["claims"]["Risk"]
    assert risk[f"recall_{low_direction}_pct"] == 62.5 and risk["failing"] is True
    assert risk["n"] == risk["private_clusters"]


def test_gate_skips_types_under_20_private_clusters(tmp_path) -> None:
    # 19 private claims, 1 public: both recalls are far below 80% but n = 19 < 20.
    assert _gate_run(tmp_path, ["alpha"], WORDS[:19]) == 0
    risk = json.loads((tmp_path / "p.json").read_text())["claims"]["Risk"]
    assert risk["private_clusters"] == 19 and risk["failing"] is False
    assert risk["recall_private_in_public_pct"] == 5.3


# --- aliases ------------------------------------------------------------------------


def _alias_recall(tmp_path: Path, private_name: str, aliases: list[str], public_name: str):
    _public_named(tmp_path / "kb.sqlite", [("Risk", public_name, "Item_1A")])
    db = _private(tmp_path / "kg.db", [("a", "Risk", DOC, None, private_name, aliases)])
    [r] = parity.compare_claims(parity.open_readonly(tmp_path / "kb.sqlite"),
                                parity.open_readonly(db), ROWS, THRESHOLD).recall
    return r


def test_alias_match_counts_as_found(tmp_path) -> None:
    r = _alias_recall(tmp_path, "Supply concentration at TSMC",
                      ["Reliance on TSMC for chip manufacturing"],
                      "Reliance on TSMC for chip manufacturing risk")
    assert (r.private_matched, r.public_matched) == (1, 1)
    assert (r.private_matched_names_only, r.public_matched_names_only) == (0, 0)


def test_short_alias_does_not_partial_match(tmp_path) -> None:
    from rapidfuzz import fuzz, utils
    # Premise: WRatio's partial-ratio fallback would score a one-word alias as a match.
    assert fuzz.WRatio("NVIDIA", "NVIDIA export control risk", processor=utils.default_process) >= 90
    r = _alias_recall(tmp_path, "Unrelated claim name", ["NVIDIA"], "NVIDIA export control risk")
    assert (r.private_matched, r.public_matched) == (0, 0)


def test_private_aliases_tolerate_a_missing_column_and_bad_json(tmp_path) -> None:
    db = _private(tmp_path / "kg.db", [("a", "Risk", DOC, None, "n1"), ("b", "Risk", DOC, None, "n2")])
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE entities SET aliases='not json' WHERE id='b'")
    claims = parity.private_claims(parity.open_readonly(db), ROWS[0])["Risk"]
    assert claims["a"].aliases == () and claims["b"].aliases == ()
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE entities DROP COLUMN aliases")
    claims = parity.private_claims(parity.open_readonly(db), ROWS[0])["Risk"]
    assert claims["a"].name == "n1" and claims["a"].aliases == ()
