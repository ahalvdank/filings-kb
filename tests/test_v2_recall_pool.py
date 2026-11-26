"""Tests for the v2 recall pool builder and blind labelling sheet."""

import csv
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from evals.v2.recall.pool import (
    bm25_rank, build_pool, section_chunks, current_accession, tokenize, write_sheet,
)
from filings_kb.chunking import Chunk
from filings_kb.sections import Section
from filings_kb.store import Store
from filings_kb.types import FilingRef

ORIG = "0000000001-25-000001"
AMEND = "0000000001-25-000002"


def _filing(accession: str, form: str, filed: date) -> FilingRef:
    return FilingRef(
        ticker="AAA", cik="1", company="AAA CORP", accession=accession, form=form,
        filed=filed, fiscal_year=2025, fiscal_year_end=date(2025, 1, 26),
        primary_doc="a.htm", html_path=Path("data/raw/a.htm"))


def _put(store: Store, accession: str, item: str, texts: list[str]) -> None:
    store.put_section(Section(accession=accession, ticker="AAA", fiscal_year=2025,
                              item=item, title="t", text=" ".join(texts)))
    store.put_chunks([
        Chunk(chunk_id=f"{accession}:{item}:{i}", accession=accession, ticker="AAA",
              fiscal_year=2025, item=item, start=i, end=i + 1, text=t)
        for i, t in enumerate(texts)])


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "kb.sqlite"
    store = Store(path)
    store.put_filing(_filing(ORIG, "10-K", date(2025, 2, 26)))
    store.put_filing(_filing(AMEND, "10-K/A", date(2025, 4, 1)))
    _put(store, ORIG, "Item_1A", [f"original chunk {i} export controls" for i in range(30)])
    _put(store, AMEND, "Item_1A", [f"amended chunk {i} export controls" for i in range(30)])
    _put(store, ORIG, "Item_7_MDA", ["only original here"])
    return path


QUESTION = {"id": "q1", "ticker": "AAA", "fiscal_year": 2025, "item": "Item_1A",
            "kind": "fact", "question": "export controls amended chunk 3"}


def test_tokenize_handles_nbsp_currency_and_curly_quotes():
    assert tokenize("Boeing’s $8,351\xa0debt") == ["boeing", "s", "8", "351", "debt"]


def test_bm25_ranks_term_match_first_and_no_match_returns_zero_scores():
    docs = {"a": "export controls on chips", "b": "dividends paid"}
    assert bm25_rank("export controls", docs)[0][0] == "a"
    ranking = bm25_rank("zebra", docs)
    assert len(ranking) == 2
    assert all(score == 0.0 for _, score in ranking)


def test_bm25_ties_break_by_chunk_id_for_determinism():
    docs = {"b": "same", "a": "same"}
    assert [cid for cid, _ in bm25_rank("same", docs)] == ["a", "b"]


def test_current_accession_matches_store_for_amended_section(db: Path):
    store = Store(db)
    assert store.section_accession("AAA", 2025, "Item_1A") == AMEND
    assert current_accession(db, "AAA", 2025, "Item_1A") == AMEND
    assert current_accession(db, "AAA", 2025, "Item_7_MDA") == store.section_accession(
        "AAA", 2025, "Item_7_MDA") == ORIG
    assert current_accession(db, "AAA", 2025, "Item_9") is None


def test_section_chunks_reads_only_the_named_section(db: Path):
    chunks = section_chunks(db, AMEND, "Item_1A")
    assert len(chunks) == 30
    assert all(cid.startswith(AMEND) for cid in chunks)


def test_current_accession_does_not_write_to_db(db: Path):
    before = db.read_bytes()
    current_accession(db, "AAA", 2025, "Item_1A")
    section_chunks(db, AMEND, "Item_1A")
    assert db.read_bytes() == before


def _fake_dense(calls: list):
    def dense(text: str, k: int, where: dict) -> list[tuple[str, float]]:
        calls.append((text, k, where))
        return [(f"{where['accession']}:{where['item']}:{i}", 1.0 - i / 100)
                for i in range(25)][:k]
    return dense


def test_build_pool_unions_dense_top5_and_bm25_top5_and_keeps_dense_top20(db: Path):
    calls: list = []
    pool = build_pool([QUESTION], db, _fake_dense(calls))
    entry = pool["questions"]["q1"]
    assert entry["accession"] == AMEND
    assert entry["item"] == "Item_1A"
    assert len(entry["dense_ranking"]) == 20
    assert len(entry["bm25_top"]) == 5
    assert entry["pooled"] == sorted(set(entry["dense_ranking"][:5]) | set(entry["bm25_top"]))
    assert calls[0][1] == 20
    assert calls[0][2] == {"ticker": "AAA", "fiscal_year": 2025, "item": "Item_1A",
                           "accession": AMEND}
    assert pool["meta"]["pool_k"] == 5 and pool["meta"]["rank_k"] == 20
    assert {"built_at", "git_commit"} <= set(pool["meta"])


def test_build_pool_uses_current_accession_for_amended_section(db: Path):
    pool = build_pool([QUESTION], db, _fake_dense([]))
    entry = pool["questions"]["q1"]
    assert all(cid.startswith(AMEND) for cid in entry["pooled"] + entry["bm25_top"])


def test_build_pool_raises_when_section_missing(db: Path):
    bad = {**QUESTION, "id": "q9", "item": "Item_9"}
    with pytest.raises(ValueError, match="q9"):
        build_pool([bad], db, _fake_dense([]))


def _pool_for_sheet(db: Path) -> dict:
    return build_pool([QUESTION], db, _fake_dense([]))


def test_write_sheet_hides_rank_and_is_deterministic(db: Path, tmp_path: Path):
    pool = _pool_for_sheet(db)
    a, b, c = (tmp_path / n for n in ("a.csv", "b.csv", "c.csv"))
    write_sheet(pool, db, a, seed=1)
    write_sheet(pool, db, b, seed=1)
    write_sheet(pool, db, c, seed=2)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    rows = list(csv.reader(a.open(encoding="utf-8", newline="")))
    assert rows[0] == ["qid", "question", "row", "chunk_id", "passage", "relevant"]
    body = rows[1:]
    ids = [r[3] for r in body]
    assert sorted(ids) == pool["questions"]["q1"]["pooled"]
    assert ids != pool["questions"]["q1"]["pooled"] or len(ids) < 3  # shuffled
    assert [r[2] for r in body] == [str(i) for i in range(1, len(body) + 1)]
    assert all(r[1] == QUESTION["question"] for r in body)
    assert all(r[5] == "" for r in body)
    assert all(r[4] for r in body)


def test_cli_writes_pool_and_sheet_without_loading_the_model(db: Path, tmp_path: Path,
                                                             monkeypatch, capsys):
    import yaml
    from evals.v2.recall import pool as pool_mod
    import filings_kb.embed

    class FakeIndex:
        def __init__(self, path, model_name, *, max_tokens):
            self.model_name = model_name

        def query(self, text, k, where):
            return _fake_dense([])(text, k, where)

    monkeypatch.setattr(filings_kb.embed, "EmbeddingIndex", FakeIndex)
    qfile = tmp_path / "q.yaml"
    qfile.write_text(yaml.safe_dump({"questions": [QUESTION]}), encoding="utf-8")
    out = tmp_path / "out"
    pool_mod.main(["--questions", str(qfile), "--db", str(db), "--chroma", str(tmp_path),
                   "--out-dir", str(out), "--seed", "7"])
    import json
    pool = json.loads((out / "pool.json").read_text(encoding="utf-8"))
    assert pool["meta"]["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert len(pool["meta"]["git_commit"]) == 40
    assert (out / "labelling_sheet.csv").exists()
