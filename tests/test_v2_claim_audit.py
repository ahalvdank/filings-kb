"""Tests for the claim-audit sample builder, verbatim check and labelling sheet."""

import csv
import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from evals.v2 import eval_claim_audit
from evals.v2.claim_audit import build
from filings_kb.sections import Section
from filings_kb.store import Store
from filings_kb.types import FilingRef

ACC = "0001045810-25-000023"
ITEM = "Item_1A"
SECTION_TEXT = "Intro " + ("filler " * 200) + "Supply is  limited\xa0by\nfoundry capacity. " + (
    "tail " * 200)
TYPES = ["Risk", "MaterialEvent", "RevenueSegment", "LegalProceeding", "CapitalReturn",
         "CapitalStructureEvent", "GuidanceUpdate", "ESGCommitment", "InsiderTransaction"]


def _store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "kb.sqlite")
    s.put_filing(FilingRef(
        ticker="NVDA", cik="1045810", company="NVIDIA CORP", accession=ACC, form="10-K",
        filed=date(2025, 2, 26), fiscal_year=2025, fiscal_year_end=date(2025, 1, 26),
        primary_doc="d.htm", html_path=Path("x.htm")))
    s.put_section(Section(accession=ACC, ticker="NVDA", fiscal_year=2025, item=ITEM,
                          title="t", text=SECTION_TEXT))
    return s


def _add_claim(s: Store, ctype: str, name: str, excerpts: list[str]) -> None:
    subject = s.put_entity("Company", "NVIDIA", {}, {})
    cid = s.put_entity(ctype, name, {"k": name}, {})
    s.add_relationship(cid, subject, "subject_of", ACC, ITEM, {})
    s.put_claim_evidence(ACC, ITEM, [(ctype, name, e) for e in excerpts])


def test_verbatim_rate_normalises_whitespace(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _add_claim(s, "Risk", "ok", ["Supply is limited by\nfoundry  capacity."])
    _add_claim(s, "Risk", "bad", ["Supply is unlimited"])
    _add_claim(s, "Risk", "mixed", ["Supply is limited by foundry capacity.", "invented"])
    res = build.verbatim_rate(tmp_path / "kb.sqlite")["whitespace"]
    assert res["rows"] == 4 and res["rows_verbatim"] == 2
    assert res["claims"] == 3 and res["claims_with_non_verbatim"] == 2
    assert {r["claim_name"] for r in res["not_verbatim"]} == {"bad", "mixed"}
    assert {r["content"] for r in res["not_verbatim"]} == {"Supply is unlimited", "invented"}


def test_fold_finds_curly_apostrophe_that_whitespace_check_misses(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.put_section(Section(accession=ACC, ticker="NVDA", fiscal_year=2025, item="Item_7",
                          title="t", text="The Firm\u2019s exposure \u2013 net \u2212 5 \u201cx\u201d."))
    _add_claim(s, "Risk", "q", [])
    s.put_claim_evidence(ACC, "Item_7", [("Risk", "q", "The Firm's exposure - net - 5 \"x\".")])
    res = build.verbatim_rate(tmp_path / "kb.sqlite")
    assert res["whitespace"]["rows_verbatim"] == 0
    assert res["fold"]["rows_verbatim"] == 1 and res["fold"]["not_verbatim"] == []


def test_no_public_section_rows_excluded_from_denominators_and_listed(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _add_claim(s, "Risk", "ok", ["Supply is limited by foundry capacity."])
    s.put_claim_evidence(ACC, "Item_9", [("Risk", "ghost", "anything"), ("Risk", "ghost", "two")])
    res = build.verbatim_rate(tmp_path / "kb.sqlite")
    assert res["no_public_section"]["rows"] == 2 and res["no_public_section"]["claims"] == 1
    assert res["no_public_section"]["items"][0]["claim_name"] == "ghost"
    assert res["whitespace"]["rows"] == 1 and res["fold"]["rows"] == 1
    assert res["whitespace"]["not_verbatim"] == []


def test_verbatim_rate_is_read_only(tmp_path: Path) -> None:
    _store(tmp_path)
    db = tmp_path / "kb.sqlite"
    before = db.read_bytes()
    build.verbatim_rate(db)
    assert db.read_bytes() == before


def test_per_type_counts_ranks_by_distinct_claims(tmp_path: Path) -> None:
    s = _store(tmp_path)
    for t, n in {"Risk": 3, "MaterialEvent": 1, "LegalProceeding": 2}.items():
        for i in range(n):
            _add_claim(s, t, f"{t}{i}", ["a", "b"])  # two excerpts: still one claim
    assert build.claim_counts_by_type(tmp_path / "kb.sqlite") == {
        "Risk": 3, "LegalProceeding": 2, "MaterialEvent": 1}


def test_default_per_type_is_six_for_top_five_five_for_rest() -> None:
    counts = {t: 100 - i for i, t in enumerate(TYPES)}
    per_type = build.default_per_type(counts)
    assert sum(per_type.values()) == 50
    assert [per_type[t] for t in TYPES] == [6] * 5 + [5] * 4


def _full_db(tmp_path: Path) -> Path:
    s = _store(tmp_path)
    for i, t in enumerate(TYPES):
        for j in range(14 - i):
            _add_claim(s, t, f"{t}-{j}", ["Supply is limited by foundry capacity.", "xx"])
    return tmp_path / "kb.sqlite"


def test_sample_claims_is_seeded_stratified_and_totals_50(tmp_path: Path) -> None:
    db = _full_db(tmp_path)
    per_type = build.default_per_type(build.claim_counts_by_type(db))
    a = build.sample_claims(db, per_type, seed=7)
    b = build.sample_claims(db, per_type, seed=7)
    c = build.sample_claims(db, per_type, seed=8)
    assert a == b and a != c
    assert len(a) == 50
    assert [x["claim_id"] for x in a] == [f"ca{i:02d}" for i in range(1, 51)]
    got = {t: sum(1 for x in a if x["claim_type"] == t) for t in per_type}
    assert got == per_type
    assert len({(x["accession"], x["item"], x["claim_type"], x["name"]) for x in a}) == 50
    first = a[0]
    assert first["ticker"] == "NVDA" and first["fiscal_year"] == 2025
    assert first["subject"] == "NVIDIA" and first["attributes"] == {"k": first["name"]}
    assert first["excerpt"] == "Supply is limited by foundry capacity."  # longest
    assert len(first["excerpts"]) == 2


def test_sample_claims_rejects_oversized_request(tmp_path: Path) -> None:
    db = _full_db(tmp_path)
    with pytest.raises(ValueError, match="InsiderTransaction"):
        build.sample_claims(db, {"InsiderTransaction": 99}, seed=1)


def test_sheet_marks_excerpt_in_paragraph(tmp_path: Path) -> None:
    db = _full_db(tmp_path)
    sample = build.sample_claims(db, {"Risk": 2}, seed=1)
    out = tmp_path / "sheet.csv"
    build.write_sheet(sample, db, out)
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["claim_id", "claim_type", "ticker", "fiscal_year", "item", "name",
                             "subject", "attributes_json", "excerpt", "locate", "paragraph",
                             "label", "note"]
    r = rows[0]
    assert ">>> Supply is limited by foundry capacity. <<<" in r["paragraph"]
    assert r["paragraph"].index(">>>") > 0 and "Intro" not in r["paragraph"]  # +-600 window
    assert r["label"] == "" and r["note"] == ""
    assert json.loads(r["attributes_json"]) == sample[0]["attributes"]


def test_sheet_locate_column_records_how_the_excerpt_was_found(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _add_claim(s, "Risk", "r", ["Supply is limited by foundry capacity."])
    db = tmp_path / "kb.sqlite"
    out = tmp_path / "sheet.csv"
    build.write_sheet(build.sample_claims(db, {"Risk": 1}, seed=1), db, out)
    assert next(csv.DictReader(out.open(newline="", encoding="utf-8")))["locate"] == "exact"


def _sheet(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "audit.csv"
    lines = ["claim_id,ticker,claim_type,statement,source_quote,label,note"]
    lines += [f"c{i},NVDA,Risk,s,q,{label},{note}" for i, (label, note, _) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_other_label_requires_note(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="c1"):
        eval_claim_audit.score_claim_audit(_sheet(tmp_path, [("TP", "", ""), ("other", "", "")]))
    ok = eval_claim_audit.score_claim_audit(
        _sheet(tmp_path, [("TP", "", ""), ("other", "odd case", "")]))
    assert ok["failure_breakdown"]["other"] == 1


def test_score_reports_wilson_interval(tmp_path: Path) -> None:
    rows = [("TP", "", "")] * 45 + [("hallucinated", "", "")] * 5
    res = eval_claim_audit.score_claim_audit(_sheet(tmp_path, rows))
    lo, hi = res["precision_wilson95"]
    assert lo == pytest.approx(0.786, abs=1e-3) and hi == pytest.approx(0.957, abs=1e-3)


def test_wrong_attribute_counts_in_breakdown(tmp_path: Path) -> None:
    res = eval_claim_audit.score_claim_audit(
        _sheet(tmp_path, [("TP", "", ""), ("wrong_attribute", "", "")]))
    assert res["failure_breakdown"]["wrong_attribute"] == 1
    assert res["precision"] == 0.5


def test_sampler_skips_claims_without_a_public_section(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _add_claim(s, "Risk", "has", ["Supply is limited by foundry capacity."])
    subject = s.put_entity("Company", "NVIDIA", {}, {})
    cid = s.put_entity("Risk", "ghost", {}, {})
    s.add_relationship(cid, subject, "subject_of", ACC, "Item_9", {})
    s.put_claim_evidence(ACC, "Item_9", [("Risk", "ghost", "x")])
    db = tmp_path / "kb.sqlite"
    assert build.claim_counts_by_type(db) == {"Risk": 1}
    assert build.excluded_no_public_section(db) == 1
    with pytest.raises(ValueError):
        build.sample_claims(db, {"Risk": 2}, seed=1)
    assert [c["name"] for c in build.sample_claims(db, {"Risk": 1}, seed=1)] == ["has"]


def test_locate_exact_fold_and_fuzzy() -> None:
    text = "Alpha beta.  The Firm\u2019s supply is limited by foundry capacity today. Omega."
    para, kind = build.locate("Firm's supply is limited by foundry capacity", text)
    assert kind == "fold" and ">>> Firm\u2019s supply is limited by foundry capacity <<<" in para
    para, kind = build.locate("beta. The Firm", text)
    assert kind == "exact" and ">>> beta. The Firm <<<" in para
    para, kind = build.locate("The Firm\u2019s supply is restricted by foundry capacity today.", text)
    assert kind.startswith("fuzzy:") and int(kind.split(":")[1]) >= 60
    assert ">>>" in para and not para.startswith("[WEAK MATCH]")


def test_locate_weak_match_is_flagged() -> None:
    text = "Alpha beta gamma delta epsilon " * 20
    para, kind = build.locate("zzzz qqqq xxxx jjjj wwww vvvv kkkk", text)
    assert kind.startswith("fuzzy:") and int(kind.split(":")[1]) < 60
    assert para.startswith("[WEAK MATCH] ")
