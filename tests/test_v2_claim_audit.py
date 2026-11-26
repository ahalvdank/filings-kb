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
    res = build.verbatim_rate(tmp_path / "kb.sqlite")
    assert res["rows"] == 4 and res["rows_verbatim"] == 2
    assert res["claims"] == 3 and res["claims_with_non_verbatim"] == 2
    assert {r["claim_name"] for r in res["not_verbatim"]} == {"bad", "mixed"}
    assert {r["content"] for r in res["not_verbatim"]} == {"Supply is unlimited", "invented"}


def test_verbatim_rate_missing_section_is_not_verbatim(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.put_claim_evidence(ACC, "Item_7", [("Risk", "r", "anything")])
    res = build.verbatim_rate(tmp_path / "kb.sqlite")
    assert res["rows_verbatim"] == 0 and len(res["not_verbatim"]) == 1


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
                             "subject", "attributes_json", "excerpt", "paragraph", "label",
                             "note"]
    r = rows[0]
    assert ">>> Supply is limited by foundry capacity. <<<" in r["paragraph"]
    assert r["paragraph"].index(">>>") > 0 and "Intro" not in r["paragraph"]  # +-600 window
    assert r["label"] == "" and r["note"] == ""
    assert json.loads(r["attributes_json"]) == sample[0]["attributes"]


def test_sheet_flags_excerpt_missing_from_section(tmp_path: Path) -> None:
    s = _store(tmp_path)
    _add_claim(s, "Risk", "r", ["not in the section"])
    db = tmp_path / "kb.sqlite"
    sample = build.sample_claims(db, {"Risk": 1}, seed=1)
    out = tmp_path / "sheet.csv"
    build.write_sheet(sample, db, out)
    para = next(csv.DictReader(out.open(newline="", encoding="utf-8")))["paragraph"]
    assert para.startswith("[EXCERPT NOT FOUND IN SECTION] Intro")
    assert len(para) == len("[EXCERPT NOT FOUND IN SECTION] ") + 1200


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
