from datetime import date
from pathlib import Path

import pytest
import yaml

from filings_kb.config import load_sample
from filings_kb.types import FilingRef

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "config" / "sample.yaml"
QUESTIONS_PATH = REPO_ROOT / "config" / "questions.yaml"

CALENDAR_YEARS = [2021, 2022, 2023, 2024, 2025]
SHIFTED_YEARS = [2022, 2023, 2024, 2025, 2026]
EXPECTED_YEARS = {
    ticker: CALENDAR_YEARS
    for ticker in ("AMD", "INTC", "BA", "UNH", "META", "JPM", "XOM", "GOOGL", "AMZN", "CVX")
}
EXPECTED_YEARS.update({"NVDA": SHIFTED_YEARS, "MSFT": SHIFTED_YEARS})
EXPECTED_TICKERS = set(EXPECTED_YEARS)


def test_load_sample_returns_60_originals_plus_1_amendment_12_tickers_5_years():
    sample = load_sample(SAMPLE_PATH)
    originals = [r for r in sample.rows if r.form == "10-K"]
    amendments = [r for r in sample.rows if r.form == "10-K/A"]

    assert len(sample.rows) == 61
    assert len(originals) == 60
    assert [(a.ticker, a.fiscal_year, a.accession, a.amends) for a in amendments] == [
        ("AMD", 2025, "0000002488-26-000021", "0000002488-26-000018")
    ]
    assert {r.ticker for r in originals} == EXPECTED_TICKERS
    for ticker in EXPECTED_TICKERS:
        years = sorted(r.fiscal_year for r in originals if r.ticker == ticker)
        assert years == EXPECTED_YEARS[ticker]
    assert len({r.accession for r in sample.rows}) == 61

    nvda = {r.fiscal_year: r.accession for r in originals if r.ticker == "NVDA"}
    assert nvda[2025] == "0001045810-25-000023"
    assert nvda[2026] == "0001045810-26-000021"

    assert len(sample.extract_items) == 22
    assert "Item_7_MDA" in sample.extract_items
    assert "Item_1A" in sample.extract_items


def test_load_sample_rejects_duplicate_accession(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "rows": [
                    {"ticker": "NVDA", "fiscal_year": 2025, "accession": "0001045810-25-000023"},
                    {"ticker": "NVDA", "fiscal_year": 2026, "accession": "0001045810-25-000023"},
                ],
                "extract_items": ["Item_1A"],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate accession"):
        load_sample(path)


@pytest.mark.parametrize("missing", ["rows", "extract_items"])
def test_load_sample_rejects_missing_top_level_key(tmp_path, missing):
    data = {
        "rows": [{"ticker": "NVDA", "fiscal_year": 2025, "accession": "0001045810-25-000023"}],
        "extract_items": ["Item_1A"],
    }
    del data[missing]
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=missing):
        load_sample(path)


def test_every_question_fiscal_year_exists_in_sample():
    sample = load_sample(SAMPLE_PATH)
    pinned = {(r.ticker, r.fiscal_year) for r in sample.rows}
    questions = yaml.safe_load(QUESTIONS_PATH.read_text())["questions"]
    for question in questions:
        assert question["filings"], f"{question['slug']} has no filings"
        for ref in question["filings"]:
            assert (ref["ticker"], ref["fiscal_year"]) in pinned


def test_sec_url_built_from_cik_and_accession():
    ref = FilingRef(
        ticker="NVDA",
        cik="0001045810",
        company="NVIDIA CORP",
        accession="0001045810-25-000023",
        form="10-K",
        filed=date(2025, 2, 26),
        fiscal_year=2025,
        fiscal_year_end=date(2025, 1, 26),
        primary_doc="nvda-20250126.htm",
        html_path=Path("data/raw/NVDA/0001045810-25-000023.htm"),
    )
    assert ref.sec_url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    )


def _sample_file(tmp_path, rows):
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump({"rows": rows, "extract_items": ["Item_1A"]}))
    return path


ORIGINAL = {"ticker": "AMD", "fiscal_year": 2025, "accession": "0000002488-26-000018"}


def test_load_sample_rejects_amends_pointing_at_unknown_row(tmp_path):
    amendment = {"ticker": "AMD", "fiscal_year": 2025, "accession": "0000002488-26-000021",
                 "form": "10-K/A", "amends": "0000002488-26-999999"}
    with pytest.raises(ValueError, match="amends"):
        load_sample(_sample_file(tmp_path, [ORIGINAL, amendment]))


def test_load_sample_rejects_10ka_row_without_amends(tmp_path):
    amendment = {"ticker": "AMD", "fiscal_year": 2025, "accession": "0000002488-26-000021",
                 "form": "10-K/A"}
    with pytest.raises(ValueError, match="amends"):
        load_sample(_sample_file(tmp_path, [ORIGINAL, amendment]))


def test_load_sample_rejects_amends_on_a_plain_10k_row(tmp_path):
    odd = {"ticker": "AMD", "fiscal_year": 2024, "accession": "0000002488-25-000012",
           "amends": ORIGINAL["accession"]}
    with pytest.raises(ValueError, match="amends"):
        load_sample(_sample_file(tmp_path, [ORIGINAL, odd]))
