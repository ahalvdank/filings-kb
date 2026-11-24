import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgar import Filing

from filings_kb import fetch
from filings_kb.types import SampleRow

FIXTURE = Path(__file__).parent / "fixtures" / "filing_index_nvda.json"
HTML = "<html><body>NVIDIA 10-K</body></html>"

NVDA_ROW = SampleRow(ticker="NVDA", fiscal_year=2025, accession="0001045810-25-000023")
NVDA_10KA_ROW = SampleRow(ticker="NVDA", fiscal_year=2025, accession="0001045810-25-000099")


class FakeFiling(Filing):
    """A real edgartools ``Filing`` with its network-backed members replaced.

    Built through the real ``Filing.__init__`` so the constructor shape is
    checked. Every overridden member must already exist on ``Filing``, so a
    renamed or invented attribute fails at class creation, not in production.
    """

    def __init__(self, record):
        super().__init__(
            cik=record["cik"],
            company=record["company"],
            form=record["form"],
            filing_date=record["filing_date"],
            accession_no=record["accession_no"],
        )
        self._record = record

    @property
    def period_of_report(self):
        return self._record["period_of_report"]

    @property
    def document(self):
        return SimpleNamespace(document=self._record["primary_document"])

    def html(self):
        return HTML


for _name in ("period_of_report", "document", "html"):
    assert hasattr(Filing, _name), f"edgartools Filing no longer has {_name}"


@pytest.fixture
def edgar_calls(monkeypatch):
    """Replace the edgartools lookup with the index fixture; record every call.

    The fixture JSON is a hand-made shape, not a real EDGAR capture.
    """
    records = {r["accession_no"]: r for r in json.loads(FIXTURE.read_text())["filings"]}
    calls = []

    def fake_get_by_accession_number(accession):
        calls.append(accession)
        record = records.get(accession)
        if record is None:
            return None
        return FakeFiling(record)

    monkeypatch.setattr(fetch, "_get_by_accession_number", fake_get_by_accession_number)
    return calls


def test_fetch_filing_resolves_pinned_accession_from_index_fixture(tmp_path, edgar_calls):
    ref = fetch.fetch_filing(NVDA_ROW, tmp_path)

    assert edgar_calls == ["0001045810-25-000023"]
    assert ref.form == "10-K"
    assert ref.fiscal_year == 2025
    assert ref.ticker == "NVDA"
    assert ref.cik == "0001045810"
    assert ref.company == "NVIDIA CORP"
    assert ref.accession == "0001045810-25-000023"
    assert ref.filed == date(2025, 2, 26)
    assert ref.fiscal_year_end == date(2025, 1, 26)
    assert ref.primary_doc == "nvda-20250126.htm"
    assert ref.html_path == tmp_path / "NVDA" / "0001045810-25-000023.htm"
    assert ref.html_path.read_text(encoding="utf-8") == HTML


def test_fetch_filing_rejects_accession_that_is_10k_a(tmp_path, edgar_calls):
    with pytest.raises(ValueError, match="10-K/A"):
        fetch.fetch_filing(NVDA_10KA_ROW, tmp_path)
    assert not (tmp_path / "NVDA" / "0001045810-25-000099.htm").exists()


def test_fetch_filing_accepts_10ka_only_when_the_row_declares_it(tmp_path, edgar_calls):
    declared = SampleRow(ticker="NVDA", fiscal_year=2025, accession="0001045810-25-000099",
                         form="10-K/A", amends="0001045810-25-000023")
    ref = fetch.fetch_filing(declared, tmp_path)
    assert ref.form == "10-K/A"
    assert ref.accession == "0001045810-25-000099"


def test_fetch_filing_rejects_original_when_row_declares_10ka(tmp_path, edgar_calls):
    wrong = SampleRow(ticker="NVDA", fiscal_year=2025, accession="0001045810-25-000023",
                      form="10-K/A", amends="0001045810-25-000001")
    with pytest.raises(ValueError, match="not a 10-K/A"):
        fetch.fetch_filing(wrong, tmp_path)


def test_fetch_filing_rejects_fiscal_year_that_disagrees_with_period(tmp_path, edgar_calls):
    wrong_year = SampleRow(ticker="NVDA", fiscal_year=2024, accession="0001045810-25-000023")
    with pytest.raises(ValueError, match="fiscal year"):
        fetch.fetch_filing(wrong_year, tmp_path)


def test_fetch_filing_raises_when_accession_not_found(tmp_path, edgar_calls):
    missing = SampleRow(ticker="NVDA", fiscal_year=2025, accession="0001045810-25-000001")
    with pytest.raises(LookupError, match="0001045810-25-000001"):
        fetch.fetch_filing(missing, tmp_path)


def test_fetch_filing_makes_no_network_call_when_html_is_cached(tmp_path, edgar_calls):
    first = fetch.fetch_filing(NVDA_ROW, tmp_path)
    assert len(edgar_calls) == 1

    second = fetch.fetch_filing(NVDA_ROW, tmp_path)

    assert len(edgar_calls) == 1
    assert second == first


def test_fetch_filing_refetches_when_html_is_missing_but_metadata_cached(tmp_path, edgar_calls):
    first = fetch.fetch_filing(NVDA_ROW, tmp_path)
    first.html_path.unlink()

    fetch.fetch_filing(NVDA_ROW, tmp_path)

    assert len(edgar_calls) == 2
    assert first.html_path.read_text(encoding="utf-8") == HTML


def test_edgar_identity_defaults_to_project_user_agent_without_email(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    assert fetch.edgar_identity() == "filings-kb (github.com/ahalvdank/filings-kb)"


def test_edgar_identity_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("EDGAR_IDENTITY", "filings-kb someone@example.com")
    assert fetch.edgar_identity() == "filings-kb someone@example.com"
