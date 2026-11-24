"""Shared value types: the pinned sample and a fetched filing."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class SampleRow:
    """One pinned filing: the company's own fiscal year, the exact accession and form.

    A ``10-K/A`` row names the original it amends in ``amends``.
    """

    ticker: str
    fiscal_year: int
    accession: str
    form: str = "10-K"
    amends: str | None = None  # accession of the original; set only on 10-K/A rows


@dataclass(frozen=True)
class Sample:
    """The pinned sample, the extraction allowlist, and items never extracted without a cache."""

    rows: list[SampleRow]
    extract_items: list[str]
    skip_items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FilingRef:
    """A 10-K that has been resolved on EDGAR and cached locally as HTML."""

    ticker: str
    cik: str
    company: str
    accession: str
    form: str
    filed: date
    fiscal_year: int
    fiscal_year_end: date
    primary_doc: str
    html_path: Path

    @property
    def sec_url(self) -> str:
        """URL of the primary document on sec.gov."""
        accession_no_dashes = self.accession.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}"
            f"/{accession_no_dashes}/{self.primary_doc}"
        )
