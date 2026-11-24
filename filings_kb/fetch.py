"""Resolve a pinned accession on EDGAR and cache its HTML locally."""

import json
import os
from datetime import date
from pathlib import Path

from edgar import Filing, get_by_accession_number, set_identity

from filings_kb.types import FilingRef, SampleRow

DEFAULT_EDGAR_IDENTITY = "filings-kb (github.com/ahalvdank/filings-kb)"
EDGAR_IDENTITY_ENV = "EDGAR_IDENTITY"


def edgar_identity() -> str:
    """User-Agent identity sent to EDGAR.

    Defaults to the project identity (no email). SEC may answer 403 to an
    identity without contact details; set ``EDGAR_IDENTITY`` to override.
    """
    return os.environ.get(EDGAR_IDENTITY_ENV, DEFAULT_EDGAR_IDENTITY)


def _get_by_accession_number(accession: str) -> Filing | None:
    """Look a filing up on EDGAR by accession; None when EDGAR has no such filing.

    The plain ``Filing`` returned here carries only cik, company, form,
    filing_date and accession_no. ``period_of_report``, ``document`` and
    ``html()`` each fetch the filing's SGML from EDGAR on first use.
    """
    set_identity(edgar_identity())
    return get_by_accession_number(accession)


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _meta_path(raw_dir: Path, row: SampleRow) -> Path:
    return raw_dir / row.ticker / f"{row.accession}.json"


def _html_path(raw_dir: Path, row: SampleRow) -> Path:
    return raw_dir / row.ticker / f"{row.accession}.htm"


def _write_meta(path: Path, ref: FilingRef) -> None:
    meta = {
        "cik": ref.cik,
        "company": ref.company,
        "form": ref.form,
        "filed": ref.filed.isoformat(),
        "fiscal_year_end": ref.fiscal_year_end.isoformat(),
        "primary_doc": ref.primary_doc,
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_meta(path: Path, row: SampleRow, html_path: Path) -> FilingRef:
    meta = json.loads(path.read_text(encoding="utf-8"))
    return FilingRef(
        ticker=row.ticker,
        cik=meta["cik"],
        company=meta["company"],
        accession=row.accession,
        form=meta["form"],
        filed=date.fromisoformat(meta["filed"]),
        fiscal_year=row.fiscal_year,
        fiscal_year_end=date.fromisoformat(meta["fiscal_year_end"]),
        primary_doc=meta["primary_doc"],
        html_path=html_path,
    )


def fetch_filing(row: SampleRow, raw_dir: Path) -> FilingRef:
    """Resolve a pinned filing and cache its HTML at ``raw_dir/<ticker>/<accession>.htm``.

    Metadata is cached beside the HTML (``<accession>.json``), so a second call
    for the same row makes no network call.

    Args:
        row: The pinned (ticker, fiscal_year, accession).
        raw_dir: Cache root (``data/raw`` by convention, gitignored).

    Returns:
        The resolved filing.

    Raises:
        LookupError: EDGAR has no filing with this accession.
        ValueError: The accession's form is not the row's declared form (an
            undeclared 10-K/A is refused), or the
            pinned fiscal year disagrees with the filing's period of report.
    """
    html_path = _html_path(raw_dir, row)
    meta_path = _meta_path(raw_dir, row)
    if html_path.exists() and meta_path.exists():
        return _read_meta(meta_path, row, html_path)

    filing = _get_by_accession_number(row.accession)
    if filing is None:
        raise LookupError(f"EDGAR has no filing with accession {row.accession}")
    if filing.form != row.form:
        raise ValueError(
            f"{row.accession} is a {filing.form}, not a {row.form}; a 10-K/A must be "
            "declared with form: 10-K/A and amends in the sample"
        )

    fiscal_year_end = _as_date(filing.period_of_report)
    if fiscal_year_end.year != row.fiscal_year:
        raise ValueError(
            f"{row.accession}: pinned fiscal year {row.fiscal_year} disagrees with "
            f"period of report {fiscal_year_end}"
        )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(filing.html(), encoding="utf-8")
    ref = FilingRef(
        ticker=row.ticker,
        cik=f"{int(filing.cik):010d}",
        company=str(filing.company),
        accession=row.accession,
        form=filing.form,
        filed=_as_date(filing.filing_date),
        fiscal_year=row.fiscal_year,
        fiscal_year_end=fiscal_year_end,
        primary_doc=str(filing.document.document),
        html_path=html_path,
    )
    _write_meta(meta_path, ref)
    return ref
