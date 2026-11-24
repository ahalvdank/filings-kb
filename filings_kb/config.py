"""Loading of the pinned sample configuration."""

from pathlib import Path

import yaml

from filings_kb.types import Sample, SampleRow


def load_sample(path: Path) -> Sample:
    """Load ``config/sample.yaml``.

    Args:
        path: Path to the sample YAML file.

    Returns:
        The pinned rows and the extraction section allowlist.

    Raises:
        ValueError: If ``rows`` or ``extract_items`` is missing, a row lacks a
            field, or two rows share an accession, a form is not
            10-K or 10-K/A, a 10-K/A has no ``amends``, a 10-K sets ``amends``, or
            ``amends`` names an accession that is not a row.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("rows", "extract_items"):
        if not data or key not in data:
            raise ValueError(f"{path}: missing required key '{key}'")

    rows: list[SampleRow] = []
    seen: set[str] = set()
    for raw in data["rows"]:
        try:
            row = SampleRow(
                ticker=str(raw["ticker"]),
                fiscal_year=int(raw["fiscal_year"]),
                accession=str(raw["accession"]),
                form=str(raw.get("form", "10-K")),
                amends=str(raw["amends"]) if raw.get("amends") else None,
            )
        except KeyError as exc:
            raise ValueError(f"{path}: row {raw!r} is missing {exc}") from exc
        if row.accession in seen:
            raise ValueError(f"{path}: duplicate accession {row.accession}")
        seen.add(row.accession)
        rows.append(row)

    for row in rows:
        if row.form not in ("10-K", "10-K/A"):
            raise ValueError(f"{path}: {row.accession} has unsupported form {row.form!r}")
        if row.form == "10-K/A" and row.amends is None:
            raise ValueError(f"{path}: 10-K/A row {row.accession} has no amends")
        if row.form == "10-K" and row.amends is not None:
            raise ValueError(f"{path}: 10-K row {row.accession} must not set amends")
        if row.amends is not None and row.amends not in seen:
            raise ValueError(
                f"{path}: {row.accession} amends {row.amends}, which is not a row in the sample"
            )

    return Sample(
        rows=rows,
        extract_items=[str(i) for i in data["extract_items"]],
        skip_items=[str(i) for i in data.get("skip_items", [])],
    )
