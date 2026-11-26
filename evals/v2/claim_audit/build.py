"""Build the v2 claim-audit sample, the verbatim check and the labelling sheet.

One claim is a distinct ``(accession, item, claim_type, claim_name)`` in ``claim_evidence``
that has at least one excerpt and a currently valid claim entity with a subject (the same
join ``Store.claims_for`` uses). The database is only ever opened read-only.

CLI: ``python -m evals.v2.claim_audit.build --db data/kb.sqlite --out-dir evals/v2/claim_audit/
--seed 20261125``.
"""

import argparse
import csv
import json
import random
import re
import sqlite3
import subprocess
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from filings_kb.store import CLAIM_TYPES, SUBJECT_RELATIONSHIP

TOP_TYPES = 5
PER_TOP_TYPE = 6
PER_OTHER_TYPE = 5
CONTEXT_CHARS = 600
NOT_FOUND_PREFIX = "[EXCERPT NOT FOUND IN SECTION] "
NOT_FOUND_CHARS = 1200
SHEET_COLUMNS = ("claim_id", "claim_type", "ticker", "fiscal_year", "item", "name", "subject",
                 "attributes_json", "excerpt", "paragraph", "label", "note")

_WS = re.compile(r"\s+")


def _connect(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _norm(text: str) -> str:
    """Collapse every run of whitespace (including ``\\xa0``) to one space and strip."""
    return _WS.sub(" ", text).strip()


def _sections(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Normalised section text keyed by ``(accession, item)``."""
    return {(a, i): _norm(t) for a, i, t in conn.execute("SELECT accession, item, text FROM sections")}


def verbatim_rate(db: Path) -> dict:
    """Whitespace-normalised containment of every evidence excerpt in its section text.

    Returns:
        ``rows``/``rows_verbatim`` over evidence rows, ``claims`` (distinct) and
        ``claims_with_non_verbatim``, ``verbatim`` (= ``rows_verbatim``), and ``not_verbatim``,
        every failing row. A row whose section is missing counts as not verbatim.
    """
    with closing(_connect(db)) as conn:
        sections = _sections(conn)
        evidence = conn.execute(
            "SELECT accession, item, claim_type, claim_name, content FROM claim_evidence "
            "ORDER BY accession, item, claim_type, claim_name, content").fetchall()
    claims: set[tuple] = set()
    bad_claims: set[tuple] = set()
    failing: list[dict] = []
    for acc, item, ctype, name, content in evidence:
        key = (acc, item, ctype, name)
        claims.add(key)
        if _norm(content) not in sections.get((acc, item), "\0"):
            bad_claims.add(key)
            failing.append({"accession": acc, "item": item, "claim_type": ctype,
                            "claim_name": name, "content": content})
    return {"rows": len(evidence), "rows_verbatim": len(evidence) - len(failing),
            "verbatim": len(evidence) - len(failing), "claims": len(claims),
            "claims_with_non_verbatim": len(bad_claims), "not_verbatim": failing}


_CLAIMS_SQL = f"""
    SELECT r.source_accession, r.source_item, c.type, c.name, c.attributes, s.name,
           f.ticker, f.fiscal_year
    FROM relationships r
    JOIN entities c ON c.id = r.src
    JOIN entities s ON s.id = r.dst
    JOIN filings f ON f.accession = r.source_accession
    WHERE r.type = '{SUBJECT_RELATIONSHIP}' AND c.type IN ({",".join("?" * len(CLAIM_TYPES))})
      AND r.valid_until IS NULL AND c.valid_until IS NULL AND s.valid_until IS NULL
    ORDER BY r.id
"""


def _evidence_claims(conn: sqlite3.Connection) -> list[dict]:
    """Every distinct evidenced claim (sorted by key), with its excerpts, subject and attributes."""
    excerpts: dict[tuple, list[str]] = defaultdict(list)
    for acc, item, ctype, name, content in conn.execute(
            "SELECT accession, item, claim_type, claim_name, content FROM claim_evidence "
            "ORDER BY rowid"):
        excerpts[(acc, item, ctype, name)].append(content)
    found: dict[tuple, dict] = {}
    for acc, item, ctype, name, attrs, subject, ticker, fy in conn.execute(
            _CLAIMS_SQL, sorted(CLAIM_TYPES)):
        key = (acc, item, ctype, name)
        if key in excerpts and key not in found:  # first relationship row wins
            ex = excerpts[key]
            found[key] = {
                "accession": acc, "item": item, "claim_type": ctype, "name": name,
                "ticker": ticker, "fiscal_year": fy, "subject": subject,
                "attributes": json.loads(attrs),
                "excerpt": max(ex, key=lambda e: (len(e), -ex.index(e))),
                "excerpts": ex,
            }
    return [found[k] for k in sorted(found)]


def claim_counts_by_type(db: Path) -> dict[str, int]:
    """Distinct evidenced claims per type, most common first (ties by type name)."""
    with closing(_connect(db)) as conn:
        claims = _evidence_claims(conn)
    counts: dict[str, int] = defaultdict(int)
    for c in claims:
        counts[c["claim_type"]] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def default_per_type(counts: dict[str, int]) -> dict[str, int]:
    """Six claims for the 5 most common types, five for the rest (ranked by ``counts``)."""
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    return {t: PER_TOP_TYPE if i < TOP_TYPES else PER_OTHER_TYPE for i, t in enumerate(ranked)}


def sample_claims(db: Path, per_type: dict[str, int], seed: int) -> list[dict]:
    """Seeded stratified sample; ``claim_id``s ``ca01``.. follow the order of ``per_type``.

    Raises:
        ValueError: if a type has fewer evidenced claims than requested.
    """
    with closing(_connect(db)) as conn:
        claims = _evidence_claims(conn)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in claims:
        by_type[c["claim_type"]].append(c)
    rng = random.Random(seed)
    out: list[dict] = []
    for ctype, n in per_type.items():
        pool = by_type.get(ctype, [])
        if len(pool) < n:
            raise ValueError(f"{ctype}: wanted {n} claims, only {len(pool)} have evidence")
        out.extend(rng.sample(pool, n))
    for i, c in enumerate(out, 1):
        c["claim_id"] = f"ca{i:02d}"
    return [{"claim_id": c.pop("claim_id"), **c} for c in out]


def paragraph_around(section_text: str | None, excerpt: str) -> str:
    """Section text within +-600 chars of the excerpt's first match, excerpt wrapped in ``>>> <<<``.

    Both are whitespace-normalised before searching. If the excerpt is absent, the first 1200
    chars of the section, prefixed with ``[EXCERPT NOT FOUND IN SECTION] ``.
    """
    text = _norm(section_text or "")
    needle = _norm(excerpt)
    pos = text.find(needle) if needle else -1
    if pos < 0:
        return NOT_FOUND_PREFIX + text[:NOT_FOUND_CHARS]
    end = pos + len(needle)
    return (text[max(0, pos - CONTEXT_CHARS):pos] + ">>> " + needle + " <<<"
            + text[end:end + CONTEXT_CHARS])


def write_sheet(sample: list[dict], db: Path, out_csv: Path) -> None:
    """Write the labelling sheet; ``label`` and ``note`` are left blank for the human auditor."""
    with closing(_connect(db)) as conn:
        sections = {(c["accession"], c["item"]): conn.execute(
            "SELECT text FROM sections WHERE accession=? AND item=?",
            (c["accession"], c["item"])).fetchone() for c in sample}
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for c in sample:
            row = sections[(c["accession"], c["item"])]
            w.writerow({
                "claim_id": c["claim_id"], "claim_type": c["claim_type"], "ticker": c["ticker"],
                "fiscal_year": c["fiscal_year"], "item": c["item"], "name": c["name"],
                "subject": c["subject"],
                "attributes_json": json.dumps(c["attributes"], sort_keys=True),
                "excerpt": c["excerpt"],
                "paragraph": paragraph_around(row[0] if row else None, c["excerpt"]),
                "label": "", "note": "",
            })


def _git_commit() -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return res.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args(argv)
    meta = {"git_commit": _git_commit(), "built_at": datetime.now(UTC).isoformat(),
            "seed": args.seed}
    counts = claim_counts_by_type(args.db)
    per_type = default_per_type(counts)
    sample = sample_claims(args.db, per_type, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_sheet(sample, args.db, args.out_dir / "sheet.csv")
    (args.out_dir / "sample.json").write_text(json.dumps(
        {"meta": meta, "distinct_claims_by_type": counts, "per_type": per_type,
         "claims": sample}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "verbatim.json").write_text(json.dumps(
        {"meta": meta, **verbatim_rate(args.db)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
