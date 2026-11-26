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

from rapidfuzz import fuzz

from filings_kb.store import CLAIM_TYPES, SUBJECT_RELATIONSHIP

TOP_TYPES = 5
PER_TOP_TYPE = 6
PER_OTHER_TYPE = 5
CONTEXT_CHARS = 600
WEAK_MATCH_SCORE = 60
WEAK_MATCH_PREFIX = "[WEAK MATCH] "
SHEET_COLUMNS = ("claim_id", "claim_type", "ticker", "fiscal_year", "item", "name", "subject",
                 "attributes_json", "excerpt", "locate", "paragraph", "label", "note")

_WS = re.compile(r"\s+")
# One-for-one character folds, so offsets in folded text equal offsets in unfolded text.
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
})


def _connect(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _norm(text: str) -> str:
    """Collapse every run of whitespace (including ``\\xa0``) to one space and strip."""
    return _WS.sub(" ", text).strip()


def _sections(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Normalised section text keyed by ``(accession, item)``."""
    return {(a, i): _norm(t) for a, i, t in conn.execute("SELECT accession, item, text FROM sections")}


def _fold(norm_text: str) -> str:
    """Typographic fold of already whitespace-normalised text (``\\xa0`` is handled by ``_norm``)."""
    return norm_text.translate(_FOLD)


def _summary(rows: list[tuple], ok: list[bool]) -> dict:
    claims = {r[:4] for r in rows}
    bad_claims = {r[:4] for r, good in zip(rows, ok) if not good}
    failing = [{"accession": a, "item": i, "claim_type": ct, "claim_name": n, "content": c}
               for (a, i, ct, n, c), good in zip(rows, ok) if not good]
    return {"rows": len(rows), "rows_verbatim": len(rows) - len(failing),
            "claims": len(claims), "claims_with_non_verbatim": len(bad_claims),
            "not_verbatim": failing}


def verbatim_rate(db: Path) -> dict:
    """Containment of every evidence excerpt in its section text, two ways.

    ``whitespace``: whitespace-normalised containment. ``fold``: the same after folding curly
    quotes, en/em dashes and the minus sign to ASCII (applied to excerpt and section alike).
    Rows whose ``(accession, item)`` has no row in ``sections`` are listed under
    ``no_public_section`` and excluded from both denominators.
    """
    with closing(_connect(db)) as conn:
        sections = _sections(conn)
        evidence = conn.execute(
            "SELECT accession, item, claim_type, claim_name, content FROM claim_evidence "
            "ORDER BY accession, item, claim_type, claim_name, content").fetchall()
    present = [r for r in evidence if (r[0], r[1]) in sections]
    missing = [r for r in evidence if (r[0], r[1]) not in sections]
    folded: dict[tuple[str, str], str] = {}
    ws_ok, fold_ok = [], []
    for acc, item, _ct, _n, content in present:
        text = sections[(acc, item)]
        needle = _norm(content)
        ws_ok.append(needle in text)
        if (acc, item) not in folded:
            folded[(acc, item)] = _fold(text)
        fold_ok.append(_fold(needle) in folded[(acc, item)])
    return {
        "rows_total": len(evidence),
        "no_public_section": {
            "rows": len(missing), "claims": len({r[:4] for r in missing}),
            "items": [{"accession": a, "item": i, "claim_type": ct, "claim_name": n,
                       "content": c} for a, i, ct, n, c in missing]},
        "whitespace": _summary(present, ws_ok),
        "fold": _summary(present, fold_ok),
    }


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


def _evidence_claims(conn: sqlite3.Connection, *, auditable: bool = True) -> list[dict]:
    """Distinct evidenced claims (sorted by key), with excerpts, subject and attributes.

    With ``auditable`` (default) only claims whose section exists in ``sections`` are kept;
    otherwise only the claims whose section is missing.
    """
    have_section = {(a, i) for a, i in conn.execute("SELECT accession, item FROM sections")}
    excerpts: dict[tuple, list[str]] = defaultdict(list)
    for acc, item, ctype, name, content in conn.execute(
            "SELECT accession, item, claim_type, claim_name, content FROM claim_evidence "
            "ORDER BY rowid"):
        excerpts[(acc, item, ctype, name)].append(content)
    found: dict[tuple, dict] = {}
    for acc, item, ctype, name, attrs, subject, ticker, fy in conn.execute(
            _CLAIMS_SQL, sorted(CLAIM_TYPES)):
        key = (acc, item, ctype, name)
        if key in excerpts and key not in found and (((acc, item) in have_section) == auditable):  # first relationship row wins
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


def excluded_no_public_section(db: Path) -> int:
    """Distinct evidenced claims (with a valid entity) dropped because their section is absent."""
    with closing(_connect(db)) as conn:
        return len(_evidence_claims(conn, auditable=False))


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


def locate(excerpt: str, section_text: str | None) -> tuple[str, str]:
    """Find ``excerpt`` in the section and return ``(paragraph, locate_kind)``.

    Tries, in order, an exact match on whitespace-normalised text (``exact``), the typographic
    fold (``fold``), then ``rapidfuzz.fuzz.partial_ratio_alignment`` (``fuzzy:<score>``). The
    paragraph is the located span +-600 chars of normalised section text, the span wrapped in
    ``>>> <<<``; a fuzzy score below 60 prefixes it with ``[WEAK MATCH] ``.
    """
    text = _norm(section_text or "")
    needle = _norm(excerpt)
    folded_text = _fold(text)
    folded_needle = _fold(needle)
    kind = "exact"
    pos = text.find(needle) if needle else -1
    end = pos + len(needle)
    if pos < 0:
        kind = "fold"
        pos = folded_text.find(folded_needle) if needle else -1
        end = pos + len(needle)
    if pos < 0:
        al = fuzz.partial_ratio_alignment(folded_needle, folded_text)
        score = int(al.score) if al else 0
        kind = f"fuzzy:{score}"
        pos, end = (al.dest_start, al.dest_end) if al else (0, 0)
    body = (text[max(0, pos - CONTEXT_CHARS):pos] + ">>> " + text[pos:end] + " <<<"
            + text[end:end + CONTEXT_CHARS])
    if kind.startswith("fuzzy:") and int(kind.split(":")[1]) < WEAK_MATCH_SCORE:
        body = WEAK_MATCH_PREFIX + body
    return body, kind


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
            para, kind = locate(c["excerpt"], row[0] if row else None)
            w.writerow({
                "claim_id": c["claim_id"], "claim_type": c["claim_type"], "ticker": c["ticker"],
                "fiscal_year": c["fiscal_year"], "item": c["item"], "name": c["name"],
                "subject": c["subject"],
                "attributes_json": json.dumps(c["attributes"], sort_keys=True),
                "excerpt": c["excerpt"],
                "locate": kind, "paragraph": para,
                "label": "", "note": "",
            })


def _git_commit() -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return res.stdout.strip() or "unknown"


def _db_counts(db: Path) -> dict[str, int]:
    with closing(_connect(db)) as conn:
        return {tbl: conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                for tbl in ("claim_evidence", "sections", "chunks")}


def _dirty() -> bool:
    res = subprocess.run(["git", "status", "--porcelain", "--", "evals/v2"],
                         capture_output=True, text=True, check=False)
    return bool(res.stdout.strip())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args(argv)
    meta = {"excluded_no_public_section": excluded_no_public_section(args.db),
            "git_commit": _git_commit(), "dirty": _dirty(),
            "db_counts": _db_counts(args.db), "built_at": datetime.now(UTC).isoformat(),
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
