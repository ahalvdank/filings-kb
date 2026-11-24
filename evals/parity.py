"""Parity of the public pipeline with the author's private KB.

Three checks, written to ``exports/parity.json`` as counts only (no claim text):

* **Section-text fidelity.** Of the extraction sections whose text the public detector
  and the private pipeline both produced as one piece, the share with the same
  ``text_sha256``. The numbers come from the ingest report, which computes them while
  ingesting. The rate is always quoted with its coverage (comparable N of M sections).
* **Matched claim recall.** Raw claim counts are not comparable: the public store keeps
  one claim per filing section (a segment named in Items 1, 7 and 15 is three claims),
  while the private KB merges claims into entities per filing. So, per filing and claim
  type, the claim names on each side are clustered with rapidfuzz ``WRatio`` at the
  linker's ``link_auto`` threshold (greedy: names sorted, each joins the first cluster
  whose representative scores at or above it, so it is not transitive). A cluster is
  matched when one of its names scores at or above the threshold against a name on the
  other side, in the same filing. Recall is matched clusters over clusters, in both
  directions, summed over filings. A private claim belongs to a filing through an open
  relationship (``valid_until IS NULL``) whose ``source_document`` is the filing's
  accession document ``filings/<TICKER>/<YYYY-MM>_<form>_<accession>.md`` or its
  accession-less manifest ``filings/<TICKER>/<YYYY-MM>_<form>.md``, which carries claims
  of its own. Each private entity also has recorded ``aliases``. A name matches a name
  with ``WRatio`` at ``link_auto``; an alias matches a name from the other side with
  ``token_sort_ratio >= 90`` only (no partial-ratio fallback, which would let a short
  alias match any longer name containing it). Recall is also computed on names alone, to
  show what the aliases add. Raw counts are kept as diagnostics only.
* **Ingest vs cache.** The public store against the committed extraction envelopes it
  was built from: the two must agree exactly, apart from claims closed out by the 10-K/A.

Run: ``python -m evals.parity --private-kb PATH_TO_PRIVATE_KB_SQLITE``
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, utils

from filings_kb.config import load_sample
from filings_kb.extract import _merge_parts, read_envelope
from filings_kb.link import load_link_thresholds
from filings_kb.store import CLAIM_TYPES
from filings_kb.types import SampleRow

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_FIDELITY_PCT = 90
MIN_RECALL_PCT = 80
MIN_N = 20  # recall below MIN_RECALL_PCT fails only for a type with at least this many private clusters
ALIAS_MIN_SCORE = 90  # token_sort_ratio for an alias against a name
# The gate is both directions: recall below MIN_RECALL_PCT either way fails a type with
# at least MIN_N private clusters (n is the private cluster count).


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open a SQLite file read-only (``mode=ro``); the file is never written.

    Raises:
        FileNotFoundError: if ``path`` does not exist (``mode=ro`` would not create it).
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _claim_marks() -> tuple[str, list[str]]:
    types = sorted(CLAIM_TYPES)
    return ",".join("?" * len(types)), types


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def _filing_documents(conn: sqlite3.Connection, row: SampleRow) -> list[str]:
    """The filing's accession documents: ``filings/<TICKER>/*_<accession without dashes>.md``."""
    pattern = f"filings/{row.ticker}/*_{row.accession.replace('-', '')}.md"
    return [d for (d,) in conn.execute(
        "SELECT DISTINCT source_document FROM relationships WHERE source_document GLOB ? "
        "ORDER BY source_document", (pattern,))]


_DOCUMENT = re.compile(r"^(?P<prefix>filings/[^/]+/\d{4}-\d{2})_.+_\d+\.md$")


def manifest_paths(documents: list[str], row: SampleRow) -> list[str]:
    """The accession-less manifests of the filing's documents: ``<prefix>_<form>.md``.

    The form is the row's own (``10-K/A`` keeps its slash: the private KB names the
    amendment's manifest ``2026-02_10-K/A.md`` but its accession document
    ``2026-02_10-K-A_<accession>.md``), so a manifest is never taken from another form.
    """
    out: list[str] = []
    for doc in documents:
        m = _DOCUMENT.match(doc)
        if m:
            path = f"{m.group('prefix')}_{row.form}.md"
            if path not in out:
                out.append(path)
    return out


@dataclass(frozen=True)
class PrivateClaim:
    """A private claim entity: its canonical name and recorded aliases."""

    name: str
    aliases: tuple[str, ...] = ()


def _parse_aliases(raw: str | None, entity_id: str) -> tuple[str, ...]:
    """The strings of a private ``aliases`` JSON list; anything else logs a warning and is ignored."""
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("private entity %s: aliases is not valid JSON, ignored", entity_id)
        return ()
    if not isinstance(parsed, list):
        log.warning("private entity %s: aliases is not a list, ignored", entity_id)
        return ()
    return tuple(a for a in parsed if isinstance(a, str) and a.strip())


def private_claims(conn: sqlite3.Connection, row: SampleRow, *,
                   include_manifests: bool = True) -> dict[str, dict[str, PrivateClaim]]:
    """Open claim entities of one filing, per type: ``{type: {entity id: PrivateClaim}}``.

    A claim counts when an open relationship (``valid_until IS NULL``) touches it with
    ``source_document`` the filing's accession document or, with ``include_manifests``,
    its accession-less manifest. Entities are filtered on ``valid_until`` too when the
    schema has that column (the current private schema does not, so the filter reduces
    to the relationship; closed-out re-ingests are closed there).
    """
    marks, types = _claim_marks()
    entity_open = " AND e.valid_until IS NULL" if _has_column(conn, "entities", "valid_until") \
        else ""
    aliases_sql = "e.aliases" if _has_column(conn, "entities", "aliases") else "NULL"
    sql = f"""
        SELECT DISTINCT e.type, e.id, e.name, {aliases_sql}
        FROM entities e
        JOIN relationships r ON r.source_id = e.id OR r.target_id = e.id
        WHERE e.type IN ({marks}){entity_open}
          AND r.valid_until IS NULL AND r.source_document = ?
    """
    docs = _filing_documents(conn, row)
    if include_manifests:
        docs += [m for m in manifest_paths(docs, row) if m not in docs]
    out: dict[str, dict[str, PrivateClaim]] = defaultdict(dict)
    for doc in docs:
        for claim_type, entity_id, name, aliases in conn.execute(sql, (*types, doc)):
            out[claim_type][entity_id] = PrivateClaim(name, _parse_aliases(aliases, entity_id))
    return dict(out)


def public_claim_names(conn: sqlite3.Connection, row: SampleRow) -> dict[str, list[str]]:
    """Names of the filing's open claims, per type, one entry per claim (per section)."""
    marks, types = _claim_marks()
    out: dict[str, list[str]] = defaultdict(list)
    for claim_type, name in conn.execute(f"""
            SELECT c.type, c.name
            FROM relationships r JOIN entities c ON c.id = r.src
            WHERE r.type = 'subject_of' AND r.source_accession = ? AND c.type IN ({marks})
              AND r.valid_until IS NULL AND c.valid_until IS NULL
            ORDER BY c.id""", (row.accession, *types)):
        out[claim_type].append(name)
    return dict(out)


def _similar(a: str, b: str, threshold: float) -> bool:
    return fuzz.WRatio(a, b, processor=utils.default_process) >= threshold


def _alias_similar(alias: str, name: str) -> bool:
    """An alias matches a name at ``token_sort_ratio >= ALIAS_MIN_SCORE``: word order is
    ignored, but a short alias cannot match a longer name that merely contains it."""
    return fuzz.token_sort_ratio(alias, name, processor=utils.default_process) >= ALIAS_MIN_SCORE


def cluster_names(names: Iterable[str], threshold: float) -> list[list[str]]:
    """Greedy clusters of ``names`` at ``WRatio >= threshold`` (the linker's scorer).

    Names are sorted (case-folded, then as written) so the result does not depend on
    input order; each joins the first cluster whose representative (its first name)
    scores at or above ``threshold``, else starts a new one. Exact repeats always join.
    Greedy clustering is not transitive: A~B and B~C can still leave A and C apart.
    """
    clusters: list[list[str]] = []
    for name in sorted(names, key=lambda n: (n.casefold(), n)):
        for cluster in clusters:
            if name == cluster[0] or _similar(name, cluster[0], threshold):
                cluster.append(name)
                break
        else:
            clusters.append([name])
    return clusters


def matched_clusters(clusters: list[list[str]], others: Iterable[str], threshold: float, *,
                     cluster_aliases: list[list[str]] | None = None,
                     other_aliases: Iterable[str] = ()) -> int:
    """How many ``clusters`` find a match on the other side.

    A cluster matches when a name of it scores ``>= threshold`` (``WRatio``) against one
    of ``others``, or when an alias on either side (``cluster_aliases[i]`` for cluster
    ``i``, or ``other_aliases``) matches a name on the opposite side by ``_alias_similar``.
    Without aliases it is the names-only match.
    """
    pool = list(others)
    foreign_aliases = list(other_aliases)
    aliases = cluster_aliases or [[] for _ in clusters]
    return sum(
        1 for cluster, own in zip(clusters, aliases, strict=True)
        if any(_similar(a, b, threshold) for a in cluster for b in pool)
        or any(_alias_similar(alias, b) for alias in own for b in pool)
        or any(_alias_similar(alias, a) for alias in foreign_aliases for a in cluster))


@dataclass(frozen=True)
class TypeRecall:
    """Matched recall for one claim type, summed over the filings.

    ``private_in_public``: private clusters with a public match, over private clusters;
    ``public_in_private`` the other way. The ``*_names_only`` counts leave aliases out.
    ``n`` is the private cluster count. A type with no private or no public claims is not
    comparable (recall undefined). ``failing``: ``n >= MIN_N`` and either direction's
    recall below ``MIN_RECALL_PCT``.
    """

    claim_type: str
    public_clusters: int
    private_clusters: int
    public_matched: int
    private_matched: int
    public_matched_names_only: int | None = None
    private_matched_names_only: int | None = None

    @property
    def n(self) -> int:
        return self.private_clusters

    @property
    def comparable(self) -> bool:
        return self.private_clusters > 0 and self.public_clusters > 0

    @property
    def private_in_public_pct(self) -> float | None:
        return _pct(self.private_matched, self.private_clusters) if self.comparable else None

    @property
    def public_in_private_pct(self) -> float | None:
        return _pct(self.public_matched, self.public_clusters) if self.comparable else None

    @property
    def private_in_public_names_only_pct(self) -> float | None:
        if not self.comparable or self.private_matched_names_only is None:
            return None
        return _pct(self.private_matched_names_only, self.private_clusters)

    @property
    def public_in_private_names_only_pct(self) -> float | None:
        if not self.comparable or self.public_matched_names_only is None:
            return None
        return _pct(self.public_matched_names_only, self.public_clusters)

    @property
    def failing(self) -> bool:
        if self.n < MIN_N:
            return False
        private_low = self.private_matched * 100 < MIN_RECALL_PCT * self.private_clusters
        public_low = (self.public_clusters > 0
                      and self.public_matched * 100 < MIN_RECALL_PCT * self.public_clusters)
        return private_low or public_low


@dataclass(frozen=True)
class ClaimComparison:
    """Recall per type plus the raw diagnostic counts (summed over filings)."""

    recall: list[TypeRecall]
    public_occurrences: dict[str, int]
    private_accession_doc: dict[str, int]
    private_with_manifest: dict[str, int]


def compare_claims(public: sqlite3.Connection, private: sqlite3.Connection,
                   rows: list[SampleRow], threshold: float) -> ClaimComparison:
    """Cluster and match claim names per filing and type; see the module docstring."""
    tally: dict[str, Counter[str]] = defaultdict(Counter)
    occurrences: Counter[str] = Counter()
    doc_only: Counter[str] = Counter()
    with_manifest: Counter[str] = Counter()
    for row in rows:
        pub = public_claim_names(public, row)
        priv = private_claims(private, row)
        for claim_type, ids in private_claims(private, row, include_manifests=False).items():
            doc_only[claim_type] += len(ids)
        for claim_type in set(pub) | set(priv):
            pub_names = pub.get(claim_type, [])
            priv_claims = list(priv.get(claim_type, {}).values())
            priv_names = [c.name for c in priv_claims]
            aliases_of: dict[str, list[str]] = defaultdict(list)
            for c in priv_claims:
                aliases_of[c.name].extend(c.aliases)
            all_aliases = [a for c in priv_claims for a in c.aliases]
            occurrences[claim_type] += len(pub_names)
            with_manifest[claim_type] += len(priv_names)
            pub_clusters = cluster_names(pub_names, threshold)
            priv_clusters = cluster_names(priv_names, threshold)
            priv_cluster_aliases = [[a for n in cluster for a in aliases_of[n]]
                                    for cluster in priv_clusters]
            t = tally[claim_type]
            t["public_clusters"] += len(pub_clusters)
            t["private_clusters"] += len(priv_clusters)
            t["public_matched"] += matched_clusters(pub_clusters, priv_names, threshold,
                                                    other_aliases=all_aliases)
            t["private_matched"] += matched_clusters(priv_clusters, pub_names, threshold,
                                                     cluster_aliases=priv_cluster_aliases)
            t["public_names_only"] += matched_clusters(pub_clusters, priv_names, threshold)
            t["private_names_only"] += matched_clusters(priv_clusters, pub_names, threshold)
    recall = [TypeRecall(claim_type, t["public_clusters"], t["private_clusters"],
                         t["public_matched"], t["private_matched"],
                         t["public_names_only"], t["private_names_only"])
              for claim_type, t in sorted(tally.items())]
    return ClaimComparison(recall, dict(occurrences), dict(doc_only), dict(with_manifest))


def private_unlinked_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Claim entities per type that no relationship touches, anywhere in the private KB."""
    marks, types = _claim_marks()
    return dict(conn.execute(f"""
        SELECT e.type, COUNT(*) FROM entities e
        WHERE e.type IN ({marks})
          AND NOT EXISTS (SELECT 1 FROM relationships r
                          WHERE r.source_id = e.id OR r.target_id = e.id)
        GROUP BY e.type""", types).fetchall())


def public_claim_counts(conn: sqlite3.Connection, rows: list[SampleRow], *,
                        open_only: bool = True) -> dict[str, int]:
    """Claim entities per type with a ``subject_of`` edge from the sampled filings.

    ``open_only`` keeps claims and edges with ``valid_until IS NULL``; without it,
    closed-out (superseded) claims are counted too.
    """
    marks, types = _claim_marks()
    open_sql = " AND r.valid_until IS NULL AND c.valid_until IS NULL" if open_only else ""
    sql = f"""
        SELECT c.type, COUNT(DISTINCT c.id)
        FROM relationships r JOIN entities c ON c.id = r.src
        WHERE r.type = 'subject_of' AND r.source_accession = ? AND c.type IN ({marks}){open_sql}
        GROUP BY c.type
    """
    counts: Counter[str] = Counter()
    for row in rows:
        for claim_type, n in conn.execute(sql, (row.accession, *types)):
            counts[claim_type] += n
    return dict(counts)


def cache_claim_counts(extractions_dir: Path, rows: list[SampleRow]) -> dict[str, int]:
    """Claim entities per type in the committed envelopes, parts merged as ingest merges them."""
    counts: Counter[str] = Counter()
    for row in rows:
        for path in sorted((extractions_dir / row.accession).glob("*.json")):
            for entity in _merge_parts(read_envelope(path).parts).entities:
                if entity.type in CLAIM_TYPES:
                    counts[entity.type] += 1
    return dict(counts)


def _pct(numerator: int, denominator: int) -> float:
    return float((Decimal(100 * numerator) / Decimal(denominator)).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP))


def fidelity_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Section-text fidelity with coverage, from an ingest report's ``fidelity.overall``.

    ``sections`` is every extraction section the check looked at: comparable ones plus
    those not comparable (the private pipeline split the section) or not detected by the
    public detector. ``match_rate_pct`` is over the comparable ones only, so it must be
    read with ``coverage_pct`` (comparable of all).

    Raises:
        ValueError: if no section was comparable.
    """
    o = report["fidelity"]["overall"]
    comparable = int(o["comparable"])
    if comparable == 0:
        raise ValueError("no comparable sections: fidelity is undefined")
    sections = comparable + int(o["not_comparable_split"]) + int(o["not_detected"])
    return {
        "match": int(o["match"]), "mismatch": int(o["mismatch"]), "comparable": comparable,
        "not_comparable_split": int(o["not_comparable_split"]),
        "not_detected": int(o["not_detected"]), "sections": sections,
        "match_rate_pct": _pct(int(o["match"]), comparable),
        "coverage_pct": _pct(comparable, sections),
    }


def _format_table(comparison: ClaimComparison, unlinked: dict[str, int]) -> str:
    lines = [f"{'claim type':<24}{'clusters pub/priv':>18}{'priv->pub':>11}{'pub->priv':>11}"
             f"{'(names)':>9}{'(names)':>9}  "
             f"{'raw: pub occ':>12}{'priv doc':>9}{'+manifest':>10}{'unlinked':>9}"]
    for r in comparison.recall:
        def pct(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.1f}%"
        mark = "FAIL" if r.failing else "" if r.comparable else "not comparable"
        t = r.claim_type
        lines.append(
            f"{t:<24}{f'{r.public_clusters}/{r.private_clusters}':>18}"
            f"{pct(r.private_in_public_pct):>11}{pct(r.public_in_private_pct):>11}"
            f"{pct(r.private_in_public_names_only_pct):>9}"
            f"{pct(r.public_in_private_names_only_pct):>9}  "
            f"{comparison.public_occurrences.get(t, 0):>12}"
            f"{comparison.private_accession_doc.get(t, 0):>9}"
            f"{comparison.private_with_manifest.get(t, 0):>10}{unlinked.get(t, 0):>9}  {mark}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the parity tables and write the counts-only JSON.

    Returns:
        1 when section-text fidelity is below 90%, the public store disagrees with the
        committed extractions it was built from (both bugs in this repo), or matched
        recall in either direction is below 80% for a type with at least 20 private
        clusters; else 0.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--private-kb", type=Path, required=True)
    ap.add_argument("--public-kb", type=Path, default=REPO_ROOT / "data" / "kb.sqlite")
    ap.add_argument("--sample", type=Path, default=REPO_ROOT / "config" / "sample.yaml")
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "data" / "committed" / "ingest_report.json")
    ap.add_argument("--extractions", type=Path,
                    default=REPO_ROOT / "data" / "committed" / "extractions")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "exports" / "parity.json")
    args = ap.parse_args(argv)

    rows = load_sample(args.sample).rows
    threshold, _ = load_link_thresholds()
    fidelity = fidelity_summary(json.loads(args.report.read_text(encoding="utf-8")))
    private = open_readonly(args.private_kb)
    public = open_readonly(args.public_kb)
    try:
        comparison = compare_claims(public, private, rows, threshold)
        pub_open = public_claim_counts(public, rows)
        pub_all = public_claim_counts(public, rows, open_only=False)
        unlinked = private_unlinked_counts(private)
    finally:
        private.close()
        public.close()
    cache = cache_claim_counts(args.extractions, rows) if args.extractions.is_dir() else {}
    ingest = {t: {"cache": cache.get(t, 0), "public_all": pub_all.get(t, 0),
                  "public_open": pub_open.get(t, 0)} for t in sorted(CLAIM_TYPES)}
    lossy = sorted(t for t, c in ingest.items() if c["cache"] != c["public_all"])
    failing = [r.claim_type for r in comparison.recall if r.failing]

    result = {
        "filings": len(rows),
        "fidelity": fidelity,
        "claims": {r.claim_type: {
            "public_clusters": r.public_clusters, "private_clusters": r.private_clusters,
            "public_matched": r.public_matched, "private_matched": r.private_matched,
            "recall_private_in_public_pct": r.private_in_public_pct,
            "recall_public_in_private_pct": r.public_in_private_pct,
            "public_matched_names_only": r.public_matched_names_only,
            "private_matched_names_only": r.private_matched_names_only,
            "recall_private_in_public_names_only_pct": r.private_in_public_names_only_pct,
            "recall_public_in_private_names_only_pct": r.public_in_private_names_only_pct,
            "n": r.n, "comparable": r.comparable, "failing": r.failing,
            "diagnostic_raw_counts": {
                "public_claims": comparison.public_occurrences.get(r.claim_type, 0),
                "private_accession_doc": comparison.private_accession_doc.get(r.claim_type, 0),
                "private_with_manifest": comparison.private_with_manifest.get(r.claim_type, 0),
                "private_unlinked_entities": unlinked.get(r.claim_type, 0),
            }} for r in comparison.recall},
        "ingest_vs_cache": ingest,
        "rule": {"cluster_threshold_wratio": threshold, "alias_min_token_sort_ratio": ALIAS_MIN_SCORE,
                 "gated_direction": "both", "n": "private_clusters",
                 "min_recall_pct": MIN_RECALL_PCT,
                 "min_n": MIN_N, "min_fidelity_pct": MIN_FIDELITY_PCT},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"section-text fidelity: {fidelity['match']} of {fidelity['comparable']} comparable "
          f"sections match ({fidelity['match_rate_pct']}%); comparable "
          f"{fidelity['comparable']} of {fidelity['sections']} sections "
          f"({fidelity['coverage_pct']}%)")
    print(f"matched recall (names + aliases; names only in brackets), per-filing WRatio "
          f"clusters at {threshold:g}, aliases at token_sort_ratio >= {ALIAS_MIN_SCORE}:")
    print(_format_table(comparison, unlinked))
    for t, c in ingest.items():
        print(f"ingest vs cache  {t:<24} cache={c['cache']:>5} public_all={c['public_all']:>5} "
              f"public_open={c['public_open']:>5}")
    print(f"wrote {args.out}")
    if lossy:
        print(f"INGEST LOSES OR INVENTS CLAIMS vs the cache: {', '.join(lossy)}")
    if failing:
        print(f"RECALL BELOW {MIN_RECALL_PCT}% IN A DIRECTION (private clusters >= {MIN_N}): "
              f"{', '.join(failing)}")
    return 1 if lossy or failing or fidelity["match_rate_pct"] < MIN_FIDELITY_PCT else 0


if __name__ == "__main__":
    sys.exit(main())
