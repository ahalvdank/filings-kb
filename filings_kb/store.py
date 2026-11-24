"""Bitemporal SQLite store: filings, sections, chunks, entities, relationships, runs.

Claims are entities of one of ``CLAIM_TYPES``; a claim's subject is reached through
a ``subject_of`` relationship (claim -> subject entity). Nothing is ever deleted:
``close_out`` stamps ``valid_until``. Costs are stored as TEXT holding a Decimal so
no float rounding can occur; timestamps are ISO-8601 UTC strings.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from filings_kb.chunking import Chunk
from filings_kb.sections import Section
from filings_kb.types import FilingRef

CLAIM_TYPES: frozenset[str] = frozenset({
    "Risk",
    "MaterialEvent",
    "CapitalStructureEvent",
    "LegalProceeding",
    "CapitalReturn",
    "GuidanceUpdate",
    "RevenueSegment",
    "ESGCommitment",
    "InsiderTransaction",
})

SUBJECT_RELATIONSHIP = "subject_of"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    cik TEXT NOT NULL,
    company TEXT NOT NULL,
    form TEXT NOT NULL,
    filed TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_year_end TEXT NOT NULL,
    primary_doc TEXT NOT NULL,
    html_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sections (
    accession TEXT NOT NULL REFERENCES filings(accession),
    item TEXT NOT NULL,
    ticker TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (accession, item)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    accession TEXT NOT NULL,
    item TEXT NOT NULL,
    ticker TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (accession, item) REFERENCES sections(accession, item)
);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    identifiers TEXT NOT NULL DEFAULT '{}',
    attributes TEXT NOT NULL DEFAULT '{}',
    valid_from TEXT NOT NULL,
    valid_until TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL REFERENCES entities(id),
    dst TEXT NOT NULL REFERENCES entities(id),
    type TEXT NOT NULL,
    source_accession TEXT NOT NULL,
    source_item TEXT NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    valid_from TEXT NOT NULL,
    valid_until TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_accession ON relationships(source_accession);
CREATE TABLE IF NOT EXISTS manual_review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS claim_evidence (
    accession TEXT NOT NULL,
    item TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    claim_name TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (accession, item, claim_type, claim_name, content)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    key TEXT NOT NULL,
    status TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ClaimRow:
    """A currently valid claim entity joined to its subject and source location."""

    entity_id: str
    claim_type: str
    name: str
    attributes: dict[str, Any]
    subject_id: str
    subject_type: str
    subject_name: str
    accession: str
    item: str


@dataclass(frozen=True)
class EntityRow:
    """A currently valid entity as seen by the linker."""

    id: str
    type: str
    name: str
    identifiers: dict[str, Any]
    attributes: dict[str, Any]


@dataclass(frozen=True)
class ReviewRow:
    """An entity queued for manual review of a possible duplicate."""

    entity_id: str
    reason: str
    resolved: bool


@dataclass(frozen=True)
class RunRow:
    """One logged pipeline run (extraction call, import, etc.)."""

    stage: str
    key: str
    status: str
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    detail: str
    created_at: str


def _now() -> str:
    """Current time as an ISO-8601 UTC string."""
    return datetime.now(UTC).isoformat()


class Store:
    """SQLite-backed knowledge store; every method opens a short-lived connection."""

    def __init__(self, path: Path) -> None:
        """Open (creating if needed) the database at ``path`` and ensure the schema."""
        self.path = Path(path)
        self._tx: sqlite3.Connection | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with foreign keys on; commit on success, roll back on error.

        Inside ``transaction()`` the transaction's connection is yielded instead and
        commit/rollback is left to the transaction.
        """
        if self._tx is not None:
            yield self._tx
            return
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run every store call in the block on one connection, atomically.

        All writes commit together when the block exits normally and are rolled
        back if it raises. Calls outside a transaction keep their own commits.

        Raises:
            RuntimeError: if a transaction is already open (no nesting).
        """
        if self._tx is not None:
            raise RuntimeError("transactions do not nest")
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            self._tx = conn
            with conn:
                yield
        finally:
            self._tx = None
            conn.close()

    def put_filing(self, f: FilingRef) -> None:
        """Insert or replace a filing keyed by accession."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO filings VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f.accession, f.ticker, f.cik, f.company, f.form, f.filed.isoformat(),
                 f.fiscal_year, f.fiscal_year_end.isoformat(), f.primary_doc,
                 str(f.html_path)),
            )

    def put_section(self, s: Section) -> None:
        """Insert or update a section keyed by (accession, item).

        Raises:
            sqlite3.IntegrityError: if the filing has not been stored.
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sections VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(accession, item) DO UPDATE SET ticker=excluded.ticker, "
                "fiscal_year=excluded.fiscal_year, title=excluded.title, "
                "text=excluded.text",
                (s.accession, s.item, s.ticker, s.fiscal_year, s.title, s.text),
            )

    def put_chunks(self, cs: list[Chunk]) -> None:
        """Insert or update chunks keyed by chunk_id.

        Raises:
            sqlite3.IntegrityError: if a chunk's section has not been stored.
        """
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET accession=excluded.accession, "
                "item=excluded.item, ticker=excluded.ticker, "
                "fiscal_year=excluded.fiscal_year, start_offset=excluded.start_offset, "
                "end_offset=excluded.end_offset, text=excluded.text",
                [(c.chunk_id, c.accession, c.item, c.ticker, c.fiscal_year, c.start,
                  c.end, c.text) for c in cs],
            )

    def put_entity(
        self, type: str, name: str, attributes: dict[str, Any],
        identifiers: dict[str, Any],
    ) -> str:
        """Create an entity valid from now; return its uuid."""
        entity_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO entities (id, type, name, identifiers, attributes, "
                "valid_from) VALUES (?,?,?,?,?,?)",
                (entity_id, type, name, json.dumps(identifiers, sort_keys=True),
                 json.dumps(attributes, sort_keys=True), _now()),
            )
        return entity_id

    def add_relationship(
        self, src: str, dst: str, type: str, source_accession: str, source_item: str,
        attributes: dict[str, Any],
    ) -> None:
        """Link two existing entities, recording where the link was sourced.

        An identical open relationship (same endpoints, type, source and
        attributes) is not inserted twice.

        Raises:
            sqlite3.IntegrityError: if ``src`` or ``dst`` is not a stored entity.
        """
        attrs = json.dumps(attributes, sort_keys=True)
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM relationships WHERE src=? AND dst=? AND type=? "
                "AND source_accession=? AND source_item=? AND attributes=? "
                "AND valid_until IS NULL",
                (src, dst, type, source_accession, source_item, attrs),
            ).fetchone()
            if exists:
                return
            conn.execute(
                "INSERT INTO relationships (src, dst, type, source_accession, "
                "source_item, attributes, valid_from) VALUES (?,?,?,?,?,?,?)",
                (src, dst, type, source_accession, source_item, attrs, _now()),
            )

    def entities_of_type(self, type: str) -> list[EntityRow]:
        """Return currently valid entities of ``type`` in creation order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, type, name, identifiers, attributes FROM entities "
                "WHERE type = ? AND valid_until IS NULL ORDER BY rowid", (type,)
            ).fetchall()
        return [EntityRow(r[0], r[1], r[2], json.loads(r[3]), json.loads(r[4]))
                for r in rows]

    def add_manual_review(self, entity_id: str, reason: str) -> None:
        """Queue ``entity_id`` for manual review.

        Raises:
            sqlite3.IntegrityError: if the entity is not stored.
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO manual_review (entity_id, reason, created_at) "
                "VALUES (?,?,?)", (entity_id, reason, _now()),
            )

    def manual_reviews(self) -> list[ReviewRow]:
        """Return every manual-review row in insertion order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT entity_id, reason, resolved_at FROM manual_review ORDER BY id"
            ).fetchall()
        return [ReviewRow(r[0], r[1], r[2] is not None) for r in rows]

    def count_rows(self, table: str, accessions: list[str]) -> int:
        """Count ``filings``, ``sections`` or ``chunks`` rows for ``accessions``.

        Raises:
            ValueError: if ``table`` is not one of those three.
        """
        if table not in {"filings", "sections", "chunks"}:
            raise ValueError(f"cannot count table {table!r}")
        marks = ",".join("?" * len(accessions))
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE accession IN ({marks})", accessions
            ).fetchone()
        return int(row[0])

    def close_out(self, entity_id: str, at: datetime) -> None:
        """Mark an entity no longer valid as of ``at``; the row is never deleted.

        An entity that is already closed keeps its first ``valid_until`` (the call
        is a no-op), so history is never rewritten.

        Raises:
            ValueError: if ``at`` is timezone-naive.
            KeyError: if the entity does not exist.
        """
        if at.tzinfo is None:
            raise ValueError("close_out requires a timezone-aware datetime")
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE entities SET valid_until=? WHERE id=? AND valid_until IS NULL",
                (at.astimezone(UTC).isoformat(), entity_id),
            )
            if cur.rowcount == 0 and conn.execute(
                    "SELECT 1 FROM entities WHERE id=?", (entity_id,)).fetchone() is None:
                raise KeyError(f"unknown entity: {entity_id}")

    def section_accession(self, ticker: str, fiscal_year: int, item: str) -> str | None:
        """Accession whose ``item`` section is current for ``(ticker, fiscal_year)``.

        A 10-K/A that carries the section supersedes the original's; otherwise the
        original is used. Returns None when no filing has the section.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT s.accession FROM sections s JOIN filings f ON f.accession = s.accession "
                "WHERE s.ticker=? AND s.fiscal_year=? AND s.item=? "
                "ORDER BY (f.form = '10-K/A') DESC, f.filed DESC, s.accession DESC LIMIT 1",
                (ticker, fiscal_year, item),
            ).fetchone()
        return row[0] if row else None

    def chunk_texts(self, accession: str, item: str) -> list[str]:
        """Return the texts of ``(accession, item)``'s chunks in document order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE accession=? AND item=? "
                "ORDER BY start_offset, chunk_id",
                (accession, item),
            ).fetchall()
        return [r[0] for r in rows]

    def filing(self, accession: str) -> FilingRef | None:
        """Return the stored filing for ``accession``, or None."""
        with self._conn() as conn:
            r = conn.execute(
                "SELECT ticker, cik, company, accession, form, filed, fiscal_year, "
                "fiscal_year_end, primary_doc, html_path FROM filings WHERE accession=?",
                (accession,),
            ).fetchone()
        if r is None:
            return None
        return FilingRef(
            ticker=r[0], cik=r[1], company=r[2], accession=r[3], form=r[4],
            filed=date.fromisoformat(r[5]), fiscal_year=r[6],
            fiscal_year_end=date.fromisoformat(r[7]), primary_doc=r[8],
            html_path=Path(r[9]),
        )

    def accessions(self) -> list[str]:
        """Return every stored filing's accession, sorted."""
        with self._conn() as conn:
            return [r[0] for r in conn.execute(
                "SELECT accession FROM filings ORDER BY accession").fetchall()]

    def filings_for(self, ticker: str, fiscal_year: int) -> list[FilingRef]:
        """Return every stored filing (10-K and any 10-K/A) of a company-year, original first."""
        with self._conn() as conn:
            accessions = [r[0] for r in conn.execute(
                "SELECT accession FROM filings WHERE ticker=? AND fiscal_year=? "
                "ORDER BY (form = '10-K/A'), filed, accession", (ticker, fiscal_year),
            ).fetchall()]
        return [f for a in accessions if (f := self.filing(a)) is not None]

    def section(self, accession: str, item: str) -> Section | None:
        """Return the stored section ``(accession, item)``, or None."""
        with self._conn() as conn:
            r = conn.execute(
                "SELECT ticker, fiscal_year, title, text FROM sections "
                "WHERE accession=? AND item=?", (accession, item),
            ).fetchone()
        if r is None:
            return None
        return Section(accession=accession, ticker=r[0], fiscal_year=r[1], item=item,
                       title=r[2], text=r[3])

    def chunks_by_id(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Return the stored chunks for the ids that exist, keyed by id."""
        out: dict[str, Chunk] = {}
        with self._conn() as conn:
            for i in range(0, len(chunk_ids), 500):
                batch = chunk_ids[i:i + 500]
                marks = ",".join("?" * len(batch))
                for r in conn.execute(
                    "SELECT chunk_id, accession, ticker, fiscal_year, item, start_offset, "
                    f"end_offset, text FROM chunks WHERE chunk_id IN ({marks})", batch,
                ):
                    out[r[0]] = Chunk(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])
        return out

    def closed_claims_for(self, accession: str, items: list[str]) -> list[ClaimRow]:
        """Return closed-out (superseded) claims sourced from ``accession`` in ``items``.

        The counterpart of ``claims_for``, which returns only open claims.
        """
        if not items:
            return []
        marks = ",".join("?" * len(items))
        type_marks = ",".join("?" * len(CLAIM_TYPES))
        sql = f"""
            SELECT c.id, c.type, c.name, c.attributes, s.id, s.type, s.name,
                   r.source_accession, r.source_item
            FROM relationships r
            JOIN entities c ON c.id = r.src
            JOIN entities s ON s.id = r.dst
            WHERE r.type = ? AND r.source_accession = ? AND r.source_item IN ({marks})
              AND c.type IN ({type_marks}) AND c.valid_until IS NOT NULL
            ORDER BY r.id
        """
        with self._conn() as conn:
            rows = conn.execute(
                sql, (SUBJECT_RELATIONSHIP, accession, *items, *sorted(CLAIM_TYPES))
            ).fetchall()
        return [
            ClaimRow(entity_id=r[0], claim_type=r[1], name=r[2],
                     attributes=json.loads(r[3]), subject_id=r[4], subject_type=r[5],
                     subject_name=r[6], accession=r[7], item=r[8])
            for r in rows
        ]

    def fiscal_years(self) -> dict[str, list[int]]:
        """Return each ticker's stored fiscal years, ascending and de-duplicated."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker, fiscal_year FROM filings ORDER BY ticker, fiscal_year"
            ).fetchall()
        out: dict[str, list[int]] = {}
        for ticker, year in rows:
            out.setdefault(ticker, []).append(year)
        return out

    def claims_for(self, accession: str) -> list[ClaimRow]:
        """Return currently valid claims sourced from ``accession``, with subjects.

        A row needs a valid claim entity of a claim type and a valid ``subject_of``
        relationship to a valid subject entity.
        """
        marks = ",".join("?" * len(CLAIM_TYPES))
        sql = f"""
            SELECT c.id, c.type, c.name, c.attributes, s.id, s.type, s.name,
                   r.source_accession, r.source_item
            FROM relationships r
            JOIN entities c ON c.id = r.src
            JOIN entities s ON s.id = r.dst
            WHERE r.type = ? AND r.source_accession = ?
              AND c.type IN ({marks})
              AND r.valid_until IS NULL AND c.valid_until IS NULL
              AND s.valid_until IS NULL
            ORDER BY r.id
        """
        with self._conn() as conn:
            rows = conn.execute(
                sql, (SUBJECT_RELATIONSHIP, accession, *sorted(CLAIM_TYPES))
            ).fetchall()
        return [
            ClaimRow(entity_id=r[0], claim_type=r[1], name=r[2],
                     attributes=json.loads(r[3]), subject_id=r[4], subject_type=r[5],
                     subject_name=r[6], accession=r[7], item=r[8])
            for r in rows
        ]

    def close_out_claims(self, accession: str, item: str, at: datetime) -> int:
        """Close out every open claim sourced from ``(accession, item)`` as of ``at``.

        Rows are never deleted; ``valid_until`` is set. Already-closed claims are
        untouched, so calling this twice is harmless.

        Invariant: a claim entity belongs to exactly one (accession, item), because
        ingest creates claims fresh per section and never links them across
        filings. Closing by (accession, item) therefore cannot touch another
        filing's claims.

        Returns:
            The number of claims closed by this call.

        Raises:
            ValueError: if ``at`` is timezone-naive.
        """
        if at.tzinfo is None:
            raise ValueError("close_out_claims requires a timezone-aware datetime")
        marks = ",".join("?" * len(CLAIM_TYPES))
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE entities SET valid_until = ? WHERE valid_until IS NULL "
                f"AND type IN ({marks}) AND id IN (SELECT src FROM relationships "
                f"WHERE type = ? AND source_accession = ? AND source_item = ?)",
                (at.astimezone(UTC).isoformat(), *sorted(CLAIM_TYPES),
                 SUBJECT_RELATIONSHIP, accession, item),
            )
            return cur.rowcount

    def closed_claim_count(self, accession: str, items: list[str]) -> int:
        """Count closed-out claims sourced from ``accession`` in any of ``items``."""
        if not items:
            return 0
        marks = ",".join("?" * len(items))
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(DISTINCT c.id) FROM relationships r "
                f"JOIN entities c ON c.id = r.src WHERE r.type = ? "
                f"AND r.source_accession = ? AND r.source_item IN ({marks}) "
                f"AND c.valid_until IS NOT NULL",
                (SUBJECT_RELATIONSHIP, accession, *items),
            ).fetchone()
        return int(row[0])

    def claims_as_of(self, accession: str, as_of: datetime) -> list[ClaimRow]:
        """Return claims sourced from ``accession`` that were known and open at ``as_of``.

        A claim counts when its source filing had been filed by ``as_of`` (UTC
        date) and it was not closed out at or before ``as_of``. This answers
        "what was known on date X", including claims later superseded.

        Raises:
            ValueError: if ``as_of`` is timezone-naive.
        """
        if as_of.tzinfo is None:
            raise ValueError("claims_as_of requires a timezone-aware datetime")
        at = as_of.astimezone(UTC)
        marks = ",".join("?" * len(CLAIM_TYPES))
        sql = f"""
            SELECT c.id, c.type, c.name, c.attributes, s.id, s.type, s.name,
                   r.source_accession, r.source_item
            FROM relationships r
            JOIN entities c ON c.id = r.src
            JOIN entities s ON s.id = r.dst
            JOIN filings f ON f.accession = r.source_accession
            WHERE r.type = ? AND r.source_accession = ?
              AND c.type IN ({marks})
              AND f.filed <= ?
              AND (c.valid_until IS NULL OR c.valid_until > ?)
              AND r.valid_until IS NULL AND s.valid_until IS NULL
            ORDER BY r.id
        """
        with self._conn() as conn:
            rows = conn.execute(
                sql, (SUBJECT_RELATIONSHIP, accession, *sorted(CLAIM_TYPES),
                      at.date().isoformat(), at.isoformat())
            ).fetchall()
        return [
            ClaimRow(entity_id=r[0], claim_type=r[1], name=r[2],
                     attributes=json.loads(r[3]), subject_id=r[4], subject_type=r[5],
                     subject_name=r[6], accession=r[7], item=r[8])
            for r in rows
        ]

    def put_claim_evidence(self, accession: str, item: str,
                           rows: list[tuple[str, str, str]]) -> None:
        """Record the verbatim extraction excerpts of a section's claims.

        ``rows`` are ``(claim_type, claim_name, content)``: the ``content`` of each
        extraction chunk attached to that claim. Keyed by section, type and name rather
        than entity id, so it can be written for a section linked earlier without
        relinking it. Existing rows are left alone (idempotent).
        """
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO claim_evidence VALUES (?,?,?,?,?)",
                [(accession, item, t, n, c) for t, n, c in rows],
            )

    def claim_evidence(self, accession: str, item: str) -> dict[tuple[str, str], list[str]]:
        """``(claim_type, claim_name) -> excerpts`` for one section, excerpts in insertion order."""
        out: dict[tuple[str, str], list[str]] = {}
        with self._conn() as conn:
            for t, n, c in conn.execute(
                "SELECT claim_type, claim_name, content FROM claim_evidence "
                "WHERE accession=? AND item=? ORDER BY rowid", (accession, item),
            ):
                out.setdefault((t, n), []).append(c)
        return out

    def log_run(
        self, stage: str, key: str, status: str, tokens_in: int, tokens_out: int,
        cost_usd: Decimal, detail: str,
    ) -> None:
        """Append a run record; cost is stored exactly as TEXT.

        Raises:
            TypeError: if ``cost_usd`` is not a Decimal (floats are refused).
        """
        if not isinstance(cost_usd, Decimal):
            raise TypeError(f"cost_usd must be Decimal, got {type(cost_usd).__name__}")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (stage, key, status, tokens_in, tokens_out, "
                "cost_usd, detail, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (stage, key, status, tokens_in, tokens_out, str(cost_usd), detail,
                 _now()),
            )

    def runs(self) -> list[RunRow]:
        """Return all logged runs in insertion order, with cost as Decimal."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT stage, key, status, tokens_in, tokens_out, cost_usd, detail, "
                "created_at FROM runs ORDER BY id"
            ).fetchall()
        return [RunRow(r[0], r[1], r[2], r[3], r[4], Decimal(r[5]), r[6], r[7])
                for r in rows]
