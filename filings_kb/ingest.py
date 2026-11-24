"""Ingest orchestration: fetch -> sections -> chunks -> embed -> extract/import -> link.

Re-running is idempotent: filings, sections and chunks are upserts; chunks already
in the vector index are not re-embedded; a section's claims are linked once, in one
transaction that also writes the ``link`` row in ``runs``. Each claim's verbatim
extraction excerpts (the payload's ``chunks[].content``) are stored on every pass, so
an older store gains them without relinking.

Without ``live`` no model is ever called: every extraction comes from the
committed envelopes in ``data/committed/extractions``, and a missing one raises
``CacheMissError`` naming the section.
"""

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from filings_kb.chunking import chunk_section
from filings_kb.config import load_sample
from filings_kb.embed import CountTokens, EmbeddingIndex
from filings_kb.extract import MODELS_PATH, cache_path, extract_section, read_envelope
from filings_kb.fetch import fetch_filing
from filings_kb.link import link_entity
from filings_kb.llm import LLMBackend, LLMResult
from filings_kb.schema import ExtractionPayload
from filings_kb.sections import Section, detect_sections, section_text_sha256
from filings_kb.store import CLAIM_TYPES, SUBJECT_RELATIONSHIP, Store
from filings_kb.types import FilingRef, SampleRow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CHUNK_MAX_TOKENS = 512
FIDELITY_STATUSES = ("match", "mismatch", "not_comparable_split")

FetchFn = Callable[[SampleRow, Path], FilingRef]
DetectFn = Callable[[FilingRef], list[Section]]


class CacheMissError(RuntimeError):
    """A section needs an extraction that is not in the committed cache."""


@dataclass(frozen=True)
class IngestReport:
    """Totals for the sample after an ingest, read back from the store.

    ``missing_sections`` lists ``extract_items`` the detector did not find, per
    accession. ``imported_not_detected`` lists the subset that still had a
    cached envelope, whose claims were ingested anyway. ``skipped_by_policy``
    lists detected ``skip_items`` sections with no envelope (not failures).
    """

    filings: int
    sections: int
    chunks: int
    claims: int
    failed_sections: list[str]
    cost_usd: Decimal
    missing_sections: dict[str, list[str]] = field(default_factory=dict)
    imported_not_detected: dict[str, list[str]] = field(default_factory=dict)
    dropped_relationships: int = 0
    skipped_by_policy: dict[str, list[str]] = field(default_factory=dict)
    amendments: list[dict[str, Any]] = field(default_factory=list)
    fidelity: dict[str, Any] = field(default_factory=dict)


class _NoLiveBackend:
    """Backend that refuses to run: a call means the cache had no usable envelope."""

    def __init__(self, key: str) -> None:
        self._key = key

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        raise CacheMissError(
            f"no usable cached extraction for {self._key}; run with --live to extract it"
        )


class _LazyAnthropicBackend:
    """``AnthropicBackend`` built on the first call, so a run whose every section is
    cached needs neither ``ANTHROPIC_API_KEY`` nor the optional ``anthropic`` package."""

    def __init__(self) -> None:
        self._backend: LLMBackend | None = None

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        if self._backend is None:
            from filings_kb import llm
            self._backend = llm.AnthropicBackend()
        return self._backend.complete(system, user, model)


def _midnight_utc(d: date) -> datetime:
    """The filing date as a UTC-midnight timestamp (when supersession takes effect)."""
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _extraction_model(path: Path = MODELS_PATH) -> str:
    return str(yaml.safe_load(path.read_text(encoding="utf-8"))["extraction"])


def _default_index_and_counter(data_dir: Path) -> tuple[EmbeddingIndex, CountTokens]:
    """Real embedding index plus the matching tokenizer-based token counter."""
    from transformers import AutoTokenizer

    model_name = str(yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))["embedding"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    index = EmbeddingIndex(data_dir / "chroma", model_name, max_tokens=CHUNK_MAX_TOKENS,
                           count_tokens=count)
    return index, count



def _fidelity_status(parts: list[str], detected: Section | None) -> str | None:
    """Classify one envelope against the detected section; None if not detected."""
    if detected is None:
        return None
    if len(parts) > 1:
        return "not_comparable_split"
    return "match" if parts[0] == section_text_sha256(detected.text) else "mismatch"


def _summarise(counts: dict[str, int]) -> dict[str, Any]:
    comparable = counts["match"] + counts["mismatch"]
    return {**counts, "comparable": comparable,
            "match_rate": counts["match"] / comparable if comparable else None}


def _build_fidelity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-envelope fidelity entries overall and per section type."""
    def blank() -> dict[str, int]:
        return {**{s: 0 for s in FIDELITY_STATUSES}, "not_detected": 0}

    overall = blank()
    by_type: dict[str, dict[str, int]] = defaultdict(blank)
    mismatches = []
    for e in entries:
        status = e["status"] or "not_detected"
        overall[status] += 1
        by_type[e["section_type"]][status] += 1
        if status == "mismatch":
            mismatches.append({
                "accession": e["accession"], "section_type": e["section_type"],
                "envelope_text_sha256": e["envelope_sha"],
                "detected_text_sha256": e["detected_sha"],
                "detected_chars": e["detected_chars"],
            })
    return {
        "overall": _summarise(overall),
        "by_section_type": {k: _summarise(v) for k, v in sorted(by_type.items())},
        "mismatches": mismatches,
    }


def _link_payload(
    store: Store, payload: ExtractionPayload, ref: FilingRef, company_id: str,
    filing_id: str, item: str,
) -> tuple[int, int]:
    """Write one section's entities and relationships; return (claims, dropped rels).

    Claims are always new entities (one per filing section, so provenance stays
    exact) with a ``subject_of`` edge to the filer. Other entities go through
    ``link_entity``. A ``Filing`` endpoint resolves to this filing; any other
    endpoint missing from the payload is dropped and counted.
    """
    ids: dict[tuple[str, str], str] = {}
    claims = 0
    for e in payload.entities:
        if e.type in CLAIM_TYPES:
            entity_id = store.put_entity(e.type, e.name, e.attributes, e.identifiers)
            store.add_relationship(entity_id, company_id, SUBJECT_RELATIONSHIP,
                                   ref.accession, item, {})
            claims += 1
        else:
            entity_id = link_entity(store, e.type, e.name, e.identifiers).entity_id
        ids[(e.name, e.type)] = entity_id

    dropped = 0
    for r in payload.relationships:
        src = filing_id if r.source_type_entity == "Filing" else ids.get(
            (r.source_name, r.source_type_entity))
        dst = filing_id if r.target_type_entity == "Filing" else ids.get(
            (r.target_name, r.target_type_entity))
        if src is None or dst is None:
            dropped += 1
            logger.warning("%s/%s: dropped %s relationship %r -> %r (endpoint not in payload)",
                           ref.accession, item, r.type, r.source_name, r.target_name)
            continue
        attrs = dict(r.attributes)
        if r.valid_from:
            attrs["payload_valid_from"] = r.valid_from
        if r.valid_until:
            attrs["payload_valid_until"] = r.valid_until
        store.add_relationship(src, dst, r.type, ref.accession, item, attrs)
    return claims, dropped


def _claim_evidence(payload: ExtractionPayload) -> list[tuple[str, str, str]]:
    """``(claim_type, claim_name, content)`` for every extraction chunk attached to a claim."""
    claims = {(e.type, e.name) for e in payload.entities if e.type in CLAIM_TYPES}
    return [(c.entity_type_entity, c.entity_name, c.content) for c in payload.chunks
            if c.entity_type_entity is not None and c.entity_name is not None
            and (c.entity_type_entity, c.entity_name) in claims]


def ingest_sample(
    sample_path: Path,
    live: bool,
    *,
    data_dir: Path = DATA_DIR,
    fetch: FetchFn = fetch_filing,
    detect: DetectFn = detect_sections,
    index: EmbeddingIndex | None = None,
    count_tokens: CountTokens | None = None,
    backend: LLMBackend | None = None,
    refresh: bool = False,
    write_committed: bool = False,
) -> IngestReport:
    """Run the whole pipeline over the pinned sample.

    Args:
        sample_path: ``config/sample.yaml``.
        live: Allow extraction calls for sections with no usable cached envelope.
            When False, such a section raises ``CacheMissError``.
        data_dir: Holds ``raw/``, ``kb.sqlite``, ``chroma/`` and
            ``committed/extractions/``.
        fetch, detect: Injection points for tests.
        index, count_tokens: Embedding index and the token counter used for
            chunking; default to the real model's.
        backend: LLM backend for ``live`` (default ``AnthropicBackend``, built on the
            first cache miss, so a fully cached run needs no API key).
        refresh: Re-extract every detected section live, ignoring (and then
            overwriting) its cached envelope. Needs ``live``, ``write_committed``
            and a store with no linked sections, because a linked section is never
            relinked. A section the detector missed has no text to send, so its
            cached envelope is used.
        write_committed: Write new extraction envelopes to the tracked
            ``committed/extractions``. Without it they go to the gitignored
            ``local/extractions``, which is read after the committed cache.

    Returns:
        Totals read back from the store, plus fidelity and missing-section detail.

    Raises:
        CacheMissError: ``live`` is False and a detected ``extract_items`` section
            has no usable cached extraction.
        ValueError: ``refresh`` without ``live`` or ``write_committed``, or into a
            store that already holds linked sections.
    """
    if refresh and not live:
        raise ValueError("refresh re-extracts with the model, so it needs live")
    if refresh and not write_committed:
        raise ValueError("refresh overwrites data/committed/extractions, so it needs "
                         "write_committed")
    sample = load_sample(sample_path)
    if index is None or count_tokens is None:
        index, count_tokens = _default_index_and_counter(data_dir)
    store = Store(data_dir / "kb.sqlite")
    cache_dir = data_dir / "committed" / "extractions"
    write_dir = cache_dir if write_committed else data_dir / "local" / "extractions"
    model = _extraction_model()

    def envelope_for(accession: str, item: str) -> Path | None:
        """The cached envelope of a section (committed first, then local), or None."""
        for d in dict.fromkeys((cache_dir, write_dir)):
            path = cache_path(d, accession, item)
            if path.exists():
                return path
        return None

    if live and backend is None:
        backend = _LazyAnthropicBackend()

    runs_before = len(store.runs())
    already_linked = {r.key for r in store.runs() if r.stage == "link" and r.status == "ok"}
    if refresh and already_linked:
        raise ValueError(f"refresh needs a fresh store: {store.path} already holds "
                         f"{len(already_linked)} linked sections; move it aside first")
    failed: list[str] = []
    missing: dict[str, list[str]] = {}
    imported_not_detected: dict[str, list[str]] = {}
    skipped: dict[str, list[str]] = {}
    fidelity_entries: list[dict[str, Any]] = []
    dropped_total = 0
    started = time.monotonic()

    # Originals first, so an amendment finds the claims it supersedes.
    rows = sorted(sample.rows, key=lambda r: r.amends is not None)
    for n, row in enumerate(rows, start=1):
        ref = fetch(row, data_dir / "raw")
        store.put_filing(ref)
        sections = detect(ref)
        for s in sections:
            store.put_section(s)
            chunks = chunk_section(s, count_tokens, CHUNK_MAX_TOKENS)
            store.put_chunks(chunks)
            have = index.existing_ids([c.chunk_id for c in chunks])
            index.add([c for c in chunks if c.chunk_id not in have])
        logger.info("[%d/%d] %s %s: %d sections (%.0fs elapsed)", n, len(rows),
                    row.ticker, row.fiscal_year, len(sections), time.monotonic() - started)

        company_id = link_entity(
            store, "Company", ref.company, {"ticker": ref.ticker, "cik": ref.cik}
        ).entity_id
        filing_id = link_entity(
            store, "Filing", f"{ref.ticker} {ref.form} FY{ref.fiscal_year}",
            {"accession": ref.accession},
            attributes={"form": ref.form, "filed": ref.filed.isoformat()},
        ).entity_id
        assert company_id is not None and filing_id is not None

        detected = {s.item: s for s in sections}
        for item in sample.extract_items:
            key = f"{ref.accession}/{item}"
            section = detected.get(item)
            if section is None:
                missing.setdefault(ref.accession, []).append(item)
                if envelope_for(ref.accession, item) is None:
                    continue
                # Detector missed it but the private run extracted it: ingest the
                # cached claims. Empty text is safe because imported envelopes skip
                # the hash check; anything else is a miss and raises.
                imported_not_detected.setdefault(ref.accession, []).append(item)
                section = Section(ref.accession, ref.ticker, ref.fiscal_year, item, "", "")
                section_backend: LLMBackend = _NoLiveBackend(key)
            else:
                if item in sample.skip_items and envelope_for(ref.accession, item) is None:
                    skipped.setdefault(ref.accession, []).append(item)
                    continue
                section_backend = backend if live and backend is not None else _NoLiveBackend(key)
            payload = extract_section(section, section_backend, model, cache_dir, store,
                                      filing_date=ref.filed.isoformat(),
                                      use_cache=not (refresh and item in detected),
                                      write_dir=write_dir)
            if payload is None:
                failed.append(key)
                continue
            if (envelope_path := envelope_for(ref.accession, item)) is not None:
                env = read_envelope(envelope_path)
                fidelity_entries.append({
                    "accession": ref.accession, "section_type": item,
                    "status": _fidelity_status([p.text_sha256 for p in env.parts],
                                               detected.get(item)),
                    "envelope_sha": env.parts[0].text_sha256,
                    "detected_sha": (section_text_sha256(detected[item].text)
                                     if item in detected else None),
                    "detected_chars": len(detected[item].text) if item in detected else None,
                })
            n_claims = sum(1 for e in payload.entities if e.type in CLAIM_TYPES)
            with store.transaction():  # all-or-nothing: a crash leaves no partial claims
                # Every pass, linked or not, so a store linked before evidence was kept
                # gets it on the next ingest (idempotent).
                store.put_claim_evidence(ref.accession, item, _claim_evidence(payload))
                if key not in already_linked:
                    _, dropped = _link_payload(
                        store, payload, ref, company_id, filing_id, item)
                    dropped_total += dropped
                    store.log_run("link", key, "ok", 0, 0, Decimal("0"), "")
                    already_linked.add(key)
                if row.amends is not None and n_claims > 0:
                    # Section-level supersession: the amendment's claims replace the
                    # original's for this item (an amendment with no claims replaces
                    # nothing). Runs on every pass, linked or not: it only touches
                    # claims that are still open, so it is idempotent, and it catches
                    # an original section that was linked after the amendment.
                    store.close_out_claims(row.amends, item, _midnight_utc(ref.filed))

    accessions = [r.accession for r in sample.rows]
    amendments = []
    for row in (r for r in sample.rows if r.amends is not None):
        have_claims = {c.item for c in store.claims_for(row.accession)}
        replaced = [i for i in sample.extract_items if i in have_claims]
        amendments.append({
            "accession": row.accession, "amends": row.amends,
            "superseded_sections": replaced,
            "claims_closed_out": store.closed_claim_count(str(row.amends), replaced),
        })
    new_runs = store.runs()[runs_before:]
    return IngestReport(
        filings=store.count_rows("filings", accessions),
        sections=store.count_rows("sections", accessions),
        chunks=store.count_rows("chunks", accessions),
        claims=sum(len(store.claims_for(a)) for a in accessions),
        failed_sections=failed,
        cost_usd=sum((r.cost_usd for r in new_runs if r.stage == "extract"), Decimal("0")),
        missing_sections=missing,
        imported_not_detected=imported_not_detected,
        skipped_by_policy=skipped,
        amendments=amendments,
        dropped_relationships=dropped_total,
        fidelity=_build_fidelity(fidelity_entries),
    )


def write_report(report: IngestReport, path: Path) -> None:
    """Write the report as deterministic JSON (cost as an exact Decimal string).

    It holds no wall-clock value (durations go to the gitignored ``data/run_log.json``),
    so an unchanged sample rewrites it byte for byte.
    """
    data = {
        "filings": report.filings, "sections": report.sections, "chunks": report.chunks,
        "claims": report.claims, "failed_sections": report.failed_sections,
        "cost_usd": str(report.cost_usd),
        "missing_sections": report.missing_sections,
        "imported_not_detected": report.imported_not_detected,
        "dropped_relationships": report.dropped_relationships,
        "skipped_by_policy": report.skipped_by_policy,
        "amendments": report.amendments,
        "fidelity": report.fidelity,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
