"""Retrieval, structured cited answers, answer checks and the answer cache.

A question names filings (ticker, fiscal year) and items. Retrieval queries the
embedding index once per (filing, item) group, gives every group an equal share
of ``max_passages`` (leftover slots go to the best remaining passages), and
numbers the passages at or above ``display_cutoff`` in score order (the cutoff is
applied before the per-group quota and ``max_passages``). Passages below the
cutoff are never shown to the model; the best ``max_rejected`` of them are kept
as rejected candidates for the trace. The model
sees only those numbered passages (plus the extracted claims as uncitable
context) and returns JSON: paragraphs of one-sentence units, each with the
passage numbers that support it.

Every unit is checked against *its own* cited passages: each number and each
quoted span in it must appear there (``check_answer``). An answer that fails is
retried once with its JSON and the failing units; a second failure drops the
question, writes a failure record and keeps the raw answers in
``data/rejected/``. A passing answer is cached together with a snapshot of every
store-derived field the export needs, so exporting never opens the store.
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import jsonschema
import yaml

from filings_kb.chunking import Chunk, context_window
from filings_kb.llm import LLMBackend
from filings_kb.store import ClaimRow, Store

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "config" / "questions.yaml"
PROMPT_PATH = REPO_ROOT / "prompts" / "answer.md"
ANSWERS_DIR = REPO_ROOT / "data" / "committed" / "answers"
REJECTED_DIR = REPO_ROOT / "data" / "rejected"  # git-ignored: raw answers that failed checks
ANSWER_SCHEMA_PATH = REPO_ROOT / "schemas" / "answer.schema.json"

CATEGORIES = ("risk_shift", "contradiction", "capital_structure", "cross_company")
MAX_CLAIMS = 60
DEFAULT_MAX_REJECTED = 10  # rejected candidates kept per trace
MAX_ATTR_CHARS = 200
SUPERSEDED_STATUS = "SUPERSEDED"
AMENDED_STATUS = "AMENDED"
AUDIT_FAILED_PREFIX = "failed human audit: "

ITEM_NAMES: dict[str, str] = {
    "Item_1": "Business",
    "Item_1A": "Risk Factors",
    "Item_1B": "Unresolved Staff Comments",
    "Item_1C": "Cybersecurity",
    "Item_2": "Properties",
    "Item_3": "Legal Proceedings",
    "Item_4": "Mine Safety Disclosures",
    "Item_5": "Market for Registrant's Common Equity",
    "Item_6": "[Reserved]",
    "Item_7_MDA": "Management's Discussion and Analysis",
    "Item_7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item_8": "Financial Statements and Supplementary Data",
    "Item_9": "Changes in and Disagreements with Accountants",
    "Item_9A": "Controls and Procedures",
    "Item_9B": "Other Information",
    "Item_9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "Item_10": "Directors, Executive Officers and Corporate Governance",
    "Item_11": "Executive Compensation",
    "Item_12": "Security Ownership",
    "Item_13": "Certain Relationships and Related Transactions",
    "Item_14": "Principal Accountant Fees and Services",
    "Item_15": "Exhibits and Financial Statement Schedules",
}


def item_label(item: str) -> str:
    """``Item_7_MDA`` -> ``Item 7``; ``Item_1A`` -> ``Item 1A``."""
    return item.removesuffix("_MDA").replace("_", " ")


def section_label(item: str) -> str:
    """Passage ``section`` string, e.g. ``Item 1A Risk Factors``."""
    return f"{item_label(item)} {ITEM_NAMES.get(item, '')}".strip()


class _Index(Protocol):
    def query(self, text: str, k: int, where: dict[str, Any]) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class QuestionSpec:
    """One candidate question and the filings and items it draws on."""

    slug: str
    title: str
    question: str
    category: str
    filings: tuple[tuple[str, int], ...]
    items: tuple[str, ...]
    featured: bool = False
    amendment: bool = False  # compare a 10-K/A's claims with the ones it superseded

    @property
    def tickers(self) -> tuple[str, ...]:
        """Distinct tickers in filing order."""
        return tuple(dict.fromkeys(t for t, _ in self.filings))


@dataclass(frozen=True)
class QuestionConfig:
    """``config/questions.yaml``: retrieval settings plus the questions."""

    top_k: int
    max_passages: int
    display_cutoff: float
    questions: list[QuestionSpec]
    max_rejected: int = DEFAULT_MAX_REJECTED
    # slug -> audit_note of every question whose human audit verdict is ``fail``.
    audit_failures: dict[str, str] = field(default_factory=dict)

    def audit_failure(self, slug: str) -> "Failure | None":
        """The failure an ``audit: fail`` question is exported as, or None (spec 6.2)."""
        note = self.audit_failures.get(slug)
        return None if note is None else Failure(slug, f"{AUDIT_FAILED_PREFIX}{note}")


def load_questions(path: Path = QUESTIONS_PATH) -> QuestionConfig:
    """Load and validate the question config.

    ``audit`` (optional) is the human audit verdict, ``pass`` or ``fail``. A ``fail``
    needs an ``audit_note`` (it becomes the failure reason) and cannot be featured.

    Raises:
        ValueError: on a missing key, an unknown category, an empty filings or
            items list, a duplicate slug, or a bad audit entry.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("top_k", "max_passages", "display_cutoff", "questions"):
        if key not in data:
            raise ValueError(f"{path}: missing required key '{key}'")
    questions: list[QuestionSpec] = []
    audit_failures: dict[str, str] = {}
    seen: set[str] = set()
    for raw in data["questions"]:
        try:
            q = QuestionSpec(
                slug=str(raw["slug"]), title=str(raw["title"]), question=str(raw["question"]),
                category=str(raw["category"]),
                filings=tuple((str(f["ticker"]), int(f["fiscal_year"])) for f in raw["filings"]),
                items=tuple(str(i) for i in raw["items"]),
                featured=bool(raw["featured"]), amendment=bool(raw.get("amendment", False)),
            )
        except KeyError as exc:
            raise ValueError(f"{path}: question {raw!r} is missing {exc}") from exc
        if q.category not in CATEGORIES:
            raise ValueError(f"{path}: {q.slug}: unknown category {q.category!r}")
        if not q.filings or not q.items:
            raise ValueError(f"{path}: {q.slug}: filings and items must be non-empty")
        if q.slug in seen:
            raise ValueError(f"{path}: duplicate slug {q.slug}")
        audit = raw.get("audit")
        if audit not in (None, "pass", "fail"):
            raise ValueError(f"{path}: {q.slug}: audit must be pass or fail, not {audit!r}")
        if audit == "fail":
            note = str(raw.get("audit_note") or "").strip()
            if not note:
                raise ValueError(f"{path}: {q.slug}: audit: fail needs an audit_note")
            if q.featured:
                raise ValueError(f"{path}: {q.slug}: an audit-failed question cannot be featured")
            audit_failures[q.slug] = note
        seen.add(q.slug)
        questions.append(q)
    return QuestionConfig(int(data["top_k"]), int(data["max_passages"]),
                          float(data["display_cutoff"]), questions,
                          int(data.get("max_rejected", DEFAULT_MAX_REJECTED)), audit_failures)


@dataclass(frozen=True)
class PassageCandidate:
    """A retrieved chunk and its cosine similarity to the question."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RunPassage:
    """A candidate after the cutoff: ``cite`` is the number the model saw (None if rejected)."""

    chunk_id: str
    score: float
    included: bool
    cite: int | None


def retrieve(
    q: QuestionSpec, index: _Index, store: Store, top_k: int, max_passages: int = 40,
    cutoff: float = 0.0, max_rejected: int = DEFAULT_MAX_REJECTED,
) -> list[PassageCandidate]:
    """Query the index per (accession, item) group and keep ``max_passages`` in total.

    The cutoff is applied first: only candidates scoring at or above ``cutoff`` take
    part in the quota and the ``max_passages`` cap, and a group with none above it
    takes no share. The best ``max_rejected`` candidates below the cutoff (from the
    same top-k hits) are appended as rejected candidates.

    Each query filters on ticker, fiscal year, item and accession. The accession is
    the one whose section is current (a 10-K/A supersedes the original's section); an
    amendment question queries the original and the amendment, so each is a group.

    Every group that returned passages gets a quota of ``max_passages // groups``
    (its best passages first), so one filing can't crowd out the others; slots left
    over go to the best remaining passages by score.

    Returns:
        Candidates sorted by score descending (ties by chunk id).

    Raises:
        ValueError: if nothing at all was retrieved.
    """
    groups: dict[tuple[str, str], dict[str, float]] = {}
    for ticker, fiscal_year in q.filings:
        for item in q.items:
            if q.amendment:
                accessions = [f.accession for f in store.filings_for(ticker, fiscal_year)]
            else:
                current = store.section_accession(ticker, fiscal_year, item)
                accessions = [current] if current else []
            if not accessions:
                logger.warning("%s: no %s section for %s FY%s", q.slug, item, ticker, fiscal_year)
                continue
            for accession in accessions:
                where = {"ticker": ticker, "fiscal_year": fiscal_year, "item": item,
                         "accession": accession}
                found = index.query(q.question, top_k, where)
                if not found:
                    logger.warning("%s: no passages for %s", q.slug, where)
                    continue
                hits = groups.setdefault((accession, item), {})
                for chunk_id, score in found:
                    hits[chunk_id] = max(score, hits.get(chunk_id, score))
    all_ids = [cid for hits in groups.values() for cid in hits]
    chunks = store.chunks_by_id(all_ids)

    def ranked(hits: dict[str, float]) -> list[PassageCandidate]:
        out = [PassageCandidate(chunks[cid], sc) for cid, sc in hits.items() if cid in chunks]
        return sorted(out, key=lambda c: (-c.score, c.chunk.chunk_id))

    pooled = [ranked(hits) for _, hits in sorted(groups.items())]
    pooled = [g for g in pooled if g]
    if not pooled:
        raise ValueError(f"{q.slug}: no passages retrieved")
    per_group = [[c for c in g if c.score >= cutoff] for g in pooled]
    per_group = [g for g in per_group if g]
    if not per_group:
        raise ValueError(f"{q.slug}: no passage reaches the display cutoff {cutoff}")
    quota = max_passages // len(per_group)
    chosen = [c for g in per_group for c in g[:quota]]
    leftovers = sorted((c for g in per_group for c in g[quota:]),
                       key=lambda c: (-c.score, c.chunk.chunk_id))
    chosen += leftovers[:max(0, max_passages - len(chosen))]
    chosen.sort(key=lambda c: (-c.score, c.chunk.chunk_id))
    below = sorted((c for g in pooled for c in g if c.score < cutoff),
                   key=lambda c: (-c.score, c.chunk.chunk_id))
    return chosen + below[:max_rejected]


def assign_cites(candidates: list[PassageCandidate], cutoff: float) -> list[RunPassage]:
    """Flag each candidate included (score >= ``cutoff``) and number the included ones 1..n.

    Candidates must already be in score order; numbering follows that order.
    """
    out: list[RunPassage] = []
    n = 0
    for c in candidates:
        included = c.score >= cutoff
        if included:
            n += 1
        out.append(RunPassage(c.chunk.chunk_id, c.score, included, n if included else None))
    return out


# --- structured answers ------------------------------------------------------


@dataclass(frozen=True)
class AnswerUnit:
    """One sentence of an answer and the prompt numbers of the passages that support it."""

    text: str
    cites: tuple[int, ...]


Paragraphs = tuple[tuple[AnswerUnit, ...], ...]

_MARKER = re.compile(r"\[\d+\]")


def _answer_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(ANSWER_SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def parse_answer(raw: str) -> tuple[Paragraphs | None, list[str]]:
    """Parse the model's JSON answer; return ``(paragraphs, [])`` or ``(None, errors)``.

    The first JSON object in ``raw`` is decoded (``raw_decode`` from the first
    ``{``), so a code fence or prose before or after it is ignored, even prose
    holding braces. The object must then match ``schemas/answer.schema.json``
    exactly: ``{"paragraphs": [[{"text", "cites"}]]}`` with non-empty text and at
    least one integer cite per unit. Unit text must not carry ``[n]`` markers.
    """
    start = raw.find("{")
    if start == -1:
        return None, ["the answer is not a JSON object"]
    try:
        data, _ = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        return None, [f"the answer is not valid JSON: {exc}"]
    errors = [
        f"answer JSON at {'/'.join(str(x) for x in e.absolute_path) or '(root)'}: "
        f"{e.message[:200]}"
        for e in sorted(_answer_validator().iter_errors(data), key=lambda e: list(e.path))
    ]
    if errors:
        return None, errors
    paragraphs = tuple(
        tuple(AnswerUnit(str(u["text"]).strip(), tuple(int(c) for c in u["cites"])) for u in para)
        for para in data["paragraphs"])
    marked = [f"{_where(i, j)}: put citations in \"cites\", not as [n] markers in the text"
              for i, j, unit in _units(paragraphs) if _MARKER.search(unit.text)]
    if marked:
        return None, marked
    return paragraphs, []


def _units(paragraphs: Paragraphs) -> list[tuple[int, int, AnswerUnit]]:
    """``(paragraph number, unit number, unit)``, both numbers 1-based."""
    return [(i, j, unit) for i, para in enumerate(paragraphs, 1)
            for j, unit in enumerate(para, 1)]


def _where(i: int, j: int) -> str:
    return f"paragraph {i} unit {j}"


def validate_citations(paragraphs: Paragraphs, included_cites: set[int]) -> list[str]:
    """Errors for every cite that is not an included passage's number (empty means valid)."""
    errors: list[str] = []
    for i, j, unit in _units(paragraphs):
        for n in sorted(set(unit.cites)):
            if n not in included_cites:
                errors.append(f"{_where(i, j)}: citation [{n}] is not an included passage")
    return errors


# --- number and quote checks -----------------------------------------------------

# Labels that contain digits but are not quantities: form names, item numbers and
# fiscal-period labels. Digits glued to letters (MI308, H20, 18A) are skipped by the
# number pattern itself.
_NON_QUANTITIES = re.compile(
    r"\b10-[KQ](?:/A)?\b|\bItems?\s+\d+[A-C]?\b|\bFY\s?\d{2,4}\b|\bQ[1-4]\b")
# A figure may carry an attached magnitude letter ("$16.6B", "16.6bn", "$1.2M") or
# "bp"/"bps" ("40bps"). The suffix is part of the match so the fraction is not cut
# off; the quantity is the numeric part and the suffix sets the scale. Any other glued
# letter ("18A", "MI308", "H20") still disqualifies the token.
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9.])(?P<num>\d(?:\d|,(?=\d))*(?:\.\d+)?)"
    r"(?P<suffix>bn|bps?|[BbMmKkTt])?(?![A-Za-z0-9])")
_YEAR = re.compile(r"^(?:19|20)\d\d$")
# A unit word after a number sets its kind (and makes it a quantity even when it looks
# like a year). "percentage points" is tried before "percent".
_UNIT_SUFFIX = re.compile(
    r"\s*(?:(?P<points>percentage\s+points?\b)|(?P<percent>%|percent\b|per\s+cent\b)"
    r"|(?P<bps>bps?\b|basis\s+points?\b)|(?P<magnitude>trillion|billion|million|thousand)\b)",
    re.IGNORECASE)
_MULTIPLIERS = {"thousand": 10**3, "k": 10**3, "million": 10**6, "m": 10**6,
                "billion": 10**9, "b": 10**9, "bn": 10**9, "trillion": 10**12, "t": 10**12}

# Kinds of number. A bare number ("2,000 aircraft", a table cell) has no stated scale.
BARE, PERCENT, POINTS, BPS, MAGNITUDE = "", "percent", "percentage points", "bps", "magnitude"
# How well a number in an answer is supported by its cited passages.
VERIFIED, SCALE_UNVERIFIED, UNSUPPORTED = "verified", "scale unverified", "unsupported"


@dataclass(frozen=True)
class NumberSpan:
    """A number found in text: its quantity, scale, whether it is a bare year, and where it sits.

    ``kind`` is ``BARE``, ``PERCENT``, ``POINTS`` (percentage points), ``BPS`` or
    ``MAGNITUDE``; ``multiplier`` is the magnitude (``10**9`` for billion, ``B`` or
    ``bn``) and 1 for every other kind.
    """

    value: str  # thousands separators removed, no magnitude letter ("16.6")
    is_year: bool
    start: int  # offsets into the text given to ``number_spans``
    end: int
    kind: str = BARE
    multiplier: int = 1


def _scale(cleaned: str, end: int, suffix: str | None) -> tuple[str, int]:
    """``(kind, multiplier)`` from an attached suffix or the unit word after ``end``."""
    if suffix:
        s = suffix.lower()
        return (BPS, 1) if s.startswith("bp") else (MAGNITUDE, _MULTIPLIERS[s])
    m = _UNIT_SUFFIX.match(cleaned, end)
    if m is None:
        return BARE, 1
    if m.group("points"):
        return POINTS, 1
    if m.group("percent"):
        return PERCENT, 1
    if m.group("bps"):
        return BPS, 1
    return MAGNITUDE, _MULTIPLIERS[m.group("magnitude").lower()]


def number_spans(text: str) -> tuple[str, list[NumberSpan]]:
    """``(cleaned, spans)``: ``text`` with non-quantity labels blanked, and its numbers.

    The quantity has thousands separators removed (``1,234`` -> ``1234``) and no
    suffix (``16.6B`` -> ``16.6``, ``40bps`` -> ``40``); the scale goes in ``kind`` and
    ``multiplier``. ``is_year`` is true only for a bare 4-digit 19xx/20xx token: no
    comma, no ``$`` before it, no suffix and no unit (``%``, ``million``, ``bps``, ...)
    after it. ``cleaned`` has the length of ``text`` (labels are blanked in place), so
    offsets index both.
    """
    cleaned = _NON_QUANTITIES.sub(lambda m: " " * len(m.group(0)), text)  # keeps offsets
    spans: list[NumberSpan] = []
    for m in _NUMBER.finditer(cleaned):
        raw = m.group("num")
        kind, multiplier = _scale(cleaned, m.end(), m.group("suffix"))
        is_year = (bool(_YEAR.match(raw)) and kind == BARE
                   and cleaned[:m.start()].rstrip()[-1:] != "$")
        spans.append(NumberSpan(raw.replace(",", ""), is_year, m.start(), m.end(),
                                kind, multiplier))
    return cleaned, spans


# Digits glued to lowercase letters ("10x", "$5tn", "12pct", "3nm"). The known magnitude
# and basis-point suffixes are quantities (see ``_NUMBER``); every other lowercase suffix
# is an error, because the number check cannot tell what scale it means. Digits glued to
# uppercase letters are product codes and node names (H20, 14A, 777X, B200, GB200).
_SUFFIXED = re.compile(r"(?<![A-Za-z0-9.])\d(?:\d|,(?=\d))*(?:\.\d+)?(?P<suffix>[a-z]{1,3})"
                       r"(?![A-Za-z0-9])")
_KNOWN_SUFFIXES = frozenset({"bn", "bp", "bps", "b", "m", "k", "t"})


def unknown_suffixed_numbers(text: str) -> list[str]:
    """Number tokens in ``text`` with a lowercase suffix the number check doesn't know.

    Form names, item numbers and fiscal-period labels are blanked first, as in
    ``number_spans``.

    Returns:
        The offending tokens as written (``10x``, ``5tn``), deduplicated, in order.
    """
    cleaned = _NON_QUANTITIES.sub(lambda m: " " * len(m.group(0)), text)
    out: list[str] = []
    for m in _SUFFIXED.finditer(cleaned):
        if m.group("suffix") not in _KNOWN_SUFFIXES and m.group(0) not in out:
            out.append(m.group(0))
    return out


def support_of(span: NumberSpan, other: NumberSpan) -> str:
    """How ``other`` (a number in a passage) supports ``span`` (a number in an answer).

    * ``VERIFIED``: same kind and the same quantity. At the same scale the digits must
      be equal (``5.5`` is not ``5.50``); across magnitudes the products must be
      exactly equal (``$1.2 billion`` is ``$1,200 million``; Decimal, never rounded).
    * ``SCALE_UNVERIFIED``: the same digits, but one side states no scale (a table
      cell under an "in millions" header), so a person must check the scale.
    * ``UNSUPPORTED``: anything else, including two explicit kinds or scales that
      disagree (``12%`` against ``12 million``, ``40 bps`` against ``40%``).
    """
    if span.kind == other.kind:
        if span.multiplier == other.multiplier:
            if span.value == other.value:
                return VERIFIED
        elif Decimal(span.value) * span.multiplier == Decimal(other.value) * other.multiplier:
            return VERIFIED
    if span.value == other.value and BARE in (span.kind, other.kind):
        return SCALE_UNVERIFIED
    return UNSUPPORTED


_RANK = {VERIFIED: 0, SCALE_UNVERIFIED: 1, UNSUPPORTED: 2}


def best_support(span: NumberSpan, others: list[NumberSpan]) -> str:
    """The best ``support_of`` ``span`` among ``others`` (``UNSUPPORTED`` when empty)."""
    return min((support_of(span, o) for o in others), key=_RANK.__getitem__,
               default=UNSUPPORTED)


def number_support(text: str, passage_texts: list[str]) -> list[tuple[NumberSpan, str]]:
    """Each number of ``text`` (bare years skipped) with its best support in the passages.

    Every number in a passage counts as a candidate, years included.
    """
    known = [sp for t in passage_texts for sp in number_spans(t)[1]]
    return [(sp, best_support(sp, known)) for sp in number_spans(text)[1] if not sp.is_year]


def unsupported_numbers(text: str, passage_texts: list[str]) -> list[str]:
    """Numbers in ``text`` that no passage supports (bare years ignored), deduplicated.

    A number is supported when a passage holds the same quantity of the same kind, or
    the same digits with no stated scale on one side (``SCALE_UNVERIFIED``, left to the
    audit list; see ``support_of``). Nothing is rounded. Only a bare year is exempt;
    ``2,000 aircraft``, ``$1,999 million`` and ``2024%`` are checked.
    """
    out: list[str] = []
    for sp, support in number_support(text, passage_texts):
        if support == UNSUPPORTED and sp.value not in out:
            out.append(sp.value)
    return out


_TYPOGRAPHIC = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'",
                              "’": "'", " ": " "})
_QUOTE = re.compile(r'"([^"]*)"')
# A single-quoted span: the opening mark follows no letter or digit and the closing
# mark precedes none, so apostrophes (company's, customers') are not quote marks.
_SINGLE_QUOTE = re.compile(r"(?<![A-Za-z0-9])'(?=\S)((?:[^']|(?<=[A-Za-z])'(?=[A-Za-z]))+?)"
                           r"(?<=\S)'(?![A-Za-z0-9])")
_MIN_SINGLE_QUOTE_WORDS = 4  # single-quoted spans of more than three words are quotes
# In a quote split by edits, a piece shorter than this is not evidence (a one- or two-word
# piece, "disrupt" of "disrupt[ed] ...", matches almost any passage), so it is ignored;
# every longer piece must verify, and the quote needs at least one.
MIN_QUOTE_FRAGMENT_WORDS = 4
_EDIT = re.compile(r"\[[^\]]*\]|\.\.\.|…")


def _normalise(text: str) -> str:
    """Straight quotes and apostrophes, single spaces."""
    return re.sub(r"\s+", " ", text.translate(_TYPOGRAPHIC)).strip()


def unverified_quotes(text: str, passage_texts: list[str]) -> list[str]:
    """Quoted spans in ``text`` that are not verbatim in any one of ``passage_texts``.

    Quoted spans are double-quoted (straight or curly) spans of any length and
    single-quoted (straight or curly) spans of more than three words. A quote is
    split at ``[bracketed edits]`` and ellipses; every remaining fragment must
    appear in the same passage, ignoring case. Quote marks, apostrophes and
    whitespace are normalised on both sides; nothing else is. An unbalanced
    double quote mark is itself an error.

    Returns:
        The offending quotes (with their quote marks), or a message for an
        unbalanced quote mark.
    """
    unit = _normalise(text)
    if unit.count('"') % 2:
        return ["unbalanced double quote mark"]
    passages = [_normalise(t).casefold() for t in passage_texts]
    out: list[str] = []
    for quote, shown in _quoted_spans(unit):
        fragments = [f.casefold() for f in _evidence_fragments(quote)]
        if not fragments:  # nothing to verify: reported by quotes_without_evidence
            continue
        if not any(all(f in p for f in fragments) for p in passages):
            out.append(shown)
    return out


def _quoted_spans(unit: str) -> list[tuple[str, str]]:
    """``(quote, quote with its marks)`` for every quoted span of a normalised unit."""
    spans = [(q, f'"{q}"') for q in _QUOTE.findall(unit)]
    spans += [(q, f"'{q}'") for q in _SINGLE_QUOTE.findall(unit)
              if len(q.split()) >= _MIN_SINGLE_QUOTE_WORDS]
    return spans


def _fragments(quote: str) -> list[str]:
    """The non-empty pieces of ``quote`` between ``[edits]`` and ellipses, trimmed."""
    pieces = [f.strip().strip(",.;:").strip() for f in _EDIT.split(quote)]
    return [f for f in pieces if f]


def _evidence_fragments(quote: str) -> list[str]:
    """The pieces of ``quote`` that must verify.

    An unedited quote is one piece, checked verbatim whatever its length. In a quote
    split by edits, pieces under ``MIN_QUOTE_FRAGMENT_WORDS`` words are not evidence
    and are dropped; the rest must all verify (the list may be empty).
    """
    pieces = _fragments(quote)
    if len(pieces) <= 1:
        return pieces
    return [f for f in pieces if len(f.split()) >= MIN_QUOTE_FRAGMENT_WORDS]


def quotes_without_evidence(text: str) -> list[str]:
    """Quotes in ``text`` split by edits into pieces that are all under 4 words.

    Such a quote has nothing that can verify it, so it fails. An unedited quote is
    never reported here: ``unverified_quotes`` checks it verbatim.

    Returns:
        The offending quotes, with their quote marks.
    """
    return [shown for quote, shown in _quoted_spans(_normalise(text))
            if _fragments(quote) and not _evidence_fragments(quote)]


# Identity, absence and universal claims ("remained the same", "no mention"). A unit
# may make one only when a passage it cites uses the same phrase (any case, any stem).
_IDENTITY_PHRASES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bremain\w*\s+the\s+same\b",
    r"\bunchanged\b",
    r"\bidentical\w*",
    r"\bmatch(?:es|ed|ing)?\s+those\b",
    r"\ball\s+other\b",
    r"\bno\s+mentions?\b",
    r"\bnot\s+present\b",
    r"\b(?:did|does|do)\s+not\s+change\b|\b(?:didn't|doesn't|don't)\s+change\b",
    r"\bsame\s+as\b",
))


def unsupported_identity_claims(text: str, passage_texts: list[str]) -> list[str]:
    """Identity/absence/universal phrases in ``text`` that none of ``passage_texts`` uses.

    Returns:
        The phrases as they appear in ``text`` (empty means none is unsupported).
    """
    unit = _normalise(text)
    passages = [_normalise(t) for t in passage_texts]
    out: list[str] = []
    for pattern in _IDENTITY_PHRASES:
        m = pattern.search(unit)
        if m and not any(pattern.search(p) for p in passages):
            out.append(m.group(0))
    return out


def check_cutoff(run: "AnswerRun") -> list[str]:
    """Errors if a snapshot disagrees with its own ``display_cutoff``.

    An included (and so citable) passage must score at or above the cutoff, a
    rejected one below it, a rejected one must carry no cite, and there are at most
    ``max_rejected`` of them.

    Returns:
        One error per offending passage (empty means consistent).
    """
    errors: list[str] = []
    cutoff = run.display_cutoff
    rejected = 0
    for p in run.passages:
        if p.included and p.score < cutoff:
            errors.append(f"passage {p.chunk_id} scores {p.score:.3f}, below the display "
                          f"cutoff {cutoff}, but is included")
        if not p.included:
            rejected += 1
            if p.score >= cutoff:
                errors.append(f"rejected passage {p.chunk_id} scores {p.score:.3f}, at or "
                              f"above the display cutoff {cutoff}")
            if p.cite is not None:
                errors.append(f"rejected passage {p.chunk_id} carries cite {p.cite}")
    if rejected > run.max_rejected:
        errors.append(f"{rejected} rejected passages exceed max_rejected {run.max_rejected}")
    return errors


def check_answer(paragraphs: Paragraphs, passages: list["PassageSnapshot"]) -> list[str]:
    """Every check an answer must pass, unit by unit, against its own cited passages.

    1. Every unit cites at least one passage, and every cite is the number of an
       included passage.
    2. Every number in a unit appears in one of the passages that unit cites, and no
       number carries a lowercase suffix the check doesn't know (``10x``, ``$5tn``).
    3. Every quoted span in a unit appears verbatim (ignoring case) in one passage
       that unit cites; in a quote split by edits, pieces under 4 words are ignored,
       every longer piece must verify, and there must be at least one.
    4. Every identity, absence or universal phrase in a unit ("remained the same",
       "no mention", "all other") is used by a passage that unit cites.

    Returns:
        Errors naming the paragraph and unit (empty means the answer passes).
    """
    by_cite = {p.cite: p.text for p in passages if p.included and p.cite is not None}
    errors = validate_citations(paragraphs, set(by_cite))
    for i, j, unit in _units(paragraphs):
        if not unit.cites:
            errors.append(f"{_where(i, j)}: cites no passage")
        texts = [by_cite[n] for n in unit.cites if n in by_cite]
        cited = "".join(f"[{n}]" for n in unit.cites)
        for n in unsupported_numbers(unit.text, texts):
            errors.append(f"{_where(i, j)}: number {n} is not in its cited passages {cited}")
        for token in unknown_suffixed_numbers(unit.text):
            errors.append(f"{_where(i, j)}: number {token} has an unknown suffix; write the "
                          "figure as the filing does")
        for q in unverified_quotes(unit.text, texts):
            errors.append(f"{_where(i, j)}: quote {q} is not verbatim in its cited "
                          f"passages {cited}")
        for q in quotes_without_evidence(unit.text):
            errors.append(f"{_where(i, j)}: quote {q} has no piece of "
                          f"{MIN_QUOTE_FRAGMENT_WORDS} or more words between its edits")
        for phrase in unsupported_identity_claims(unit.text, texts):
            errors.append(f"{_where(i, j)}: identity/absence claim \"{phrase}\" is not "
                          f"stated in its cited passages {cited}")
    return errors


# --- rendering -----------------------------------------------------------------


_CLOSERS = ".!?\"”)]"
_SINGLE_CLOSERS = "'’"
_OPENS_SINGLE_QUOTE = re.compile(r"(?<![A-Za-z0-9])['‘](?=\S)")


def _split_tail(text: str) -> tuple[str, str]:
    """``text`` split before its trailing run of end punctuation and closing marks.

    A trailing ``'`` or ``’`` counts as a closing mark only when the unit opens a
    single quote somewhere, so a possessive (``customers'.``) is left alone.
    """
    closers = _CLOSERS + (_SINGLE_CLOSERS if _OPENS_SINGLE_QUOTE.search(text) else "")
    end = len(text)
    while end > 0 and text[end - 1] in closers:
        end -= 1
    return text[:end].rstrip(), text[end:]


def render_answer(paragraphs: Paragraphs) -> tuple[str, dict[int, int]]:
    """Render the answer as markdown, renumbering cites by first appearance.

    Each unit becomes ``text [a][b].``: the markers go before the unit's trailing
    run of ``.!?``, closing quotes and closing brackets (so ``slowed."`` renders as
    ``slowed [1]."``), or after the text when it has none, in ascending display
    order. Units are joined by a space and paragraphs by a blank line.

    Returns:
        ``(answer_md, mapping)`` where ``mapping`` takes the prompt number the
        model cited to the number shown in ``answer_md``.
    """
    mapping: dict[int, int] = {}
    for _, _, unit in _units(paragraphs):
        for n in unit.cites:
            mapping.setdefault(n, len(mapping) + 1)
    rendered: list[str] = []
    for para in paragraphs:
        parts: list[str] = []
        for unit in para:
            markers = "".join(f"[{d}]" for d in sorted({mapping[n] for n in unit.cites}))
            body, tail = _split_tail(unit.text)
            parts.append(f"{body} {markers}{tail}")
        rendered.append(" ".join(parts))
    return "\n\n".join(rendered), mapping


# --- claims ------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimView:
    """A claim chosen for a trace.

    ``cites`` holds the numbers of every included passage (same accession and item)
    whose text overlaps the claim's name, description or extraction excerpt, ascending.
    """

    row: ClaimRow
    ticker: str
    cites: tuple[int, ...]
    superseded: bool | None  # None outside the amendment question

    @property
    def cite(self) -> int | None:
        """The lowest overlapping passage number, or None."""
        return self.cites[0] if self.cites else None

    def cite_among(self, cited: set[int]) -> int | None:
        """The lowest overlapping passage number that is in ``cited``, or None."""
        return next((n for n in self.cites if n in cited), None)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _padded(words: list[str]) -> str:
    return " " + " ".join(words) + " "


# Free-text attributes an extraction may carry. Most claims have none (their names are
# generated labels), so overlap is rare by construction.
DESCRIPTION_KEYS = ("description", "note", "subject")


# A claim's extraction excerpt supports a passage when they share this many consecutive
# words (the whole excerpt when it is shorter, but never fewer than the minimum).
EVIDENCE_SPAN_WORDS = 8
MIN_EVIDENCE_WORDS = 4


def _spans(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _evidence_overlaps(excerpts: list[str], passage_words: list[str]) -> bool:
    """True when a passage shares an ``EVIDENCE_SPAN_WORDS`` run with one of ``excerpts``."""
    for excerpt in excerpts:
        words = _words(excerpt)
        n = min(EVIDENCE_SPAN_WORDS, len(words))
        if n >= MIN_EVIDENCE_WORDS and _spans(words, n) & _spans(passage_words, n):
            return True
    return False


def _overlaps(claim: ClaimRow, passage_words: str) -> bool:
    """True when the passage holds the claim's name or a 6-word span of its description."""
    name = _words(claim.name)
    if name and _padded(name) in passage_words:
        return True
    for key in DESCRIPTION_KEYS:
        desc = _words(str(claim.attributes.get(key, "")))
        if any(_padded(desc[i:i + 6]) in passage_words for i in range(len(desc) - 5)):
            return True
    return False


def select_claims(
    q: QuestionSpec, passages: list[RunPassage], store: Store, max_claims: int = MAX_CLAIMS,
) -> list[ClaimView]:
    """Choose the claims a trace shows (and the model sees as context).

    Ordinary questions take the open claims of every (accession, item) that has an
    included passage, interleaved round-robin across those sections (so one filing
    can't fill the cap); claims whose name or description overlaps an included
    passage come first, and at most ``max_claims`` are kept. An amendment question
    takes the claims the 10-K/A superseded (flagged) and the claims of the
    amendment, for every item it carries.

    A view's ``cites`` are the included passages of its own (accession, item) that
    hold its name or description, or share a run of ``EVIDENCE_SPAN_WORDS`` words
    with one of its stored extraction excerpts. The excerpts only add cites: the
    order (and so which claims the model sees) depends on name and description
    alone, so storing excerpts never changes an answer's input.
    """
    included = [p for p in passages if p.included]
    chunks = store.chunks_by_id([p.chunk_id for p in included])
    by_section: dict[tuple[str, str], list[tuple[int, str, list[str]]]] = {}
    for p in included:
        c = chunks[p.chunk_id]
        words = _words(c.text)
        by_section.setdefault((c.accession, c.item), []).append(
            (p.cite or 0, _padded(words), words))
    evidence: dict[tuple[str, str], dict[tuple[str, str], list[str]]] = {}

    def excerpts(row: ClaimRow) -> list[str]:
        key = (row.accession, row.item)
        if key not in evidence:
            evidence[key] = store.claim_evidence(*key)
        return evidence[key].get((row.claim_type, row.name), [])

    def cites_for(row: ClaimRow) -> tuple[int, ...]:
        mine = excerpts(row)
        return tuple(number for number, text, words
                     in sorted(by_section.get((row.accession, row.item), []))
                     if _overlaps(row, text) or _evidence_overlaps(mine, words))

    def named_in_a_passage(row: ClaimRow) -> bool:
        section = by_section.get((row.accession, row.item), [])
        return any(_overlaps(row, text) for _, text, _ in section)

    tickers: dict[str, str] = {}

    def ticker_of(accession: str) -> str:
        if accession not in tickers:
            filing = store.filing(accession)
            tickers[accession] = filing.ticker if filing else ""
        return tickers[accession]

    views: list[ClaimView] = []
    if q.amendment:
        for ticker, fiscal_year in q.filings:
            filings = store.filings_for(ticker, fiscal_year)
            originals = [f for f in filings if f.form == "10-K"]
            for amendment in (f for f in filings if f.form == "10-K/A"):
                amended = store.claims_for(amendment.accession)
                items = sorted({c.item for c in amended})
                for orig in originals:
                    views += [ClaimView(c, ticker, cites_for(c), True)
                              for c in store.closed_claims_for(orig.accession, items)]
                views += [ClaimView(c, ticker, cites_for(c), False) for c in amended]
        return views[:max_claims]

    sections: dict[tuple[str, str], list[ClaimView]] = {}
    for p in included:
        c = chunks[p.chunk_id]
        if (c.accession, c.item) in sections:
            continue
        sections[(c.accession, c.item)] = [
            ClaimView(row, ticker_of(row.accession), cites_for(row), None)
            for row in store.claims_for(c.accession) if row.item == c.item]
    rounds = max((len(v) for v in sections.values()), default=0)
    views = [group[i] for i in range(rounds) for group in sections.values() if i < len(group)]
    views.sort(key=lambda v: not named_in_a_passage(v.row))  # stable: named claims first
    return views[:max_claims]


def claim_line(view: ClaimView) -> str:
    """One claim as the model sees it: ``[Type] subject: name | key=value; ...``."""
    row = view.row
    attrs = "; ".join(f"{k}={str(v)[:MAX_ATTR_CHARS]}" for k, v in row.attributes.items())
    status = ""
    if view.superseded is not None:
        status = (SUPERSEDED_STATUS if view.superseded else AMENDED_STATUS) + " "
    return f"{status}[{row.claim_type}] {row.subject_name}: {row.name} | {attrs}".rstrip(" |")


# --- snapshot ------------------------------------------------------------------


@dataclass(frozen=True)
class PassageSnapshot:
    """A candidate passage plus every store-derived field the trace export needs.

    ``cite`` is the number the model saw (None for a rejected candidate) and
    ``context["match"]`` is the chunk text, which the answer checks read.
    """

    chunk_id: str
    score: float
    included: bool
    cite: int | None
    ticker: str
    company: str
    form: str
    fiscal_year: int
    filed: str  # ISO date
    item: str
    accession: str
    sec_url: str
    context: dict[str, str]

    @property
    def text(self) -> str:
        """The chunk text."""
        return self.context["match"]


def _context(store: Store, chunk: Chunk) -> dict[str, str]:
    """Chunk plus two paragraphs each side, from the stored section.

    Falls back to the chunk alone when the stored section is missing or no longer
    contains the chunk at its recorded offsets.
    """
    section = store.section(chunk.accession, chunk.item)
    if section is None or section.text[chunk.start:chunk.end] != chunk.text:
        logger.warning("context unavailable for %s; using the match only", chunk.chunk_id)
        return {"before": "", "match": chunk.text, "after": ""}
    return context_window(section, chunk, paragraphs=2)


def snapshot_passages(passages: list[RunPassage], store: Store) -> list[PassageSnapshot]:
    """Copy each passage's filing, section and context fields out of the store.

    Raises:
        ValueError: if a passage's chunk or filing is not in the store.
    """
    chunks = store.chunks_by_id([p.chunk_id for p in passages])
    out: list[PassageSnapshot] = []
    for p in passages:
        chunk = chunks.get(p.chunk_id)
        filing = store.filing(chunk.accession) if chunk else None
        if chunk is None or filing is None:
            raise ValueError(f"passage {p.chunk_id} is not in the store")
        out.append(PassageSnapshot(
            chunk_id=p.chunk_id, score=p.score, included=p.included, cite=p.cite,
            ticker=filing.ticker, company=filing.company, form=filing.form,
            fiscal_year=filing.fiscal_year, filed=filing.filed.isoformat(), item=chunk.item,
            accession=filing.accession, sec_url=filing.sec_url, context=_context(store, chunk)))
    return out


# --- failures --------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    """Why a question has no exported trace, and where its rejected raw answers are."""

    slug: str
    reason: str
    rejected: tuple[str, ...] = ()


def _failure_path(slug: str, cache_dir: Path) -> Path:
    return cache_dir / f"{slug}.failed.json"


def record_failure(cache_dir: Path, failure: Failure, *, keep_answer: bool = False) -> None:
    """Commit-able record of a failed question.

    The record holds no timestamp or absolute path of its own making, so a
    cache-only re-export reproduces ``exports/failed.json`` byte for byte.

    Args:
        keep_answer: False (an answer failed validation twice) removes any stale
            cached answer; True (a transient error such as a CLI failure) leaves a
            committed answer in place.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not keep_answer:
        (cache_dir / f"{failure.slug}.json").unlink(missing_ok=True)
    _failure_path(failure.slug, cache_dir).write_text(
        json.dumps(asdict(failure), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")


def load_failure(slug: str, cache_dir: Path = ANSWERS_DIR) -> Failure | None:
    """The failure record of ``slug``, or None when there is none.

    Raises:
        ValueError: if the record exists but is malformed.
    """
    path = _failure_path(slug, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Failure(str(data["slug"]), str(data["reason"]),
                       tuple(str(r) for r in data["rejected"]))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: not a valid failure record: {exc!r}") from exc


# --- synthesis and cache -------------------------------------------------------


@dataclass(frozen=True)
class AnswerRun:
    """One checked answer plus the snapshot its trace is built from (no store needed)."""

    spec: QuestionSpec
    paragraphs: Paragraphs
    passages: list[PassageSnapshot]
    claims: list[ClaimView]
    models: dict[str, str]
    top_k: int
    max_passages: int
    display_cutoff: float
    prompt_sha256: str
    input_sha256: str
    run_timestamp: str
    tokens_in: int = 0
    tokens_out: int = 0
    max_rejected: int = DEFAULT_MAX_REJECTED

    @property
    def answer_md(self) -> str:
        """The rendered answer, cites renumbered by first appearance."""
        return render_answer(self.paragraphs)[0]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_sha256(prompt_path: Path = PROMPT_PATH) -> str:
    """The cache key of the system prompt at ``prompt_path``."""
    return _sha256(prompt_path.read_text(encoding="utf-8"))


def build_user_message(
    q: QuestionSpec, passages: list[PassageSnapshot], claims: list[ClaimView],
) -> str:
    """Render the model's input: the question, numbered included passages, then claims."""
    parts = ["QUESTION", q.question, "", "PASSAGES"]
    for p in passages:
        if not p.included:
            continue
        parts += [f"[{p.cite}] {p.ticker} FY{p.fiscal_year} {p.form}, {item_label(p.item)}",
                  p.text, ""]
    parts.append("CLAIMS (context only, not citable)")
    parts += [f"- {claim_line(v)}" for v in claims] or ["(none)"]
    return "\n".join(parts)


def _run_to_json(run: AnswerRun) -> dict[str, Any]:
    return asdict(run)


def _run_from_json(data: dict[str, Any]) -> AnswerRun:
    s = data["spec"]
    spec = QuestionSpec(
        slug=s["slug"], title=s["title"], question=s["question"], category=s["category"],
        filings=tuple((t, int(y)) for t, y in s["filings"]), items=tuple(s["items"]),
        featured=bool(s["featured"]), amendment=bool(s["amendment"]),
    )
    paragraphs = tuple(tuple(AnswerUnit(u["text"], tuple(int(c) for c in u["cites"]))
                             for u in para) for para in data["paragraphs"])
    claims = [ClaimView(row=ClaimRow(**c["row"]), ticker=c["ticker"],
                        cites=tuple(int(n) for n in c["cites"]), superseded=c["superseded"])
              for c in data["claims"]]
    return AnswerRun(
        spec=spec, paragraphs=paragraphs,
        passages=[PassageSnapshot(**p) for p in data["passages"]], claims=claims,
        models=dict(data["models"]), top_k=int(data["top_k"]),
        max_passages=int(data["max_passages"]), display_cutoff=float(data["display_cutoff"]),
        prompt_sha256=data["prompt_sha256"], input_sha256=data["input_sha256"],
        run_timestamp=data["run_timestamp"], tokens_in=int(data["tokens_in"]),
        tokens_out=int(data["tokens_out"]),
        max_rejected=int(data.get("max_rejected", 0)),
    )


def load_run(slug: str, cache_dir: Path = ANSWERS_DIR) -> AnswerRun:
    """Read a cached answer.

    Raises:
        FileNotFoundError: if the question has no cached answer.
        ValueError: if the cache file is not a valid answer record.
    """
    path = cache_dir / f"{slug}.json"
    text = path.read_text(encoding="utf-8")
    try:
        return _run_from_json(json.loads(text))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: not a valid answer cache: {exc!r}") from exc


def _cached(slug: str, cache_dir: Path, model: str, prompt_sha: str, input_sha: str,
            ) -> AnswerRun | None:
    try:
        run = load_run(slug, cache_dir)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        logger.warning("ignoring unreadable answer cache: %s", exc)
        return None
    if (run.models.get("synthesis"), run.prompt_sha256, run.input_sha256) != (
            model, prompt_sha, input_sha):
        return None
    errors = check_answer(run.paragraphs, run.passages)
    if errors:
        logger.warning("%s: cached answer fails the current checks; calling again: %s",
                       slug, errors)
        return None
    return run


def _display_path(path: Path) -> str:
    """``path`` relative to the repo when it is inside it (stable across clones)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _save_rejected(rejected_dir: Path, slug: str, attempt: int, raw: str, errors: list[str],
                   meta: dict[str, str]) -> str:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    path = rejected_dir / f"{slug}.attempt{attempt}.json"
    record = {"slug": slug, "attempt": attempt, "raw": raw, "errors": errors, **meta}
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return _display_path(path)


_UNIT_ERROR = re.compile(r"^paragraph (\d+) unit (\d+): (.*)$", re.DOTALL)


def _retry_message(user: str, raw: str, paragraphs: Paragraphs | None,
                   errors: list[str]) -> str:
    """The original input plus the rejected JSON and exactly which units failed and why."""
    lines: list[str] = []
    for e in errors:
        m = _UNIT_ERROR.match(e)
        if m and paragraphs is not None:
            i, j = int(m.group(1)), int(m.group(2))
            lines.append(f"- {_where(i, j)} {json.dumps(paragraphs[i - 1][j - 1].text)}: "
                         f"{m.group(3)}")
        else:
            lines.append(f"- {e}")
    return (user + "\n\nYOUR PREVIOUS ANSWER WAS REJECTED.\n\nPREVIOUS ANSWER:\n" + raw.strip()
            + "\n\nFAILING UNITS:\n" + "\n".join(lines)
            + "\n\nReturn the complete answer again as JSON in the same format. Fix every "
              "failing unit: cite a passage that contains the exact number, quoted words "
              "or identity/absence phrase, or remove the number, quote or claim. Keep the "
              "units that passed.")


def _failure_reason(errors: list[str]) -> str:
    shown = "; ".join(errors[:3])
    more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
    return f"answer failed validation on both attempts: {shown}{more}"


def _write_run(run: AnswerRun, cache_dir: Path) -> AnswerRun:
    """Cache ``run``, clear any failure record, and return it as it reads back."""
    slug = run.spec.slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{slug}.json").write_text(
        json.dumps(_run_to_json(run), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    _failure_path(slug, cache_dir).unlink(missing_ok=True)
    return load_run(slug, cache_dir)


def run_question(
    q: QuestionSpec, index: _Index, store: Store, backend: LLMBackend, model: str,
    top_k: int, cutoff: float, cache_dir: Path, *, models: dict[str, str],
    max_passages: int = 40, max_rejected: int = DEFAULT_MAX_REJECTED,
    prompt_path: Path = PROMPT_PATH, rejected_dir: Path = REJECTED_DIR,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AnswerRun | None:
    """Retrieve, synthesize, check and cache an answer; None if it never passes.

    A cached answer is reused when its model, system-prompt hash and input hash all
    match and it still passes ``check_answer`` (so a re-run costs no model call). Otherwise the backend is called and the
    answer must parse (``parse_answer``) and pass ``check_answer``. A rejected answer
    is saved to ``rejected_dir`` and retried once with its JSON and failing units;
    after a second rejection a failure record replaces any cached answer and None
    is returned.

    Args:
        models: ``{embedding, extraction, synthesis, sentiment}`` recorded on the run;
            its ``synthesis`` entry is overridden by ``model``, the one actually called.

    Returns:
        The run exactly as it reads back from the cache, or None.
    """
    system = prompt_path.read_text(encoding="utf-8")
    prompt_sha = _sha256(system)
    passages = assign_cites(
        retrieve(q, index, store, top_k, max_passages, cutoff, max_rejected), cutoff)
    snapshots = snapshot_passages(passages, store)
    claims = select_claims(q, passages, store)
    user = build_user_message(q, snapshots, claims)
    input_sha = _sha256(user)

    hit = _cached(q.slug, cache_dir, model, prompt_sha, input_sha)
    if hit is not None:
        logger.info("%s: answer cache hit", q.slug)
        settings = (hit.top_k, hit.max_passages, hit.display_cutoff, hit.max_rejected)
        if settings == (top_k, max_passages, cutoff, max_rejected) and (
                hit.passages == snapshots and hit.claims == claims):
            return hit
        # Same model input, different cutoff bookkeeping (for example rejected candidates
        # added): the answer stands, so refresh the snapshot instead of paying for a call.
        refreshed = replace(hit, passages=snapshots, claims=claims, top_k=top_k,
                            max_passages=max_passages, display_cutoff=cutoff,
                            max_rejected=max_rejected)
        errors = check_answer(refreshed.paragraphs, refreshed.passages)
        if not errors:
            return _write_run(refreshed, cache_dir)
        logger.warning("%s: refreshed snapshot fails the checks; calling again: %s",
                       q.slug, errors)

    meta = {"model": model, "prompt_sha256": prompt_sha, "input_sha256": input_sha}
    message = user
    tokens_in = tokens_out = 0
    rejected: list[str] = []
    errors: list[str] = []
    for attempt in (1, 2):
        result = backend.complete(system, message, model)
        tokens_in += result.tokens_in
        tokens_out += result.tokens_out
        paragraphs, errors = parse_answer(result.text)
        if paragraphs is not None:
            errors = check_answer(paragraphs, snapshots)
        if paragraphs is not None and not errors:
            run = AnswerRun(
                spec=q, paragraphs=paragraphs, passages=snapshots, claims=claims,
                models={**models, "synthesis": model}, top_k=top_k, max_passages=max_passages,
                display_cutoff=cutoff, prompt_sha256=prompt_sha, input_sha256=input_sha,
                run_timestamp=now().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                tokens_in=tokens_in, tokens_out=tokens_out, max_rejected=max_rejected,
            )
            return _write_run(run, cache_dir)
        rejected.append(_save_rejected(rejected_dir, q.slug, attempt, result.text, errors, meta))
        logger.warning("%s: attempt %d rejected: %s", q.slug, attempt, errors)
        message = _retry_message(user, result.text, paragraphs, errors)
    record_failure(cache_dir, Failure(q.slug, _failure_reason(errors), tuple(rejected)))
    return None
