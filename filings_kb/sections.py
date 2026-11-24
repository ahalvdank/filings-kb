"""10-K section detection with table-of-contents stub rejection.

Detection reads the cached HTML of a filing (no network). The primary path is
edgartools' ``TenK`` item accessor; when it yields no item at all, a regex
scan over the document text is used instead. A section whose text looks like a
table-of-contents entry rather than a real body is rejected.

Section text is the item body with surrounding whitespace stripped, and
``section_text_sha256`` hashes exactly that text, so a hash computed here can
be compared with one computed over the same text elsewhere.
"""

import hashlib
import logging
import re
from dataclasses import dataclass

from edgar.company_reports import TenK
from lxml import etree
from lxml import html as lxml_html

from filings_kb.types import FilingRef

logger = logging.getLogger(__name__)

# Section tokens in document order. Item 7 is "Item_7_MDA"; the rest are
# "Item_<id>". A test keeps this equal to config/sample.yaml: extract_items.
ITEM_TOKENS: tuple[str, ...] = (
    "Item_1", "Item_1A", "Item_1B", "Item_1C", "Item_2", "Item_3", "Item_4",
    "Item_5", "Item_6", "Item_7_MDA", "Item_7A", "Item_8", "Item_9", "Item_9A",
    "Item_9B", "Item_9C", "Item_10", "Item_11", "Item_12", "Item_13",
    "Item_14", "Item_15",
)

STUB_MIN_WORDS = 200
STUB_MAX_PAGE_LINE_FRACTION = 0.30
_TITLE_MAX_CHARS = 200
_PAGE_NUMBER_END = re.compile(r"(?:^|[\s.\u2026])\d{1,3}$")
_DOTTED_LEADER = re.compile(r"\.{4,}|\u2026")
# A line-start heading: "Item 7", "Item 7.", "ITEM 1A - Risk Factors". After the
# id comes end of line, punctuation, or a capitalised word; a lowercase
# continuation ("Item 7 of this report ...") is a wrapped cross-reference.
_HEADING = re.compile(
    r"^\s*Item\s+(\d{1,2}[ABC]?)(?=\s*$|\s*[.:\u2013\u2014-]|\s+(?-i:[A-Z]))", re.IGNORECASE
)
# Same values as the private pipeline's item-body guards.
_HTML_TAG = re.compile(r"<[a-zA-Z/!][^>]*>")
_HTML_TAG_DENSITY_THRESHOLD = 1.0  # tags per 1000 chars
_ITEM_MAX_CHARS = 2_500_000


@dataclass(frozen=True)
class Section:
    """One detected 10-K item body."""

    accession: str
    ticker: str
    fiscal_year: int
    item: str
    title: str
    text: str


def _edgar_item_id(token: str) -> str:
    """Map a section token to the item id edgartools understands."""
    return token.removesuffix("_MDA").replace("_", " ")


def _regex_item_id(token: str) -> str:
    """Map a section token to the bare id a heading carries (``7A``)."""
    return _edgar_item_id(token).removeprefix("Item ")


def section_text_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8.

    Args:
        text: Section text exactly as stored on a ``Section``.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_toc_stub(text: str) -> bool:
    """Return True when ``text`` looks like a table-of-contents entry.

    A stub has more than 30% of its non-empty lines ending in a page number
    (one to three digits), or fewer than 200 words *and* looks like a ToC
    entry: it ends in a page number or contains dotted leaders. Short real
    items ("None.", "[Reserved]", "incorporated by reference") are not stubs.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    page_lines = sum(1 for ln in lines if _PAGE_NUMBER_END.search(ln))
    if page_lines / len(lines) > STUB_MAX_PAGE_LINE_FRACTION:
        return True
    if len(text.split()) < STUB_MIN_WORDS:
        return bool(_PAGE_NUMBER_END.search(lines[-1]) or _DOTTED_LEADER.search(text))
    return False


def _strip_html_if_markup(text: str) -> str:
    """Strip markup from a body that edgartools returned as raw HTML.

    A body with at least one tag per 1000 characters is treated as
    un-extracted HTML and reduced to its text; anything else, or a body lxml
    cannot parse, is returned unchanged.
    """
    if not text:
        return text
    density = len(_HTML_TAG.findall(text)) / max(1.0, len(text) / 1000.0)
    if density < _HTML_TAG_DENSITY_THRESHOLD:
        return text
    try:
        return lxml_html.fromstring(text).text_content() or text
    except (ValueError, etree.ParserError) as exc:
        logger.warning("HTML strip of item body failed, keeping raw: %s", exc)
        return text


def _clean_body(body: object, token: str, accession: str) -> str | None:
    """Strip markup and whitespace; None when empty or over the size cap."""
    if not isinstance(body, str):
        return None
    cleaned = _strip_html_if_markup(body).strip()
    if not cleaned:
        return None
    if len(cleaned) > _ITEM_MAX_CHARS:
        logger.warning("%s: dropping over-captured %s (%d chars)", accession, token, len(cleaned))
        return None
    return cleaned


class _LocalFiling:
    """The two attributes edgartools' ``TenK`` needs, served from a local file."""

    form = "10-K"

    def __init__(self, html: str) -> None:
        self._html = html

    def html(self) -> str:
        return self._html


def _edgartools_bodies(html: str, tokens: tuple[str, ...], accession: str) -> dict[str, str]:
    """Item bodies from edgartools' ``TenK``; empty when it cannot parse."""
    try:
        report = TenK(_LocalFiling(html))
    except Exception as exc:  # noqa: BLE001
        # edgartools raises assorted types on malformed filings; any failure
        # here must fall through to the regex path, not abort the run.
        logger.warning("%s: edgartools could not parse filing: %s", accession, exc)
        return {}
    bodies: dict[str, str] = {}
    for token in tokens:
        try:
            body = report[_edgar_item_id(token)]
        except (KeyError, TypeError, AttributeError, ValueError):
            continue  # edgartools signals "item not in this filing" this way
        cleaned = _clean_body(body, token, accession)
        if cleaned:
            bodies[token] = cleaned
    return bodies


def _regex_bodies(html: str, tokens: tuple[str, ...], accession: str) -> dict[str, str]:
    """Item bodies by scanning heading lines in the document text.

    Every heading for an item is a candidate (table of contents and body).
    A candidate runs to the next heading for a different item; one with no
    later heading is dropped rather than run to the end of the document. The
    longest candidate wins.
    """
    text = lxml_html.fromstring(html).text_content()
    wanted = {_regex_item_id(t): t for t in tokens}
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _HEADING.match(line)
        if match:
            headings.append((offset + len(line) - len(line.lstrip()), match.group(1).upper()))
        offset += len(line)
    best: dict[str, str] = {}
    for idx, (start, item_id) in enumerate(headings):
        token = wanted.get(item_id)
        end = next((off for off, nxt in headings[idx + 1:] if nxt != item_id), None)
        if token is None or end is None:
            continue
        cleaned = _clean_body(text[start:end], token, accession)
        if cleaned and len(cleaned) > len(best.get(token, "")):
            best[token] = cleaned
    return best


def _title(text: str) -> str:
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first_line[:_TITLE_MAX_CHARS]


def detect_sections(
    f: FilingRef, tokens: tuple[str, ...] = ITEM_TOKENS
) -> list[Section]:
    """Detect the item sections of a cached 10-K, rejecting ToC stubs.

    Args:
        f: Filing whose ``html_path`` points at the cached HTML.
        tokens: Section tokens to look for; defaults to every 10-K item.

    Returns:
        Sections in document order. Items that are absent, empty or stubs are
        omitted rather than returned empty.

    Raises:
        FileNotFoundError: If ``f.html_path`` does not exist.
    """
    html = f.html_path.read_text(encoding="utf-8")
    bodies = _edgartools_bodies(html, tokens, f.accession)
    if not bodies:
        logger.info("%s: edgartools found no items; using regex fallback", f.accession)
        bodies = _regex_bodies(html, tokens, f.accession)

    sections: list[Section] = []
    for token in tokens:
        body = bodies.get(token)
        if body is None:
            continue
        if is_toc_stub(body):
            logger.info("%s: rejected %s as ToC stub", f.accession, token)
            continue
        sections.append(Section(
            accession=f.accession, ticker=f.ticker, fiscal_year=f.fiscal_year,
            item=token, title=_title(body), text=body,
        ))
    return sections
