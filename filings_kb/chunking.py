"""Sentence-boundary chunking and paragraph context windows.

Chunks are packed greedily from whole sentences, so every chunk starts and ends
on a sentence boundary and ``Section.text[chunk.start:chunk.end] == chunk.text``.
Chunks are in document order and together cover every sentence of the section;
only the whitespace between two chunks is left out. The token counter is
injected so this module loads no model.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from filings_kb.sections import Section

# Good enough for filings: a sentence ends at . ! or ? followed by whitespace and
# a capital letter or opening parenthesis ("vs. last year" does not split).
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_NEWLINE = re.compile(r"\n")
_WHITESPACE = re.compile(r"\s+")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A run of whole sentences from one section; offsets index ``Section.text``."""

    chunk_id: str
    accession: str
    ticker: str
    fiscal_year: int
    item: str
    start: int
    end: int
    text: str


def _split(
    text: str, start: int, end: int, separator: re.Pattern[str]
) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` on ``separator``; return trimmed non-empty spans."""
    spans: list[tuple[int, int]] = []
    pos = start
    for m in [*separator.finditer(text, start, end), None]:
        piece_end = end if m is None else m.start()
        piece = text[pos:piece_end]
        stripped = piece.lstrip()
        if stripped.strip():
            lead = len(piece) - len(stripped)
            spans.append((pos + lead, pos + lead + len(stripped.rstrip())))
        if m is not None:
            pos = m.end()
    return spans


def _spans(text: str, separator: re.Pattern[str]) -> list[tuple[int, int]]:
    """Return trimmed spans of the non-empty pieces of ``text`` between separators."""
    return _split(text, 0, len(text), separator)


def _units(text: str) -> list[tuple[int, int]]:
    """Sentences, never spanning a paragraph break."""
    return [
        sentence
        for p_start, p_end in _spans(text, _PARAGRAPH_BREAK)
        for sentence in _split(text, p_start, p_end, _SENTENCE_BREAK)
    ]


def chunk_section(
    s: Section, count_tokens: Callable[[str], int], max_tokens: int = 512
) -> list[Chunk]:
    """Split a section into chunks of whole sentences, each within ``max_tokens``.

    Sentences end at ``.``, ``!`` or ``?`` before a capital letter or ``(``, so
    a capitalised abbreviation such as "Inc. The" splits; paragraph breaks are
    boundaries too. A unit longer than ``max_tokens`` is split on single
    newlines, then on whitespace, so no chunk exceeds ``max_tokens`` (a single
    word that is itself too long is the only exception); a warning is logged for
    every chunk that came from such a split.

    Args:
        s: The section to chunk.
        count_tokens: Returns the token count of a string.
        max_tokens: Upper bound on tokens per chunk.

    Returns:
        Chunks in document order; ``chunk_id`` is ``accession:item:NNNN``.

    Raises:
        ValueError: If ``max_tokens`` is not positive.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    pieces: list[tuple[int, int, bool]] = []  # (start, end, came from a force-split)
    for start, end in _units(s.text):
        if count_tokens(s.text[start:end]) <= max_tokens:
            pieces.append((start, end, False))
            continue
        for l_start, l_end in _split(s.text, start, end, _NEWLINE):
            if count_tokens(s.text[l_start:l_end]) <= max_tokens:
                pieces.append((l_start, l_end, True))
            else:
                pieces.extend(
                    (w_start, w_end, True)
                    for w_start, w_end in _split(s.text, l_start, l_end, _WHITESPACE)
                )
    groups: list[tuple[int, int, bool]] = []
    cur: tuple[int, int, bool] | None = None
    for start, end, forced in pieces:
        if cur is not None and count_tokens(s.text[cur[0]:end]) <= max_tokens:
            cur = (cur[0], end, cur[2] or forced)
            continue
        if cur is not None:
            groups.append(cur)
        cur = (start, end, forced)
    if cur is not None:
        groups.append(cur)
    chunks: list[Chunk] = []
    for i, (start, end, forced) in enumerate(groups):
        chunk = Chunk(
            chunk_id=f"{s.accession}:{s.item}:{i:04d}",
            accession=s.accession,
            ticker=s.ticker,
            fiscal_year=s.fiscal_year,
            item=s.item,
            start=start,
            end=end,
            text=s.text[start:end],
        )
        if forced:
            logger.warning(
                "force-split oversized unit: %s has %d tokens (max %d)",
                chunk.chunk_id, count_tokens(chunk.text), max_tokens,
            )
        chunks.append(chunk)
    return chunks


def context_window(s: Section, c: Chunk, paragraphs: int = 2) -> dict[str, str]:
    """Return the chunk with up to ``paragraphs`` whole paragraphs on each side.

    Paragraphs are blocks separated by blank lines. The remainder of the
    chunk's own first and last paragraph is included, so
    ``before + match + after`` is one contiguous slice of ``s.text``.

    Args:
        s: The section the chunk came from.
        c: A chunk of ``s``.
        paragraphs: Whole paragraphs to add before and after the chunk's own.

    Returns:
        Dict with keys ``before``, ``match`` (the chunk text) and ``after``.

    Raises:
        ValueError: If ``paragraphs`` is negative.
    """
    if paragraphs < 0:
        raise ValueError(f"paragraphs must be >= 0, got {paragraphs}")
    spans = _spans(s.text, _PARAGRAPH_BREAK)
    overlapping = [i for i, (a, b) in enumerate(spans) if a < c.end and b > c.start]
    first, last = overlapping[0], overlapping[-1]
    window_start = spans[max(0, first - paragraphs)][0]
    window_end = spans[min(len(spans) - 1, last + paragraphs)][1]
    return {
        "before": s.text[min(window_start, c.start):c.start],
        "match": c.text,
        "after": s.text[c.end:max(window_end, c.end)],
    }
