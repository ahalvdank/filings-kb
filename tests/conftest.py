"""Shared fakes for the answer and export tests: a tiny store, a fake index, a fake backend."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from filings_kb.chunking import Chunk, chunk_section
from filings_kb.llm import LLMResult
from filings_kb.sections import Section
from filings_kb.store import Store
from filings_kb.types import FilingRef

NVDA_24 = "0001045810-24-000029"
NVDA_25 = "0001045810-25-000023"
AMD_ORIG = "0000002488-26-000018"
AMD_AMEND = "0000002488-26-000021"

SECTION_TEXT = (
    "Export controls could reduce our data center revenue by $5.5 billion in a quarter. "
    "We rely on a few customers.\n\n"
    "Customer concentration remains high. Two customers accounted for 24% of revenue.\n\n"
    "Supply constraints may limit shipments. We depend on third-party foundries."
)


def count_tokens(text: str) -> int:
    """Whitespace token counter, so tests load no tokenizer."""
    return len(text.split())


def _filing(acc: str, ticker: str, fy: int, form: str = "10-K", filed: date = date(2025, 2, 26),
            cik: str = "1045810", company: str = "NVIDIA CORP") -> FilingRef:
    return FilingRef(ticker=ticker, cik=cik, company=company, accession=acc, form=form,
                     filed=filed, fiscal_year=fy, fiscal_year_end=date(fy, 1, 26),
                     primary_doc=f"{ticker.lower()}-{fy}.htm", html_path=Path("x.htm"))


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """NVDA FY2024 and FY2025 Item_1A sections with chunks; AMD FY2025 10-K and 10-K/A."""
    s = Store(tmp_path / "kb.sqlite")
    for acc, fy in ((NVDA_24, 2024), (NVDA_25, 2025)):
        s.put_filing(_filing(acc, "NVDA", fy))
        sec = Section(acc, "NVDA", fy, "Item_1A", "Risk Factors", SECTION_TEXT)
        s.put_section(sec)
        s.put_chunks(chunk_section(sec, count_tokens, max_tokens=14))
    s.put_filing(_filing(AMD_ORIG, "AMD", 2025, filed=date(2026, 2, 4), cik="2488",
                         company="ADVANCED MICRO DEVICES INC"))
    s.put_filing(_filing(AMD_AMEND, "AMD", 2025, form="10-K/A", filed=date(2026, 2, 4),
                         cik="2488", company="ADVANCED MICRO DEVICES INC"))
    return s


class FakeIndex:
    """Scores chunks from a fixed table; honours ``where`` like the real index."""

    def __init__(self, store: Store, scores: dict[str, float]) -> None:
        self.scores = scores
        self.store = store
        self.calls: list[tuple[str, int, dict]] = []

    def query(self, text: str, k: int, where: dict) -> list[tuple[str, float]]:
        self.calls.append((text, k, where))
        chunks = self.store.chunks_by_id(list(self.scores))
        hits = [
            (cid, sc) for cid, sc in self.scores.items()
            if cid in chunks and all(
                getattr(chunks[cid], {"fiscal_year": "fiscal_year"}.get(key, key)) == val
                for key, val in where.items()
            )
        ]
        return sorted(hits, key=lambda h: -h[1])[:k]


class FakeBackend:
    """Returns queued replies and records every call."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, str]] = []

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        self.calls.append((system, user, model))
        return LLMResult(self.replies.pop(0), 100, 20)


def fixed_now() -> datetime:
    return datetime(2026, 11, 23, 12, 0, 0, tzinfo=UTC)


def answer_json(*paragraphs: list[tuple[str, list[int]]]) -> str:
    """A structured answer as the model returns it: one ``(text, cites)`` per unit."""
    import json

    return json.dumps({"paragraphs": [[{"text": t, "cites": c} for t, c in p]
                                      for p in paragraphs]})
