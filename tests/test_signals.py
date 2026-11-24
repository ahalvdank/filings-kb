"""Tests for YoY signals (fake embedder, fake classifier, fake LLM: no model, no CLI)."""

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from filings_kb.chunking import Chunk
from filings_kb.embed import EmbeddingIndex
from filings_kb.llm import LLMResult
from filings_kb.sections import Section
from filings_kb.signals import (
    ClaimDiff, compute_signals, claim_diff, sentiment_shift, tone_shift,
)
from filings_kb.store import ClaimRow, Store
from filings_kb.types import FilingRef

VECS = {
    "x": [1.0, 0.0], "y": [0.0, 1.0], "x2": [1.0, 0.0],
    "pos": [1.0, 0.0], "neg": [0.0, 1.0],
}


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [VECS[t] for t in texts]


@pytest.fixture
def index(tmp_path: Path) -> EmbeddingIndex:
    return EmbeddingIndex(tmp_path / "chroma", "fake", embed_fn=fake_embed,
                          max_tokens=10, count_tokens=lambda t: 1)


def _chunk(cid: str, text: str, year: int, ticker: str = "NVDA",
           item: str = "Item_1A", acc: str | None = None) -> Chunk:
    return Chunk(cid, acc or f"acc-{ticker}-{year}", ticker, year, item, 0, 1, text)


def test_tone_shift_zero_for_identical_sections(index: EmbeddingIndex) -> None:
    index.add([_chunk("p1", "x", 2024), _chunk("p2", "y", 2024),
               _chunk("c1", "x", 2025), _chunk("c2", "y", 2025)])
    assert tone_shift(index, "NVDA", "Item_1A", 2024, 2025) == pytest.approx(0.0, abs=1e-9)


def test_tone_shift_is_one_minus_cosine_of_centroids(index: EmbeddingIndex) -> None:
    index.add([_chunk("p1", "x", 2024), _chunk("c1", "y", 2025)])
    assert tone_shift(index, "NVDA", "Item_1A", 2024, 2025) == pytest.approx(1.0)


def test_tone_shift_none_when_prior_missing(index: EmbeddingIndex) -> None:
    index.add([_chunk("c1", "x", 2025)])
    assert tone_shift(index, "NVDA", "Item_1A", 2024, 2025) is None


def test_tone_shift_none_when_current_missing(index: EmbeddingIndex) -> None:
    index.add([_chunk("p1", "x", 2024)])
    assert tone_shift(index, "NVDA", "Item_1A", 2024, 2025) is None


def test_tone_shift_accession_filter_excludes_other_filings_of_same_year(
    index: EmbeddingIndex,
) -> None:
    index.add([_chunk("p1", "x", 2024), _chunk("c1", "x", 2025, acc="orig"),
               _chunk("c2", "y", 2025, acc="amend")])
    both = tone_shift(index, "NVDA", "Item_1A", 2024, 2025)
    only_orig = tone_shift(index, "NVDA", "Item_1A", 2024, 2025, current_accession="orig")
    only_amend = tone_shift(index, "NVDA", "Item_1A", 2024, 2025, current_accession="amend")
    assert only_orig == pytest.approx(0.0, abs=1e-9)
    assert only_amend == pytest.approx(1.0)
    assert 0.0 < both < 1.0


def _filing(ticker: str, year: int, acc: str, form: str = "10-K") -> FilingRef:
    return FilingRef(
        ticker=ticker, cik="1", company=ticker, accession=acc, form=form,
        filed=date(year + 1, 2, 1), fiscal_year=year, fiscal_year_end=date(year, 12, 31),
        primary_doc="d.htm", html_path=Path("x.htm"),
    )


def _put(store: Store, ticker: str, year: int, item: str, texts: list[str],
         acc: str | None = None, form: str = "10-K") -> str:
    acc = acc or f"acc-{ticker}-{year}"
    store.put_filing(_filing(ticker, year, acc, form))
    store.put_section(Section(accession=acc, ticker=ticker, fiscal_year=year, item=item,
                              title=item, text=" ".join(texts)))
    store.put_chunks([Chunk(f"{acc}-{item}-{i}", acc, ticker, year, item, i, i + 1, t)
                      for i, t in enumerate(texts)])
    return acc


def fake_scorer(texts: list[str]) -> list[float]:
    """pos-neg score: +1 for 'good', -1 for 'bad'."""
    return [1.0 if "good" in t else -1.0 for t in texts]


class CountingScorer:
    """fake_scorer that also reports truncation like FinbertScorer does."""

    truncated = 3
    scored = 0

    def __call__(self, texts: list[str]) -> list[float]:
        self.scored += len(texts)
        return fake_scorer(texts)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "kb.sqlite")


def test_sentiment_shift_sign(store: Store) -> None:
    _put(store, "NVDA", 2024, "Item_1A", ["bad a", "bad b"])
    _put(store, "NVDA", 2025, "Item_1A", ["good a", "bad b"])
    # prior mean -1.0, current mean 0.0 -> improvement of +1.0
    assert sentiment_shift(store, "NVDA", "Item_1A", 2024, 2025, scorer=fake_scorer) == 1.0
    assert sentiment_shift(store, "NVDA", "Item_1A", 2025, 2024, scorer=fake_scorer) == -1.0


def test_sentiment_shift_none_when_a_side_has_no_chunks(store: Store) -> None:
    _put(store, "NVDA", 2025, "Item_1A", ["good"])
    assert sentiment_shift(store, "NVDA", "Item_1A", 2024, 2025, scorer=fake_scorer) is None


def _claim(name: str, ctype: str = "Risk", **attrs: object) -> ClaimRow:
    return ClaimRow("e", ctype, name, dict(attrs), "s", "Company", "NVDA", "acc", "Item_1A")


class FakeBackend:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, str]] = []

    def complete(self, system: str, user: str, model: str) -> LLMResult:
        self.calls.append((system, user, model))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(reply, 10, 5)


GOOD_JSON = json.dumps({
    "added": ["Export controls"], "removed": ["Crypto demand"],
    "changed": [{"before": "Buyback $10B", "after": "Buyback $25B"}],
})


def test_claim_diff_parses_llm_json_into_added_removed_changed() -> None:
    backend = FakeBackend([GOOD_JSON])
    out = claim_diff([_claim("Crypto demand")], [_claim("Export controls", severity="high")],
                     backend, "m-synth")
    assert out == ClaimDiff(added=["Export controls"], removed=["Crypto demand"],
                            changed=[("Buyback $10B", "Buyback $25B")])
    system, user, model = backend.calls[0]
    assert model == "m-synth"
    assert "Crypto demand" in user and "Export controls" in user and "severity=high" in user


def test_claim_diff_accepts_fenced_json() -> None:
    backend = FakeBackend([f"```json\n{GOOD_JSON}\n```"])
    out = claim_diff([_claim("a")], [_claim("b")], backend, "m")
    assert out is not None and out.added == ["Export controls"]


def test_claim_diff_retries_once_on_invalid_then_succeeds() -> None:
    backend = FakeBackend(["not json", GOOD_JSON])
    out = claim_diff([_claim("a")], [_claim("b")], backend, "m")
    assert out is not None and len(backend.calls) == 2


def test_claim_diff_invalid_json_returns_none_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    backend = FakeBackend(["not json", '{"added": "oops"}'])
    with caplog.at_level(logging.WARNING, logger="filings_kb.signals"):
        out = claim_diff([_claim("a")], [_claim("b")], backend, "m")
    assert out is None
    assert len(backend.calls) == 2  # one retry, no more
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_claim_diff_backend_error_counts_as_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = FakeBackend([RuntimeError("boom"), RuntimeError("boom")])
    with caplog.at_level(logging.WARNING, logger="filings_kb.signals"):
        assert claim_diff([_claim("a")], [_claim("b")], backend, "m") is None
    assert len(backend.calls) == 2


def _claim_in(store: Store, acc: str, item: str, name: str) -> None:
    eid = store.put_entity("Risk", name, {}, {})
    subj = store.put_entity("Company", "NVDA", {}, {"ticker": "NVDA"})
    store.add_relationship(eid, subj, "subject_of", acc, item, {})


def _two_year_kb(store: Store, index: EmbeddingIndex) -> None:
    for year, text in ((2024, "x"), (2025, "y")):
        acc = _put(store, "NVDA", year, "Item_1A", [text])
        index.add([_chunk(f"c{year}", text, year, acc=acc)])
        _claim_in(store, acc, "Item_1A", f"risk {year}")


def test_compute_signals_writes_deterministic_json_and_caches_claim_diffs(
    store: Store, index: EmbeddingIndex, tmp_path: Path,
) -> None:
    _two_year_kb(store, index)
    backend = FakeBackend([GOOD_JSON])
    out = tmp_path / "signals.json"
    cache = tmp_path / "claim_diffs"

    def run() -> list:
        return compute_signals(store, index, backend, "m-synth", scorer=CountingScorer(),
                               out_path=out, cache_dir=cache, items=["Item_1A"],
                               prompt_path=Path("prompts/claim_diff.md"))

    rows = run()
    assert len(rows) == 1
    r = rows[0]
    assert (r.ticker, r.item, r.prior_fy, r.current_fy) == ("NVDA", "Item_1A", 2024, 2025)
    assert r.tone_shift == pytest.approx(1.0)
    assert r.sentiment_shift == 0.0
    assert r.claim_diff == ClaimDiff(added=["Export controls"], removed=["Crypto demand"],
                                     changed=[("Buyback $10B", "Buyback $25B")])
    assert r.status == "shadow"
    first = out.read_bytes()
    data = json.loads(first)
    assert data["signals"][0]["status"] == "shadow"
    assert data["meta"]["sentiment_truncated_chunks"] == 3
    assert data["meta"]["sentiment_scored_chunks"] == 2
    cached = json.loads((cache / "NVDA_Item_1A_2024_2025.json").read_text())
    assert cached["model"] == "m-synth" and cached["prompt_sha256"] and cached["created_at"]

    run()  # second run: no backend replies left, so a call would raise IndexError
    assert len(backend.calls) == 1
    assert out.read_bytes() == first


def test_compute_signals_records_none_rows_and_skips_claim_diff_when_not_requested(
    store: Store, index: EmbeddingIndex, tmp_path: Path,
) -> None:
    _two_year_kb(store, index)
    backend = FakeBackend([])
    rows = compute_signals(store, index, backend, "m", scorer=fake_scorer,
                           out_path=tmp_path / "s.json", cache_dir=tmp_path / "c",
                           items=["Item_1A", "Item_7_MDA"], claim_diff_pairs=set(),
                           prompt_path=Path("prompts/claim_diff.md"))
    assert [(r.item, r.tone_shift is None, r.claim_diff) for r in rows] == [
        ("Item_1A", False, None), ("Item_7_MDA", True, None)]
    assert rows[1].sentiment_shift is None
    assert backend.calls == []


class _FakeTokenizer:
    def __call__(self, text: str, truncation: bool, max_length: int) -> dict[str, list[int]]:
        return {"input_ids": list(range(min(len(text.split()), max_length)))}


class _FakePipe:
    tokenizer = _FakeTokenizer()

    def __call__(self, texts: list[str], **_: object) -> list[list[dict[str, object]]]:
        return [[{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.1},
                 {"label": "neutral", "score": 0.2}] for _ in texts]


def test_finbert_scorer_scores_positive_minus_negative_and_counts_truncation() -> None:
    from filings_kb.signals import FinbertScorer

    scorer = FinbertScorer(max_length=3)
    scorer._pipe = _FakePipe()  # skip the model download
    assert scorer(["a b", "a b c d e"]) == [pytest.approx(0.6)] * 2
    assert (scorer.scored, scorer.truncated) == (2, 1)
    assert scorer([]) == []


# --- final review C2: committed claim diffs are read, never rewritten, without the flag ----


def _signals(store, index, backend, out, cache, read=()):
    return compute_signals(store, index, backend, "m-synth", scorer=CountingScorer(),
                           out_path=out, cache_dir=cache, read_cache_dirs=read,
                           items=["Item_1A"], prompt_path=Path("prompts/claim_diff.md"))


def test_claim_diff_cache_hit_leaves_the_file_and_its_created_at_untouched(
    store: Store, index: EmbeddingIndex, tmp_path: Path,
) -> None:
    _two_year_kb(store, index)
    cache = tmp_path / "claim_diffs"
    _signals(store, index, FakeBackend([GOOD_JSON]), tmp_path / "s.json", cache)
    path = cache / "NVDA_Item_1A_2024_2025.json"
    first = path.read_bytes()
    _signals(store, index, FakeBackend([]), tmp_path / "s.json", cache)
    assert path.read_bytes() == first


def test_committed_claim_diffs_are_read_and_misses_go_to_the_local_dir(
    store: Store, index: EmbeddingIndex, tmp_path: Path,
) -> None:
    _two_year_kb(store, index)
    committed, local = tmp_path / "committed", tmp_path / "local"
    _signals(store, index, FakeBackend([GOOD_JSON]), tmp_path / "c.json", committed)
    before = {p: p.read_bytes() for p in committed.iterdir()}
    rows = _signals(store, index, FakeBackend([]), tmp_path / "l.json", local, read=(committed,))
    assert rows[0].claim_diff_status == "ok" and not local.exists()  # a hit, nothing written

    stale = committed / "NVDA_Item_1A_2024_2025.json"  # inputs changed: committed copy is stale
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "input_sha256": "0" * 64}))
    before = {p: p.read_bytes() for p in committed.iterdir()}
    _signals(store, index, FakeBackend([GOOD_JSON]), tmp_path / "l.json", local, read=(committed,))
    assert (local / "NVDA_Item_1A_2024_2025.json").exists()
    assert {p: p.read_bytes() for p in committed.iterdir()} == before


def test_signals_drift_lists_changed_rows_and_fields(tmp_path: Path) -> None:
    from filings_kb.signals import signals_drift
    row = {"ticker": "NVDA", "item": "Item_1A", "prior_fy": 2024, "current_fy": 2025,
           "tone_shift": 0.1, "sentiment_shift": -0.2, "claim_diff": None,
           "claim_diff_status": "no_claims", "status": "shadow"}
    committed, local = tmp_path / "c.json", tmp_path / "l.json"
    committed.write_text(json.dumps({"meta": {}, "signals": [row]}))
    local.write_text(json.dumps({"meta": {}, "signals": [row]}))
    assert signals_drift(local, committed) == []
    local.write_text(json.dumps({"meta": {}, "signals": [{**row, "sentiment_shift": -0.200001}]}))
    assert signals_drift(local, committed) == [
        "NVDA Item_1A 2024->2025: sentiment_shift -0.2 -> -0.200001"]
    assert signals_drift(local, tmp_path / "missing.json") == ["no committed signals to compare"]
