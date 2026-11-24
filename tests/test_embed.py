"""Tests for the Chroma embedding index (fake embedder; no model downloads)."""

from pathlib import Path

import pytest

from filings_kb.chunking import Chunk
from filings_kb.embed import EmbeddingIndex

VECTORS = {
    "alpha": [1.0, 0.0, 0.0, 0.0],
    "beta": [0.9, 0.1, 0.0, 0.0],
    "gamma": [0.0, 1.0, 0.0, 0.0],
    "delta": [-1.0, 0.0, 0.0, 0.0],
}


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [VECTORS[t] for t in texts]


def word_count(text: str) -> int:
    return len(text.split())


def _chunk(cid: str, text: str, ticker: str = "NVDA", year: int = 2025) -> Chunk:
    return Chunk(cid, f"acc-{ticker}", ticker, year, "Item_1A", 0, len(text), text)


@pytest.fixture
def index(tmp_path: Path) -> EmbeddingIndex:
    return EmbeddingIndex(tmp_path / "chroma", "fake-model", embed_fn=fake_embed,
                          max_tokens=10, count_tokens=word_count)


def test_query_returns_cosine_similarity_descending(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha"), _chunk("b", "beta"), _chunk("g", "gamma"),
               _chunk("d", "delta")])
    hits = index.query("alpha", k=4, where={})
    assert [cid for cid, _ in hits] == ["a", "b", "g", "d"]
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert scores[1] == pytest.approx(0.9 / (0.9**2 + 0.1**2) ** 0.5, abs=1e-5)
    assert scores[2] == pytest.approx(0.0, abs=1e-5)
    assert scores[3] == pytest.approx(-1.0, abs=1e-5)


def test_query_where_filters_by_ticker(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha", "NVDA"), _chunk("b", "beta", "AMD")])
    hits = index.query("alpha", k=5, where={"ticker": "AMD"})
    assert [cid for cid, _ in hits] == ["b"]


def test_query_where_filters_by_fiscal_year_int(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha", year=2024), _chunk("b", "beta", year=2025)])
    assert [c for c, _ in index.query("alpha", 5, {"fiscal_year": 2024})] == ["a"]


def test_query_multiple_where_keys_are_anded(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha", "NVDA", 2024), _chunk("b", "beta", "NVDA", 2025)])
    hits = index.query("alpha", 5, {"ticker": "NVDA", "fiscal_year": 2025})
    assert [c for c, _ in hits] == ["b"]


def test_query_empty_index_returns_empty(index: EmbeddingIndex) -> None:
    assert index.query("alpha", k=3, where={}) == []


def test_add_is_idempotent_on_chunk_id(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha")])
    index.add([_chunk("a", "alpha")])
    assert len(index.query("alpha", k=10, where={})) == 1


def test_index_persists_across_instances(tmp_path: Path) -> None:
    first = EmbeddingIndex(tmp_path / "c", "m", embed_fn=fake_embed, max_tokens=10,
                           count_tokens=word_count)
    first.add([_chunk("a", "alpha")])
    second = EmbeddingIndex(tmp_path / "c", "m", embed_fn=fake_embed, max_tokens=10,
                            count_tokens=word_count)
    assert [c for c, _ in second.query("alpha", 1, {})] == ["a"]


def test_add_raises_naming_chunk_over_max_tokens(index: EmbeddingIndex) -> None:
    long = Chunk("too-long", "acc", "NVDA", 2025, "Item_1A", 0, 50, " ".join(["w"] * 11))
    with pytest.raises(ValueError, match="too-long"):
        index.add([_chunk("a", "alpha"), long])
    # Nothing was written: validation happens before any embedding or upsert.
    assert index.query("alpha", 5, {}) == []


def test_add_accepts_chunk_exactly_at_max_tokens(index: EmbeddingIndex) -> None:
    edge = Chunk("edge", "acc", "NVDA", 2025, "Item_1A", 0, 50, " ".join(["w"] * 10))
    VECTORS[edge.text] = [0.0, 0.0, 1.0, 0.0]
    index.add([edge])
    assert [c for c, _ in index.query(edge.text, 1, {})] == ["edge"]


def test_default_embedder_is_not_loaded_at_construction(tmp_path: Path) -> None:
    # No embed_fn: constructing must not import or load sentence-transformers.
    idx = EmbeddingIndex(tmp_path / "c", "Qwen/Qwen3-Embedding-0.6B", max_tokens=512)
    assert idx.query("anything", k=1, where={}) == []  # empty: no embedding needed


def test_existing_ids_reports_only_stored_ids(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha"), _chunk("b", "beta")])
    assert index.existing_ids(["a", "b", "zzz"]) == {"a", "b"}
    assert index.existing_ids([]) == set()


def test_vectors_returns_stored_embeddings_for_where(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha", "NVDA"), _chunk("b", "beta", "NVDA"),
               _chunk("g", "gamma", "AMD")])
    got = index.vectors({"ticker": "NVDA", "fiscal_year": 2025, "item": "Item_1A"})
    assert len(got) == 2
    for want, have in zip(sorted([VECTORS["alpha"], VECTORS["beta"]]), sorted(got)):
        assert have == pytest.approx(want, abs=1e-6)  # Chroma stores float32


def test_vectors_empty_when_no_match(index: EmbeddingIndex) -> None:
    index.add([_chunk("a", "alpha", "NVDA")])
    assert index.vectors({"ticker": "XOM"}) == []
