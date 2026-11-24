"""Chroma-backed embedding index over chunks, with a no-truncation guard.

The default embedder loads ``sentence-transformers`` lazily on first use, never at
import or construction. ``embed_fn`` and ``count_tokens`` are injectable so tests
need no model. ``add`` refuses any chunk longer than the effective token limit
instead of letting the model silently truncate it.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb

from filings_kb.chunking import Chunk

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]
CountTokens = Callable[[str], int]

COLLECTION = "chunks"


def _chroma_where(where: dict[str, Any]) -> dict[str, Any] | None:
    """Translate exact-match metadata filters into Chroma's ``where`` form (ANDed)."""
    if len(where) > 1:
        return {"$and": [{k: v} for k, v in where.items()]}
    return dict(where) or None


class EmbeddingIndex:
    """Persistent cosine-similarity index of chunk text."""

    def __init__(
        self,
        path: Path,
        model_name: str,
        *,
        max_tokens: int,
        embed_fn: EmbedFn | None = None,
        count_tokens: CountTokens | None = None,
    ) -> None:
        """Open the Chroma collection at ``path``.

        Args:
            path: directory for the persistent Chroma client.
            model_name: sentence-transformers model, used only by the default embedder.
            max_tokens: caller's chunk token budget; the effective limit is the smaller
                of this and the model's ``max_seq_length`` when the default embedder
                is used.
            embed_fn: texts -> vectors; defaults to a lazily loaded SentenceTransformer.
            count_tokens: text -> token count; defaults to the model's tokenizer.
        """
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._embed_fn = embed_fn
        self._count_tokens = count_tokens
        self._model: Any = None
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def _load_model(self) -> Any:
        """Load the SentenceTransformer once, on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` with the injected function or the default model."""
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        model = self._load_model()
        return model.encode(texts, normalize_embeddings=True).tolist()

    def _limit(self) -> int:
        """Effective token limit: min(max_tokens, model.max_seq_length) if default model."""
        if self._embed_fn is not None:
            return self._max_tokens
        return min(self._max_tokens, int(self._load_model().max_seq_length))

    def _count(self, text: str) -> int:
        """Token count via the injected counter or the model's tokenizer."""
        if self._count_tokens is not None:
            return self._count_tokens(text)
        tokenizer = self._load_model().tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert chunks by ``chunk_id``.

        Raises:
            ValueError: naming the first chunk whose token count exceeds the limit.
                Checked for all chunks before anything is embedded or written.
        """
        if not chunks:
            return
        limit = self._limit()
        for c in chunks:
            n = self._count(c.text)
            if n > limit:
                raise ValueError(
                    f"chunk {c.chunk_id} has {n} tokens, over the limit of {limit}; "
                    "it would be truncated by the embedding model"
                )
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=self._embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[
                {"accession": c.accession, "ticker": c.ticker,
                 "fiscal_year": c.fiscal_year, "item": c.item}
                for c in chunks
            ],
        )

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ``ids`` already stored, so callers can skip re-embedding."""
        if not ids:
            return set()
        return set(self._collection.get(ids=ids, include=[])["ids"])

    def query(self, text: str, k: int, where: dict[str, Any]) -> list[tuple[str, float]]:
        """Return up to ``k`` (chunk_id, cosine similarity) pairs, best first.

        ``where`` maps metadata keys to exact values; several keys are ANDed.
        Similarity is ``1 - cosine distance``, so it lies in [-1, 1].
        """
        if self._collection.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=self._embed([text]),
            n_results=k,
            where=_chroma_where(where),
        )
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def vectors(self, where: dict[str, Any]) -> list[list[float]]:
        """Return the stored embeddings of every chunk matching ``where``.

        Reads back what ``add`` wrote; nothing is re-embedded. ``where`` has the
        same meaning as in ``query``. Returns ``[]`` when nothing matches.
        """
        res = self._collection.get(where=_chroma_where(where), include=["embeddings"])
        return [[float(x) for x in v] for v in res["embeddings"]]
