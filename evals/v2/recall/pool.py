"""Recall pool builder and blind labelling sheet for the v2 recall@k eval.

For each frozen question the pool is the union of the dense top ``pool_k`` and a BM25 top
``pool_k`` over the same section, so a labeller sees passages the dense retriever would miss.
The labelling sheet shuffles those passages and hides rank and source, so labels cannot be
anchored on the retriever's order. The knowledge base is only ever opened read-only.

Usage:
    python -m evals.v2.recall.pool --questions evals/v2/recall/questions.yaml \
        --db data/kb.sqlite --chroma data/chroma --out-dir evals/v2/recall/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sqlite3
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
DenseFn = Callable[[str, int, dict], list[tuple[str, float]]]
SHEET_COLUMNS = ["qid", "question", "row", "chunk_id", "passage", "relevant"]

# Same rule as filings_kb.store.Store.section_accession; kept as SQL so the DB opens read-only
# (Store.__init__ runs a schema script, which writes).
_CURRENT_ACCESSION_SQL = (
    "SELECT s.accession FROM sections s JOIN filings f ON f.accession = s.accession "
    "WHERE s.ticker=? AND s.fiscal_year=? AND s.item=? "
    "ORDER BY (f.form = '10-K/A') DESC, f.filed DESC, s.accession DESC LIMIT 1"
)


def _connect_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def tokenize(text: str) -> list[str]:
    """Lowercase NFKC text and split it into alphanumeric runs."""
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def bm25_rank(query: str, docs: dict[str, str], k1: float = 1.5,
              b: float = 0.75) -> list[tuple[str, float]]:
    """Rank ``docs`` against ``query`` with Okapi BM25, best first.

    IDF is ``ln((N - df + 0.5) / (df + 0.5) + 1)``. Every doc is returned; docs sharing no
    term with the query score 0.0. Ties break on doc id so the ranking is deterministic.
    """
    tokens = {doc_id: tokenize(text) for doc_id, text in docs.items()}
    n = len(tokens)
    avg_len = (sum(len(t) for t in tokens.values()) / n) if n else 0.0
    df: Counter[str] = Counter()
    for toks in tokens.values():
        df.update(set(toks))
    query_terms = set(tokenize(query))
    scored: list[tuple[str, float]] = []
    for doc_id, toks in tokens.items():
        tf = Counter(toks)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1)
            denom = tf[term] + k1 * (1 - b + b * len(toks) / avg_len)
            score += idf * tf[term] * (k1 + 1) / denom
        scored.append((doc_id, score))
    return sorted(scored, key=lambda p: (-p[1], p[0]))


def current_accession(db: Path, ticker: str, fiscal_year: int, item: str) -> str | None:
    """Accession whose ``item`` section is current (a 10-K/A supersedes the original)."""
    with closing(_connect_ro(db)) as conn:
        row = conn.execute(_CURRENT_ACCESSION_SQL, (ticker, fiscal_year, item)).fetchone()
    return row[0] if row else None


def section_chunks(db: Path, accession: str, item: str) -> dict[str, str]:
    """Map chunk_id to text for every chunk of one section, in chunk_id order."""
    with closing(_connect_ro(db)) as conn:
        rows = conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE accession=? AND item=? ORDER BY chunk_id",
            (accession, item)).fetchall()
    return dict(rows)


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def build_pool(questions: list[dict], db: Path, dense: DenseFn, pool_k: int = 5,
               rank_k: int = 20, meta_extra: dict[str, Any] | None = None) -> dict:
    """Build the candidate pool for each question.

    Args:
        questions: rows with id, ticker, fiscal_year, item, question.
        db: path to the KB sqlite; opened read-only.
        dense: ``EmbeddingIndex.query``-shaped callable (text, k, where) -> [(chunk_id, score)].
        pool_k: how many top chunks from each of dense and BM25 enter the pool.
        rank_k: how many dense results are kept as the recall ranking.
        meta_extra: extra provenance (e.g. the embedding model name) merged into ``meta``.

    Raises:
        ValueError: a question's section does not exist, or has no chunks.
    """
    out: dict[str, dict] = {}
    for q in questions:
        qid, ticker, fy, item = q["id"], q["ticker"], q["fiscal_year"], q["item"]
        accession = current_accession(db, ticker, fy, item)
        if accession is None:
            raise ValueError(f"{qid}: no {item} section for {ticker} FY{fy}")
        chunks = section_chunks(db, accession, item)
        if not chunks:
            raise ValueError(f"{qid}: {item} of {accession} has no chunks")
        where = {"ticker": ticker, "fiscal_year": fy, "item": item, "accession": accession}
        dense_ranking = [cid for cid, _ in dense(q["question"], rank_k, where)][:rank_k]
        bm25_top = [cid for cid, _ in bm25_rank(q["question"], chunks)[:pool_k]]
        out[qid] = {
            "question": q["question"],
            "accession": accession,
            "item": item,
            "dense_ranking": dense_ranking,
            "bm25_top": bm25_top,
            "pooled": sorted(set(dense_ranking[:pool_k]) | set(bm25_top)),
        }
    meta = {"pool_k": pool_k, "rank_k": rank_k,
            "built_at": datetime.now(UTC).isoformat(), "git_commit": _git_commit(),
            **(meta_extra or {})}
    return {"meta": meta, "questions": out}


def write_sheet(pool: dict, db: Path, out_csv: Path, seed: int) -> None:
    """Write the blind labelling sheet: shuffled passages, no rank or source, ``relevant`` blank."""
    rng = random.Random(seed)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SHEET_COLUMNS)
        for qid, entry in pool["questions"].items():
            chunks = section_chunks(db, entry["accession"], entry["item"])
            ids = list(entry["pooled"])
            rng.shuffle(ids)
            for row, chunk_id in enumerate(ids, start=1):
                writer.writerow([qid, entry["question"], row, chunk_id,
                                 chunks[chunk_id], ""])


def main(argv: list[str] | None = None) -> None:
    """Build ``pool.json`` and ``labelling_sheet.csv`` from the frozen question set."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args(argv)

    from filings_kb.embed import EmbeddingIndex
    from filings_kb.extract import MODELS_PATH

    models = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))["questions"]
    index = EmbeddingIndex(args.chroma, str(models["embedding"]), max_tokens=512)
    pool = build_pool(questions, args.db, index.query,
                      meta_extra={"embedding_model": str(models["embedding"])})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pool.json").write_text(
        json.dumps(pool, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_sheet(pool, args.db, args.out_dir / "labelling_sheet.csv", args.seed)


if __name__ == "__main__":
    main()
