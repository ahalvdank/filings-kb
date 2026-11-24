"""Year-over-year signals between consecutive 10-Ks: tone, sentiment, claim diff.

Motivation: Cohen, Malloy and Nguyen, "Lazy Prices" (Journal of Finance, 2020),
found that year-over-year changes in the language of 10-K sections carry
information. Everything here is research output: every signal is labelled
``"status": "shadow"`` and nothing in this repository trades on it.

* ``tone_shift``: 1 - cosine(centroid of prior-year chunk embeddings, centroid of
  current-year ones), read back from the vector index (nothing is re-embedded).
* ``sentiment_shift``: mean FinBERT (positive - negative) per chunk, current minus
  prior. FinBERT truncates at 512 BERT tokens; truncated chunks are counted.
* ``claim_diff``: an LLM compares the two years' extracted claims. Results are
  cached on disk so a re-run never calls the backend again.

A pair is skipped (all signals None) when either side has no section. For a
fiscal year with a 10-K/A, the amendment's section supersedes the original's.
"""

import hashlib
import json
import logging
import math
import re
import statistics
import subprocess
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from filings_kb.embed import EmbeddingIndex
from filings_kb.llm import LLMBackend
from filings_kb.store import ClaimRow, Store

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "prompts" / "claim_diff.md"
SIGNALS_PATH = REPO_ROOT / "data" / "committed" / "signals.json"
CLAIM_DIFF_DIR = REPO_ROOT / "data" / "committed" / "claim_diffs"
# Where a run without --write-committed writes (gitignored).
LOCAL_SIGNALS_PATH = REPO_ROOT / "data" / "local" / "signals.json"
LOCAL_CLAIM_DIFF_DIR = REPO_ROOT / "data" / "local" / "claim_diffs"

SIGNAL_ITEMS: tuple[str, ...] = ("Item_1A", "Item_7_MDA")
SIGNAL_STATUS = "shadow"
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_TOKENS = 512
MAX_ATTR_CHARS = 200
_ROUND_DIGITS = 6

Scorer = Callable[[list[str]], list[float]]
"""Texts -> one (positive - negative) sentiment score per text."""

PairKey = tuple[str, str, int, int]
"""(ticker, item, prior_fy, current_fy)."""


class ClaimDiff(BaseModel):
    """What changed between two years' claims for one section."""

    added: list[str]
    removed: list[str]
    changed: list[tuple[str, str]]


class _Change(BaseModel):
    before: str
    after: str


class _RawDiff(BaseModel):
    """The JSON the model is asked for; ``changed`` entries are objects, not pairs."""

    added: list[str]
    removed: list[str]
    changed: list[_Change]


@dataclass(frozen=True)
class SignalRow:
    """All signals for one ticker, item and consecutive fiscal-year pair.

    ``claim_diff_status`` is one of ``ok``, ``failed``, ``no_claims``,
    ``no_section`` or ``not_requested`` and says why ``claim_diff`` is None.
    """

    ticker: str
    item: str
    prior_fy: int
    current_fy: int
    tone_shift: float | None
    sentiment_shift: float | None
    claim_diff: ClaimDiff | None
    claim_diff_status: str
    status: str = SIGNAL_STATUS


def _where(ticker: str, fiscal_year: int, item: str, accession: str | None) -> dict[str, Any]:
    where: dict[str, Any] = {"ticker": ticker, "fiscal_year": fiscal_year, "item": item}
    if accession is not None:
        where["accession"] = accession
    return where


def _centroid(vectors: list[list[float]]) -> list[float]:
    """Component-wise mean of equal-length vectors."""
    return [math.fsum(col) / len(vectors) for col in zip(*vectors)]


def tone_shift(
    index: EmbeddingIndex, ticker: str, item: str, prior_fy: int, current_fy: int,
    *, prior_accession: str | None = None, current_accession: str | None = None,
) -> float | None:
    """1 - cosine(centroid_prior, centroid_current) over the stored chunk embeddings.

    Args:
        index: vector index holding the chunks' embeddings.
        ticker, item, prior_fy, current_fy: the section pair to compare.
        prior_accession, current_accession: restrict a side to one filing, needed
            when a fiscal year has both a 10-K and a 10-K/A carrying the section.

    Returns:
        A value in [0, 2] (0 = identical direction), or None if either side has no
        chunks or a centroid is the zero vector (cosine undefined; logged).
    """
    prior = index.vectors(_where(ticker, prior_fy, item, prior_accession))
    current = index.vectors(_where(ticker, current_fy, item, current_accession))
    if not prior or not current:
        return None
    a, b = _centroid(prior), _centroid(current)
    na, nb = math.sqrt(math.fsum(x * x for x in a)), math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        logger.warning("zero-norm centroid for %s %s %s->%s; tone_shift undefined",
                       ticker, item, prior_fy, current_fy)
        return None
    cosine = math.fsum(x * y for x, y in zip(a, b)) / (na * nb)
    return max(0.0, 1.0 - cosine)  # cosine <= 1 mathematically; clamp float noise


class FinbertScorer:
    """Local FinBERT scorer: per text, P(positive) - P(negative).

    The model loads on first call. Inputs longer than ``max_length`` BERT tokens
    are truncated by the tokenizer; ``truncated`` counts them and ``scored`` counts
    every text scored, so the loss is reported rather than hidden.
    """

    def __init__(self, model_name: str = FINBERT_MODEL, max_length: int = FINBERT_MAX_TOKENS,
                 batch_size: int = 16) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._pipe: Any = None
        self.truncated = 0
        self.scored = 0

    def _load(self) -> Any:
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info("loading %s on %s", self._model_name, device)
            self._pipe = pipeline("text-classification", model=self._model_name,
                                  top_k=None, device=device)
        return self._pipe

    def __call__(self, texts: list[str]) -> list[float]:
        """Score ``texts``; raises ValueError if the model lacks positive/negative labels."""
        if not texts:
            return []
        pipe = self._load()
        for t in texts:
            ids = pipe.tokenizer(t, truncation=True, max_length=self._max_length + 1)["input_ids"]
            if len(ids) > self._max_length:
                self.truncated += 1
        outputs = pipe(texts, batch_size=self._batch_size, truncation=True,
                       max_length=self._max_length)
        scores: list[float] = []
        for labels in outputs:
            by_label = {d["label"].lower(): float(d["score"]) for d in labels}
            if "positive" not in by_label or "negative" not in by_label:
                raise ValueError(f"{self._model_name} labels {sorted(by_label)} lack positive/negative")
            scores.append(by_label["positive"] - by_label["negative"])
        self.scored += len(texts)
        return scores


@lru_cache(maxsize=1)
def _default_scorer() -> FinbertScorer:
    """One shared FinBERT scorer, so the model loads once per process."""
    return FinbertScorer()


SentimentCache = dict[tuple[str, str], float | None]


def _section_sentiment(
    store: Store, accession: str, item: str, scorer: Scorer, cache: SentimentCache | None,
) -> float | None:
    """Mean chunk score of one filing's section, memoised in ``cache`` when given."""
    key = (accession, item)
    if cache is not None and key in cache:
        return cache[key]
    texts = store.chunk_texts(accession, item)
    value = statistics.fmean(scorer(texts)) if texts else None
    if cache is not None:
        cache[key] = value
    return value


def sentiment_shift(
    store: Store, ticker: str, item: str, prior_fy: int, current_fy: int,
    *, scorer: Scorer | None = None, cache: SentimentCache | None = None,
) -> float | None:
    """Mean chunk sentiment (positive - negative) of the current year minus the prior year.

    Args:
        store: knowledge store holding the chunk texts.
        ticker, item, prior_fy, current_fy: the section pair to compare.
        scorer: texts -> scores; defaults to a shared local FinBERT.
        cache: optional memo of per-section means, so a section used by two
            adjacent pairs is scored once.

    Returns:
        current mean - prior mean, or None if either side has no chunks.
    """
    prior_acc = store.section_accession(ticker, prior_fy, item)
    current_acc = store.section_accession(ticker, current_fy, item)
    if prior_acc is None or current_acc is None:
        return None
    score = scorer or _default_scorer()
    prior = _section_sentiment(store, prior_acc, item, score, cache)
    current = _section_sentiment(store, current_acc, item, score, cache)
    if prior is None or current is None:
        return None
    return current - prior


def _claim_line(c: ClaimRow) -> str:
    """One prompt line: ``[Type] name | key=value; ...`` (long values truncated)."""
    attrs = "; ".join(f"{k}={str(v)[:MAX_ATTR_CHARS]}" for k, v in sorted(c.attributes.items()))
    return f"[{c.claim_type}] {c.name}" + (f" | {attrs}" if attrs else "")


def build_claim_diff_input(prior: list[ClaimRow], current: list[ClaimRow],
                           context: str = "") -> str:
    """The user message for one comparison: both years' claims, one per line."""
    head = f"{context}\n\n" if context else ""
    return (f"{head}PRIOR YEAR CLAIMS:\n" + "\n".join(_claim_line(c) for c in prior)
            + "\n\nCURRENT YEAR CLAIMS:\n" + "\n".join(_claim_line(c) for c in current))


def _parse_diff(text: str) -> ClaimDiff:
    """Parse and validate the model's reply.

    Raises:
        ValueError: if the reply is not JSON or does not match the schema.
    """
    body = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        raw = _RawDiff.model_validate_json(body)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return ClaimDiff(added=raw.added, removed=raw.removed,
                     changed=[(c.before, c.after) for c in raw.changed])


def claim_diff(
    prior: list[ClaimRow], current: list[ClaimRow], backend: LLMBackend, model: str,
    *, context: str = "", prompt_path: Path = PROMPT_PATH,
) -> ClaimDiff | None:
    """Ask the model what changed between two years' claims.

    Invalid output (or a backend error) is retried once; after that the failure is
    logged and None returned.

    Args:
        prior, current: the claims of the two years for one ticker and item.
        backend: LLM backend; model: model name passed to it.
        context: optional line naming the ticker, item and years.
        prompt_path: system prompt file.
    """
    system = prompt_path.read_text(encoding="utf-8")
    user = build_claim_diff_input(prior, current, context)
    reminder = "\n\nYour previous reply was invalid. Reply with the JSON object only."
    last_error = ""
    for attempt in (1, 2):
        message = user if attempt == 1 else user + reminder
        try:
            reply = backend.complete(system, message, model)
            return _parse_diff(reply.text)
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("claim_diff attempt %d failed (%s): %s", attempt, context, last_error)
    logger.warning("claim_diff giving up (%s): %s", context, last_error)
    return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cached_claim_diff(
    prior: list[ClaimRow], current: list[ClaimRow], backend: LLMBackend, model: str,
    key: PairKey, cache_dir: Path, prompt_path: Path, counters: dict[str, int],
    read_cache_dirs: Sequence[Path] = (),
) -> ClaimDiff | None:
    """``claim_diff`` behind an on-disk cache keyed by pair, model, prompt and input hashes.

    ``read_cache_dirs`` are tried first, then ``cache_dir``; a hit never writes (so its
    ``created_at`` stays). A miss is computed and written to ``cache_dir`` only.
    """
    ticker, item, prior_fy, current_fy = key
    context = f"Ticker {ticker}, section {item}, fiscal year {prior_fy} -> {current_fy}."
    prompt_sha = _sha256(prompt_path.read_text(encoding="utf-8"))
    input_sha = _sha256(build_claim_diff_input(prior, current, context))
    name = f"{ticker}_{item}_{prior_fy}_{current_fy}.json"
    path = cache_dir / name
    for candidate in dict.fromkeys([*(d / name for d in read_cache_dirs), path]):
        if not candidate.exists():
            continue
        try:
            cached = json.loads(candidate.read_text(encoding="utf-8"))
            if (cached["model"], cached["prompt_sha256"], cached["input_sha256"]) == (
                    model, prompt_sha, input_sha):
                counters["cache_hits"] += 1
                return ClaimDiff.model_validate(cached["result"])
            logger.info("claim_diff cache stale for %s", candidate)
        except (json.JSONDecodeError, KeyError, ValidationError) as exc:
            logger.warning("claim_diff cache unreadable for %s (%s)", candidate, exc)
    counters["calls"] += 1
    result = claim_diff(prior, current, backend, model, context=context,
                        prompt_path=prompt_path)
    if result is None:
        counters["failures"] += 1
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ticker": ticker, "item": item, "prior_fy": prior_fy, "current_fy": current_fy,
        "model": model, "prompt_sha256": prompt_sha, "input_sha256": input_sha,
        "created_at": datetime.now(UTC).isoformat(),
        "result": result.model_dump(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _year_pairs(years: list[int]) -> list[tuple[int, int]]:
    """Consecutive fiscal-year pairs (y, y+1) present in ``years``."""
    present = set(years)
    return [(y, y + 1) for y in sorted(present) if y + 1 in present]


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, _ROUND_DIGITS)


def _row_json(r: SignalRow) -> dict[str, Any]:
    return {
        "ticker": r.ticker, "item": r.item, "prior_fy": r.prior_fy,
        "current_fy": r.current_fy, "tone_shift": _round(r.tone_shift),
        "sentiment_shift": _round(r.sentiment_shift),
        "claim_diff": r.claim_diff.model_dump() if r.claim_diff else None,
        "claim_diff_status": r.claim_diff_status, "status": r.status,
    }


def compute_signals(
    store: Store, index: EmbeddingIndex, backend: LLMBackend, model: str,
    *, scorer: Scorer | None = None, claim_diff_pairs: Collection[PairKey] | None = None,
    out_path: Path = SIGNALS_PATH, cache_dir: Path = CLAIM_DIFF_DIR,
    items: Collection[str] = SIGNAL_ITEMS, prompt_path: Path = PROMPT_PATH,
    read_cache_dirs: Sequence[Path] = (),
) -> list[SignalRow]:
    """Compute signals for every consecutive fiscal-year pair and write ``out_path``.

    Args:
        store, index: the knowledge store and its vector index.
        backend, model: LLM backend and model for ``claim_diff`` (cost guard below).
        scorer: sentiment scorer; defaults to local FinBERT.
        claim_diff_pairs: ``(ticker, item, prior_fy, current_fy)`` tuples to run
            claim_diff for. Default None means the latest pair of each ticker for
            every item; pass a wider set (or ``set()`` for none) to change it.
        out_path: where the deterministic JSON (sorted keys and rows) is written.
        cache_dir: claim_diff cache directory; new claim diffs are written here.
        read_cache_dirs: further claim_diff caches read before ``cache_dir`` and never
            written (the committed cache, for a run that writes locally).
        items: sections to cover.
        prompt_path: claim_diff system prompt.

    Returns:
        Rows sorted by (ticker, item, prior_fy); pairs missing a section are kept
        with None signals.
    """
    score = scorer or _default_scorer()
    years_by_ticker = store.fiscal_years()
    if claim_diff_pairs is None:
        wanted: set[PairKey] = set()
        for ticker, years in years_by_ticker.items():
            pairs = _year_pairs(years)
            if pairs:
                wanted.update((ticker, item, *pairs[-1]) for item in items)
    else:
        wanted = set(claim_diff_pairs)

    sentiment_cache: SentimentCache = {}
    counters = {"calls": 0, "failures": 0, "cache_hits": 0}
    rows: list[SignalRow] = []
    for ticker in sorted(years_by_ticker):
        for item in sorted(items):
            for prior_fy, current_fy in _year_pairs(years_by_ticker[ticker]):
                key: PairKey = (ticker, item, prior_fy, current_fy)
                prior_acc = store.section_accession(ticker, prior_fy, item)
                current_acc = store.section_accession(ticker, current_fy, item)
                if prior_acc is None or current_acc is None:
                    rows.append(SignalRow(ticker, item, prior_fy, current_fy, None, None,
                                          None, "no_section"))
                    continue
                tone = tone_shift(index, ticker, item, prior_fy, current_fy,
                                  prior_accession=prior_acc, current_accession=current_acc)
                sent = sentiment_shift(store, ticker, item, prior_fy, current_fy,
                                       scorer=score, cache=sentiment_cache)
                diff: ClaimDiff | None = None
                diff_status = "not_requested"
                if key in wanted:
                    prior_claims = [c for c in store.claims_for(prior_acc) if c.item == item]
                    current_claims = [c for c in store.claims_for(current_acc) if c.item == item]
                    if not prior_claims or not current_claims:
                        diff_status = "no_claims"
                    else:
                        diff = _cached_claim_diff(prior_claims, current_claims, backend, model,
                                                  key, cache_dir, prompt_path, counters,
                                                  read_cache_dirs)
                        diff_status = "ok" if diff is not None else "failed"
                rows.append(SignalRow(ticker, item, prior_fy, current_fy, tone, sent,
                                      diff, diff_status))

    meta = {
        "status": SIGNAL_STATUS,
        "sentiment_model": FINBERT_MODEL,
        "claim_diff_model": model,
        "claim_diff_prompt_sha256": _sha256(prompt_path.read_text(encoding="utf-8")),
        "rows": len(rows),
        "rows_with_none_tone": sum(r.tone_shift is None for r in rows),
        "rows_with_none_sentiment": sum(r.sentiment_shift is None for r in rows),
        "sentiment_scored_chunks": getattr(score, "scored", None),
        "sentiment_truncated_chunks": getattr(score, "truncated", None),
        "sentiment_max_tokens": FINBERT_MAX_TOKENS,
        "claim_diff_requested": len(wanted),
        "claim_diff_ok": sum(r.claim_diff_status == "ok" for r in rows),
        "claim_diff_failed": sum(r.claim_diff_status == "failed" for r in rows),
        "claim_diff_no_claims": sum(r.claim_diff_status == "no_claims" for r in rows),
    }
    if meta["sentiment_truncated_chunks"]:
        logger.info("FinBERT truncated %s of %s chunks at %d tokens",
                    meta["sentiment_truncated_chunks"], meta["sentiment_scored_chunks"],
                    FINBERT_MAX_TOKENS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "signals": [_row_json(r) for r in rows]},
                                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("claim_diff: %d calls, %d failures, %d cache hits", counters["calls"],
                counters["failures"], counters["cache_hits"])
    return rows


def signals_drift(path: Path, committed: Path) -> list[str]:
    """How the signal rows in ``path`` differ from the ``committed`` ones, one line each.

    Rows are matched on (ticker, item, prior_fy, current_fy); every field that differs is
    listed with both values (``meta`` is not compared).

    Returns:
        The differences (empty means none), or a single note when ``committed`` is missing.
    """
    if not committed.exists():
        return ["no committed signals to compare"]

    def rows(p: Path) -> dict[PairKey, dict[str, Any]]:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {(r["ticker"], r["item"], r["prior_fy"], r["current_fy"]): r
                for r in data["signals"]}

    def show(value: Any) -> str:
        text = json.dumps(value, sort_keys=True)
        return text if len(text) <= 80 else text[:80] + "..."

    before, after = rows(committed), rows(path)
    out: list[str] = []
    for key in sorted(set(before) | set(after)):
        label = f"{key[0]} {key[1]} {key[2]}->{key[3]}"
        if key not in before:
            out.append(f"{label}: not in the committed signals")
            continue
        if key not in after:
            out.append(f"{label}: missing from this run")
            continue
        for name in sorted(set(before[key]) | set(after[key])):
            if before[key].get(name) != after[key].get(name):
                out.append(f"{label}: {name} {show(before[key].get(name))} -> "
                           f"{show(after[key].get(name))}")
    return out
