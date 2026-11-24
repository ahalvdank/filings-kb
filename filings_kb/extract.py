"""Section extraction: v3 prompt, pydantic validation, one retry, on-disk cache.

Cache layout: ``<cache_dir>/<accession>/<item>.json`` holds an envelope::

    {accession, item, prompt_version, model, origin: "live"|"imported",
     parts: [{text_sha256, extracted_at, payload}]}

A live extraction writes one part. An import from the private pipeline can
carry several parts, because that pipeline split long sections. A cache hit
needs a matching ``prompt_version`` and ``model``; ``live`` envelopes also need
every part's text hash to match the section. ``imported`` envelopes skip the
hash check (see ``_read_cached``), so they never trigger a paid re-run.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from filings_kb.llm import LLMBackend
from filings_kb.schema import ExtractionPayload
from filings_kb.sections import Section, section_text_sha256
from filings_kb.store import Store

logger = logging.getLogger(__name__)
PROMPT_VERSION = "v3"
_REPO = Path(__file__).resolve().parent.parent
PROMPT_PATH = _REPO / "prompts" / "extract_v3.md"
MODELS_PATH = _REPO / "config" / "models.yaml"
_MTOK = Decimal(1_000_000)


def load_price(model: str, path: Path = MODELS_PATH) -> tuple[Decimal, Decimal]:
    """Return (input, output) USD per million tokens for ``model``.

    Raises:
        ValueError: if the model has no entry in the ``prices`` table.
    """
    prices = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("prices", {})
    if model not in prices:
        raise ValueError(f"no price for model {model!r} in {path}")
    entry = prices[model]
    return Decimal(str(entry["input_per_mtok"])), Decimal(str(entry["output_per_mtok"]))


def cache_path(cache_dir: Path, accession: str, item: str) -> Path:
    """Path of the cache envelope for one section."""
    return cache_dir / accession / f"{item}.json"


def write_envelope(path: Path, envelope: dict[str, Any]) -> None:
    """Write an envelope deterministically (sorted keys, fixed indent, newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class EnvelopePart:
    """One extraction call's output with its provenance."""

    payload: ExtractionPayload
    text_sha256: str
    extracted_at: str
    source_file: str


@dataclass(frozen=True)
class Envelope:
    """A parsed cache file."""

    origin: str
    prompt_version: str
    model: str
    parts: list[EnvelopePart]


def read_envelope(path: Path) -> Envelope:
    """Parse and validate a cache envelope, keeping per-part provenance.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        pydantic.ValidationError: if a part's payload fails the schema.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for part in raw["parts"]:
        payload = ExtractionPayload.model_validate(part["payload"])
        parts.append(EnvelopePart(
            payload=payload, text_sha256=part["text_sha256"],
            extracted_at=part["extracted_at"], source_file=payload.source_file,
        ))
    return Envelope(raw["origin"], raw["prompt_version"], raw["model"], parts)


def _canonical(obj: BaseModel) -> str:
    return json.dumps(obj.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def _merge_parts(parts: list[EnvelopePart]) -> ExtractionPayload:
    """Concatenate parts, dropping exact duplicates (canonical JSON) across them."""
    merged: dict[str, Any] = {
        "source_file": parts[0].source_file if parts else "",
        "source_type": "sec_filing",
        "entities": [], "relationships": [], "chunks": [],
    }
    for key in ("entities", "relationships", "chunks"):
        seen: set[str] = set()
        for part in parts:
            for item in getattr(part.payload, key):
                canon = _canonical(item)
                if canon not in seen:
                    seen.add(canon)
                    merged[key].append(item.model_dump(mode="json"))
    return ExtractionPayload.model_validate(merged)


def _read_cached(path: Path, model: str, text_sha256: str) -> ExtractionPayload | None:
    """Return the merged payload of a usable cache file, else None.

    A hit needs matching prompt_version and model. For ``origin == "live"``
    every part must also match ``text_sha256``, so a changed section text is a
    miss. ``imported`` envelopes skip the hash check: the private pipeline split
    long sections, so their hashes never equal the whole-section hash, and Task
    6's fidelity check reports those hashes instead.
    """
    if not path.exists():
        return None
    envelope = read_envelope(path)
    if envelope.prompt_version != PROMPT_VERSION or envelope.model != model:
        return None
    if envelope.origin == "live" and any(
        p.text_sha256 != text_sha256 for p in envelope.parts
    ):
        logger.info("cache miss for %s: section text changed since extraction", path)
        return None
    return _merge_parts(envelope.parts)


def build_user_message(s: Section, filing_date: str | None = None) -> str:
    """Build the input block in the format the prompt's examples use."""
    header = [f"section_type: {s.item}"]
    if filing_date:
        header.append(f"filing_date: {filing_date}")
    header += [
        f"ticker: {s.ticker}",
        f"accession_number: {s.accession}",
        f"filing_entity_name: {s.ticker} 10-K {s.fiscal_year}",
        "text:",
        s.text,
    ]
    return "\n".join(header)


def _backend_errors() -> tuple[type[BaseException], ...]:
    """Exceptions a backend call raises when the call itself fails.

    ``RuntimeError`` (the CLI's exit or error result), ``subprocess.TimeoutExpired``
    (the CLI ran past its timeout) and, when the optional package is installed,
    ``anthropic.APIError``. It is imported here, after a cache miss, so a cache-only
    run never needs it.
    """
    errors: tuple[type[BaseException], ...] = (RuntimeError, subprocess.TimeoutExpired)
    try:
        import anthropic
    except ImportError:
        return errors
    return (*errors, anthropic.APIError)


def _first_json_object(text: str) -> Any:
    """Decode the first JSON object in ``text``; a code fence or prose around it is ignored.

    Raises:
        json.JSONDecodeError: if ``text`` holds no ``{`` or the object is not valid JSON.
    """
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object in the response", text, 0)
    data, _ = json.JSONDecoder().raw_decode(text, start)
    return data


def _cost(tokens_in: int, tokens_out: int, model: str) -> Decimal:
    price_in, price_out = load_price(model)
    return (Decimal(tokens_in) * price_in + Decimal(tokens_out) * price_out) / _MTOK


def extract_section(
    s: Section,
    backend: LLMBackend,
    model: str,
    cache_dir: Path,
    store: Store,
    filing_date: str | None = None,
    *,
    use_cache: bool = True,
    write_dir: Path | None = None,
) -> ExtractionPayload | None:
    """Extract one section, using the cache when possible (and ``use_cache`` is True).

    The first JSON object in the model output (a code fence or prose around it is
    ignored) is validated against ``ExtractionPayload``. On a parse or validation
    error the request is retried once with the error text appended. A second
    failure is logged as a failed run and ``None`` is returned; the cache is
    written only after validation passes. A backend error (``RuntimeError``,
    ``subprocess.TimeoutExpired`` or ``anthropic.APIError``) is logged as a failed
    run with the tokens and cost spent so far, and re-raised.

    Args:
        s: The section to extract.
        backend: Where completions come from.
        model: Model id; must have a price in ``config/models.yaml``.
        cache_dir: Root of the committed extraction cache.
        store: Receives one ``runs`` row per call (ok / failed / cache_hit).
        filing_date: Optional ISO date passed to the prompt as ``filing_date``.
        use_cache: False skips the cache read (a refresh); the result is still
            written to the cache, replacing what was there.
        write_dir: where a new envelope is written (default ``cache_dir``). A
            cache read tries ``cache_dir`` first, then ``write_dir``.

    Returns:
        The payload (parts concatenated on a cache hit), or None on failure.

    Raises:
        ValueError: if ``model`` has no price entry (checked before any call).
        RuntimeError, subprocess.TimeoutExpired, anthropic.APIError: if the backend
            itself fails (logged, not swallowed).
    """
    key = f"{s.accession}/{s.item}"
    write_dir = write_dir or cache_dir
    path = cache_path(write_dir, s.accession, s.item)
    text_sha = section_text_sha256(s.text)
    if use_cache:
        for read_dir in dict.fromkeys((cache_dir, write_dir)):
            cached = _read_cached(cache_path(read_dir, s.accession, s.item), model, text_sha)
            if cached is not None:
                store.log_run("extract", key, "cache_hit", 0, 0, Decimal("0"), "")
                return cached

    load_price(model)  # fail fast on an unpriced model, before spending anything
    system = PROMPT_PATH.read_text(encoding="utf-8")
    user = build_user_message(s, filing_date)
    backend_errors = _backend_errors()
    tokens_in = tokens_out = 0
    error = ""
    prompt = user
    for attempt in (1, 2):
        try:
            result = backend.complete(system, prompt, model)
        except backend_errors as exc:
            store.log_run("extract", key, "failed", tokens_in, tokens_out,
                          _cost(tokens_in, tokens_out, model),
                          f"backend error on attempt {attempt}: "
                          f"{type(exc).__name__}: {exc}"[:2000])
            raise
        tokens_in += result.tokens_in
        tokens_out += result.tokens_out
        try:
            data = _first_json_object(result.text)
            payload = ExtractionPayload.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            error = str(exc)
            prompt = (
                f"{user}\n\nYour previous response failed validation:\n{error}\n\n"
                f"Previous response:\n{result.text}\n\n"
                "Return corrected JSON only, matching the schema exactly."
            )
            continue
        write_envelope(path, {
            "accession": s.accession, "item": s.item,
            "prompt_version": PROMPT_VERSION, "model": model, "origin": "live",
            "parts": [{
                "text_sha256": text_sha,
                "extracted_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "payload": data,
            }],
        })
        store.log_run("extract", key, "ok", tokens_in, tokens_out,
                      _cost(tokens_in, tokens_out, model), f"attempts={attempt}")
        return payload

    store.log_run("extract", key, "failed", tokens_in, tokens_out,
                  _cost(tokens_in, tokens_out, model), error[:2000])
    return None
