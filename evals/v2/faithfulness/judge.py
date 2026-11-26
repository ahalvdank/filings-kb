"""LLM judge for faithfulness units, with a hard Decimal budget and resume.

Each unit gets one Messages API call with structured output (``output_config.format``,
a JSON schema whose ``verdict`` enum is ``supported | partial | unsupported``). The harness
adds two verdicts of its own: ``refusal`` (``stop_reason == "refusal"``) and
``parse_error`` (any other non-``end_turn`` stop, no text block, or output that is not
exactly the schema). The judge model is fixed: no fallback model is configured.

Records are append-only. A rerun on the same ``--out`` file resumes: it skips every unit
already recorded (``parse_error`` and ``refusal`` included) and counts the spend already
recorded toward ``--max-usd``. To retry failures, run again with a new ``--run-label`` and
a new ``--out`` file over the units to retry; never delete or edit lines in a run file.

``--provider openai`` judges with OpenAI's Responses API instead (``judge_unit_openai``):
the same prompt as ``instructions``, the same user message as ``input``, and the same
schema as a strict Structured Outputs ``json_schema`` format. A ``refusal`` content item
gives ``refusal``; a response whose status is not ``completed`` (``incomplete`` at
``max_output_tokens``, for one) or whose text is not exactly the schema gives
``parse_error``. Its records carry ``provider: "openai"``; records with no ``provider``
field are Anthropic's.

CLI (makes paid calls; needs ``ANTHROPIC_API_KEY``, or ``OPENAI_API_KEY`` with
``--provider openai``)::

    python -m evals.v2.faithfulness.judge --units evals/v2/faithfulness/units.jsonl \
        --out evals/v2/faithfulness/judge_run1.jsonl --run-label run1 --max-usd 15 [--limit N] \
        [--provider openai] [--model gpt-6.2-sol]
"""

import argparse
import hashlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from evals.v2.faithfulness.units import format_passages, read_units

MODEL = "claude-opus-5-5"
EFFORT = "high"
# Thinking at high effort counts against max_tokens; too small a cap turns real verdicts
# into parse errors. 8000 also keeps the non-streaming call inside the SDK's time limit.
MAX_TOKENS = 8000
PROMPT_PATH = Path(__file__).with_name("judge_prompt.md")
# USD per million tokens.
PRICING: dict[str, dict[str, Decimal]] = {
    "claude-opus-5-5": {"input": Decimal("4"), "output": Decimal("20"),
                        "cache_read": Decimal("0.2"), "cache_write": Decimal("5")},
}
# USD per million tokens. Source: https://developers.openai.com/api/docs/pricing, Standard
# tier, <=272K context, read 2026-11-26. ``cached_tokens`` is a subset of ``input_tokens``
# and ``reasoning_tokens`` a subset of ``output_tokens`` (openai 3.60.0,
# types/responses/response_usage.py), so each token is billed once.
OPENAI_MODEL = "gpt-6.2-sol"
OPENAI_PRICING: dict[str, dict[str, Decimal]] = {
    "gpt-6.2-sol": {"input": Decimal("2.00"), "cached_input": Decimal("0.10"),
                    "output": Decimal("10.00")},
}
PROVIDERS = ("anthropic", "openai")
DEFAULT_MODELS = {"anthropic": MODEL, "openai": OPENAI_MODEL}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
_PER_TOKEN = Decimal(1_000_000)
# Used only to project the first call's cost; deliberately high (English runs ~4 chars/token).
CHARS_PER_TOKEN_ESTIMATE = 3

JUDGE_VERDICTS = ("supported", "partial", "unsupported")
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(JUDGE_VERDICTS)},
        "unsupported_span": {"type": ["string", "null"]},
    },
    "required": ["verdict", "unsupported_span"],
    "additionalProperties": False,
}
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens")
# OpenAI usage is stored in OpenAI's own terms; ``input_tokens`` includes ``cached_tokens``.
OPENAI_USAGE_FIELDS = ("input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens",
                       "reasoning_tokens")
OPENAI_FORMAT_NAME = "faithfulness_verdict"


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def user_message(unit: dict) -> str:
    """The judge's input: the sentence, then each cited passage under its cite number."""
    return "\n".join(["SENTENCE", unit["sentence"], "", "CITED PASSAGES", format_passages(unit)])


def call_cost(usage: dict[str, int], model: str = MODEL) -> Decimal:
    """Exact USD cost of one call from its token usage.

    Raises:
        KeyError: if ``model`` has no entry in ``PRICING`` or ``OPENAI_PRICING``.
    """
    if model in OPENAI_PRICING:
        return _openai_call_cost(usage, model)
    price = PRICING[model]
    micro = (Decimal(usage.get("input_tokens", 0)) * price["input"]
             + Decimal(usage.get("output_tokens", 0)) * price["output"]
             + Decimal(usage.get("cache_read_input_tokens", 0)) * price["cache_read"]
             + Decimal(usage.get("cache_creation_input_tokens", 0)) * price["cache_write"])
    return micro / _PER_TOKEN


def _openai_call_cost(usage: dict[str, int], model: str) -> Decimal:
    """Cached input at the cached rate, the rest of the input at the input rate, and all
    output (reasoning included) at the output rate. ``cache_write_tokens`` has no separate
    price here and is billed inside ``input_tokens``."""
    price = OPENAI_PRICING[model]
    cached = Decimal(usage.get("cached_tokens", 0))
    micro = ((Decimal(usage.get("input_tokens", 0)) - cached) * price["input"]
             + cached * price["cached_input"]
             + Decimal(usage.get("output_tokens", 0)) * price["output"])
    return micro / _PER_TOKEN


def projected_call_cost(prompt: str, unit: dict, model: str = MODEL) -> Decimal:
    """Worst-case cost of one call for ``unit``: full ``MAX_TOKENS`` output plus estimated input."""
    chars = len(prompt) + len(user_message(unit))
    est_input = -(-chars // CHARS_PER_TOKEN_ESTIMATE)  # ceiling division
    return call_cost({"input_tokens": est_input, "output_tokens": MAX_TOKENS}, model)


def _usage(response: Any) -> dict[str, int]:
    return {f: int(getattr(response.usage, f, None) or 0) for f in USAGE_FIELDS}


def _parse(text: str) -> tuple[str, str | None] | None:
    """``(verdict, unsupported_span)`` if ``text`` is exactly the schema, else None."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != {"verdict", "unsupported_span"}:
        return None
    verdict, span = data["verdict"], data["unsupported_span"]
    if verdict not in JUDGE_VERDICTS or not (span is None or isinstance(span, str)):
        return None
    return verdict, span


def judge_unit(client: Any, unit: dict, prompt: str, model: str = MODEL,
               effort: str = EFFORT) -> dict:
    """One judge call for one unit.

    Args:
        client: an ``anthropic.Anthropic`` (or a fake with ``messages.create``).
        unit: a unit from ``units.build_units``.
        prompt: the judge instructions, sent as the system prompt.

    Returns:
        ``{"unit_id", "verdict", "unsupported_span", "raw", "stop_reason", "usage"}``;
        ``verdict`` is a schema verdict, ``refusal`` or ``parse_error``.
    """
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=prompt,
        messages=[{"role": "user", "content": user_message(unit)}],
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    )
    raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    verdict, span = "parse_error", None
    if response.stop_reason == "refusal":
        verdict = "refusal"
    elif response.stop_reason == "end_turn" and raw:
        parsed = _parse(raw)
        if parsed is not None:
            verdict, span = parsed
    return {"unit_id": unit["unit_id"], "verdict": verdict, "unsupported_span": span, "raw": raw,
            "stop_reason": response.stop_reason, "usage": _usage(response)}


def _openai_usage(response: Any) -> dict[str, int]:
    """Token counts from an ``openai.types.responses.ResponseUsage``; zeros if absent."""
    usage = response.usage
    if usage is None:
        return {f: 0 for f in OPENAI_USAGE_FIELDS}
    inp = getattr(usage, "input_tokens_details", None)
    out = getattr(usage, "output_tokens_details", None)
    return {"input_tokens": int(usage.input_tokens or 0),
            "output_tokens": int(usage.output_tokens or 0),
            "cached_tokens": int(getattr(inp, "cached_tokens", None) or 0),
            "cache_write_tokens": int(getattr(inp, "cache_write_tokens", None) or 0),
            "reasoning_tokens": int(getattr(out, "reasoning_tokens", None) or 0)}


def judge_unit_openai(client: Any, unit: dict, prompt: str, model: str = OPENAI_MODEL,
                      effort: str = EFFORT) -> dict:
    """One OpenAI Responses API judge call for one unit.

    Args:
        client: an ``openai.OpenAI`` (or a fake with ``responses.create``).
        unit: a unit from ``units.build_units``.
        prompt: the judge instructions, sent as ``instructions``.

    Returns:
        The same keys as ``judge_unit``. ``stop_reason`` is the response status, with
        ``incomplete_details.reason`` appended after a colon when present; ``raw`` is
        ``output_text``, or the refusal text for a refusal.
    """
    response = client.responses.create(
        model=model,
        instructions=prompt,
        input=user_message(unit),
        max_output_tokens=MAX_TOKENS,
        reasoning={"effort": effort},
        text={"format": {"type": "json_schema", "name": OPENAI_FORMAT_NAME,
                         "schema": VERDICT_SCHEMA, "strict": True}},
    )
    refusals = [c.refusal for item in response.output if getattr(item, "type", None) == "message"
                for c in item.content if getattr(c, "type", None) == "refusal"]
    raw = response.output_text or ""
    stop_reason = response.status
    reason = getattr(response.incomplete_details, "reason", None)
    if reason:
        stop_reason = f"{stop_reason}:{reason}"
    verdict, span = "parse_error", None
    if refusals:
        verdict, raw = "refusal", "".join(refusals)
    elif response.status == "completed" and raw:
        parsed = _parse(raw)
        if parsed is not None:
            verdict, span = parsed
    return {"unit_id": unit["unit_id"], "verdict": verdict, "unsupported_span": span, "raw": raw,
            "stop_reason": stop_reason, "usage": _openai_usage(response)}


JUDGES = {"anthropic": judge_unit, "openai": judge_unit_openai}
_PRICING_BY_PROVIDER = {"anthropic": PRICING, "openai": OPENAI_PRICING}


def _existing(out_jsonl: Path, run_label: str, model: str, effort: str,
              prompt_sha: str, provider: str = "anthropic") -> list[dict]:
    """Records already in ``out_jsonl``; raises if any came from a different setup.

    A record with no ``provider`` field counts as ``anthropic``.

    Raises:
        ValueError: if a record's run label, model, effort, prompt hash or provider
            differs, so one file never mixes two judge configurations.
    """
    if not out_jsonl.exists():
        return []
    records = read_units(out_jsonl)
    want = {"run_label": run_label, "model": model, "effort": effort, "prompt_sha256": prompt_sha}
    for r in records:
        diff = {k: r.get(k) for k, v in want.items() if r.get(k) != v}
        if r.get("provider", "anthropic") != provider:
            diff["provider"] = r.get("provider", "anthropic")
        if diff:
            raise ValueError(f"{out_jsonl}: record {r.get('unit_id')} differs from this run: {diff}")
    return records


def run(units: list[dict], out_jsonl: Path, run_label: str, max_usd: Decimal = Decimal("15"), *,
        client: Any, prompt_path: Path = PROMPT_PATH, model: str = MODEL, effort: str = EFFORT,
        limit: int | None = None, provider: str = "anthropic") -> dict:
    """Judge every unit not already in ``out_jsonl``, appending one record per call.

    Spend already recorded in ``out_jsonl`` counts toward ``max_usd``, so restarting does
    not reset the cap. Before each call the cost is projected as the larger of the most
    expensive call so far and ``projected_call_cost`` for this unit (full ``MAX_TOKENS``
    output), so a unit larger than any seen cannot push spend over the cap, and the run
    stops if ``spent + projected > max_usd``.

    ``provider`` picks the judge call (``JUDGES``); ``model`` must be priced for it.
    Non-Anthropic records carry a ``provider`` field; Anthropic records are written as
    before, without one.

    Returns:
        ``{"called", "skipped", "spent_usd": Decimal, "stopped_on_budget": bool}``.

    Raises:
        ValueError: on an unknown provider, a model with no price for that provider, or
            an ``out_jsonl`` written with a different configuration.
    """
    if model not in _PRICING_BY_PROVIDER.get(provider, {}):
        raise ValueError(f"no {provider!r} price for model {model!r}")
    judge_call = JUDGES[provider]
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_file(prompt_path)
    existing = _existing(out_jsonl, run_label, model, effort, prompt_sha, provider)
    done = {r["unit_id"] for r in existing}
    costs = [call_cost(r["usage"], r["model"]) for r in existing]
    spent = sum(costs, Decimal(0))
    max_cost = max(costs) if costs else None
    called = skipped = 0
    stopped = False
    with out_jsonl.open("a", encoding="utf-8") as out:
        for unit in units:
            if unit["unit_id"] in done:
                skipped += 1
                continue
            if limit is not None and called >= limit:
                break
            projected = projected_call_cost(prompt, unit, model)
            if max_cost is not None:
                projected = max(projected, max_cost)
            if spent + projected > max_usd:
                stopped = True
                break
            result = judge_call(client, unit, prompt, model=model, effort=effort)
            cost = call_cost(result["usage"], model)
            spent += cost
            max_cost = cost if max_cost is None else max(max_cost, cost)
            called += 1
            record = {**result, "model": model, "effort": effort, "prompt_sha256": prompt_sha,
                      "run_label": run_label, "cost_usd": str(cost)}
            if provider != "anthropic":
                record["provider"] = provider
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return {"called": called, "skipped": skipped, "spent_usd": spent, "stopped_on_budget": stopped}


def api_key(env_file: Path | None, provider: str = "anthropic") -> str | None:
    """The provider's key (``KEY_ENV``) from the environment, else from ``env_file``.

    Only that one variable is read from the file, and nothing is put into
    ``os.environ``, so the file's other settings never reach this process.
    """
    name = KEY_ENV[provider]
    key = os.environ.get(name)
    if not key and env_file is not None:
        from dotenv import dotenv_values

        key = dotenv_values(env_file).get(name)
    return key or None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge faithfulness units (paid API calls).")
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--max-usd", type=Decimal, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--env-file", type=Path, default=None,
                        help="read the provider's API key from this .env when it is not set")
    parser.add_argument("--provider", choices=PROVIDERS, default="anthropic")
    parser.add_argument("--model", default=None,
                        help="judge model (default: claude-opus-5-5 or gpt-6.2-sol by provider)")
    args = parser.parse_args(argv)

    key = api_key(args.env_file, args.provider)
    if not key:
        print(f"{KEY_ENV[args.provider]} is not set; the judge makes paid API calls and needs it.",
              file=sys.stderr)
        sys.exit(2)
    if args.provider == "openai":
        import openai  # the optional extra; only needed for a real run

        client: Any = openai.OpenAI(api_key=key)
    else:
        import anthropic  # the optional extra; only needed for a real run

        client = anthropic.Anthropic(api_key=key)

    summary = run(read_units(args.units), args.out, args.run_label, args.max_usd,
                  client=client, model=args.model or DEFAULT_MODELS[args.provider],
                  limit=args.limit, provider=args.provider)
    print(f"called {summary['called']}, skipped {summary['skipped']}, "
          f"spent ${summary['spent_usd']}, stopped on budget: {summary['stopped_on_budget']}")


if __name__ == "__main__":
    main()
