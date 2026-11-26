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

CLI (makes paid calls; needs ``ANTHROPIC_API_KEY``)::

    python -m evals.v2.faithfulness.judge --units evals/v2/faithfulness/units.jsonl \
        --out evals/v2/faithfulness/judge_run1.jsonl --run-label run1 --max-usd 15 [--limit N]
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


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def user_message(unit: dict) -> str:
    """The judge's input: the sentence, then each cited passage under its cite number."""
    return "\n".join(["SENTENCE", unit["sentence"], "", "CITED PASSAGES", format_passages(unit)])


def call_cost(usage: dict[str, int], model: str = MODEL) -> Decimal:
    """Exact USD cost of one call from its token usage.

    Raises:
        KeyError: if ``model`` has no entry in ``PRICING``.
    """
    price = PRICING[model]
    micro = (Decimal(usage.get("input_tokens", 0)) * price["input"]
             + Decimal(usage.get("output_tokens", 0)) * price["output"]
             + Decimal(usage.get("cache_read_input_tokens", 0)) * price["cache_read"]
             + Decimal(usage.get("cache_creation_input_tokens", 0)) * price["cache_write"])
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


def _existing(out_jsonl: Path, run_label: str, model: str, effort: str,
              prompt_sha: str) -> list[dict]:
    """Records already in ``out_jsonl``; raises if any came from a different setup.

    Raises:
        ValueError: if a record's run label, model, effort or prompt hash differs, so one
            file never mixes two judge configurations.
    """
    if not out_jsonl.exists():
        return []
    records = read_units(out_jsonl)
    want = {"run_label": run_label, "model": model, "effort": effort, "prompt_sha256": prompt_sha}
    for r in records:
        diff = {k: r.get(k) for k, v in want.items() if r.get(k) != v}
        if diff:
            raise ValueError(f"{out_jsonl}: record {r.get('unit_id')} differs from this run: {diff}")
    return records


def run(units: list[dict], out_jsonl: Path, run_label: str, max_usd: Decimal = Decimal("15"), *,
        client: Any, prompt_path: Path = PROMPT_PATH, model: str = MODEL, effort: str = EFFORT,
        limit: int | None = None) -> dict:
    """Judge every unit not already in ``out_jsonl``, appending one record per call.

    Spend already recorded in ``out_jsonl`` counts toward ``max_usd``, so restarting does
    not reset the cap. Before each call the cost is projected as the larger of the most
    expensive call so far and ``projected_call_cost`` for this unit (full ``MAX_TOKENS``
    output), so a unit larger than any seen cannot push spend over the cap, and the run
    stops if ``spent + projected > max_usd``.

    Returns:
        ``{"called", "skipped", "spent_usd": Decimal, "stopped_on_budget": bool}``.
    """
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_file(prompt_path)
    existing = _existing(out_jsonl, run_label, model, effort, prompt_sha)
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
            result = judge_unit(client, unit, prompt, model=model, effort=effort)
            cost = call_cost(result["usage"], model)
            spent += cost
            max_cost = cost if max_cost is None else max(max_cost, cost)
            called += 1
            record = {**result, "model": model, "effort": effort, "prompt_sha256": prompt_sha,
                      "run_label": run_label, "cost_usd": str(cost)}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return {"called": called, "skipped": skipped, "spent_usd": spent, "stopped_on_budget": stopped}


def api_key(env_file: Path | None) -> str | None:
    """``ANTHROPIC_API_KEY`` from the environment, else from ``env_file`` if given.

    Only that one variable is read from the file, and nothing is put into
    ``os.environ``, so the file's other settings never reach this process.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and env_file is not None:
        from dotenv import dotenv_values

        key = dotenv_values(env_file).get("ANTHROPIC_API_KEY")
    return key or None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge faithfulness units (paid API calls).")
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--max-usd", type=Decimal, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--env-file", type=Path, default=None,
                        help="read ANTHROPIC_API_KEY from this .env when it is not set")
    args = parser.parse_args(argv)

    key = api_key(args.env_file)
    if not key:
        print("ANTHROPIC_API_KEY is not set; the judge makes paid API calls and needs it.",
              file=sys.stderr)
        sys.exit(2)
    import anthropic  # the optional extra; only needed for a real run

    summary = run(read_units(args.units), args.out, args.run_label, args.max_usd,
                  client=anthropic.Anthropic(api_key=key), limit=args.limit)
    print(f"called {summary['called']}, skipped {summary['skipped']}, "
          f"spent ${summary['spent_usd']}, stopped on budget: {summary['stopped_on_budget']}")


if __name__ == "__main__":
    main()
