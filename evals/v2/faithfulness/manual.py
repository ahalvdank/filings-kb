"""Manual judge path: export a packet to paste into a chat model, import its pasted reply.

``export`` writes two files: ``PROMPT.md`` (the judge prompt verbatim, then the batch
instructions and the reply format) and ``units_for_manual_judge.md`` (every unit as
``### <unit_id>``, its sentence, then exactly ``units.format_passages(unit)``, the text the
API judges saw). ``import`` turns the reply into judge records with the same fields as
an API run, ``provider: "manual"``, ``usage`` null and ``cost_usd`` "0", so
``evaluate_faithfulness_v2`` scores them like any other run.

The reply must hold one JSON object per line. Code-fence lines and blank lines are
skipped; any other line that is not a JSON object with a ``unit_id`` raises. A line whose
verdict is not exactly the schema becomes ``parse_error``.

CLI (no API calls)::

    python -m evals.v2.faithfulness.manual export --units evals/v2/faithfulness/units.jsonl \\
        --out-dir evals/v2/faithfulness/manual/
    python -m evals.v2.faithfulness.manual import --units evals/v2/faithfulness/units.jsonl \\
        --reply reply.txt --out evals/v2/faithfulness/judge_gemini1.jsonl \\
        --run-label gemini1 --model <model name>
"""

import argparse
import json
import sys
from pathlib import Path

from evals.v2.faithfulness.judge import PROMPT_PATH, _parse, sha256_file
from evals.v2.faithfulness.units import format_passages, read_units

PROMPT_FILE = "PROMPT.md"
UNITS_FILE = "units_for_manual_judge.md"
REPLY_FORMAT = ('{"unit_id": "...", "verdict": "supported|partial|unsupported", '
                '"unsupported_span": "..."|null}')


def render_unit(unit: dict) -> str:
    """One unit for the packet: id heading, sentence, then its passages as the API judge saw."""
    return f"### {unit['unit_id']}\n\nSentence: {unit['sentence']}\n\n{format_passages(unit)}"


def render_prompt(prompt: str, n_units: int) -> str:
    """``PROMPT.md``: the judge prompt verbatim, then the batch instructions."""
    return f"""{prompt}
---

## How to answer this batch

The file `{UNITS_FILE}` holds {n_units} units. Each unit starts with a `### <unit_id>`
heading, then the sentence, then the passages it cites. Judge every unit independently,
by the rules above: use only that unit's own passages, and let no other unit's sentence or
passages affect its verdict.

Reply with one JSON object per line, in this exact form, and nothing else:

{REPLY_FORMAT}

- `unit_id` is the unit's heading, copied exactly.
- `unsupported_span` is null for `supported`, and an exact quote from the sentence otherwise.
- Every unit_id appears exactly once: {n_units} lines in all. No commentary, headings or
  explanations.

The units file is long, so your reply may come in several parts. If you have to stop
before the end, stop after a complete line; when asked to continue, carry on from the
next unit. Keep going until all {n_units} units are done.
"""


def export(units: list[dict], out_dir: Path, prompt_path: Path = PROMPT_PATH) -> None:
    """Write ``PROMPT.md`` and ``units_for_manual_judge.md`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_path.read_text(encoding="utf-8")
    (out_dir / PROMPT_FILE).write_text(render_prompt(prompt, len(units)), encoding="utf-8")
    body = "\n\n".join(render_unit(u) for u in units) + "\n"
    (out_dir / UNITS_FILE).write_text(body, encoding="utf-8")


def _reply_rows(reply: str) -> list[tuple[str, str, dict]]:
    """``(unit_id, line, object)`` for each JSON line, skipping fences and blank lines.

    Raises:
        ValueError: on a line that is not a JSON object with a string ``unit_id``.
    """
    rows = []
    for number, raw_line in enumerate(reply.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"reply line {number} is not JSON: {line[:80]!r}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("unit_id"), str):
            raise ValueError(f"reply line {number} has no unit_id: {line[:80]!r}")
        rows.append((data["unit_id"], line, data))
    return rows


def import_reply(units: list[dict], reply: str, run_label: str, model: str,
                 prompt_path: Path = PROMPT_PATH) -> list[dict]:
    """Judge records from a pasted reply, in the order of ``units``.

    Args:
        units: the units the packet was exported from.
        reply: the pasted reply text.
        run_label: the run label written on every record.
        model: the name of the model that wrote the reply.
        prompt_path: the judge prompt the packet embeds; its hash goes on every record.

    Returns:
        One record per unit: ``unit_id``, ``verdict`` (a schema verdict or
        ``parse_error``), ``unsupported_span``, ``raw`` (the reply line), ``stop_reason``
        None, ``usage`` None, ``model``, ``effort`` None, ``prompt_sha256``,
        ``run_label``, ``cost_usd`` "0" and ``provider`` "manual".

    Raises:
        ValueError: if a line cannot be attributed to a unit, or the reply does not name
            every unit exactly once (missing, duplicate and unknown ids are all listed).
    """
    rows = _reply_rows(reply)
    wanted = [u["unit_id"] for u in units]
    seen: dict[str, tuple[str, dict]] = {}
    duplicate: set[str] = set()
    for unit_id, line, data in rows:
        if unit_id in seen:
            duplicate.add(unit_id)
        else:
            seen[unit_id] = (line, data)
    missing = [u for u in wanted if u not in seen]
    unknown = sorted(set(seen) - set(wanted))
    if missing or duplicate or unknown:
        raise ValueError(f"reply must name every unit exactly once: missing {missing}, "
                         f"duplicate {sorted(duplicate)}, unknown {unknown}")
    prompt_sha = sha256_file(prompt_path)
    records = []
    for unit_id in wanted:
        line, data = seen[unit_id]
        verdict_only = {k: v for k, v in data.items() if k != "unit_id"}
        parsed = _parse(json.dumps(verdict_only))
        verdict, span = parsed if parsed is not None else ("parse_error", None)
        records.append({"unit_id": unit_id, "verdict": verdict, "unsupported_span": span,
                        "raw": line, "stop_reason": None, "usage": None, "model": model,
                        "effort": None, "prompt_sha256": prompt_sha, "run_label": run_label,
                        "cost_usd": "0", "provider": "manual"})
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manual judge packet (no API calls).")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="write PROMPT.md and the units file")
    exp.add_argument("--units", type=Path, required=True)
    exp.add_argument("--out-dir", type=Path, required=True)
    imp = sub.add_parser("import", help="turn a pasted reply into a judge run file")
    imp.add_argument("--units", type=Path, required=True)
    imp.add_argument("--reply", type=Path, required=True)
    imp.add_argument("--out", type=Path, required=True)
    imp.add_argument("--run-label", required=True)
    imp.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    units = read_units(args.units)
    if args.command == "export":
        export(units, args.out_dir)
        print(f"wrote {len(units)} units to {args.out_dir}")
        return
    if args.out.exists():
        print(f"{args.out} exists; run files are never overwritten.", file=sys.stderr)
        sys.exit(2)
    records = import_reply(units, args.reply.read_text(encoding="utf-8"), args.run_label,
                           args.model)
    with args.out.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    errors = sum(r["verdict"] == "parse_error" for r in records)
    print(f"imported {len(records)} units ({errors} parse errors) to {args.out}")


if __name__ == "__main__":
    main()
