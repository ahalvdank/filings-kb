"""Faithfulness units: one cited answer sentence plus the passages it cites.

A unit is one answer sentence (``paragraphs[i-1][j-1]``, numbered 1-based as in
``filings_kb.answer``) of a question that has an exported trace. Its passages are the
``chunks.text`` of each cited chunk, which is the text the answer model saw: the model
input renders ``PassageSnapshot.text``, which is ``context["match"]``, which is the chunk
text. Each passage also carries the answer passage's ticker, fiscal year, form and item,
so ``format_passages`` renders the same ``[cite] TICKER FY2025 10-K, Item 1A`` header the
answer model saw (``filings_kb.answer.build_user_message``); a sentence's fiscal year is
often stated only there. The database is only ever opened read-only.

CLI::

    python -m evals.v2.faithfulness.units build --answers data/committed/answers \
        --traces exports/traces --db data/kb.sqlite --out evals/v2/faithfulness/units.jsonl
    python -m evals.v2.faithfulness.units spot-sheet --units evals/v2/faithfulness/units.jsonl \
        --judge evals/v2/faithfulness/judge_run1.jsonl --out evals/v2/faithfulness/spot_sheet.csv
"""

import argparse
import csv
import json
import random
import sqlite3
from contextlib import closing
from pathlib import Path

from filings_kb.answer import item_label

SPOT_N = 40
SPOT_SEED = 20261125
# Every unit the judge did not call supported, including units it gave no verdict on.
FLAGGED_VERDICTS = ("partial", "unsupported", "parse_error", "refusal")
HEADER_FIELDS = ("ticker", "fiscal_year", "form", "item")
SPOT_COLUMNS = ("unit_id", "sentence", "passages", "human_verdict", "note")


def _connect(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _chunk_texts(db: Path, chunk_ids: set[str]) -> dict[str, str]:
    """``chunk_id -> chunks.text`` for the ids found; missing ids are simply absent."""
    out: dict[str, str] = {}
    with closing(_connect(db)) as conn:
        for chunk_id in sorted(chunk_ids):
            row = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if row is not None:
                out[chunk_id] = row[0]
    return out


def _cite_map(slug: str, answer: dict) -> dict[int, dict]:
    """``cite -> answer passage`` for the included passages; raises on a cite given twice."""
    out: dict[int, dict] = {}
    for p in answer["passages"]:
        if p["cite"] is None:
            continue
        if p["cite"] in out:
            raise ValueError(f"{slug}: cite {p['cite']} is given to two passages")
        if not p["included"]:
            raise ValueError(f"{slug}: cite {p['cite']} is on a passage that is not included")
        out[p["cite"]] = p
    return out


def build_units(answers_dir: Path, traces_dir: Path, db: Path) -> list[dict]:
    """Every cited sentence of every answer that has an exported trace, in slug order.

    Returns:
        Units ``{"unit_id", "slug", "sentence", "cites", "passages": [{"cite", "chunk_id",
        "ticker", "fiscal_year", "form", "item", "text"}]}``, one passage per cite in the
        sentence's cite order.

    Raises:
        ValueError: if a trace has no answer, a cite has no passage or points at a passage
            that was not included, or a cited chunk is missing from the database.
    """
    raw: list[tuple[str, int, int, dict, dict[int, dict]]] = []
    for trace in sorted(traces_dir.glob("*.json")):
        slug = trace.stem
        answer_path = answers_dir / f"{slug}.json"
        if not answer_path.exists():
            raise ValueError(f"{slug}: has a trace but no answer at {answer_path}")
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        cites = _cite_map(slug, answer)
        for i, para in enumerate(answer["paragraphs"], 1):
            for j, unit in enumerate(para, 1):
                for c in unit["cites"]:
                    if c not in cites:
                        raise ValueError(f"{slug}:p{i}u{j}: cite {c} has no passage")
                raw.append((slug, i, j, unit, cites))

    texts = _chunk_texts(db, {cites[c]["chunk_id"] for *_, unit, cites in raw
                              for c in unit["cites"]})
    units: list[dict] = []
    for slug, i, j, unit, cites in raw:
        passages = []
        for c in unit["cites"]:
            source = cites[c]
            chunk_id = source["chunk_id"]
            if chunk_id not in texts:
                raise ValueError(f"{slug}:p{i}u{j}: chunk {chunk_id} is not in {db}")
            passages.append({"cite": c, "chunk_id": chunk_id,
                             **{f: source[f] for f in HEADER_FIELDS}, "text": texts[chunk_id]})
        units.append({"unit_id": f"{slug}:p{i}u{j}", "slug": slug, "sentence": unit["text"],
                      "cites": list(unit["cites"]), "passages": passages})
    return units


def write_units(units: list[dict], path: Path) -> None:
    """One JSON unit per line."""
    path.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units),
                    encoding="utf-8")


def read_units(path: Path) -> list[dict]:
    """Units written by ``write_units``."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def passage_header(p: dict) -> str:
    """The header the answer model saw for a passage, e.g. ``[1] NVDA FY2026 10-K, Item 1A``."""
    return f"[{p['cite']}] {p['ticker']} FY{p['fiscal_year']} {p['form']}, {item_label(p['item'])}"


def format_passages(unit: dict) -> str:
    """The unit's passages, each as its model-input header followed by its text."""
    return "\n\n".join(f"{passage_header(p)}\n{p['text']}" for p in unit["passages"])


def _judge_verdicts(judge_jsonl: Path) -> dict[str, str]:
    return {r["unit_id"]: r["verdict"] for r in read_units(judge_jsonl)}


def write_spot_sheet(units: list[dict], judge_jsonl: Path, out_csv: Path, n: int = SPOT_N,
                     seed: int = SPOT_SEED) -> int:
    """Blind human-labelling sheet: every flagged unit plus a seeded sample of supported ones.

    Flagged means anything but ``supported``: ``partial``, ``unsupported``, and the
    harness verdicts ``parse_error`` and ``refusal`` (scoring leaves those out of the 2x2,
    but a human label on them is still worth having). Supported units are sampled
    (seeded) until the sheet has ``n`` rows; if more than ``n`` units are flagged, all of
    them are kept. Rows are shuffled with the same seed so their
    order does not reveal the judge's verdict, and there is no judge column.

    Returns:
        The number of rows written.

    Raises:
        ValueError: if the judge file names a unit that is not in ``units``, or does not
            cover every unit (the sheet is drawn only from a complete run).
    """
    verdicts = _judge_verdicts(judge_jsonl)
    by_id = {u["unit_id"]: u for u in units}
    unknown = sorted(set(verdicts) - set(by_id))
    if unknown:
        raise ValueError(f"judge file names units not in the unit list: {unknown[:5]}")
    missing = sorted(set(by_id) - set(verdicts))
    if missing:
        raise ValueError(f"judge file does not cover {len(missing)} units, e.g. {missing[:5]}")
    flagged = sorted(u for u, v in verdicts.items() if v in FLAGGED_VERDICTS)
    supported = sorted(u for u, v in verdicts.items() if v == "supported")
    rng = random.Random(seed)
    chosen = flagged + rng.sample(supported, max(0, min(n - len(flagged), len(supported))))
    rng.shuffle(chosen)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SPOT_COLUMNS)
        for unit_id in chosen:
            unit = by_id[unit_id]
            writer.writerow([unit_id, unit["sentence"], format_passages(unit), "", ""])
    return len(chosen)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write units.jsonl")
    build.add_argument("--answers", type=Path, required=True)
    build.add_argument("--traces", type=Path, required=True)
    build.add_argument("--db", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    spot = sub.add_parser("spot-sheet", help="write the blind human spot sheet")
    spot.add_argument("--units", type=Path, required=True)
    spot.add_argument("--judge", type=Path, required=True)
    spot.add_argument("--out", type=Path, required=True)
    spot.add_argument("--n", type=int, default=SPOT_N)
    spot.add_argument("--seed", type=int, default=SPOT_SEED)
    args = parser.parse_args(argv)

    if args.command == "build":
        built = build_units(args.answers, args.traces, args.db)
        write_units(built, args.out)
        pairs = sum(len(u["cites"]) for u in built)
        print(f"{len(built)} units, {pairs} (unit, cite) pairs, "
              f"{len({u['slug'] for u in built})} slugs -> {args.out}")
    else:
        written = write_spot_sheet(read_units(args.units), args.judge, args.out, args.n, args.seed)
        print(f"{written} spot rows -> {args.out}")


if __name__ == "__main__":
    main()
