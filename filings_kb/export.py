"""Build, validate and write the D2 trace JSON files, the manifest and audit notes.

``build_trace`` turns a cached ``AnswerRun`` (which carries a snapshot of every
store-derived field) into the ``Trace`` shape the portfolio site renders
(schemas/trace.schema.json). Export never opens the store, so a clean clone with
no ``kb.sqlite`` reproduces ``exports/``. Every answer is re-checked here
(citations, per-unit numbers and quotes) whether or not it came from the cache;
one that fails is not exported and is listed in ``exports/failed.json``.
Everything is schema-validated before anything is written, and files are written
with sorted keys so a re-export from the same cache is byte-identical.
"""

import json
import logging
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jsonschema

from filings_kb.answer import (
    AnswerRun, ClaimView, Failure, check_answer, check_cutoff, render_answer, section_label,
)
from filings_kb.config import load_sample
from filings_kb.signals import ClaimDiff, SignalRow

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"
EXPORTS_DIR = REPO_ROOT / "exports"
SAMPLE_PATH = REPO_ROOT / "config" / "sample.yaml"
MODEL_KEYS = ("embedding", "extraction", "synthesis", "sentiment")
EXPORT_SCRIPT_URL = "https://github.com/ahalvdank/filings-kb/blob/{commit}/filings_kb/export.py"
# Outputs of `make traces`: changes here don't make the code tree dirty.
OUTPUT_PATHSPECS = (":(exclude)exports", ":(exclude)data/committed/answers")
# Entity types that must never reach a public trace: claims about them are dropped from
# the trace and its audit note (they don't exist in the sample; the guard is defence in
# depth for a future private KB run).
EXCLUDED_ENTITY_TYPES = frozenset({"ThesisPage", "ResearchSource", "own_research"})
MAX_STATEMENT_ATTRS = 4
MAX_STATEMENT_VALUE_CHARS = 120


# What a trace's build_commit vouches for: a cache export may reuse that commit only
# while these are unchanged at HEAD.
BUILD_INPUT_PATHS = ("filings_kb", "schemas", "prompts", "config")


class DirtyTreeError(RuntimeError):
    """The git working tree has uncommitted changes, so a trace's commit would lie."""


class StaleBuildError(RuntimeError):
    """Code, schemas, prompts or config changed since ``build_commit``, so reusing it would lie."""


def require_code_unchanged(repo: Path, commit: str) -> None:
    """Refuse to reuse ``commit`` when anything in ``BUILD_INPUT_PATHS`` differs at HEAD.

    Raises:
        StaleBuildError: naming the changed files, or when git cannot compare (not a
            git checkout, or ``commit`` is not in its history).
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", commit, "HEAD", "--",
         *BUILD_INPUT_PATHS], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise StaleBuildError(
            f"cannot compare build_commit {commit[:12]} with HEAD in {repo}: "
            f"{proc.stderr.strip()}")
    changed = proc.stdout.split()
    if changed:
        raise StaleBuildError(
            f"{len(changed)} file(s) under {', '.join(BUILD_INPUT_PATHS)} changed since "
            f"build_commit {commit[:12]}; re-run make traces at a clean HEAD to re-stamp: "
            + ", ".join(changed))


def require_clean_tree(repo: Path) -> str:
    """Return ``HEAD`` of ``repo`` if its working tree is clean.

    Untracked files count as dirty. The pipeline's own outputs (``exports/`` and
    ``data/committed/answers/``) are ignored so the export can be re-run.

    Raises:
        DirtyTreeError: if code or config has uncommitted changes.
        RuntimeError: if ``repo`` is not a git repository.
    """
    def git(*args: str) -> str:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}")
        return proc.stdout

    dirty = git("status", "--porcelain", "--untracked-files=all", "--", ".", *OUTPUT_PATHSPECS)
    if dirty.strip():
        raise DirtyTreeError(
            f"{repo} has uncommitted changes; commit them first so the trace's build_commit "
            f"is true:\n{dirty.rstrip()}")
    return git("rev-parse", "HEAD").strip()


def commit_from_manifest(path: Path) -> str:
    """Return ``build_commit`` of a committed manifest, so a cache-only re-export keeps it.

    Raises:
        FileNotFoundError: if there is no manifest yet (run ``make traces`` first).
        ValueError: if the manifest has no valid ``build_commit``.
    """
    commit = json.loads(path.read_text(encoding="utf-8")).get("build_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise ValueError(f"{path}: no valid build_commit")
    return str(commit)


def load_signal_rows(path: Path) -> list[SignalRow]:
    """Read ``data/committed/signals.json`` back into ``SignalRow`` objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[SignalRow] = []
    for r in data["signals"]:
        diff = r["claim_diff"]
        rows.append(SignalRow(
            ticker=r["ticker"], item=r["item"], prior_fy=r["prior_fy"], current_fy=r["current_fy"],
            tone_shift=r["tone_shift"], sentiment_shift=r["sentiment_shift"],
            claim_diff=None if diff is None else ClaimDiff(
                added=diff["added"], removed=diff["removed"],
                changed=[(c[0], c[1]) if isinstance(c, list) else (c["before"], c["after"])
                         for c in diff["changed"]]),
            claim_diff_status=r["claim_diff_status"], status=r["status"],
        ))
    return rows


# --- trace ---------------------------------------------------------------------


def _fmt_value(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text if len(text) <= MAX_STATEMENT_VALUE_CHARS else text[:MAX_STATEMENT_VALUE_CHARS] + "..."


def _statement(view: ClaimView) -> str:
    """A readable sentence: ``Subject: claim name (key value; key value).``"""
    attrs = "; ".join(f"{k.replace('_', ' ')} {_fmt_value(v)}"
                      for k, v in list(view.row.attributes.items())[:MAX_STATEMENT_ATTRS])
    detail = f" ({attrs})" if attrs else ""
    return f"{view.row.subject_name}: {view.row.name}{detail}."


def _wikilink(text: str) -> str:
    return "[[" + re.sub(r"\s+", " ", re.sub(r"[\[\]|#^]", "", text)).strip() + "]]"


def _cites_text(cites: list[int]) -> str:
    return ", ".join(f"[{n}]" for n in sorted(set(cites))) or "none cited"


def build_audit_note(
    title: str, question: str, passages: list[dict[str, Any]], views: list[ClaimView],
    sample_tickers: set[str],
) -> str:
    """Obsidian-style note: the trace's sample entities as ``[[links]]`` with cited passages.

    Only entities whose ticker is in the sample and whose type isn't an excluded
    one (ThesisPage, ResearchSource, own_research) are listed. A claim the 10-K/A
    superseded is marked as such.

    Args:
        passages: exported passage dicts (``cite`` is None when not cited).
        views: the trace's claims, with ``cites`` in displayed numbers.
        sample_tickers: tickers of the public sample.
    """
    companies: dict[tuple[str, str], list[int]] = {}
    for p in passages:
        if p["ticker"] in sample_tickers:
            cites = companies.setdefault((p["company"], p["ticker"]), [])
            if p["cite"] is not None:
                cites.append(p["cite"])
    lines = [f"# {title}", "", f"Question: {question}", "", "## Companies", ""]
    lines += [f"- {_wikilink(name)} ({ticker}): {_cites_text(cites)}"
              for (name, ticker), cites in companies.items()]
    lines += ["", "## Claims", ""]
    for v in views:
        if v.ticker not in sample_tickers or v.row.subject_type in EXCLUDED_ENTITY_TYPES:
            continue
        cites = [] if v.cite is None else [v.cite]
        mark = " (superseded by the amendment)" if v.superseded else ""
        lines.append(f"- {_wikilink(v.row.name)} ({v.row.claim_type}, {v.ticker}) on "
                     f"{_wikilink(v.row.subject_name)}: {_cites_text(cites)}{mark}")
    return "\n".join(lines) + "\n"


def _signals_for(run: AnswerRun, signals: list[SignalRow]) -> list[dict[str, Any]]:
    years: dict[str, set[int]] = {}
    for ticker, fiscal_year in run.spec.filings:
        years.setdefault(ticker, set()).add(fiscal_year)
    chosen = sorted(
        (r for r in signals
         if r.item in run.spec.items and r.prior_fy in years.get(r.ticker, ())
         and r.current_fy in years.get(r.ticker, ())),
        key=lambda r: (r.ticker, r.item, r.prior_fy))
    out: list[dict[str, Any]] = []
    for r in chosen:
        if r.tone_shift is None and r.sentiment_shift is None and r.claim_diff is None:
            continue
        entry: dict[str, Any] = {
            "ticker": r.ticker, "section": section_label(r.item), "prior_fy": r.prior_fy,
            "current_fy": r.current_fy, "tone_shift": r.tone_shift,
            "sentiment_shift": r.sentiment_shift,
        }
        if r.claim_diff is not None:
            entry["claim_diff"] = {
                "added": list(r.claim_diff.added), "removed": list(r.claim_diff.removed),
                "changed": [{"before": b, "after": a} for b, a in r.claim_diff.changed],
            }
        out.append(entry)
    return out


def build_trace(run: AnswerRun, signals: list[SignalRow], commit: str,
                sample_tickers: set[str]) -> dict[str, Any]:
    """Assemble the D2 ``Trace`` dict for one answer; pure over the run's snapshot.

    Cites are renumbered by first appearance in the answer (``render_answer``).
    ``cite`` on a passage is its displayed number only when the answer cites it;
    passages the model saw but didn't cite keep ``included: true`` with ``cite:
    null``. A claim's ``cite`` is the lowest displayed number among the cited
    passages it overlaps, else null. Claims whose subject type is in
    ``EXCLUDED_ENTITY_TYPES`` are dropped.

    Args:
        sample_tickers: tickers of the public sample; the audit note lists only these.
    """
    spec = run.spec
    answer_md, shown = render_answer(run.paragraphs)
    passages: list[dict[str, Any]] = []
    for p in sorted(run.passages, key=lambda p: (-p.score, p.chunk_id)):
        passages.append({
            "cite": shown.get(p.cite) if p.cite is not None else None,
            "included": p.included,
            "score": round(p.score, 6),
            "ticker": p.ticker, "company": p.company, "form": p.form,
            "fiscal_year": p.fiscal_year, "filed": p.filed,
            "section": section_label(p.item),
            "accession": p.accession, "sec_url": p.sec_url,
            "context": dict(p.context),
        })

    # A claim about an excluded entity type (private research) never reaches the trace.
    shown_views = [ClaimView(v.row, v.ticker, tuple(sorted(shown[n] for n in v.cites if n in shown)),
                             v.superseded)
                   for v in run.claims if v.row.subject_type not in EXCLUDED_ENTITY_TYPES]
    claims: list[dict[str, Any]] = []
    for v in shown_views:
        claim: dict[str, Any] = {
            "claim_type": v.row.claim_type, "entity_type": v.row.subject_type,
            "entity": v.row.subject_name, "ticker": v.ticker,
            "statement": _statement(v), "raw_extraction": v.row.attributes,
            "cite": v.cite,
        }
        if v.superseded is not None:
            claim["superseded"] = v.superseded
        claims.append(claim)

    trace: dict[str, Any] = {
        "slug": spec.slug, "title": spec.title, "question": spec.question,
        "category": spec.category, "tickers": list(spec.tickers),
        "meta": {"run_timestamp": run.run_timestamp, "build_commit": commit,
                 "models": {k: run.models[k] for k in MODEL_KEYS},
                 "top_k": run.top_k, "max_passages": run.max_passages,
                 "max_rejected": run.max_rejected,
                 "display_cutoff": run.display_cutoff},
        "answer_md": answer_md, "passages": passages, "claims": claims,
    }
    sigs = _signals_for(run, signals)
    if sigs:
        trace["signals"] = sigs
    trace["audit_note_md"] = build_audit_note(
        spec.title, spec.question, passages, shown_views, sample_tickers)
    return trace


# --- manifest and export -----------------------------------------------------


def _pipeline_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def sample_summary(path: Path = SAMPLE_PATH) -> dict[str, Any]:
    """The manifest's ``sample`` block from the pinned ``config/sample.yaml``.

    Returns:
        ``{"companies": sorted tickers, "filings": number of pinned rows}``.
    """
    rows = load_sample(path).rows
    return {"companies": sorted({r.ticker for r in rows}), "filings": len(rows)}


def build_manifest(
    runs: list[AnswerRun], build_commit: str, full_corpus: dict[str, int],
    pipeline_version: str, sample: dict[str, Any],
) -> dict[str, Any]:
    """The D2 ``TraceManifest``; ``sample`` comes from ``sample_summary``."""
    return {
        "pipeline_version": pipeline_version,
        "build_commit": build_commit,
        "export_script_url": EXPORT_SCRIPT_URL.format(commit=build_commit),
        "run_timestamp": max(r.run_timestamp for r in runs),
        "sample": {"companies": list(sample["companies"]), "filings": int(sample["filings"])},
        "full_corpus": {"companies": int(full_corpus["companies"]),
                        "filings": int(full_corpus["filings"])},
        "traces": [{"slug": r.spec.slug, "title": r.spec.title, "question": r.spec.question,
                    "category": r.spec.category, "tickers": list(r.spec.tickers),
                    "featured": r.spec.featured}
                   for r in runs],
    }


def _validator(schemas_dir: Path, name: str) -> jsonschema.Draft202012Validator:
    schema = json.loads((schemas_dir / f"{name}.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_against(obj: dict[str, Any], schemas_dir: Path, name: str, label: str) -> None:
    """Validate ``obj`` against ``schemas/<name>.schema.json``.

    Raises:
        ValueError: naming ``label`` and the first offending path.
    """
    errors = sorted(_validator(schemas_dir, name).iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "(root)"
        raise ValueError(f"{label}: {name} schema violation at {where}: {e.message[:200]}")


def _dump(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class ExportResult:
    """Where the manifest went and which questions were not exported (sorted by slug)."""

    manifest_path: Path
    failures: list[Failure]


def export_all(
    runs: list[AnswerRun], out_dir: Path, build_commit: str, *,
    signals: list[SignalRow], full_corpus: dict[str, int], sample: dict[str, Any],
    failures: list[Failure] | None = None, schemas_dir: Path = SCHEMAS_DIR,
) -> ExportResult:
    """Re-check every answer, then write traces, audit notes, the manifest and failed.json.

    Each run is re-checked with ``check_answer`` and ``check_cutoff`` against its own snapshot; a run
    that fails is dropped and added to ``failures``. ``failed.json`` lists every
    failure (``slug``, ``reason``, ``rejected`` paths), sorted by slug, and is
    written even when empty. Every trace and the manifest are built and
    schema-validated before the first file is written, so a schema violation
    leaves ``out_dir`` untouched. Files from earlier exports that are no longer
    exported are removed.

    Raises:
        ValueError: if any trace or the manifest violates its schema, or if no
            answer passes its checks.
    """
    failed = list(failures or [])
    passing: list[AnswerRun] = []
    for run in runs:
        errors = check_answer(run.paragraphs, run.passages) + check_cutoff(run)
        if errors:
            logger.warning("%s: not exported, re-check failed: %s", run.spec.slug, errors)
            shown = "; ".join(errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3
                                              else "")
            failed.append(Failure(run.spec.slug, f"export re-check failed: {shown}"))
        else:
            passing.append(run)
    if not passing:
        raise ValueError("no answer passed its checks; nothing to export")
    failed.sort(key=lambda f: f.slug)

    tickers = set(sample["companies"])
    traces = {r.spec.slug: build_trace(r, signals, build_commit, tickers) for r in passing}
    for slug, trace in traces.items():
        validate_against(trace, schemas_dir, "trace", slug)
    manifest = build_manifest(passing, build_commit, full_corpus, _pipeline_version(), sample)
    validate_against(manifest, schemas_dir, "manifest", "manifest")

    for sub, suffix in (("traces", ".json"), ("audit_notes", ".md")):
        directory = out_dir / sub
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob(f"*{suffix}"):
            if stale.stem not in traces:
                stale.unlink()
    for slug, trace in traces.items():
        (out_dir / "traces" / f"{slug}.json").write_text(_dump(trace), encoding="utf-8")
        (out_dir / "audit_notes" / f"{slug}.md").write_text(trace["audit_note_md"],
                                                            encoding="utf-8")
    (out_dir / "failed.json").write_text(
        _dump({"failed": [{**asdict(f), "rejected": list(f.rejected)} for f in failed]}),
        encoding="utf-8")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(_dump(manifest), encoding="utf-8")
    return ExportResult(manifest_path, failed)
