"""Command line: ``python -m filings_kb ingest [--live | --refresh] [--write-committed]``,
``signals [--write-committed]`` and ``traces [--from-cache]``.

Tracked files under ``data/committed/`` are written only with ``--write-committed``
(``make sample-refresh``). Without it, ingest and signals write to the gitignored
``data/local/``, and wall-clock durations always go to the gitignored
``data/run_log.json``, so ``make ingest && make traces`` never finds a dirty tree.
"""

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from filings_kb.extract import MODELS_PATH
from filings_kb.ingest import DATA_DIR, REPO_ROOT, ingest_sample, write_report

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from filings_kb.answer import AnswerRun, Failure, QuestionConfig

SAMPLE_PATH = REPO_ROOT / "config" / "sample.yaml"
REPORT_PATH = DATA_DIR / "committed" / "ingest_report.json"
LOCAL_REPORT_PATH = DATA_DIR / "local" / "ingest_report.json"
RUN_LOG_PATH = DATA_DIR / "run_log.json"  # gitignored: durations change every run


def _claude_available() -> bool:
    """True when ``claude --version`` runs."""
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"claude CLI check failed: {exc}")
        return False
    return True


def _log_duration(command: str, duration_s: float, path: Path | None = None) -> None:
    """Append ``{command, duration_s, finished_at}`` to the gitignored run log."""
    path = path or RUN_LOG_PATH
    runs: list[dict[str, Any]] = []
    if path.exists():
        try:
            runs = list(json.loads(path.read_text(encoding="utf-8"))["runs"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("%s is unreadable (%r); starting a new run log", path, exc)
    runs.append({"command": command, "duration_s": round(duration_s, 1),
                 "finished_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")


def _run_signals(write_committed: bool = False) -> int:
    """Compute signals over the ingested sample with the ``claude`` CLI as backend.

    With ``write_committed`` the tracked ``data/committed/signals.json`` and
    ``claim_diffs/`` are written. Without it, signals go to ``data/local/signals.json``,
    committed claim diffs are read but never rewritten (new ones go to
    ``data/local/claim_diffs/``), and every difference from the committed signals is
    printed.

    Returns 1 (without computing anything) if ``claude --version`` does not work.
    """
    import yaml

    from filings_kb.embed import EmbeddingIndex
    from filings_kb.llm import ClaudeCliBackend
    from filings_kb.signals import (
        CLAIM_DIFF_DIR, LOCAL_CLAIM_DIFF_DIR, LOCAL_SIGNALS_PATH, SIGNALS_PATH, compute_signals,
        signals_drift,
    )
    from filings_kb.store import Store

    if not _claude_available():
        print("claude CLI unavailable; claim_diff needs it")
        return 1

    def no_embedding(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("signals reads stored embeddings and never embeds")

    models = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    store = Store(DATA_DIR / "kb.sqlite")
    index = EmbeddingIndex(DATA_DIR / "chroma", str(models["embedding"]), max_tokens=512,
                           embed_fn=no_embedding)
    if write_committed:
        out, cache_dir, read_dirs = SIGNALS_PATH, CLAIM_DIFF_DIR, ()
    else:
        out, cache_dir, read_dirs = LOCAL_SIGNALS_PATH, LOCAL_CLAIM_DIFF_DIR, (CLAIM_DIFF_DIR,)
    rows = compute_signals(store, index, ClaudeCliBackend(), str(models["synthesis"]),
                           out_path=out, cache_dir=cache_dir, read_cache_dirs=read_dirs)
    print(f"rows={len(rows)} report={out}")
    if not write_committed:
        drift = signals_drift(out, SIGNALS_PATH)
        print(f"drift vs {SIGNALS_PATH.relative_to(REPO_ROOT)}: "
              f"{len(drift) or 'none'}{' differences' if drift else ''}")
        for line in drift:
            print(f"  {line}")
    return 0


@dataclass(frozen=True)
class TracePaths:
    """Every file ``traces`` reads or writes (tests point these at a temp clone)."""

    repo: Path
    questions: Path
    prompt: Path
    answers: Path
    rejected: Path
    exports: Path
    signals: Path
    models: Path
    sample: Path
    data: Path

    @classmethod
    def default(cls) -> "TracePaths":
        """The repository's own paths."""
        from filings_kb.answer import ANSWERS_DIR, PROMPT_PATH, QUESTIONS_PATH, REJECTED_DIR
        from filings_kb.export import EXPORTS_DIR
        from filings_kb.signals import SIGNALS_PATH

        return cls(repo=REPO_ROOT, questions=QUESTIONS_PATH, prompt=PROMPT_PATH,
                   answers=ANSWERS_DIR, rejected=REJECTED_DIR, exports=EXPORTS_DIR,
                   signals=SIGNALS_PATH, models=MODELS_PATH, sample=SAMPLE_PATH, data=DATA_DIR)


def _cached_runs(cfg: "QuestionConfig", paths: TracePaths,
                 ) -> tuple[list["AnswerRun"], list["Failure"]]:
    """Committed answers and failure records; never opens the store.

    A question with a committed failure record is failed: without a cached answer it
    keeps the record's reason (so a clean clone reproduces ``failed.json``); with one,
    the failure was transient and the answer was retained, which the reason says. Either
    way the cache export matches what the live run reported. A cached answer made with
    a different system prompt also counts as failed.
    """
    from filings_kb.answer import Failure, load_failure, load_run, prompt_sha256

    current = prompt_sha256(paths.prompt)
    runs: list[AnswerRun] = []
    failures: list[Failure] = []
    for q in cfg.questions:
        if (audit := cfg.audit_failure(q.slug)) is not None:
            failures.append(audit)  # the same reason the live run gives
            continue
        try:
            record = load_failure(q.slug, paths.answers)
        except ValueError as exc:
            failures.append(Failure(q.slug, str(exc)))
            continue
        try:
            run = load_run(q.slug, paths.answers)
        except FileNotFoundError:
            failures.append(record or Failure(q.slug, "no cached answer"))
            continue
        except ValueError as exc:
            failures.append(Failure(q.slug, str(exc)))
            continue
        if record is not None:
            failures.append(Failure(q.slug, "transient; committed answer retained",
                                    record.rejected))
            continue
        if run.prompt_sha256 != current:
            failures.append(Failure(q.slug, "cached answer was made with a different prompt; "
                                            "re-run make traces"))
            continue
        # featured is curation in questions.yaml, not part of the answer: the cached spec
        # snapshot predates the audit, so the current config decides.
        runs.append(replace(run, spec=replace(run.spec, featured=q.featured)))
    return runs, failures


def _live_runs(cfg: "QuestionConfig", paths: TracePaths, models: dict[str, Any],
               ) -> tuple[list["AnswerRun"], list["Failure"]]:
    """Answer every question with the claude CLI (cache hits cost no call)."""
    from filings_kb.answer import Failure, load_failure, record_failure, run_question
    from filings_kb.embed import EmbeddingIndex
    from filings_kb.llm import ClaudeCliBackend
    from filings_kb.store import Store

    store = Store(paths.data / "kb.sqlite")
    index = EmbeddingIndex(paths.data / "chroma", str(models["embedding"]), max_tokens=512)
    backend = ClaudeCliBackend()
    run_models = {k: str(models[k]) for k in ("embedding", "extraction", "synthesis",
                                              "sentiment")}
    runs: list[AnswerRun] = []
    failures: list[Failure] = []
    for q in cfg.questions:
        if (audit := cfg.audit_failure(q.slug)) is not None:
            failures.append(audit)  # never retrieved or answered; its cache is left alone
            continue
        try:
            run = run_question(
                q, index, store, backend, str(models["synthesis"]), cfg.top_k,
                cfg.display_cutoff, paths.answers, models=run_models,
                max_passages=cfg.max_passages, max_rejected=cfg.max_rejected,
                prompt_path=paths.prompt,
                rejected_dir=paths.rejected)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            # A transient error (CLI exit, I/O) says nothing about a committed answer.
            logger.warning("%s: %s; keeping any committed answer", q.slug, exc)
            failure = Failure(q.slug, f"{type(exc).__name__}: {str(exc)[:300]}")
            record_failure(paths.answers, failure, keep_answer=True)
            failures.append(failure)
            continue
        if run is None:
            failures.append(load_failure(q.slug, paths.answers)
                            or Failure(q.slug, "answer failed validation"))
        else:
            # As in _cached_runs: featured is curation in questions.yaml, and a cache hit
            # carries the spec snapshot from before the audit, so the current config decides.
            runs.append(replace(run, spec=replace(run.spec, featured=q.featured)))
    return runs, failures


def _failed_slugs(path: Path) -> set[str]:
    """Slugs listed in an existing ``failed.json`` (empty when it is missing or unreadable)."""
    if not path.exists():
        return set()
    try:
        return {str(f["slug"]) for f in json.loads(path.read_text(encoding="utf-8"))["failed"]}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("%s is unreadable (%r); every failure counts as new", path, exc)
        return set()


def _run_traces(from_cache: bool, paths: TracePaths | None = None) -> int:
    """Answer every configured question (or reuse the cache) and export the traces.

    A question whose human audit verdict is ``fail`` is never answered or exported, in
    either mode; ``failed.json`` gives the reason ``failed human audit: <audit_note>``.
    Live mode needs a clean git tree (the commit is recorded in every trace) and the
    ``claude`` CLI; it exits 1 when a question failed (other than on audit) that the
    previous ``exports/failed.json`` did not already list. ``from_cache`` never opens
    the store or calls a model: it re-exports the committed answers, keeps the
    committed manifest's ``build_commit`` and the committed failure reasons, and
    exits 0 even when some questions are listed in ``exports/failed.json``. It exits
    2, writing nothing, when ``filings_kb/``, ``schemas/``, ``prompts/`` or ``config/``
    differ between that ``build_commit`` and HEAD (the commit would no longer be true).
    """
    import yaml

    from filings_kb.answer import load_questions
    from filings_kb.export import (
        StaleBuildError, commit_from_manifest, export_all, load_signal_rows, require_clean_tree,
        require_code_unchanged, sample_summary,
    )

    paths = paths or TracePaths.default()
    cfg = load_questions(paths.questions)
    models = yaml.safe_load(paths.models.read_text(encoding="utf-8"))
    if from_cache:
        try:
            commit = commit_from_manifest(paths.exports / "manifest.json")
        except (OSError, ValueError) as exc:
            print(f"cannot reuse build_commit: {exc}")
            return 1
        try:
            require_code_unchanged(paths.repo, commit)
        except StaleBuildError as exc:
            print(f"cannot reuse build_commit: {exc}")
            return 2
        runs, failures = _cached_runs(cfg, paths)
    else:
        commit = require_clean_tree(paths.repo)
        if not _claude_available():
            print("answer synthesis needs the claude CLI")
            return 1
        prior_failed = _failed_slugs(paths.exports / "failed.json")
        runs, failures = _live_runs(cfg, paths, models)
    try:
        result = export_all(runs, paths.exports, commit,
                            signals=load_signal_rows(paths.signals),
                            full_corpus=models["full_corpus"],
                            sample=sample_summary(paths.sample), failures=failures)
    except ValueError as exc:
        print(f"export failed: {exc}")
        return 1
    for f in result.failures:
        print(f"FAILED {f.slug}: {f.reason}")
    exported = len(cfg.questions) - len(result.failures)
    print(f"traces={exported} failed={len(result.failures)} build_commit={commit} "
          f"manifest={result.manifest_path}")
    if from_cache:
        return 0
    # A failed human audit is curation, not a pipeline regression, so it never fails the run.
    new = sorted({f.slug for f in result.failures} - prior_failed - set(cfg.audit_failures))
    if new:
        print(f"new failures: {', '.join(new)}")
    return 1 if new else 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns the process exit code (1 if any section failed)."""
    parser = argparse.ArgumentParser(prog="filings_kb")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="fetch, chunk, embed, extract/import and link the sample")
    ingest.add_argument("--live", action="store_true",
                        help="call the extraction model for sections with no cached envelope")
    ingest.add_argument("--refresh", action="store_true",
                        help="re-extract every detected section live, ignoring the committed "
                             "cache (implies --live; needs --write-committed and a fresh "
                             "data/kb.sqlite)")
    ingest.add_argument("--write-committed", action="store_true",
                        help="write the tracked data/committed/ outputs (default: data/local/)")
    traces = sub.add_parser("traces", help="cited answers for every question -> exports/")
    traces.add_argument("--from-cache", action="store_true",
                        help="re-export the committed answers; no model call, keeps build_commit")
    signals = sub.add_parser("signals",
                             help="YoY tone, sentiment and claim-diff signals -> signals.json")
    signals.add_argument("--write-committed", action="store_true",
                         help="write data/committed/signals.json and claim_diffs/ "
                              "(default: data/local/, with drift against the committed copy)")
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")  # EDGAR_IDENTITY lives there; never printed
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "signals":
        return _run_signals(args.write_committed)
    if args.command == "traces":
        return _run_traces(args.from_cache)

    if args.refresh and not args.write_committed:
        print("--refresh overwrites data/committed/extractions; pass --write-committed "
              "(make sample-refresh does)")
        return 2
    started = time.monotonic()
    report = ingest_sample(SAMPLE_PATH, live=args.live or args.refresh, refresh=args.refresh,
                           write_committed=args.write_committed)
    duration = time.monotonic() - started
    report_path = REPORT_PATH if args.write_committed else LOCAL_REPORT_PATH
    write_report(report, report_path)
    _log_duration("ingest", duration)
    print(
        f"filings={report.filings} sections={report.sections} chunks={report.chunks} "
        f"claims={report.claims} failed_sections={len(report.failed_sections)} "
        f"cost_usd={report.cost_usd} duration_s={duration:.0f} report={report_path}"
    )
    return 1 if report.failed_sections else 0


if __name__ == "__main__":
    raise SystemExit(main())
