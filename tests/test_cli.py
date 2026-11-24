"""Tests for the command-line entry point (ingest itself is faked)."""

from decimal import Decimal
from pathlib import Path

import pytest

from filings_kb import __main__ as cli
from filings_kb.ingest import IngestReport


def _fake_report(failed: list[str]) -> IngestReport:
    return IngestReport(filings=1, sections=2, chunks=3, claims=4,
                        failed_sections=failed, cost_usd=Decimal("0"))


def test_ingest_defaults_to_cache_only_and_writes_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []
    monkeypatch.setattr(cli, "ingest_sample", _fake_ingest(calls))
    _ingest_paths(monkeypatch, tmp_path)
    assert cli.main(["ingest"]) == 0
    assert calls == [(False, False, False)]
    assert (tmp_path / "local" / "ingest_report.json").exists()
    assert "filings=1" in capsys.readouterr().out


def test_ingest_live_flag_is_passed_through(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    calls: list = []
    monkeypatch.setattr(cli, "ingest_sample", _fake_ingest(calls))
    _ingest_paths(monkeypatch, tmp_path)
    cli.main(["ingest", "--live"])
    assert calls == [(True, False, False)]


def test_ingest_refresh_flag_implies_live_and_is_passed_through(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list = []
    monkeypatch.setattr(cli, "ingest_sample", _fake_ingest(calls))
    _ingest_paths(monkeypatch, tmp_path)
    cli.main(["ingest", "--refresh", "--write-committed"])
    cli.main(["ingest"])
    assert calls == [(True, True, True), (False, False, False)]


def test_failed_sections_give_nonzero_exit(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "ingest_sample", lambda p, live, refresh=False,
                        write_committed=False: _fake_report(["a/Item_1A"]))
    _ingest_paths(monkeypatch, tmp_path)
    assert cli.main(["ingest"]) == 1


def _ingest_paths(monkeypatch: pytest.MonkeyPatch, data: Path) -> None:
    monkeypatch.setattr(cli, "REPORT_PATH", data / "committed" / "ingest_report.json")
    monkeypatch.setattr(cli, "LOCAL_REPORT_PATH", data / "local" / "ingest_report.json")
    monkeypatch.setattr(cli, "RUN_LOG_PATH", data / "run_log.json")


def _fake_ingest(calls: list | None = None):
    def fake(sample_path: Path, live: bool, refresh: bool = False,
             write_committed: bool = False) -> IngestReport:
        if calls is not None:
            calls.append((live, refresh, write_committed))
        return _fake_report([])
    return fake


# --- traces (C1, I7) -------------------------------------------------------------

import json  # noqa: E402

import yaml  # noqa: E402

from filings_kb.answer import Failure, record_failure  # noqa: E402
from filings_kb.extract import MODELS_PATH  # noqa: E402
from filings_kb.store import Store  # noqa: E402
from tests.conftest import NVDA_24, NVDA_25, FakeBackend, FakeIndex, answer_json  # noqa: E402

COMMIT = "b" * 40
GOOD = answer_json([("Export controls cut revenue by $5.5 billion.", [1])])
BAD = answer_json([("Revenue fell 77%.", [1])])
SCORES = {f"{NVDA_25}:Item_1A:0000": 0.9, f"{NVDA_24}:Item_1A:0000": 0.7}
QUESTION = {"title": "T", "question": "How did risk change?", "category": "risk_shift",
            "filings": [{"ticker": "NVDA", "fiscal_year": 2024},
                        {"ticker": "NVDA", "fiscal_year": 2025}],
            "items": ["Item_1A"], "featured": False}


def _paths(tmp_path: Path) -> "cli.TracePaths":
    root = tmp_path / "clone"
    (root / "config").mkdir(parents=True)
    (root / "prompts").mkdir()
    (root / "config" / "questions.yaml").write_text(yaml.safe_dump({
        "top_k": 20, "max_passages": 40, "display_cutoff": 0.6,
        "questions": [{"slug": "nvda-risk", **QUESTION}, {"slug": "other-q", **QUESTION}]}))
    (root / "prompts" / "answer.md").write_text("prompt v2")
    (root / "data" / "committed").mkdir(parents=True)
    (root / "data" / "committed" / "signals.json").write_text('{"meta": {}, "signals": []}')
    return cli.TracePaths(
        repo=root, questions=root / "config" / "questions.yaml",
        prompt=root / "prompts" / "answer.md", answers=root / "data" / "committed" / "answers",
        rejected=root / "data" / "rejected", exports=root / "exports",
        signals=root / "data" / "committed" / "signals.json", models=MODELS_PATH,
        sample=cli.SAMPLE_PATH, data=root / "data")


def _live(monkeypatch: pytest.MonkeyPatch, store: Store, *replies: str) -> None:
    monkeypatch.setattr("filings_kb.export.require_clean_tree", lambda repo: COMMIT)
    monkeypatch.setattr(cli, "_claude_available", lambda: True)
    monkeypatch.setattr("filings_kb.llm.ClaudeCliBackend", lambda: FakeBackend(*replies))
    monkeypatch.setattr("filings_kb.embed.EmbeddingIndex",
                        lambda *a, **k: FakeIndex(store, SCORES))
    monkeypatch.setattr("filings_kb.store.Store", lambda path: store)


def _no_store(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("traces --from-cache must never open the Store")

    monkeypatch.setattr("filings_kb.store.Store", refuse)
    monkeypatch.setattr("filings_kb.embed.EmbeddingIndex", refuse)
    # The temp clone is not a git repo; the build_commit guard has its own tests.
    monkeypatch.setattr("filings_kb.export.require_code_unchanged", lambda repo, commit: None)


def _files(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_traces_live_exits_1_on_failures_and_records_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, BAD, BAD)  # nvda-risk passes; other-q fails twice
    assert cli._run_traces(False, paths) == 1
    assert (paths.exports / "traces" / "nvda-risk.json").exists()
    assert not (paths.exports / "traces" / "other-q.json").exists()
    [failed] = json.loads((paths.exports / "failed.json").read_text())["failed"]
    assert failed["slug"] == "other-q" and "validation" in failed["reason"]
    assert [Path(r).name for r in failed["rejected"]] == ["other-q.attempt1.json",
                                                         "other-q.attempt2.json"]
    manifest = json.loads((paths.exports / "manifest.json").read_text())
    assert manifest["build_commit"] == COMMIT and manifest["sample"]["filings"] == 61


def test_traces_live_exits_0_when_every_question_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    assert cli._run_traces(False, paths) == 0
    assert json.loads((paths.exports / "failed.json").read_text()) == {"failed": []}


def test_traces_from_cache_never_opens_the_store_and_needs_no_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, BAD, BAD)
    cli._run_traces(False, paths)
    live = _files(paths.exports)

    for p in paths.exports.rglob("*"):  # a clean clone holds only what was committed
        if p.is_file() and p.name != "manifest.json":
            p.unlink()
    _no_store(monkeypatch)
    assert cli._run_traces(True, paths) == 0  # I7: exit 0 although other-q is not exported
    assert not (paths.data / "kb.sqlite").exists()
    assert _files(paths.exports) == live  # byte-identical, failed.json included


def test_traces_from_cache_takes_featured_from_the_current_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)
    flags = lambda: {t["slug"]: t["featured"]  # noqa: E731
                     for t in json.loads((paths.exports / "manifest.json").read_text())["traces"]}
    assert flags() == {"nvda-risk": False, "other-q": False}

    cfg = yaml.safe_load(paths.questions.read_text())
    cfg["questions"][0]["featured"] = True  # audit decision made after the answers were cached
    paths.questions.write_text(yaml.safe_dump(cfg))
    _no_store(monkeypatch)
    assert cli._run_traces(True, paths) == 0
    assert flags() == {"nvda-risk": True, "other-q": False}


def test_traces_from_cache_without_cache_or_record_reports_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)
    (paths.answers / "other-q.json").unlink()
    _no_store(monkeypatch)
    assert cli._run_traces(True, paths) == 0
    [failed] = json.loads((paths.exports / "failed.json").read_text())["failed"]
    assert failed == {"slug": "other-q", "reason": "no cached answer", "rejected": []}


def test_traces_from_cache_treats_an_answer_from_another_prompt_as_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)
    cached = paths.answers / "nvda-risk.json"
    data = json.loads(cached.read_text())
    cached.write_text(json.dumps({**data, "prompt_sha256": "0" * 64}))
    _no_store(monkeypatch)
    assert cli._run_traces(True, paths) == 0
    [failed] = json.loads((paths.exports / "failed.json").read_text())["failed"]
    assert failed["slug"] == "nvda-risk" and "prompt" in failed["reason"]
    assert not (paths.exports / "traces" / "nvda-risk.json").exists()
    assert (paths.exports / "traces" / "other-q.json").exists()
    paths.prompt.write_text("prompt v3")  # now no answer matches: nothing to export
    assert cli._run_traces(True, paths) == 1


def test_record_failure_roundtrips(tmp_path: Path) -> None:
    from filings_kb.answer import load_failure
    f = Failure("q", "answer failed validation", ("data/rejected/q.attempt1.json",))
    record_failure(tmp_path, f)
    assert load_failure("q", tmp_path) == f


def test_transient_error_keeps_committed_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)
    committed = (paths.answers / "nvda-risk.json").read_bytes()
    paths.prompt.write_text("prompt v3")  # force a cache miss so the backend is called

    class Flaky:
        def complete(self, system: str, user: str, model: str) -> None:
            raise RuntimeError("claude CLI exited 1")

    monkeypatch.setattr("filings_kb.llm.ClaudeCliBackend", lambda: Flaky())
    assert cli._run_traces(False, paths) == 1
    assert (paths.answers / "nvda-risk.json").read_bytes() == committed
    from filings_kb.answer import load_failure
    record = load_failure("nvda-risk", paths.answers)
    assert record is not None and record.reason.startswith("RuntimeError")


def test_live_exit_only_for_new_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, BAD, BAD)
    assert cli._run_traces(False, paths) == 1  # other-q is a new failure
    # Same failure again (cache hit for nvda-risk, other-q fails twice): not new -> 0.
    _live(monkeypatch, store, BAD, BAD)
    assert cli._run_traces(False, paths) == 0
    # nvda-risk now fails too: a new failure -> 1.
    (paths.answers / "nvda-risk.json").unlink()
    _live(monkeypatch, store, BAD, BAD, BAD, BAD)
    assert cli._run_traces(False, paths) == 1


def test_from_cache_failure_record_wins_over_a_retained_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    """A transient live failure keeps the answer file; the cache export must match live."""
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)
    record_failure(paths.answers, Failure("nvda-risk", "RuntimeError: claude CLI exited 1"),
                   keep_answer=True)
    assert (paths.answers / "nvda-risk.json").exists()
    _no_store(monkeypatch)
    assert cli._run_traces(True, paths) == 0
    [failed] = json.loads((paths.exports / "failed.json").read_text())["failed"]
    assert failed == {"slug": "nvda-risk", "reason": "transient; committed answer retained",
                      "rejected": []}
    assert not (paths.exports / "traces" / "nvda-risk.json").exists()
    assert (paths.exports / "traces" / "other-q.json").exists()


# --- final review C1: an audit-failed question is never exported -------------------------


def _fail_audit(paths: "cli.TracePaths", slug: str) -> None:
    cfg = yaml.safe_load(paths.questions.read_text())
    for q in cfg["questions"]:
        if q["slug"] == slug:
            q.update(audit="fail", audit_note="three weak sentences")
    paths.questions.write_text(yaml.safe_dump(cfg))


def test_audit_failed_question_is_not_answered_or_exported_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _fail_audit(paths, "other-q")
    backend = FakeBackend(GOOD)  # one reply: other-q must never reach the model
    _live(monkeypatch, store)
    monkeypatch.setattr("filings_kb.llm.ClaudeCliBackend", lambda: backend)
    assert cli._run_traces(False, paths) == 0  # an audit verdict is curation, not a regression
    assert len(backend.calls) == 1
    assert not (paths.exports / "traces" / "other-q.json").exists()
    [failed] = json.loads((paths.exports / "failed.json").read_text())["failed"]
    assert failed == {"slug": "other-q", "reason": "failed human audit: three weak sentences",
                      "rejected": []}
    manifest = json.loads((paths.exports / "manifest.json").read_text())
    assert [t["slug"] for t in manifest["traces"]] == ["nvda-risk"]


def test_audit_failed_question_with_a_cached_answer_is_dropped_from_cache_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)  # both exported and cached
    _fail_audit(paths, "other-q")
    _no_store(monkeypatch)
    assert cli._run_traces(True, paths) == 0
    assert (paths.answers / "other-q.json").exists()  # the cached answer is kept
    assert not (paths.exports / "traces" / "other-q.json").exists()
    assert not (paths.exports / "audit_notes" / "other-q.md").exists()
    [failed] = json.loads((paths.exports / "failed.json").read_text())["failed"]
    assert failed["reason"] == "failed human audit: three weak sentences"


# --- final review C2: plain ingest writes nothing tracked ------------------------------


def test_plain_ingest_writes_its_report_locally_and_duration_to_the_run_log(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _ingest_paths(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(cli, "ingest_sample", _fake_ingest(calls))
    assert cli.main(["ingest"]) == 0
    assert calls == [(False, False, False)]
    assert not (tmp_path / "committed").exists()
    report = json.loads((tmp_path / "local" / "ingest_report.json").read_text())
    assert "duration_s" not in report and report["filings"] == 1
    [entry] = json.loads((tmp_path / "run_log.json").read_text())["runs"]
    assert entry["command"] == "ingest" and entry["duration_s"] >= 0


def test_ingest_write_committed_writes_the_committed_report_without_duration(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _ingest_paths(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(cli, "ingest_sample", _fake_ingest(calls))
    assert cli.main(["ingest", "--refresh", "--write-committed"]) == 0
    assert calls == [(True, True, True)]
    first = (tmp_path / "committed" / "ingest_report.json").read_bytes()
    assert "duration_s" not in json.loads(first)
    assert cli.main(["ingest", "--refresh", "--write-committed"]) == 0
    assert (tmp_path / "committed" / "ingest_report.json").read_bytes() == first  # stable
    assert len(json.loads((tmp_path / "run_log.json").read_text())["runs"]) == 2


def test_refresh_without_write_committed_is_refused(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _ingest_paths(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(cli, "ingest_sample", _fake_ingest(calls))
    assert cli.main(["ingest", "--refresh"]) == 2
    assert calls == [] and "--write-committed" in capsys.readouterr().out


def test_make_ingest_then_make_traces_never_hits_a_dirty_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    """Fixture-level `make ingest && make traces`: ingest writes only ignored files."""
    import shutil
    import subprocess as sp

    from filings_kb.export import require_clean_tree

    paths = _paths(tmp_path)
    repo = paths.repo
    shutil.copy(cli.REPO_ROOT / ".gitignore", repo / ".gitignore")
    (paths.data / "committed" / "ingest_report.json").write_text("{}\n")
    git = lambda *a: sp.run(["git", "-C", str(repo), *a], check=True,  # noqa: E731
                            capture_output=True, text=True).stdout
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init")

    _ingest_paths(monkeypatch, paths.data)
    monkeypatch.setattr(cli, "ingest_sample", _fake_ingest())
    assert cli.main(["ingest"]) == 0
    assert (paths.data / "run_log.json").exists()
    assert git("status", "--porcelain", "--untracked-files=all") == ""
    head = require_clean_tree(repo)

    _live(monkeypatch, store, GOOD, GOOD)
    monkeypatch.setattr("filings_kb.export.require_clean_tree", require_clean_tree)  # the real one
    assert cli._run_traces(False, paths) == 0
    manifest = json.loads((paths.exports / "manifest.json").read_text())
    assert manifest["build_commit"] == head


# --- final review I2: from-cache refuses a stale build_commit ---------------------------


def test_traces_from_cache_exits_2_and_writes_nothing_when_code_changed_since_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from filings_kb.export import StaleBuildError

    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)
    before = _files(paths.exports)
    _no_store(monkeypatch)

    def stale(repo: Path, commit: str) -> None:
        assert commit == COMMIT
        raise StaleBuildError("filings_kb/answer.py changed since the build commit")

    monkeypatch.setattr("filings_kb.export.require_code_unchanged", stale)
    assert cli._run_traces(True, paths) == 2
    assert "filings_kb/answer.py" in capsys.readouterr().out
    assert _files(paths.exports) == before


def test_traces_live_takes_featured_from_the_current_config_on_a_cache_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, store: Store,
) -> None:
    paths = _paths(tmp_path)
    _live(monkeypatch, store, GOOD, GOOD)
    cli._run_traces(False, paths)  # cached with featured: false
    cfg = yaml.safe_load(paths.questions.read_text())
    cfg["questions"][0]["featured"] = True  # audit decision made after the answers were cached
    paths.questions.write_text(yaml.safe_dump(cfg))
    _live(monkeypatch, store)  # no replies: both questions must be cache hits
    assert cli._run_traces(False, paths) == 0
    flags = {t["slug"]: t["featured"]
             for t in json.loads((paths.exports / "manifest.json").read_text())["traces"]}
    assert flags == {"nvda-risk": True, "other-q": False}
