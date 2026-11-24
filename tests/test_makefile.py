"""Tests for the Makefile targets the README documents (dry runs only: ``make -n``)."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")


def _dry_run(*args: str) -> str:
    done = subprocess.run(["make", "-n", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True, timeout=30)
    return done.stdout


def test_evals_runs_the_trace_check_and_parity_only_with_a_private_kb() -> None:
    without = _dry_run("evals", "PRIVATE_KB=")
    assert "evals.trace_check exports/traces" in without and "evals.parity" not in without
    with_kb = _dry_run("evals", "PRIVATE_KB=/kb/knowledge_graph.db")
    assert "evals.trace_check exports/traces" in with_kb
    assert "evals.parity --private-kb /kb/knowledge_graph.db" in with_kb


def test_sample_refresh_bypasses_the_cache_and_sample_does_not() -> None:
    assert "filings_kb ingest --refresh" in _dry_run("sample-refresh")
    assert "--refresh" not in _dry_run("sample")


def test_only_sample_refresh_writes_committed_outputs() -> None:
    refresh = _dry_run("sample-refresh")
    assert "filings_kb ingest --refresh --write-committed" in refresh
    assert "filings_kb signals --write-committed" in refresh
    for target in ("sample", "ingest", "signals"):
        assert "--write-committed" not in _dry_run(target), target
