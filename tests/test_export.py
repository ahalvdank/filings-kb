"""Trace building, audit notes, clean-tree guard and export (fake index and backend)."""

import json
import subprocess
from pathlib import Path

import pytest

import dataclasses

from filings_kb.answer import (
    AnswerRun, AnswerUnit, ClaimView, Failure, QuestionSpec, run_question,
)
from filings_kb.export import (
    EXCLUDED_ENTITY_TYPES, DirtyTreeError, build_audit_note, build_manifest, build_trace,
    export_all,
    SCHEMAS_DIR, load_signal_rows, require_clean_tree, sample_summary, validate_against,
)
from filings_kb.signals import ClaimDiff, SignalRow
from filings_kb.store import ClaimRow, Store
from tests.conftest import (
    AMD_AMEND, AMD_ORIG, NVDA_24, NVDA_25, FakeBackend, FakeIndex, answer_json, fixed_now,
)

COMMIT = "a" * 40
FULL = {"companies": 186, "filings": 973}
MODELS = {"embedding": "emb", "extraction": "ext", "synthesis": "syn", "sentiment": "sen"}
GOOD = answer_json([("Export controls cut revenue by $5.5 billion.", [1]),
                    ("Customers are concentrated.", [2])])
SAMPLE = {"companies": ["AMD", "NVDA"], "filings": 4}
TICKERS = set(SAMPLE["companies"])


def spec(**kw) -> QuestionSpec:
    base = dict(slug="nvda-risk", title="NVDA risk", question="How did risk change?",
                category="risk_shift", filings=(("NVDA", 2024), ("NVDA", 2025)),
                items=("Item_1A",))
    base.update(kw)
    return QuestionSpec(**base)


def make_run(store: Store, q: QuestionSpec | None = None, answer: str = GOOD) -> AnswerRun:
    q = q or spec()
    scores = {f"{NVDA_25}:Item_1A:0000": 0.9, f"{NVDA_24}:Item_1A:0000": 0.7,
              f"{NVDA_25}:Item_1A:0001": 0.5, f"{NVDA_24}:Item_1A:0003": -0.1}
    return run_question(q, FakeIndex(store, scores), store, FakeBackend(answer), "syn", 20,
                        0.6, _tmp(store), models=MODELS, now=fixed_now,
                        rejected_dir=_tmp(store) / "rejected")


def trace(run: AnswerRun, signals: list[SignalRow] | None = None) -> dict:
    return build_trace(run, signals or [], COMMIT, TICKERS)


def _tmp(store: Store) -> Path:
    return store.path.parent / "answers"


def sig(**kw) -> SignalRow:
    base = dict(ticker="NVDA", item="Item_1A", prior_fy=2024, current_fy=2025,
                tone_shift=0.01, sentiment_shift=-0.02,
                claim_diff=ClaimDiff(added=["A"], removed=["R"], changed=[("b", "a")]),
                claim_diff_status="ok")
    base.update(kw)
    return SignalRow(**base)


# --- build_trace ---------------------------------------------------------------


def test_build_trace_has_exactly_the_contract_keys(store):
    t = trace(make_run(store), [sig()])
    assert set(t) == {"slug", "title", "question", "category", "tickers", "meta", "answer_md",
                      "passages", "claims", "signals", "audit_note_md"}
    assert t["tickers"] == ["NVDA"]
    assert t["meta"] == {"run_timestamp": "2026-11-23T12:00:00Z", "build_commit": COMMIT,
                         "models": MODELS, "top_k": 20, "max_passages": 40,
                         "max_rejected": 10, "display_cutoff": 0.6}


def test_build_trace_meta_models_has_four_keys_including_sentiment(store):
    assert set(trace(make_run(store))["meta"]["models"]) == {
        "embedding", "extraction", "synthesis", "sentiment"}


def test_build_trace_is_pure_over_the_run_and_needs_no_store(store):
    run = make_run(store)
    before = trace(run, [sig()])
    store.path.unlink()  # C1: nothing below may read the store
    assert trace(run, [sig()]) == before
    assert not store.path.exists()


def test_build_trace_passages_sorted_by_score_with_contract_fields(store):
    ps = trace(make_run(store))["passages"]
    assert [p["score"] for p in ps] == sorted((p["score"] for p in ps), reverse=True)
    first = ps[0]
    assert set(first) == {"cite", "included", "score", "ticker", "company", "form",
                          "fiscal_year", "filed", "section", "accession", "sec_url", "context"}
    assert first["cite"] == 1 and first["included"] is True
    assert first["company"] == "NVIDIA CORP" and first["form"] == "10-K"
    assert first["section"] == "Item 1A Risk Factors" and first["filed"] == "2025-02-26"
    assert first["sec_url"] == ("https://www.sec.gov/Archives/edgar/data/1045810/"
                                f"{NVDA_25.replace('-', '')}/nvda-2025.htm")
    assert set(first["context"]) == {"before", "match", "after"}
    assert first["context"]["match"].startswith("Export controls")
    assert "Customer concentration" in first["context"]["after"]


def test_build_trace_rejected_passages_have_null_cite_and_included_false(store):
    ps = trace(make_run(store))["passages"]
    rejected = [p for p in ps if not p["included"]]
    assert len(rejected) == 2 and all(p["cite"] is None for p in rejected)


def test_build_trace_included_but_uncited_passage_has_null_cite(store):
    # Contract: cite is null "when not cited in the answer". The answer cites [1] only.
    run = make_run(store, answer=answer_json([("Export controls cut revenue.", [1])]))
    by_score = {p["score"]: p for p in trace(run)["passages"]}
    assert by_score[0.9]["cite"] == 1
    assert by_score[0.7]["included"] is True and by_score[0.7]["cite"] is None


def test_build_trace_renumbers_cites_by_first_appearance_in_the_answer(store):
    # M3: the model cites [2] (score 0.7) first, so it is shown as [1].
    run = make_run(store, answer=answer_json([("Customers are concentrated.", [2]),
                                              ("Revenue may fall $5.5 billion.", [1])]))
    t = trace(run)
    assert t["answer_md"] == "Customers are concentrated [1]. Revenue may fall $5.5 billion [2]."
    by_score = {p["score"]: p["cite"] for p in t["passages"]}
    assert by_score[0.7] == 1 and by_score[0.9] == 2


def test_snapshot_context_falls_back_to_match_only_when_section_missing(store, caplog):
    with store._conn() as conn:  # noqa: SLF001
        conn.execute("UPDATE sections SET text = 'changed text' WHERE accession = ?", (NVDA_25,))
    ps = trace(make_run(store))["passages"]
    p = next(p for p in ps if p["accession"] == NVDA_25 and p["score"] == 0.9)
    assert p["context"] == {"before": "", "match": p["context"]["match"], "after": ""}
    assert p["context"]["match"].startswith("Export controls")
    assert "context" in caplog.text


def test_build_trace_claims_map_type_subject_and_statement(store):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    cid = store.put_entity("Risk", "Export control risk", {"severity": "high", "n": 3}, {})
    store.add_relationship(cid, nvda, "subject_of", NVDA_25, "Item_1A", {})
    [claim] = trace(make_run(store))["claims"]
    assert claim["claim_type"] == "Risk" and claim["entity_type"] == "Company"
    assert claim["entity"] == "NVIDIA CORP" and claim["ticker"] == "NVDA"
    assert claim["statement"] == "NVIDIA CORP: Export control risk (n 3; severity high)."
    assert claim["raw_extraction"] == {"severity": "high", "n": 3}
    assert claim["cite"] is None and "superseded" not in claim and "relationship" not in claim


def test_claim_cite_set_only_when_the_overlapping_passage_is_cited(store):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    cid = store.put_entity("Risk", "Customer concentration remains high", {}, {})
    store.add_relationship(cid, nvda, "subject_of", NVDA_25, "Item_1A", {})
    # The overlapping chunk (NVDA_25 #1) is included at cutoff 0.4 but the answer cites only [1].
    q = spec()
    scores = {f"{NVDA_25}:Item_1A:0000": 0.9, f"{NVDA_25}:Item_1A:0001": 0.5}
    run = run_question(q, FakeIndex(store, scores), store,
                       FakeBackend(answer_json([("Only first.", [1])])), "syn", 20, 0.4,
                       _tmp(store), models=MODELS, now=fixed_now,
                       rejected_dir=_tmp(store) / "rej")
    assert trace(run)["claims"][0]["cite"] is None
    run2 = run_question(q, FakeIndex(store, scores), store,
                        FakeBackend(answer_json([("Second.", [2]), ("First.", [1])])),
                        "syn", 20, 0.4, _tmp(store) / "b", models=MODELS, now=fixed_now,
                        rejected_dir=_tmp(store) / "rej")
    # Prompt passage [2] is cited first, so it is displayed (and the claim points to it) as [1].
    assert trace(run2)["claims"][0]["cite"] == 1


def test_build_trace_attaches_signals_for_trace_tickers_items_and_year_pairs(store):
    rows = [sig(), sig(ticker="AMD"), sig(item="Item_7_MDA"), sig(prior_fy=2023, current_fy=2024),
            sig(tone_shift=None, sentiment_shift=None, claim_diff=None,
                claim_diff_status="not_requested", prior_fy=2025, current_fy=2026)]
    t = trace(make_run(store), rows)
    assert t["signals"] == [{
        "ticker": "NVDA", "section": "Item 1A Risk Factors", "prior_fy": 2024, "current_fy": 2025,
        "tone_shift": 0.01, "sentiment_shift": -0.02,
        "claim_diff": {"added": ["A"], "removed": ["R"],
                       "changed": [{"before": "b", "after": "a"}]}}]


def test_build_trace_omits_signals_key_when_none_apply(store):
    assert "signals" not in trace(make_run(store))


def _amendment_run(store: Store) -> AnswerRun:
    from datetime import UTC, datetime
    amd = store.put_entity("Company", "ADVANCED MICRO DEVICES INC", {}, {})
    old = store.put_entity("GuidanceUpdate", "Old guidance", {}, {})
    new = store.put_entity("GuidanceUpdate", "New guidance", {}, {})
    store.add_relationship(old, amd, "subject_of", AMD_ORIG, "Item_7_MDA", {})
    store.add_relationship(new, amd, "subject_of", AMD_AMEND, "Item_7_MDA", {})
    store.close_out(old, datetime(2026, 2, 4, tzinfo=UTC))
    from filings_kb.chunking import chunk_section
    from filings_kb.sections import Section
    from tests.conftest import count_tokens
    for acc in (AMD_ORIG, AMD_AMEND):
        sec = Section(acc, "AMD", 2025, "Item_7_MDA", "MD&A", "Guidance text here. More words.")
        store.put_section(sec)
        store.put_chunks(chunk_section(sec, count_tokens, 14))
    q = spec(slug="amd", filings=(("AMD", 2025),), items=("Item_7_MDA",), amendment=True,
             title="AMD amendment", question="What changed?")
    scores = {f"{AMD_ORIG}:Item_7_MDA:0000": 0.8, f"{AMD_AMEND}:Item_7_MDA:0000": 0.7}
    return run_question(q, FakeIndex(store, scores), store,
                        FakeBackend(answer_json([("Guidance changed.", [1, 2])])),
                        "syn", 20, 0.0, _tmp(store), models=MODELS, now=fixed_now,
                        rejected_dir=_tmp(store) / "rej")


def test_amendment_trace_flags_superseded_claims_and_shows_both_forms(store):
    t = trace(_amendment_run(store))
    assert [(c["statement"].split(":")[1].split("(")[0].strip(" ."), c["superseded"])
            for c in t["claims"]] == [("Old guidance", True), ("New guidance", False)]
    assert {p["form"] for p in t["passages"]} == {"10-K", "10-K/A"}


def test_amendment_audit_note_marks_superseded_claims(store):
    note = trace(_amendment_run(store))["audit_note_md"]
    [old] = [line for line in note.splitlines() if "Old guidance" in line]
    [new] = [line for line in note.splitlines() if "New guidance" in line]
    assert "superseded" in old.lower() and "superseded" not in new.lower()


# --- audit note ----------------------------------------------------------------


def _view(subject_type="Company", subject="NVIDIA CORP", name="Risk X", ticker="NVDA", cite=1,
          superseded=None):
    row = ClaimRow("e1", "Risk", name, {}, "s1", subject_type, subject, NVDA_25, "Item_1A")
    return ClaimView(row, ticker, () if cite is None else (cite,), superseded)


def test_audit_note_lists_entities_as_wikilinks_with_cited_passages():
    note = build_audit_note("T", "Q?", [{"company": "NVIDIA CORP", "ticker": "NVDA", "cite": 2}],
                            [_view(cite=2)], {"NVDA"})
    assert "[[NVIDIA CORP]]" in note and "[[Risk X]]" in note
    assert "[2]" in note and note.startswith("# T")


def test_audit_note_marks_superseded_claim_lines():
    note = build_audit_note("T", "Q?", [], [_view(name="Old", superseded=True),
                                            _view(name="New", superseded=False),
                                            _view(name="Plain")], {"NVDA"})
    lines = {n: next(line for line in note.splitlines() if f"[[{n}]]" in line)
             for n in ("Old", "New", "Plain")}
    assert "superseded" in lines["Old"].lower()
    assert "superseded" not in lines["New"].lower() and "superseded" not in lines["Plain"].lower()


def test_audit_note_excludes_non_sample_entities():
    views = [_view(subject_type="ThesisPage", subject="My Thesis", name="Thesis claim"),
             _view(subject="Other Corp", name="Other claim", ticker="ZZZZ"),
             _view(subject_type="ResearchSource", subject="own_research", name="Memo claim")]
    note = build_audit_note("T", "Q?", [{"company": "NVIDIA CORP", "ticker": "NVDA", "cite": 1},
                                        {"company": "Other Corp", "ticker": "ZZZZ", "cite": 3}],
                            views, {"NVDA"})
    for banned in ("Thesis", "own_research", "Other", "ZZZZ", "Memo"):
        assert banned not in note
    assert "[[NVIDIA CORP]]" in note


def test_audit_note_sanitises_wikilink_text():
    note = build_audit_note("T", "Q?", [], [_view(name="Risk [x] | y")], {"NVDA"})
    assert "[[Risk x y]]" in note


# --- clean tree ----------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "code.py").write_text("x = 1\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_require_clean_tree_returns_head(repo):
    assert require_clean_tree(repo) == _git(repo, "rev-parse", "HEAD")


def test_export_refuses_dirty_tree(repo):
    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(DirtyTreeError):
        require_clean_tree(repo)


def test_export_refuses_untracked_file(repo):
    (repo / "new.py").write_text("y = 1\n")
    with pytest.raises(DirtyTreeError):
        require_clean_tree(repo)


def test_require_clean_tree_ignores_its_own_outputs(repo):
    (repo / "exports").mkdir()
    (repo / "exports" / "manifest.json").write_text("{}")
    (repo / "data" / "committed" / "answers").mkdir(parents=True)
    (repo / "data" / "committed" / "answers" / "a.json").write_text("{}")
    assert require_clean_tree(repo) == _git(repo, "rev-parse", "HEAD")


def test_require_clean_tree_outside_a_repo_raises(tmp_path):
    with pytest.raises(RuntimeError, match="git"):
        require_clean_tree(tmp_path)


# --- manifest and export_all ---------------------------------------------------


def test_manifest_full_corpus_is_186_companies_973_filings():
    cfg = yaml_models()
    assert cfg["full_corpus"] == {"companies": 186, "filings": 973}


def yaml_models() -> dict:
    import yaml
    return yaml.safe_load((Path(__file__).resolve().parent.parent / "config" /
                           "models.yaml").read_text())


def test_sample_summary_reads_the_pinned_sample_config():
    got = sample_summary()
    assert got["filings"] == 61 and len(got["companies"]) == 12
    assert got["companies"] == sorted(got["companies"]) and "NVDA" in got["companies"]


def test_build_manifest_takes_sample_from_config_not_the_store(store):
    run = make_run(store)
    m = build_manifest([run], COMMIT, FULL, "0.1.0", SAMPLE)
    assert m["sample"] == SAMPLE
    assert m["full_corpus"] == FULL and m["build_commit"] == COMMIT
    assert m["run_timestamp"] == "2026-11-23T12:00:00Z" and m["pipeline_version"] == "0.1.0"
    assert m["export_script_url"] == (
        f"https://github.com/ahalvdank/filings-kb/blob/{COMMIT}/filings_kb/export.py")
    assert m["traces"] == [{"slug": "nvda-risk", "title": "NVDA risk",
                            "question": "How did risk change?", "category": "risk_shift",
                            "tickers": ["NVDA"], "featured": False}]


def test_manifest_carries_featured_and_the_schema_accepts_it(store):
    run = make_run(store, spec(featured=True))
    m = build_manifest([run], COMMIT, FULL, "0.1.0", SAMPLE)
    assert m["traces"][0]["featured"] is True
    validate_against(m, SCHEMAS_DIR, "manifest", "manifest")
    m["traces"][0]["featured"] = "yes"
    with pytest.raises(ValueError, match="featured"):
        validate_against(m, SCHEMAS_DIR, "manifest", "manifest")


def export(runs, out, commit=COMMIT, failures=(), signals=None):
    return export_all(runs, out, commit, signals=signals or [], full_corpus=FULL, sample=SAMPLE,
                      failures=list(failures))


def test_export_all_writes_validated_sorted_files(store, tmp_path):
    out = tmp_path / "exports"
    result = export([make_run(store)], out, signals=[sig()])
    assert result.manifest_path == out / "manifest.json" and result.failures == []
    text = (out / "traces" / "nvda-risk.json").read_text()
    assert text.endswith("}\n") and '\n  "answer_md"' in text
    data = json.loads(text)
    assert list(data) == sorted(data)
    assert (out / "audit_notes" / "nvda-risk.md").read_text() == data["audit_note_md"]
    assert json.loads(result.manifest_path.read_text())["traces"][0]["slug"] == "nvda-risk"
    assert json.loads((out / "failed.json").read_text()) == {"failed": []}


def test_export_all_needs_no_store(store, tmp_path):
    run = make_run(store)
    store.path.unlink()
    export([run], tmp_path / "exports")
    assert (tmp_path / "exports" / "traces" / "nvda-risk.json").exists()
    assert not store.path.exists()


def test_export_all_is_byte_identical_on_rerun(store, tmp_path):
    out = tmp_path / "exports"
    run = make_run(store)
    fail = [Failure("other", "answer failed validation", ("data/rejected/other.attempt1.json",))]
    export([run], out, failures=fail, signals=[sig()])
    first = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    export([run], out, failures=fail, signals=[sig()])
    assert first == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_export_all_revalidates_every_answer_and_records_failures(store, tmp_path):
    # I2: a cached answer whose number is not in its cited passage is not exported.
    good = make_run(store)
    bad_spec = dataclasses.replace(good.spec, slug="tampered")
    bad = dataclasses.replace(good, spec=bad_spec, paragraphs=(
        (AnswerUnit("Revenue fell $34.6 billion.", (1,)),),))
    out = tmp_path / "exports"
    result = export([good, bad], out, failures=[Failure("missing", "no cached answer")])
    assert [f.slug for f in result.failures] == ["missing", "tampered"]
    assert not (out / "traces" / "tampered.json").exists()
    assert (out / "traces" / "nvda-risk.json").exists()
    failed = json.loads((out / "failed.json").read_text())["failed"]
    assert [f["slug"] for f in failed] == ["missing", "tampered"]
    assert "34.6" in failed[1]["reason"] and failed[1]["rejected"] == []
    slugs = [t["slug"] for t in json.loads((out / "manifest.json").read_text())["traces"]]
    assert slugs == ["nvda-risk"]


def test_export_all_removes_stale_traces_and_notes(store, tmp_path):
    out = tmp_path / "exports"
    (out / "traces").mkdir(parents=True)
    (out / "audit_notes").mkdir()
    (out / "traces" / "old.json").write_text("{}")
    (out / "audit_notes" / "old.md").write_text("x")
    export([make_run(store)], out)
    assert not (out / "traces" / "old.json").exists()
    assert not (out / "audit_notes" / "old.md").exists()


def test_export_all_writes_nothing_when_a_trace_fails_validation(store, tmp_path):
    out = tmp_path / "exports"
    bad = make_run(store)
    with pytest.raises(ValueError, match="nvda-risk"):
        export([bad], out, commit="not-a-commit")
    assert not out.exists()


def test_load_signal_rows_roundtrips_signals_json(tmp_path):
    path = tmp_path / "signals.json"
    path.write_text(json.dumps({"meta": {}, "signals": [{
        "ticker": "NVDA", "item": "Item_1A", "prior_fy": 2024, "current_fy": 2025,
        "tone_shift": 0.1, "sentiment_shift": None,
        "claim_diff": {"added": ["a"], "removed": [], "changed": [["b", "c"]]},
        "claim_diff_status": "ok", "status": "shadow"}]}))
    [row] = load_signal_rows(path)
    assert row.claim_diff == ClaimDiff(added=["a"], removed=[], changed=[("b", "c")])
    assert row.sentiment_shift is None


def test_commit_from_manifest_reads_build_commit(tmp_path):
    from filings_kb.export import commit_from_manifest
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"build_commit": COMMIT}))
    assert commit_from_manifest(path) == COMMIT
    path.write_text(json.dumps({"build_commit": "nope"}))
    with pytest.raises(ValueError, match="build_commit"):
        commit_from_manifest(path)
    with pytest.raises(FileNotFoundError):
        commit_from_manifest(tmp_path / "missing.json")


# --- display cutoff (Task 9) ----------------------------------------------------


def _cutoff_run(store: Store, max_rejected: int) -> AnswerRun:
    scores = {f"{NVDA_25}:Item_1A:0000": 0.9, f"{NVDA_25}:Item_1A:0001": 0.5,
              f"{NVDA_25}:Item_1A:0002": 0.4, f"{NVDA_25}:Item_1A:0003": 0.3,
              f"{NVDA_24}:Item_1A:0000": 0.2, f"{NVDA_24}:Item_1A:0001": 0.1}
    return run_question(
        spec(), FakeIndex(store, scores), store,
        FakeBackend(answer_json([("Export controls cut revenue by $5.5 billion.", [1])])),
        "syn", 20, 0.8, _tmp(store), models=MODELS, now=fixed_now,
        rejected_dir=_tmp(store) / "rejected", max_rejected=max_rejected)


def test_rejected_candidates_exported_with_included_false_and_capped(store):
    t = trace(_cutoff_run(store, max_rejected=3))
    included = [p for p in t["passages"] if p["included"]]
    rejected = [p for p in t["passages"] if not p["included"]]
    assert [p["score"] for p in included] == [0.9]
    assert [p["score"] for p in rejected] == [0.5, 0.4, 0.3]  # best three of the five
    assert all(p["cite"] is None for p in rejected)
    assert all(p["score"] < t["meta"]["display_cutoff"] for p in rejected)
    assert t["meta"]["display_cutoff"] == 0.8 and t["meta"]["max_rejected"] == 3
    assert [p["score"] for p in t["passages"]] == [0.9, 0.5, 0.4, 0.3]


def test_cited_passage_never_below_cutoff(store, tmp_path):
    good = _cutoff_run(store, max_rejected=3)
    cited = good.passages[0]
    assert cited.cite == 1 and cited.score >= good.display_cutoff
    # A snapshot where a cited passage scores under the cutoff is not exported.
    bad_passage = dataclasses.replace(cited, score=0.1)
    bad = dataclasses.replace(good, spec=dataclasses.replace(good.spec, slug="tampered"),
                              passages=[bad_passage, *good.passages[1:]])
    out = tmp_path / "exports"
    result = export([good, bad], out)
    assert [f.slug for f in result.failures] == ["tampered"]
    assert "below the display cutoff" in result.failures[0].reason
    assert not (out / "traces" / "tampered.json").exists()
    assert (out / "traces" / "nvda-risk.json").exists()
    # The same check holds for a rejected passage that scores at or above the cutoff.
    leaked = dataclasses.replace(good.passages[1], score=0.95)
    bad2 = dataclasses.replace(good, spec=dataclasses.replace(good.spec, slug="leaked"),
                               passages=[good.passages[0], leaked, *good.passages[2:]])
    result = export([good, bad2], tmp_path / "exports2")
    assert [f.slug for f in result.failures] == ["leaked"]
    assert "at or above the display cutoff" in result.failures[0].reason


# --- final review I6: excluded entity types never reach a trace's claims ----------------


@pytest.mark.parametrize("excluded", sorted(EXCLUDED_ENTITY_TYPES))
def test_build_trace_drops_claims_whose_subject_type_is_excluded(store, excluded):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    private = store.put_entity(excluded, "My private page", {}, {})
    for name, subject in (("Export control risk", nvda), ("Private thesis risk", private)):
        claim = store.put_entity("Risk", name, {}, {})
        store.add_relationship(claim, subject, "subject_of", NVDA_25, "Item_1A", {})
    run = make_run(store)
    assert {v.row.subject_type for v in run.claims} == {"Company", excluded}  # the input is untouched
    t = trace(run)
    assert [c["entity"] for c in t["claims"]] == ["NVIDIA CORP"]
    assert "My private page" not in json.dumps(t)


# --- final review I2: a cache export refuses a build_commit whose code has changed -------


def _commit_file(repo: Path, rel: str, text: str) -> str:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {rel}")
    return _git(repo, "rev-parse", "HEAD")


def test_build_commit_guard_passes_when_only_data_and_docs_changed(repo):
    from filings_kb.export import require_code_unchanged
    build = _commit_file(repo, "filings_kb/answer.py", "v = 1\n")
    _commit_file(repo, "exports/manifest.json", "{}\n")
    _commit_file(repo, "README.md", "docs\n")
    _commit_file(repo, "data/committed/answers/a.json", "{}\n")
    require_code_unchanged(repo, build)  # no error


@pytest.mark.parametrize("rel", ["filings_kb/answer.py", "schemas/trace.schema.json",
                                 "prompts/answer.md", "config/questions.yaml"])
def test_build_commit_guard_refuses_changed_code_schemas_prompts_or_config(repo, rel):
    from filings_kb.export import StaleBuildError, require_code_unchanged
    build = _commit_file(repo, "filings_kb/answer.py", "v = 1\n")
    _commit_file(repo, rel, "changed\n")
    with pytest.raises(StaleBuildError, match=rel):
        require_code_unchanged(repo, build)


def test_build_commit_guard_refuses_an_unknown_commit(repo):
    from filings_kb.export import StaleBuildError, require_code_unchanged
    with pytest.raises(StaleBuildError, match="cannot compare"):
        require_code_unchanged(repo, "f" * 40)
