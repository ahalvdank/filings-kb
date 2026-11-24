"""Retrieval, citation validation and cached synthesis (fake index, fake backend)."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from filings_kb.answer import (
    AnswerUnit, PassageSnapshot, QuestionSpec, assign_cites, check_answer, load_failure,
    load_questions, load_run, number_spans, parse_answer, render_answer, retrieve, run_question,
    number_support, select_claims, unsupported_numbers, unverified_quotes, validate_citations,
)
from filings_kb.store import Store
from tests.conftest import (
    AMD_AMEND, AMD_ORIG, NVDA_24, NVDA_25, FakeBackend, FakeIndex, answer_json, fixed_now,
)

MODELS = {"embedding": "emb", "extraction": "ext", "synthesis": "syn", "sentiment": "sen"}


def spec(**kw) -> QuestionSpec:
    base = dict(slug="nvda-risk", title="T", question="How did risk change?",
                category="risk_shift", filings=(("NVDA", 2024), ("NVDA", 2025)),
                items=("Item_1A",))
    base.update(kw)
    return QuestionSpec(**base)


def cid(acc: str, n: int) -> str:
    return f"{acc}:Item_1A:{n:04d}"


@pytest.fixture
def scores() -> dict[str, float]:
    return {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.5, cid(NVDA_24, 0): 0.7,
            cid(NVDA_24, 3): -0.1}


@pytest.fixture
def index(store: Store, scores) -> FakeIndex:
    return FakeIndex(store, scores)


# --- retrieval ---------------------------------------------------------------


def test_retrieve_queries_each_filing_and_item_with_where_filter(store, index):
    retrieve(spec(items=("Item_1A", "Item_3")), index, store, top_k=20)
    wheres = [c[2] for c in index.calls]
    # Item_3 has no section in the store, so only Item_1A is queried, once per filing year.
    assert {(w["ticker"], w["fiscal_year"], w["item"]) for w in wheres} == {
        ("NVDA", 2024, "Item_1A"), ("NVDA", 2025, "Item_1A")}
    assert all(c[1] == 20 for c in index.calls)
    assert {w["accession"] for w in wheres} == {NVDA_24, NVDA_25}


def test_retrieve_sorts_by_score_and_caps_total(store, index):
    got = retrieve(spec(), index, store, top_k=20, max_passages=3, max_rejected=0)
    assert [round(c.score, 2) for c in got] == [0.9, 0.7, 0.5]
    assert got[0].chunk.chunk_id == cid(NVDA_25, 0)


def test_retrieve_raises_when_nothing_found(store):
    with pytest.raises(ValueError, match="no passages"):
        retrieve(spec(), FakeIndex(store, {}), store, top_k=20)


def test_retrieve_amendment_queries_original_and_amendment(store):
    idx = FakeIndex(store, {})
    with pytest.raises(ValueError):
        retrieve(spec(slug="amd", filings=(("AMD", 2025),), items=("Item_7_MDA",),
                      amendment=True), idx, store, top_k=20)
    assert {c[2]["accession"] for c in idx.calls} == {AMD_ORIG, AMD_AMEND}


def test_passages_include_rejected_below_cutoff_flagged_included_false(store, index):
    cands = retrieve(spec(), index, store, top_k=20)
    passages = assign_cites(cands, cutoff=0.6)
    by_id = {p.chunk_id: p for p in passages}
    assert by_id[cid(NVDA_25, 1)].included is False
    assert by_id[cid(NVDA_24, 3)].included is False
    assert by_id[cid(NVDA_25, 0)].included is True
    assert len(passages) == 4  # rejected candidates are kept, not dropped


def test_cite_numbers_assigned_only_to_included_passages_in_score_order(store, index):
    passages = assign_cites(retrieve(spec(), index, store, top_k=20), cutoff=0.6)
    assert [(p.chunk_id, p.cite) for p in passages] == [
        (cid(NVDA_25, 0), 1), (cid(NVDA_24, 0), 2), (cid(NVDA_25, 1), None),
        (cid(NVDA_24, 3), None)]


def test_cutoff_is_inclusive(store, index):
    passages = assign_cites(retrieve(spec(), index, store, top_k=20), cutoff=0.5)
    assert [p.cite for p in passages] == [1, 2, 3, None]


# --- retrieval quota (I4) -----------------------------------------------------


def test_retrieve_quota_keeps_every_filing_item_group(store):
    # NVDA_25 holds the three best chunks; a global cap of 2 would starve NVDA_24.
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.85, cid(NVDA_25, 2): 0.8,
              cid(NVDA_24, 0): 0.3}
    got = retrieve(spec(), FakeIndex(store, scores), store, top_k=20, max_passages=2)
    assert [c.chunk.chunk_id for c in got] == [cid(NVDA_25, 0), cid(NVDA_24, 0)]


def test_retrieve_quota_leftovers_filled_by_score(store):
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.85, cid(NVDA_25, 2): 0.8,
              cid(NVDA_24, 0): 0.3, cid(NVDA_24, 1): 0.2}
    got = retrieve(spec(), FakeIndex(store, scores), store, top_k=20, max_passages=3)
    # quota 3 // 2 = 1 each, then the single leftover slot goes to the best remaining.
    assert [c.chunk.chunk_id for c in got] == [cid(NVDA_25, 0), cid(NVDA_25, 1),
                                               cid(NVDA_24, 0)]


def test_retrieve_quota_counts_only_groups_that_returned_passages(store):
    # Item_3 has no section, so it must not take a share of the quota.
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.85, cid(NVDA_24, 0): 0.3}
    got = retrieve(spec(items=("Item_1A", "Item_3")), FakeIndex(store, scores), store,
                   top_k=20, max_passages=2)
    assert {c.chunk.accession for c in got} == {NVDA_24, NVDA_25}


# --- display cutoff (Task 9) ----------------------------------------------------


def test_passages_below_cutoff_not_sent_to_model(store, tmp_path):
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.4, cid(NVDA_24, 0): 0.7,
              cid(NVDA_24, 3): 0.2}
    low = [c.text for c in store.chunks_by_id([cid(NVDA_25, 1), cid(NVDA_24, 3)]).values()]
    backend = FakeBackend(GOOD)
    got = run(store, FakeIndex(store, scores), backend, tmp_path, cutoff=0.5)
    user = backend.calls[0][1]
    assert "[1] NVDA FY2025" in user and "[2] NVDA FY2024" in user and "[3]" not in user
    assert all(text not in user for text in low)
    rejected = [p for p in got.passages if not p.included]
    assert sorted(p.score for p in rejected) == [0.2, 0.4]
    assert all(p.cite is None for p in rejected)


def test_retrieve_applies_cutoff_before_quota_and_cap(store):
    # NVDA_24 has nothing above the cutoff, so it takes no share of the quota.
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.8, cid(NVDA_25, 2): 0.7,
              cid(NVDA_24, 0): 0.3, cid(NVDA_24, 1): 0.2}
    got = retrieve(spec(), FakeIndex(store, scores), store, top_k=20, max_passages=2,
                   cutoff=0.5, max_rejected=10)
    kept = [c for c in got if c.score >= 0.5]
    assert [c.chunk.chunk_id for c in kept] == [cid(NVDA_25, 0), cid(NVDA_25, 1)]
    assert [round(c.score, 1) for c in got if c.score < 0.5] == [0.3, 0.2]


def test_retrieve_raises_when_nothing_reaches_the_cutoff(store, index):
    with pytest.raises(ValueError, match="cutoff"):
        retrieve(spec(), index, store, top_k=20, cutoff=0.99)


def test_retrieve_caps_rejected_by_score(store):
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.4, cid(NVDA_25, 2): 0.3,
              cid(NVDA_24, 0): 0.2, cid(NVDA_24, 1): 0.1}
    got = retrieve(spec(), FakeIndex(store, scores), store, top_k=20, cutoff=0.5,
                   max_rejected=2)
    assert [round(c.score, 1) for c in got] == [0.9, 0.4, 0.3]


def test_cutoff_change_reuses_the_answer_when_the_model_input_is_unchanged(store, tmp_path):
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_24, 0): 0.7, cid(NVDA_25, 1): 0.1}
    first = run(store, FakeIndex(store, scores), FakeBackend(GOOD), tmp_path, cutoff=0.0)
    assert all(p.included for p in first.passages)
    # cutoff 0.5 drops the 0.1 passage: different model input, so the backend is called.
    backend = FakeBackend(GOOD)
    run(store, FakeIndex(store, scores), backend, tmp_path, cutoff=0.5)
    assert len(backend.calls) == 1
    # cutoff 0.6 leaves the model input as it was: no call, but the snapshot is refreshed.
    backend = FakeBackend()
    again = run(store, FakeIndex(store, scores), backend, tmp_path, cutoff=0.6)
    assert backend.calls == [] and again.display_cutoff == 0.6
    assert load_run("nvda-risk", tmp_path / "answers") == again


# --- structured answer parsing (I3) -----------------------------------------------

ONE = answer_json([("Export controls cut revenue by $5.5 billion.", [1])])


def test_parse_answer_accepts_plain_json():
    paragraphs, errors = parse_answer(ONE)
    assert errors == []
    assert paragraphs == ((AnswerUnit("Export controls cut revenue by $5.5 billion.", (1,)),),)


@pytest.mark.parametrize("wrap", [
    "```json\n{}\n```", "```\n{}\n```", "Here is the answer:\n{}\nDone.",
])
def test_parse_answer_strips_code_fence_and_surrounding_prose(wrap):
    paragraphs, errors = parse_answer(wrap.format(ONE))
    assert errors == [] and paragraphs is not None and paragraphs[0][0].cites == (1,)


@pytest.mark.parametrize("raw, fragment", [
    ("not json at all", "JSON"),
    ('{"paragraphs": []}', "paragraphs"),
    ('{"paragraphs": [[]]}', "paragraphs"),
    ('{"paragraphs": [[{"text": "A.", "cites": []}]]}', "cites"),
    ('{"paragraphs": [[{"text": "A.", "cites": ["1"]}]]}', "cites"),
    ('{"paragraphs": [[{"text": "", "cites": [1]}]]}', "text"),
    ('{"paragraphs": [[{"text": "A.", "cites": [1], "extra": 1}]]}', "extra"),
    ('{"answer": "A."}', "paragraphs"),
])
def test_parse_answer_rejects_malformed_output(raw, fragment):
    paragraphs, errors = parse_answer(raw)
    assert paragraphs is None and errors
    assert any(fragment in e for e in errors), errors


def test_parse_answer_rejects_inline_markers_in_unit_text():
    _, errors = parse_answer(answer_json([("Revenue rose [2].", [1])]))
    assert any("[n]" in e for e in errors)


# --- citation validation -----------------------------------------------------


def test_validate_citations_flags_unknown_and_rejected_numbers():
    paragraphs, _ = parse_answer(answer_json(
        [("Risk rose.", [1]), ("Other thing.", [3]), ("Ghost.", [9, 2])]))
    errors = validate_citations(paragraphs, {1, 2})
    assert any("[3]" in e for e in errors) and any("[9]" in e for e in errors)
    assert not any("[1]" in e or "[2]" in e for e in errors)
    assert any("paragraph 1 unit 2" in e for e in errors)


def test_validate_citations_accepts_known_numbers():
    paragraphs, _ = parse_answer(answer_json([("A.", [1, 2])], [("B.", [2])]))
    assert validate_citations(paragraphs, {1, 2}) == []


# --- number check (C2/I1) --------------------------------------------------------


def test_unsupported_numbers_flags_numbers_missing_from_passages():
    got = unsupported_numbers("Revenue was $5.5 billion and 99% of units.",
                              ["we lost $5.5 billion", "other"])
    assert got == ["99"]


def test_unsupported_numbers_ignores_years():
    assert unsupported_numbers("In fiscal 2025 it grew.", ["text"]) == []


@pytest.mark.parametrize("text", [
    "AMD filed a 10-K/A.", "The 10-K and 10-Q differ.", "Item 7 and Item 1A changed.",
    "In FY2025 and FY24 it grew.", "Q4 shipments of the MI308 and H20 rose.",
    "Intel 18A ramps.",
])
def test_unsupported_numbers_ignores_form_item_period_and_product_labels(text):
    assert unsupported_numbers(text, ["no figures here"]) == []


def test_unsupported_numbers_does_not_match_digits_inside_identifiers():
    # "20" inside the passage's "H20" is not the quantity 20.
    assert unsupported_numbers("Sales fell 20%.", ["H20 sales were restricted"]) == ["20"]


@pytest.mark.parametrize("figure, passage", [
    ("$16.6B", "Data Center revenue was $16.6 billion."),
    ("16.6bn", "revenue of 16.6 billion"),
    ("$1.2M", "a charge of $1.2 million"),
    ("$16.6B", "revenue of $16.6B"),
])
def test_letter_suffixed_figure_keeps_fraction(figure, passage):
    # The fraction survives the suffix: "16.6B" is the quantity 16.6, not 16.
    assert unsupported_numbers(f"Revenue was {figure} in the quarter.", [passage]) == []
    assert unsupported_numbers("Revenue was 16 in the quarter.", [passage]) == ["16"]


@pytest.mark.parametrize("text, passage, missing", [
    ("Revenue was $16.9B.", "Revenue was $16.6B.", ["16.9"]),
    ("Revenue was $16.9bn.", "Revenue was $16.6 billion.", ["16.9"]),
    ("A charge of $1.3M.", "A charge of $1.2M.", ["1.3"]),
])
def test_wrong_decimal_with_suffix_is_flagged(text, passage, missing):
    assert unsupported_numbers(text, [passage]) == missing


def test_suffix_letters_do_not_reopen_the_identifier_hole():
    # "18A", "MI308" and "H20" stay labels; a suffix must be followed by a non-letter.
    assert unsupported_numbers("Intel 18A and the MI308 and H20 and 5Bx.", ["none"]) == []


def test_number_spans_offsets_index_the_original_text():
    text = "In the 10-K/A for Item 7, Q4 sales were $16.6B, up 12%."
    cleaned, spans = number_spans(text)
    assert len(cleaned) == len(text)
    assert [(sp.value, text[sp.start:sp.end]) for sp in spans] == [("16.6", "16.6B"),
                                                                   ("12", "12")]


def test_unsupported_numbers_normalises_thousands_separators():
    assert unsupported_numbers("It had 1,234 staff.", ["1234 staff"]) == []


# --- scale and kind (T10 r1 ruling 4) ----------------------------------------------


@pytest.mark.parametrize("text, passage", [
    ("Sales rose 12%.", "Sales were 12 million units."),          # percent against a magnitude
    ("Revenue was $12 billion.", "Revenue was $12 million."),     # wrong magnitude
    ("Revenue was $16.6B.", "Revenue was $16.6 million."),
    ("Margin rose 40 bps.", "Margin rose 40%."),                  # bps is not percent
    ("Margin rose 3 percentage points.", "Margin rose 3%."),      # points are not percent
])
def test_explicit_scale_or_kind_mismatch_fails(text, passage):
    assert unsupported_numbers(text, [passage]) == [number_spans(text)[1][0].value]


@pytest.mark.parametrize("text, passage", [
    ("Buybacks were $1.2 billion.", "Buybacks were $1,200 million."),
    ("Buybacks were $1,200 million.", "Buybacks were $1.2 billion."),
    ("Buybacks were $1.2B.", "Buybacks were $1,200 million."),
    ("Assets were $2.5 trillion.", "Assets were $2,500 billion."),
    ("Sales rose 12 percent.", "Sales rose 12%."),
])
def test_same_quantity_at_another_scale_or_spelling_is_supported(text, passage):
    assert unsupported_numbers(text, [passage]) == []


def test_same_scale_still_compares_digits_exactly():
    # Only a change of magnitude is converted; 5.5 and 5.50 at the same scale differ.
    assert unsupported_numbers("A $5.5 billion charge.", ["a $5.50 billion charge"]) == ["5.5"]


@pytest.mark.parametrize("text, passage", [
    ("Margin rose 40bps.", "Margin rose 40 basis points."),
    ("Margin rose 40 bps.", "Margin rose 40bps."),
    ("Margin rose 25bp.", "Margin rose 25 bps."),
    ("Margin rose 25 bp.", "Margin rose 25 basis points."),
    ("Margin rose 25 bps.", "Margin rose 25 bp."),
])
def test_bps_is_parsed_and_matches_basis_points(text, passage):
    [span] = number_spans(text)[1]
    assert span.kind == "bps"
    assert unsupported_numbers(text, [passage]) == []
    assert unsupported_numbers(text, ["no figures here"]) == [span.value]


def test_bare_passage_figure_is_scale_unverified_not_a_failure():
    # A table under an "(in millions)" header holds the bare 1,702.
    text, table = "Payments were $1,702 million.", "Total 1,702 5,108"
    assert unsupported_numbers(text, [table]) == []
    [(span, status)] = number_support(text, [table])
    assert (span.value, status) == ("1702", "scale unverified")


def test_number_support_prefers_a_verified_match():
    [(_, status)] = number_support("Revenue was $4.2 billion.",
                                   ["Total 4.2", "Revenue was $4.2 billion."])
    assert status == "verified"
    [(_, status)] = number_support("It had 2,000 aircraft.", ["2,000 aircraft"])
    assert status == "verified"  # bare against bare: nothing to verify, same kind
    [(_, status)] = number_support("Sales rose 12%.", ["12 million"])
    assert status == "unsupported"


# --- quote check (I5/M1) ---------------------------------------------------------


def test_unverified_quotes_accepts_verbatim_quote():
    assert unverified_quotes('It said "customer concentration remains high".',
                             ["Customer concentration remains high."]) == []  # case ignored
    assert unverified_quotes('It said "concentration remains high".',
                             ["Customer concentration remains high."]) == []


def test_unverified_quotes_flags_paraphrase():
    got = unverified_quotes('It warned of "very high concentration".',
                            ["Customer concentration remains high."])
    assert got == ['"very high concentration"']


def test_unverified_quotes_normalises_curly_quotes_apostrophes_and_whitespace():
    passage = "We can’t   guarantee\nsupply."
    assert unverified_quotes("It said “We can't guarantee supply”.", [passage]) == []


def test_unverified_quotes_treats_brackets_and_ellipses_as_fragment_boundaries():
    passage = "the Company relies on a few customers and on third-party foundries"
    assert unverified_quotes('"[The Company] relies on a few customers ... foundries"',
                             [passage]) == []
    assert unverified_quotes('"[It] relies on many customers… foundries"',
                             [passage]) == ['"[It] relies on many customers… foundries"']


def test_unverified_quotes_needs_every_fragment_in_one_passage():
    got = unverified_quotes('"relies on a few customers ... depends on third-party foundries"',
                            ["relies on a few customers", "depends on third-party foundries"])
    assert len(got) == 1


def test_unverified_quotes_flags_unbalanced_quote_mark():
    assert unverified_quotes('It said "never closed.', ["never closed"]) != []


# --- per-unit check over cited passages (C2, I1, I5) ----------------------------------


def _snap(cite: int, text: str) -> PassageSnapshot:
    return PassageSnapshot(
        chunk_id=f"c{cite}", score=1.0 - cite / 100, included=True, cite=cite, ticker="NVDA",
        company="NVIDIA CORP", form="10-K", fiscal_year=2025, filed="2025-02-26",
        item="Item_1A", accession=NVDA_25, sec_url="https://www.sec.gov/Archives/edgar/data/1/",
        context={"before": "", "match": text, "after": ""})


def test_check_answer_number_must_be_in_that_units_cited_passages():
    # C2 regression: $34.6 billion is in passage [3], but the unit cites only [1] and [2].
    passages = [_snap(1, "Client revenue grew."), _snap(2, "Units rose 15%."),
                _snap(3, "Net revenue was $34.6 billion.")]
    paragraphs, _ = parse_answer(answer_json(
        [("Net revenue was $34.6 billion.", [1, 2]), ("Units rose 15%.", [2])]))
    errors = check_answer(paragraphs, passages)
    assert len(errors) == 1
    assert "paragraph 1 unit 1" in errors[0] and "34.6" in errors[0]


def test_check_answer_quote_must_be_in_that_units_cited_passages():
    passages = [_snap(1, "Demand is volatile."), _snap(2, "Supply is constrained.")]
    paragraphs, _ = parse_answer(answer_json([('It says "Supply is constrained".', [1])]))
    errors = check_answer(paragraphs, passages)
    assert len(errors) == 1 and "quote" in errors[0]


def test_check_answer_ignores_rejected_passages_even_when_numbered():
    rejected = PassageSnapshot(**{**_snap(1, "Revenue $9 billion.").__dict__, "included": False,
                                  "cite": None})
    paragraphs, _ = parse_answer(answer_json([("Revenue was $9 billion.", [1])]))
    errors = check_answer(paragraphs, [rejected])
    assert any("[1]" in e for e in errors)


def test_check_answer_passes_a_faithful_answer():
    passages = [_snap(1, "Net revenue was $34.6 billion."), _snap(2, "Units rose 15%.")]
    paragraphs, _ = parse_answer(answer_json(
        [("Net revenue was $34.6 billion.", [1])], [('Units "rose 15%".', [2, 1])]))
    assert check_answer(paragraphs, passages) == []


# --- deterministic render and renumbering (I3, M3) ------------------------------------


def test_render_answer_renumbers_cites_by_first_appearance():
    paragraphs, _ = parse_answer(answer_json(
        [("Export risk rose.", [7, 3]), ("Customers are few.", [3])],
        [("Supply is tight", [12, 7])]))
    md, mapping = render_answer(paragraphs)
    assert mapping == {7: 1, 3: 2, 12: 3}
    assert md == ("Export risk rose [1][2]. Customers are few [2].\n\n"
                  "Supply is tight [1][3]")


def test_render_answer_is_deterministic():
    paragraphs, _ = parse_answer(answer_json([("A.", [2]), ("B?", [1])]))
    assert render_answer(paragraphs) == render_answer(paragraphs)
    assert render_answer(paragraphs)[0] == "A [1]. B [2]?"


# --- synthesis, retry, cache -------------------------------------------------

GOOD = answer_json([("Export controls cut revenue by $5.5 billion.", [1]),
                    ("Customers are concentrated.", [2])])


def run(store, index, backend, tmp_path, **kw):
    args = dict(q=spec(), index=index, store=store, backend=backend, model="syn",
                top_k=20, cutoff=0.6, cache_dir=tmp_path / "answers", models=MODELS,
                now=fixed_now, rejected_dir=tmp_path / "rejected")
    args.update(kw)
    return run_question(**args)


def test_run_question_returns_run_and_caches(store, index, tmp_path):
    backend = FakeBackend(GOOD)
    got = run(store, index, backend, tmp_path)
    assert got is not None
    assert got.answer_md == "Export controls cut revenue by $5.5 billion [1]. Customers are concentrated [2]."
    assert got.run_timestamp == "2026-11-23T12:00:00Z"
    assert got.models == MODELS and got.top_k == 20 and got.display_cutoff == 0.6
    assert got.max_passages == 40
    user = backend.calls[0][1]
    assert "QUESTION" in user and "[1] NVDA FY2025 10-K, Item 1A" in user
    assert "$5.5 billion" in user
    assert "rejected" not in user.lower()  # rejected passages are never shown to the model
    assert (tmp_path / "answers" / "nvda-risk.json").exists()
    assert load_run("nvda-risk", tmp_path / "answers") == got


def test_run_snapshots_every_store_field_the_export_needs(store, index, tmp_path):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    _claim(store, "Risk", "Customer concentration remains high", NVDA_25, "Item_1A", {}, nvda)
    got = run(store, index, FakeBackend(GOOD), tmp_path, cutoff=0.4)
    first = got.passages[0]
    assert (first.ticker, first.company, first.form, first.fiscal_year, first.filed,
            first.item, first.accession) == ("NVDA", "NVIDIA CORP", "10-K", 2025,
                                             "2025-02-26", "Item_1A", NVDA_25)
    assert first.sec_url.startswith("https://www.sec.gov/Archives/edgar/data/1045810/")
    assert first.context["match"].startswith("Export controls")
    assert "Customer concentration" in first.context["after"]
    assert [v.row.name for v in got.claims] == ["Customer concentration remains high"]
    assert got.claims[0].cites == (3,)
    cached = json.loads((tmp_path / "answers" / "nvda-risk.json").read_text())
    assert cached["passages"][0]["company"] == "NVIDIA CORP"
    assert cached["claims"][0]["row"]["name"] == "Customer concentration remains high"


def test_rerun_with_same_inputs_does_not_call_the_backend(store, index, tmp_path):
    run(store, index, FakeBackend(GOOD), tmp_path)
    backend = FakeBackend()  # would IndexError if called
    again = run(store, index, backend, tmp_path, now=lambda: fixed_now().replace(year=2030))
    assert backend.calls == [] and again.run_timestamp == "2026-11-23T12:00:00Z"


def test_rerun_after_prompt_or_model_change_calls_again(store, index, tmp_path):
    run(store, index, FakeBackend(GOOD), tmp_path)
    other_prompt = tmp_path / "p.md"
    other_prompt.write_text("different prompt")
    b = FakeBackend(GOOD)
    run(store, index, b, tmp_path, prompt_path=other_prompt)
    assert len(b.calls) == 1
    b2 = FakeBackend(GOOD)
    run(store, index, b2, tmp_path, model="other-model")
    assert len(b2.calls) == 1


def test_cache_key_records_model_prompt_and_input_hashes(store, index, tmp_path):
    backend = FakeBackend(GOOD)
    run(store, index, backend, tmp_path)
    cached = json.loads((tmp_path / "answers" / "nvda-risk.json").read_text())
    assert cached["models"]["synthesis"] == "syn"
    assert cached["input_sha256"] == hashlib.sha256(backend.calls[0][1].encode()).hexdigest()
    assert cached["prompt_sha256"] == hashlib.sha256(backend.calls[0][0].encode()).hexdigest()


BAD = answer_json([("Bad cite.", [7])])


def test_answer_with_bad_citation_retried_once_then_none(store, index, tmp_path):
    backend = FakeBackend(BAD, answer_json([("Still bad.", [8])]))
    assert run(store, index, backend, tmp_path) is None
    assert len(backend.calls) == 2
    assert "[7]" in backend.calls[1][1]  # the retry tells the model what was wrong
    assert not (tmp_path / "answers" / "nvda-risk.json").exists()


def test_retry_shows_previous_json_and_the_exact_failing_units(store, index, tmp_path):
    first = answer_json([("Customers are concentrated.", [2]), ("Revenue fell 77%.", [1])])
    backend = FakeBackend(first, GOOD)
    assert run(store, index, backend, tmp_path) is not None
    retry = backend.calls[1][1]
    assert first in retry
    assert "paragraph 1 unit 2" in retry and "Revenue fell 77%." in retry and "77" in retry
    assert "paragraph 1 unit 1" not in retry  # the passing unit is not listed as failing


def test_number_not_in_cited_passage_fails_validation_then_retry_then_none(store, index, tmp_path):
    # C2: 77% is in no passage; $5.5 billion is in [1] and [2] but this unit cites only [3].
    backend = FakeBackend(answer_json([("Revenue fell 77%.", [1])]),
                          answer_json([("Revenue fell $5.5 billion.", [3])]))
    assert run(store, index, backend, tmp_path, cutoff=0.4) is None
    assert len(backend.calls) == 2


def test_unparseable_answer_is_rejected_and_retried(store, index, tmp_path):
    backend = FakeBackend("Export controls cut revenue [1].", GOOD)
    got = run(store, index, backend, tmp_path)
    assert got is not None and len(backend.calls) == 2
    assert "JSON" in backend.calls[1][1]


def test_rejected_raw_answers_and_failure_record_are_persisted(store, index, tmp_path):
    backend = FakeBackend(BAD, "not json")
    assert run(store, index, backend, tmp_path) is None
    one = json.loads((tmp_path / "rejected" / "nvda-risk.attempt1.json").read_text())
    two = json.loads((tmp_path / "rejected" / "nvda-risk.attempt2.json").read_text())
    assert one["raw"] == BAD and two["raw"] == "not json"
    assert one["errors"] and two["errors"] and one["attempt"] == 1
    failure = load_failure("nvda-risk", tmp_path / "answers")
    assert failure is not None and failure.slug == "nvda-risk"
    assert "validation" in failure.reason
    assert [Path(p).name for p in failure.rejected] == ["nvda-risk.attempt1.json",
                                                        "nvda-risk.attempt2.json"]


def test_failed_run_removes_a_stale_cached_answer(store, index, tmp_path):
    run(store, index, FakeBackend(GOOD), tmp_path)
    other_prompt = tmp_path / "p.md"
    other_prompt.write_text("different prompt")
    assert run(store, index, FakeBackend(BAD, BAD), tmp_path, prompt_path=other_prompt) is None
    assert not (tmp_path / "answers" / "nvda-risk.json").exists()


def test_answer_fixed_on_retry_is_accepted_and_clears_failure_record(store, index, tmp_path):
    run(store, index, FakeBackend(BAD, BAD), tmp_path)
    assert load_failure("nvda-risk", tmp_path / "answers") is not None
    got = run(store, index, FakeBackend(BAD, GOOD), tmp_path)
    assert got is not None and got.answer_md.startswith("Export controls")
    assert load_failure("nvda-risk", tmp_path / "answers") is None


# --- claims ------------------------------------------------------------------


def _claim(store, ctype, name, acc, item, attrs, subject_id):
    cid_ = store.put_entity(ctype, name, attrs, {})
    store.add_relationship(cid_, subject_id, "subject_of", acc, item, {})
    return cid_


def test_select_claims_uses_claims_of_included_sections_cited_first(store, index):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    _claim(store, "Risk", "Unrelated risk", NVDA_25, "Item_1A", {}, nvda)
    _claim(store, "Risk", "Customer concentration remains high", NVDA_25, "Item_1A", {}, nvda)
    _claim(store, "Risk", "Other filing", "0000000000-00-000000", "Item_1A", {}, nvda)
    passages = assign_cites(retrieve(spec(), index, store, top_k=20), cutoff=0.4)
    views = select_claims(spec(), passages, store)
    assert [v.row.name for v in views][:2] == ["Customer concentration remains high",
                                               "Unrelated risk"]
    assert views[0].cite == 3  # the chunk that holds that sentence
    assert views[1].cite is None
    assert all(v.row.accession != "0000000000-00-000000" for v in views)


def test_select_claims_interleaves_sections_so_one_filing_cannot_fill_the_cap(store, index):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    for i in range(3):  # the top-scoring filing alone has more claims than the cap
        _claim(store, "Risk", f"Top filing claim {i}", NVDA_25, "Item_1A", {}, nvda)
    _claim(store, "Risk", "Other filing claim", NVDA_24, "Item_1A", {}, nvda)
    passages = assign_cites(retrieve(spec(), index, store, top_k=20), cutoff=0.4)
    views = select_claims(spec(), passages, store, max_claims=2)
    assert {v.row.accession for v in views} == {NVDA_25, NVDA_24}


def test_claim_cite_among_picks_lowest_cited_overlapping_passage(store, index):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    # "customers" sentence sits in chunks 1 and 2 of NVDA_25 via two overlapping spans.
    _claim(store, "Risk", "Two customers accounted for 24% of revenue", NVDA_25, "Item_1A",
           {"note": "We rely on a few customers. Customer concentration remains high"}, nvda)
    scores = {cid(NVDA_25, 0): 0.9, cid(NVDA_25, 1): 0.8, cid(NVDA_25, 2): 0.7}
    passages = assign_cites(retrieve(spec(), FakeIndex(store, scores), store, top_k=20),
                            cutoff=0.0)
    [view] = select_claims(spec(), passages, store)
    assert view.cites == (2, 3) and view.cite == 2
    assert view.cite_among({3}) == 3 and view.cite_among({1}) is None


def test_claim_cite_null_when_no_overlap(store, index):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    _claim(store, "Risk", "Quantum computing risk", NVDA_25, "Item_1A",
           {"description": "totally different words appear here in this description"}, nvda)
    passages = assign_cites(retrieve(spec(), index, store, top_k=20), cutoff=0.0)
    [view] = select_claims(spec(), passages, store)
    assert view.cite is None


def test_claim_cite_set_on_six_word_description_overlap(store, index):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    _claim(store, "Risk", "Foundry risk", NVDA_25, "Item_1A",
           {"description": "We depend on third-party foundries. Extra words"}, nvda)
    # The 6-word span "we depend on third party foundries" is in chunk 3 of NVDA_25.
    scores = {cid(NVDA_25, 3): 0.8, cid(NVDA_25, 0): 0.9}
    passages = assign_cites(retrieve(spec(), FakeIndex(store, scores), store, top_k=20),
                            cutoff=0.0)
    [view] = select_claims(spec(), passages, store)
    assert view.cite == 2


def test_select_claims_for_amendment_pairs_superseded_with_amended(store):
    from datetime import UTC, datetime
    amd = store.put_entity("Company", "ADVANCED MICRO DEVICES INC", {}, {})
    old = _claim(store, "GuidanceUpdate", "Old guidance", AMD_ORIG, "Item_7_MDA", {}, amd)
    _claim(store, "GuidanceUpdate", "New guidance", AMD_AMEND, "Item_7_MDA", {}, amd)
    store.close_out(old, datetime(2026, 2, 4, tzinfo=UTC))
    q = spec(slug="amd", filings=(("AMD", 2025),), items=("Item_7_MDA",), amendment=True)
    views = select_claims(q, [], store)
    assert [(v.row.name, v.superseded) for v in views] == [
        ("Old guidance", True), ("New guidance", False)]


# --- config ------------------------------------------------------------------


def test_load_questions_parses_spec_and_globals(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump({
        "top_k": 20, "max_passages": 40, "display_cutoff": 0.0,
        "questions": [{"slug": "a-b", "title": "T", "question": "Q?", "category": "risk_shift",
                       "filings": [{"ticker": "NVDA", "fiscal_year": 2025}],
                       "items": ["Item_1A"], "featured": False}]}))
    cfg = load_questions(path)
    assert (cfg.top_k, cfg.max_passages, cfg.display_cutoff) == (20, 40, 0.0)
    [q] = cfg.questions
    assert q.filings == (("NVDA", 2025),) and q.items == ("Item_1A",)
    assert q.tickers == ("NVDA",) and q.amendment is False


def test_load_questions_rejects_unknown_category(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump({
        "top_k": 20, "max_passages": 40, "display_cutoff": 0.0,
        "questions": [{"slug": "a", "title": "T", "question": "Q?", "category": "nope",
                       "filings": [{"ticker": "NVDA", "fiscal_year": 2025}],
                       "items": ["Item_1A"], "featured": False}]}))
    with pytest.raises(ValueError, match="category"):
        load_questions(path)


def test_repo_questions_yaml_loads_with_13_questions_in_four_categories():
    cfg = load_questions(Path(__file__).resolve().parent.parent / "config" / "questions.yaml")
    assert len(cfg.questions) == 13
    assert len({q.slug for q in cfg.questions}) == 13
    by_cat = {}
    for q in cfg.questions:
        by_cat[q.category] = by_cat.get(q.category, 0) + 1
    assert by_cat == {"risk_shift": 4, "contradiction": 3, "capital_structure": 3,
                      "cross_company": 3}
    assert sum(q.amendment for q in cfg.questions) == 1


# --- round 2: case-insensitive quotes, single quotes, identity guard, years ---------


def test_quote_check_is_case_insensitive():
    # NVDA regression: a sentence-initial capital lowercased inside the quote.
    passages = [_snap(1, "The loss of any of our large customers could harm us.")]
    paragraphs, _ = parse_answer(answer_json(
        [('NVIDIA warns that "the loss of any of our large customers" could harm it.', [1])]))
    assert check_answer(paragraphs, passages) == []


def test_single_quoted_span_checked():
    passages = [_snap(1, "Supply of advanced packaging remains constrained.")]
    bad, _ = parse_answer(answer_json(
        [("It calls supply 'severely constrained for several years' now.", [1])]))
    good, _ = parse_answer(answer_json(
        [("It says ‘supply of advanced packaging remains constrained’ today.", [1])]))
    errors = check_answer(bad, passages)
    assert len(errors) == 1 and "quote" in errors[0] and "severely" in errors[0]
    assert check_answer(good, passages) == []
    # Apostrophes and short single-quoted labels (three words or fewer) are not quotes.
    short, _ = parse_answer(answer_json(
        [("The company's 'AI factory' plan and its customers' 'very long' waits.", [1])]))
    assert check_answer(short, passages) == []


@pytest.mark.parametrize("unit", [
    "All other elements of the segment discussion remained the same.",
    "The broader figures in the amended filing match those in the original filing.",
    "Risk factor wording was unchanged.",
    "The two sections are identical.",
    "The filing makes no mention of China.",
    "Guidance did not change.",
    "Margins are the same as last year.",
    "The table is not present in the amendment.",
])
def test_identity_phrase_fails_unless_in_cited_passage(unit):
    passages = [_snap(1, "Client segment revenue was reported for the year.")]
    paragraphs, _ = parse_answer(answer_json([(unit, [1])]))
    errors = check_answer(paragraphs, passages)
    assert errors
    assert all(e.startswith("paragraph 1 unit 1: ") and "identity" in e for e in errors)


def test_identity_phrase_passes_when_cited_passage_contains_it():
    passages = [_snap(1, "Other information in the filing remains unchanged."),
                _snap(2, "All other terms remain the same as previously reported.")]
    paragraphs, _ = parse_answer(answer_json(
        [("The amendment says other information was Unchanged.", [1]),
         ("It states that all other terms remained the same.", [2])]))
    assert check_answer(paragraphs, passages) == []
    # The phrase must be in a passage *this unit* cites, not just any passage.
    elsewhere, _ = parse_answer(answer_json([("Other information was unchanged.", [2])]))
    assert len(check_answer(elsewhere, passages)) == 1


@pytest.mark.parametrize("text, missing", [
    ("Boeing delivered 2,000 aircraft.", ["2000"]),
    ("Costs were $1,999 million.", ["1999"]),
    ("Costs were $2025.", ["2025"]),
    ("Margin was 2019% of plan.", ["2019"]),
    ("Revenue was 2024 million.", ["2024"]),
])
def test_comma_number_not_treated_as_year(text, missing):
    assert unsupported_numbers(text, ["Revenue grew."]) == missing


def test_bare_year_still_exempt():
    assert unsupported_numbers("In 2024 and in 1999, demand rose.", ["Demand rose."]) == []
    assert unsupported_numbers("Boeing delivered 2,000 aircraft.",
                               ["It delivered 2,000 aircraft."]) == []


def test_marker_placed_before_closing_quote():
    paragraphs, _ = parse_answer(answer_json(
        [('It warns that "demand may slow."', [1]), ("Supply is tight (per Item 1A).", [2]),
         ("Its plan is 'to grow the foundry business'", [1])]))
    md, _ = render_answer(paragraphs)
    assert md == ('It warns that "demand may slow [1]." Supply is tight (per Item 1A [2]). '
                  "Its plan is 'to grow the foundry business [1]'")


def test_parse_answer_ignores_trailing_text():
    paragraphs, errors = parse_answer(ONE + "\nNote: I cited {1} only.")
    assert errors == [] and paragraphs is not None and paragraphs[0][0].cites == (1,)


class _Raising:
    """A backend whose call fails the way a flaky CLI does."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, model: str):
        self.calls += 1
        raise RuntimeError("claude CLI exited 1")


def test_cached_answer_failing_new_checks_is_recalled(store, index, tmp_path):
    run(store, index, FakeBackend(GOOD), tmp_path)
    path = tmp_path / "answers" / "nvda-risk.json"
    data = json.loads(path.read_text())
    data["paragraphs"][0][1]["text"] = "All other customers are unchanged."
    path.write_text(json.dumps(data))
    backend = FakeBackend(GOOD)
    got = run(store, index, backend, tmp_path)
    assert len(backend.calls) == 1 and got is not None
    assert "unchanged" not in got.answer_md


def test_committed_questions_are_audited_and_featured_ones_passed() -> None:
    """Every committed question has an audit verdict; only audit-passed ones may be featured."""
    import yaml

    from filings_kb.answer import QUESTIONS_PATH
    raw = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    assert raw
    for q in raw:
        assert q["audit"] in ("pass", "fail"), q["slug"]
        assert q.get("audit_note"), q["slug"]
        assert q["audit"] == "pass" or q["featured"] is False, q["slug"]
    assert sum(q["featured"] for q in raw) == 8


# --- final review C1: the human audit verdict ----------------------------------------


def _questions_yaml(tmp_path, **extra) -> Path:
    path = tmp_path / "q.yaml"
    q = {"slug": "a-b", "title": "T", "question": "Q?", "category": "risk_shift",
         "filings": [{"ticker": "NVDA", "fiscal_year": 2025}], "items": ["Item_1A"],
         "featured": False, **extra}
    path.write_text(yaml.safe_dump({"top_k": 20, "max_passages": 40, "display_cutoff": 0.0,
                                    "questions": [q]}))
    return path


def test_load_questions_records_audit_failures_with_their_note(tmp_path):
    cfg = load_questions(_questions_yaml(tmp_path, audit="fail", audit_note="weak sentence"))
    assert cfg.audit_failures == {"a-b": "weak sentence"}
    assert load_questions(_questions_yaml(tmp_path, audit="pass")).audit_failures == {}
    assert load_questions(_questions_yaml(tmp_path)).audit_failures == {}


@pytest.mark.parametrize("extra, match", [
    ({"audit": "maybe"}, "audit"),
    ({"audit": "fail"}, "audit_note"),
    ({"audit": "fail", "audit_note": "x", "featured": True}, "featured"),
])
def test_load_questions_rejects_bad_audit_entries(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        load_questions(_questions_yaml(tmp_path, **extra))


def test_repo_config_fails_unh_on_audit():
    cfg = load_questions(Path(__file__).resolve().parent.parent / "config" / "questions.yaml")
    assert set(cfg.audit_failures) == {"unh-cyber-regulatory-risk-fy23-fy25"}


# --- final review I3, I4 and minor 4: stricter per-unit checks -------------------------


def test_check_answer_rejects_a_unit_with_no_cites():
    passages = [_snap(1, "Demand is volatile.")]
    paragraphs = ((AnswerUnit("Demand is volatile.", (1,)), AnswerUnit("Supply is fine.", ())),)
    errors = check_answer(paragraphs, passages)
    assert errors == ["paragraph 1 unit 2: cites no passage"]


@pytest.mark.parametrize("text, token", [
    ("Capex rose 10x.", "10x"), ("A $5tn market.", "5tn"), ("Margin was 12pct.", "12pct"),
    ("Built on 3nm.", "3nm"), ("Growth in the 2020s.", "2020s"),
])
def test_unknown_lowercase_suffixed_number_is_an_error(text, token):
    from filings_kb.answer import unknown_suffixed_numbers
    assert unknown_suffixed_numbers(text) == [token]
    errors = check_answer(((AnswerUnit(text, (1,)),),), [_snap(1, text)])
    assert any(f"number {token} has an unknown suffix" in e for e in errors), errors


@pytest.mark.parametrize("text", [
    "The H20, B200 and GB200 need licences.", "Intel 14A and 18A nodes.", "The 777X slipped.",
    "Revenue was $16.6bn.", "Spreads widened 40bps.", "A $5B charge and a $3m fee.",
    "It filed a 10-K in Q4 for FY2025.",
])
def test_known_suffixes_and_uppercase_product_codes_are_not_suffix_errors(text):
    from filings_kb.answer import unknown_suffixed_numbers
    assert unknown_suffixed_numbers(text) == []


PASSAGE_SC = "Export controls could disrupt our supply chain and distribution channels."


def test_bracket_edit_with_long_fragment_passes():
    # "disrupt" (1 word) is not evidence and is ignored; the 6-word piece verifies.
    quote = 'It said they "disrupt[ed] our supply chain and distribution channels".'
    assert check_answer(((AnswerUnit(quote, (1,)),),), [_snap(1, PASSAGE_SC)]) == []


def test_ellipsis_short_lead_fragment_ignored():
    passage = ("Governments have already encouraged buyers. Overseas governments may request "
               "that our customers purchase from our competitors.")
    quote = ('"already encouraged...overseas governments to request that our customers '
             'purchase from our competitors".')
    # "already encouraged" is ignored; the long piece is not verbatim ("to request") and fails.
    errors = check_answer(((AnswerUnit(quote, (1,)),),), [_snap(1, passage)])
    assert len(errors) == 1 and "not verbatim" in errors[0]
    ok = '"already encouraged...overseas governments may request that our customers".'
    assert check_answer(((AnswerUnit(ok, (1,)),),), [_snap(1, passage)]) == []


def test_quote_with_only_short_fragments_fails():
    quote = 'It said "export controls [...] could disrupt [...] our supply".'
    errors = check_answer(((AnswerUnit(quote, (1,)),),), [_snap(1, PASSAGE_SC)])
    assert len(errors) == 1 and "no piece of 4 or more words" in errors[0], errors


def test_long_fragment_not_in_passage_fails():
    quote = 'It said "disrupt[ed] our factories and retail distribution channels".'
    errors = check_answer(((AnswerUnit(quote, (1,)),),), [_snap(1, PASSAGE_SC)])
    assert len(errors) == 1 and "not verbatim" in errors[0], errors


def test_edited_quote_whose_fragments_all_have_four_words_passes():
    passage = "Export controls could disrupt our supply chain and distribution channels."
    quote = 'It said "export controls could disrupt [...] our supply chain and distribution".'
    assert check_answer(((AnswerUnit(quote, (1,)),),), [_snap(1, passage)]) == []


def test_unedited_short_quote_is_still_checked_verbatim_not_by_length():
    passage = "Supply may be insufficient."
    ok = check_answer(((AnswerUnit('Supply "may be insufficient".', (1,)),),), [_snap(1, passage)])
    bad = check_answer(((AnswerUnit('Supply "is insufficient".', (1,)),),), [_snap(1, passage)])
    assert ok == [] and len(bad) == 1 and "not verbatim" in bad[0]


# --- final review I5: a claim is cited from its extraction chunk content ----------------


def test_claim_cites_passages_holding_its_extraction_content(store, index):
    from filings_kb.answer import build_user_message, snapshot_passages
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    _claim(store, "Risk", "Customer concentration remains high", NVDA_25, "Item_1A", {}, nvda)
    _claim(store, "Risk", "NVDA Supply Risk", NVDA_25, "Item_1A", {}, nvda)  # name matches nothing
    passages = assign_cites(retrieve(spec(), index, store, top_k=20), cutoff=0.4)
    before = select_claims(spec(), passages, store)
    store.put_claim_evidence(NVDA_25, "Item_1A", [
        ("Risk", "NVDA Supply Risk",
         "Export controls could reduce our data center revenue by $5.5 billion.")])
    after = select_claims(spec(), passages, store)
    assert [v.row.name for v in after] == [v.row.name for v in before]  # order unchanged
    snaps = snapshot_passages(passages, store)
    assert build_user_message(spec(), snaps, after) == build_user_message(spec(), snaps, before)
    supply = next(v for v in after if v.row.name == "NVDA Supply Risk")
    assert next(v for v in before if v.row.name == "NVDA Supply Risk").cites == ()
    chunk_texts = {p.cite: s.text for p, s in zip(passages, snaps) if p.included}
    assert supply.cites == (1,) and chunk_texts[1].startswith("Export controls could reduce")


def test_claim_evidence_shorter_than_four_words_cites_nothing(store, index):
    nvda = store.put_entity("Company", "NVIDIA CORP", {}, {})
    _claim(store, "Risk", "NVDA Supply Risk", NVDA_25, "Item_1A", {}, nvda)
    store.put_claim_evidence(NVDA_25, "Item_1A", [("Risk", "NVDA Supply Risk", "We depend.")])
    passages = assign_cites(retrieve(spec(), index, store, top_k=20), cutoff=0.0)
    [view] = select_claims(spec(), passages, store)
    assert view.cites == ()
