"""Tests for evals/trace_check.py: citations, numbers, FY labels, claims to judge, network."""

import json
from pathlib import Path

import pytest

from evals import trace_check as tc
from filings_kb.answer import number_support


def _passage(cite: int | None, match: str, *, fy: int = 2025, ticker: str = "NVDA",
             included: bool = True, section: str = "Item 1A Risk Factors",
             url: str | None = None) -> dict:
    return {"accession": "0001045810-25-000023", "cite": cite, "company": "NVIDIA CORP",
            "context": {"before": "", "match": match, "after": ""}, "filed": "2025-02-26",
            "fiscal_year": fy, "form": "10-K", "included": included, "score": 0.5,
            "sec_url": url or f"https://www.sec.gov/Archives/edgar/data/1/{fy}/{cite}.htm",
            "section": section, "ticker": ticker}


def _trace(answer_md: str, passages: list[dict], slug: str = "q") -> dict:
    return {"slug": slug, "title": "A title", "question": "What changed?", "answer_md": answer_md,
            "passages": passages, "claims": [], "tickers": ["NVDA"]}


P1 = _passage(1, "Revenue was $4.2 billion, up 12%, and customers cannot easily switch.")
P2 = _passage(2, "Demand was strong. Export rules apply to all customers.", fy=2024)


def test_number_present_passes() -> None:
    trace = _trace("Revenue was $4.2 billion [1]. It grew 12% [1].", [P1])
    assert tc.check_trace(trace).problems == []


def test_number_not_in_cited_passage_flagged() -> None:
    trace = _trace("Revenue was $4.9 billion [1].", [P1])
    [problem] = tc.check_trace(trace).problems
    assert "p1u1" in problem and "4.9" in problem


def test_number_is_checked_against_the_passages_that_unit_cites() -> None:
    p2 = _passage(2, "Margins were 77% last year.")
    trace = _trace("Revenue was $4.2 billion [1]. Margins were 77% [1].", [P1, p2])
    [problem] = tc.check_trace(trace).problems
    assert "p1u2" in problem and "77" in problem


def test_wrong_decimal_with_magnitude_letter_is_flagged() -> None:
    trace = _trace("Revenue was $4.9B [1].", [_passage(1, "Revenue was $4.2 billion.")])
    assert len(tc.check_trace(trace).problems) == 1
    trace = _trace("Revenue was $4.2B [1].", [_passage(1, "Revenue was $4.2 billion.")])
    assert tc.check_trace(trace).problems == []


def test_citation_markers_are_not_read_as_numbers() -> None:
    trace = _trace("Customers could switch [1][2]. They did not [2].", [P1, P2])
    assert tc.check_trace(trace).problems == []


def test_fiscal_year_labels_not_flagged() -> None:
    trace = _trace("In FY2025 and FY24, and fiscal 2023, demand was strong [2].", [P2])
    result = tc.check_trace(trace)
    assert result.problems == []
    assert [f.label for f in result.fiscal_years] == ["FY2025", "FY24", "fiscal 2023"]


def test_fiscal_year_label_is_listed_with_the_cited_filings_years() -> None:
    result = tc.check_trace(_trace("In FY2025 demand was strong [2].", [P1, P2]))
    [fy] = result.fiscal_years
    assert fy.year == 2025 and fy.cited_years == {("NVDA", 2024)} and fy.matches_cited is False
    result = tc.check_trace(_trace("In FY2025 demand was strong [1].", [P1, P2]))
    assert result.fiscal_years[0].matches_cited is True


def test_citation_to_a_missing_or_rejected_passage_is_flagged() -> None:
    rejected = _passage(None, "Rejected text.", included=False)
    trace = _trace("Demand was strong [2]. Other text [7].", [P1, rejected])
    problems = tc.check_trace(trace).problems
    assert len(problems) == 2 and all("not an included passage" in p for p in problems)


def test_unit_without_a_citation_is_flagged() -> None:
    [problem] = tc.check_trace(_trace("Demand was strong.", [P1])).problems
    assert "p1u1" in problem and "no citation" in problem


def test_units_split_by_paragraph_and_marker_group() -> None:
    answer = 'He said "demand may slow [1]." Next unit [2].\n\nSecond paragraph [1][2].'
    units = tc.split_units(answer)
    assert [(u.paragraph, u.index, u.cites) for u in units] == [
        (1, 1, (1,)), (1, 2, (2,)), (2, 1, (1, 2))]
    assert units[0].text == 'He said "demand may slow."'


# The three units the T8 ruling pinned, verbatim from the exported traces (markers removed).
UNH_P1U1 = ('The FY2023 cybersecurity risk factor described threats in general terms, noting the '
            'company was regularly the target of attempted "cyber-attacks" and listing '
            'vulnerabilities including "human error, malicious social engineering," but '
            'referenced no specific incident.')
NVDA_P2U3 = ('The FY2026 disclosures also added entirely new concerns absent from FY2022, including '
             "China's government publicly questioning whether H20 products \"have built-in "
             'vulnerabilities, discouraging customers from purchasing our products," and risks '
             'from customers pivoting to open-source foundation models originating in China '
             'such as "DeepSeek, Qwen, or KIMMI."')
INTC_P2U1 = ('On partner co-investments, both the FY2024 and FY2025 filings disclose the same two '
             'Semiconductor Co-Investment Program arrangements—the 2022 joint investment with '
             'Brookfield in Arizona manufacturing and the 2024 joint investment with Apollo '
             'related to Fab 34 in Ireland—and both note the $755 million charge recognized in the '
             'fourth quarter of 2024 for penalties related to Ireland SCIP construction delays.')


@pytest.mark.parametrize("text, triggers", [
    (UNH_P1U1, ["no + specific/mention/reference/disclose"]),
    (NVDA_P2U3, ["entirely new", "absent"]),
    (INTC_P2U1, ["same + count"]),
    ("By the end of 2025, all three major rating agencies had revised the outlooks.",
     ["all/every + count"]),
    ("The filing does not mention a buyback.", ["no + specific/mention/reference/disclose"]),
    ("The FY2025 filing makes no reference to it, and the amount was not disclosed.",
     ["no + specific/mention/reference/disclose"]),
    ("The wording was unchanged, identical and never seen for the first time.",
     ["unchanged", "identical", "never", "first time"]),
    ("The program remained the same.", ["remained the same"]),
])
def test_claims_to_judge_flags_absence_identity_and_universal_claims(text, triggers) -> None:
    trace = _trace(f"{text} [1]", [P1])
    [flag] = tc.check_trace(trace).claims_to_judge
    assert flag.triggers == triggers and flag.cites == (1,)


@pytest.mark.parametrize("text", [
    # Noise the first word list flagged in the real traces (T10 review): proper nouns,
    # connectives, the filing's own hedging and ordinary uses of "new", "all", "only".
    "The company's New Energies organization is focused on hydrogen.",
    "At the same time, the 10-K makes clear that growing hydrocarbons remains a priority.",
    "Intel may not be successful in developing a competitive foundry business.",
    "The Board authorized a new $30 billion common share repurchase program.",
    "Boeing remained in full compliance with all debt covenants.",
    "Termination rights allow for potential repayment of some or all awards.",
    "Intel has no long-term contract with TSMC.",
    "The FY2022 10-K framed such harm only as something that could occur.",
    "NVIDIA does not assemble, test, or package its products itself.",
    "Customers cannot switch, and small renewable newer products ship.",
    'It said "there is no mention of that" in the filing.',  # a verbatim quote is the filing's
])
def test_claims_to_judge_skips_ordinary_wording_and_quoted_text(text) -> None:
    assert tc.check_trace(_trace(f"{text} [1]", [P1])).claims_to_judge == []


def test_number_scale_mismatch_is_a_problem() -> None:
    trace = _trace("Units rose 12% [1].", [_passage(1, "We sold 12 million units.")])
    [problem] = tc.check_trace(trace).problems
    assert "p1u1" in problem and "12" in problem


def test_passage_years_count_as_candidates_like_the_answer_time_check() -> None:
    # "2024 units" is a bare year-shaped figure in the passage; answer-time number_support
    # accepts it as support, so the re-check must too and not call the number unsupported.
    passage = _passage(1, "We shipped 2024 units.")
    trace = _trace("We shipped 2,024 units [1].", [passage])
    result = tc.check_trace(trace)
    assert result.problems == []
    [item] = result.numbers
    assert item.status == number_support("We shipped 2,024 units.", ["We shipped 2024 units."])[0][1]
    assert item.status == "verified"


def test_default_out_is_local_without_network_and_committed_with_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tc, "_http_status", lambda url, agent: 200)
    monkeypatch.setattr(tc.time, "sleep", lambda s: None)
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "q.json").write_text(json.dumps(_trace("Demand was strong [2].", [P1, P2])))
    assert tc.main([str(traces)]) == 0
    assert (tmp_path / "data" / "audit_list.local.md").is_file()
    assert not (tmp_path / "exports" / "audit_list.md").exists()
    assert tc.main([str(traces), "--network"]) == 0
    assert (tmp_path / "exports" / "audit_list.md").is_file()


def test_bare_passage_figure_is_scale_unverified_not_a_problem() -> None:
    table = _passage(1, "Total 1,702 5,108", section="Item 8 Financial Statements")
    result = tc.check_trace(_trace("Payments were $1,702 million [1].", [table]))
    assert result.problems == []
    [item] = result.numbers
    assert (item.display, item.status) == ("$1,702 million", "scale unverified")
    assert tc.check_trace(_trace("Revenue was $4.2 billion [1].", [P1])).numbers[0].status \
        == "verified"


@pytest.mark.parametrize("text, display", [
    ("The plan runs 18 months [1].", "18 months"),
    ("Margin rose 3 percentage points [1].", "3 percentage points"),
    ("Margin rose 40bps [1].", "40bps"),
])
def test_number_display_keeps_its_unit(text, display) -> None:
    passage = _passage(1, "It runs 18 months; margin rose 3 percentage points, or 40bps.")
    assert [n.display for n in tc.check_trace(_trace(text, [passage])).numbers] == [display]


TABLE = _passage(3, "Total 1,702 5,108", section="Item 8 Financial Statements")


def _audit(tmp_path: Path) -> str:
    answer = ('Revenue was $4.2 billion and grew 12% in FY2025 [1]. '
              'The factor referenced no specific incident [2]. '
              'Payments were $1,702 million in FY2023 [3].')
    trace = _trace(answer, [P1, P2, TABLE, _passage(None, "Rejected.", included=False)])
    out = tmp_path / "audit_list.md"
    tc.write_audit_list([trace], out, network=None)
    return out.read_text()


def test_audit_list_collapses_verified_numbers_and_matching_labels_into_counts(tmp_path) -> None:
    text = _audit(tmp_path)
    assert "`$4.2 billion` —" not in text and "`12%` —" not in text  # verified: counted only
    assert "- [ ] `FY2025`" not in text                          # matches a cited filing
    assert "3 numbers: 2 verified" in text and "1 scale unverified" in text
    assert "2 fiscal-year labels: 1 match" in text


def test_audit_list_lists_scale_unverified_numbers_first_with_source(tmp_path) -> None:
    text = _audit(tmp_path)
    first_check = text.index("- [ ] ")
    assert text[first_check:].startswith("- [ ] `$1,702 million`")
    assert "Item 8 Financial Statements" in text and "Total 1,702 5,108" in text
    assert "https://www.sec.gov/Archives/edgar/data/1/2025/3.htm" in text
    assert "- [ ] `FY2023`" in text and "no cited filing has this fiscal year" in text


def test_audit_list_shows_claims_to_judge_with_their_passages(tmp_path) -> None:
    text = _audit(tmp_path)
    _, _, judge = text.partition("## Claims to judge")
    assert "referenced **no specific** incident" in judge  # trigger shown in bold
    assert "<details>" in judge and "Export rules apply to all customers." in judge


# --- network ---------------------------------------------------------------------


def test_network_check_dedupes_urls_and_limits_to_two_requests_per_second() -> None:
    now = [0.0]
    sleeps: list[float] = []
    seen: list[tuple[str, str]] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    def fetch(url: str, agent: str) -> int:
        seen.append((url, agent))
        return 200

    result = tc.check_urls(["u1", "u2", "u1", "u3", "u4", "u5"], "agent", fetch=fetch,
                           sleep=sleep, clock=lambda: now[0])
    assert result.checked == 5 and result.failures == []
    assert [u for u, _ in seen] == ["u1", "u2", "u3", "u4", "u5"]
    assert all(a == "agent" for _, a in seen)
    assert now[0] >= 4 * 0.5 - 1e-9  # five requests need at least four 0.5 s gaps


def test_network_check_reports_non_200_and_errors() -> None:
    statuses = {"ok": 200, "gone": 404, "boom": None}
    result = tc.check_urls(["ok", "gone", "boom"], "agent", fetch=lambda u, a: statuses[u],
                           sleep=lambda s: None, clock=lambda: 0.0)
    assert result.failures == [("gone", 404), ("boom", None)]


def test_network_user_agent_comes_from_env_and_is_never_printed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    secret = "Jane Doe jane@example.com"
    monkeypatch.setenv("EDGAR_IDENTITY", secret)
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "q.json").write_text(json.dumps(_trace("Demand was strong [2].", [P1, P2])))
    agents: list[str] = []
    monkeypatch.setattr(tc, "_http_status", lambda url, agent: agents.append(agent) or 200)
    monkeypatch.setattr(tc.time, "sleep", lambda s: None)
    code = tc.main([str(traces), "--network", "--out", str(tmp_path / "audit.md")])
    out = capsys.readouterr()
    assert code == 0 and set(agents) == {secret}
    assert secret not in out.out + out.err and secret not in (tmp_path / "audit.md").read_text()


def test_main_exits_1_on_a_flagged_number_and_writes_the_list(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "q.json").write_text(json.dumps(_trace("Revenue was $9.9 billion [1].", [P1])))
    out = tmp_path / "audit.md"
    assert tc.main([str(traces), "--out", str(out)]) == 1
    assert "9.9" in out.read_text()


def test_main_exits_0_for_a_clean_trace(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "q.json").write_text(json.dumps(_trace("Revenue was $4.2 billion [1].", [P1])))
    assert tc.main([str(traces), "--out", str(tmp_path / "audit.md")]) == 0


# --- final review minor 3: the answer-time quote and identity checks are reused -------


def test_quote_not_verbatim_in_cited_passage_is_a_problem() -> None:
    trace = _trace('Customers "can switch at will" [1].', [P1])
    [problem] = tc.check_trace(trace).problems
    assert "p1u1" in problem and "can switch at will" in problem


def test_verbatim_quote_before_its_marker_passes() -> None:
    trace = _trace('Customers "cannot easily switch" [1].', [P1])
    assert tc.check_trace(trace).problems == []


def test_identity_claim_not_in_cited_passage_is_a_problem() -> None:
    trace = _trace("Demand was unchanged [1].", [P1])
    [problem] = tc.check_trace(trace).problems
    assert "p1u1" in problem and "unchanged" in problem


def test_unknown_suffixed_number_and_quote_without_a_long_piece_are_problems() -> None:
    p = _passage(1, "Export controls could disrupt our supply chain and distribution channels.")
    trace = _trace('Capacity rose 10x [1]. It "disrupt[ed] our supply chain and distribution '
                   'channels" [1]. It "could disrupt [...] our supply" [1].', [p])
    problems = tc.check_trace(trace).problems
    assert len(problems) == 2  # p1u2's short "disrupt" piece is ignored, not a problem
    assert "p1u1" in problems[0] and "10x" in problems[0]
    assert "p1u3" in problems[1] and "no piece of 4 or more words" in problems[1]
