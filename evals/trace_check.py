"""Check the exported traces and write the hand-audit list.

Run: ``python -m evals.trace_check exports/traces [--network]``

For every trace this re-checks, from the exported JSON alone (no store, no model):

* every ``[n]`` marker in ``answer_md`` is an included passage with that cite number,
  and has a ``sec.gov`` URL;
* every unit (sentence plus its marker group) cites something;
* every number in a unit is supported by a passage that unit cites. This reuses the
  answer-time matcher (``filings_kb.answer.number_support``), so the two cannot drift:
  thousands separators are removed, the scale (``%``, percentage points, ``bps``,
  thousand to trillion, ``B``/``M``/``K``) is part of the comparison, so ``12%``
  against ``12 million`` fails and ``$1.2 billion`` matches ``$1,200 million``; nothing
  is rounded; bare years and fiscal-year labels are not quantities. A figure that only
  appears bare in the passage (a table cell under an "in millions" header) passes as
  *scale unverified* and goes on the audit list;
* the rest of the answer-time check (``filings_kb.answer.check_answer``), with the same
  functions: no number with an unknown lowercase suffix (``10x``), every quoted span
  verbatim in one cited passage (pieces under 4 words between edits are ignored, and an
  edited quote needs at least one longer piece), and every
  identity or absence phrase ("unchanged", "no mention") used by a cited passage.

With ``--network`` it also requests every distinct ``sec_url`` (at most 2 per second,
with the ``EDGAR_IDENTITY`` User-Agent from ``.env``) and requires HTTP 200.

``exports/audit_list.md`` is the checklist for the human audit. Verified numbers and
fiscal-year labels that match a cited filing are counted, not listed. Listed in full,
in this order: scale-unverified numbers with their source passage and link, fiscal-year
labels no cited filing has, and the units that make an absence, identity or universal
claim about the filings ("no specific incident", "the same two", "all three"), which no
number check can see. These are for judgement, not failures.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filings_kb.answer import (
    MIN_QUOTE_FRAGMENT_WORDS, SCALE_UNVERIFIED, UNSUPPORTED, VERIFIED, NumberSpan, best_support,
    number_spans, quotes_without_evidence, unknown_suffixed_numbers, unsupported_identity_claims,
    unsupported_numbers, unverified_quotes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EDGAR_IDENTITY = "filings-kb (github.com/ahalvdank/filings-kb)"
REQUESTS_PER_SECOND = 2
SEC_PREFIX = "https://www.sec.gov/"

# Claims no number check can see: absence, identity or universal statements about what
# the filings say. Tightened in T10 r1: bare "no", "not", "new", "all", "same" and
# "only" mostly matched proper nouns ("New Energies"), connectives ("at the same time")
# and the filing's own hedging ("may not be successful"), so each now needs the context
# that makes it a claim about the filings.
_COUNT = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"
TRIGGERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("no + specific/mention/reference/disclose", re.compile(
        r"\b(?:no|not)\s+(?:\w+\s+){0,2}?(?:specific\w*|mention\w*|referenc\w*|disclos\w*)\b",
        re.IGNORECASE)),
    ("absent", re.compile(r"\babsent\b", re.IGNORECASE)),
    ("entirely new", re.compile(r"\bentirely\s+new\b", re.IGNORECASE)),
    ("same + count", re.compile(rf"\bsame\s+{_COUNT}\b", re.IGNORECASE)),
    ("remained the same", re.compile(r"\b(?:remain(?:s|ed)?|stay(?:s|ed)?)\s+the\s+same\b",
                                     re.IGNORECASE)),
    ("all/every + count", re.compile(rf"\b(?:all|every)\s+{_COUNT}\b", re.IGNORECASE)),
    ("unchanged", re.compile(r"\bunchanged\b", re.IGNORECASE)),
    ("identical", re.compile(r"\bidentical\b", re.IGNORECASE)),
    ("never", re.compile(r"\bnever\b", re.IGNORECASE)),
    ("first time", re.compile(r"\bfirst\s+time\b", re.IGNORECASE)),
)
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
_MARKER_GROUP = re.compile(r"(?P<text>.+?)\s*(?P<markers>(?:\[\d+\])+)(?P<tail>[.!?\"”’')\]]*)"
                           r"(?:\s+|$)", re.DOTALL)
_MARKER = re.compile(r"\[(\d+)\]")
_FY_LABEL = re.compile(r"\bFY\s?(\d{4}|\d{2})\b|\bfiscal(?:\s+year)?\s+(\d{4})\b", re.IGNORECASE)
_NUMBER_UNIT = re.compile(
    r"\s?(?:%|percentage\s+points?\b|percent\b|billion\b|million\b|trillion\b|thousand\b"
    r"|bps\b|basis\s+points?\b|months?\b|years?\b)", re.IGNORECASE)
SNIPPET_RADIUS = 50


@dataclass(frozen=True)
class Unit:
    """One sentence of the answer and the passage numbers it cites (``p1u2`` = paragraph 1, unit 2)."""

    paragraph: int
    index: int
    text: str  # the sentence as the model wrote it: markers removed, end punctuation kept
    cites: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"p{self.paragraph}u{self.index}"


@dataclass(frozen=True)
class NumberItem:
    """A number in a unit as written (``$4.2 billion``) and where it was found."""

    unit: Unit
    display: str
    value: str
    found: tuple[tuple[int, str], ...]  # (cite, snippet) per supporting passage, best first
    status: str  # VERIFIED, SCALE_UNVERIFIED or UNSUPPORTED (see ``answer.support_of``)


@dataclass(frozen=True)
class FiscalLabel:
    """A fiscal-year label in a unit and the fiscal years of the filings that unit cites."""

    unit: Unit
    label: str
    year: int
    cited_years: set[tuple[str, int]]

    @property
    def matches_cited(self) -> bool:
        return any(fy == self.year for _, fy in self.cited_years)


@dataclass(frozen=True)
class Flag:
    """A unit with absence, identity or universal wording, for human judgement."""

    unit: Unit
    triggers: list[str]
    marked: str  # the unit text with each trigger in **bold**

    @property
    def cites(self) -> tuple[int, ...]:
        return self.unit.cites


@dataclass
class TraceResult:
    """Everything ``check_trace`` found in one trace."""

    slug: str
    title: str
    question: str
    problems: list[str] = field(default_factory=list)
    numbers: list[NumberItem] = field(default_factory=list)
    fiscal_years: list[FiscalLabel] = field(default_factory=list)
    claims_to_judge: list[Flag] = field(default_factory=list)


def split_units(answer_md: str) -> list[Unit]:
    """Split a rendered answer into units: text up to a marker group, plus its end punctuation.

    A trailing run of text with no marker group is a unit with no cites.
    """
    units: list[Unit] = []
    for p, paragraph in enumerate((x for x in answer_md.split("\n\n") if x.strip()), start=1):
        pos, j = 0, 0
        for m in _MARKER_GROUP.finditer(paragraph):
            j += 1
            cites = tuple(sorted({int(n) for n in _MARKER.findall(m.group("markers"))}))
            units.append(Unit(p, j, (m.group("text") + m.group("tail")).strip(), cites))
            pos = m.end()
        rest = paragraph[pos:].strip()
        if rest:
            units.append(Unit(p, j + 1, rest, ()))
    return units


def _display_number(text: str, start: int, end: int) -> str:
    """The number as written, with a leading ``$`` and a trailing unit word or ``%``."""
    if text[:start].rstrip().endswith("$"):
        start = text[:start].rstrip().rfind("$")
    unit = _NUMBER_UNIT.match(text, end)
    return text[start:unit.end() if unit else end].strip()


def _snippet(text: str, start: int, end: int) -> str:
    lo, hi = max(0, start - SNIPPET_RADIUS), min(len(text), end + SNIPPET_RADIUS)
    body = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return f"{'…' if lo else ''}{body}{'…' if hi < len(text) else ''}"


def _find_number(passage_text: str, span: NumberSpan) -> tuple[int, int, str] | None:
    """``(start, end, support)`` of the passage number that best supports ``span``, or None.

    Passage years are candidates like any other number, as in ``number_support``.
    """
    best: tuple[int, int, str] | None = None
    for sp in number_spans(passage_text)[1]:
        support = best_support(span, [sp])
        if support == VERIFIED:
            return sp.start, sp.end, support
        if support == SCALE_UNVERIFIED and best is None:
            best = (sp.start, sp.end, support)
    return best


def _mask_quotes(text: str) -> str:
    """``text`` with double-quoted spans blanked (a verbatim quote is the filing's words)."""
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), text)


def _flag(unit: Unit) -> Flag | None:
    masked = _mask_quotes(unit.text)
    hits = sorted((m.start(), m.end(), name) for name, rx in TRIGGERS for m in rx.finditer(masked))
    if not hits:
        return None
    spans: list[list[int]] = []  # overlapping hits are bolded as one span
    for start, end, _ in hits:
        if spans and start < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    marked, last = [], 0
    for start, end in spans:
        marked += [unit.text[last:start], "**", unit.text[start:end], "**"]
        last = end
    marked.append(unit.text[last:])
    first_seen: list[str] = []
    for _, _, name in hits:
        if name not in first_seen:
            first_seen.append(name)
    return Flag(unit, first_seen, "".join(marked))


def check_trace(trace: dict[str, Any]) -> TraceResult:
    """Check one exported trace and collect what the audit list needs."""
    result = TraceResult(trace["slug"], trace.get("title", trace["slug"]),
                         trace.get("question", ""))
    by_cite = {p["cite"]: p for p in trace["passages"]
               if p.get("included") and p.get("cite") is not None}
    for p in by_cite.values():
        if not str(p.get("sec_url", "")).startswith(SEC_PREFIX):
            result.problems.append(f"passage [{p['cite']}] has no sec.gov URL")
    for unit in split_units(trace["answer_md"]):
        if not unit.cites:
            result.problems.append(f"{unit.label}: no citation")
        for n in unit.cites:
            if n not in by_cite:
                result.problems.append(f"{unit.label}: [{n}] is not an included passage")
        cited = [by_cite[n] for n in unit.cites if n in by_cite]
        texts = [p["context"]["match"] for p in cited]
        cites = "".join(f"[{n}]" for n in unit.cites)
        for missing in unsupported_numbers(unit.text, texts):
            result.problems.append(
                f"{unit.label}: number {missing} is not in its cited passages {cites}")
        # The answer-time checks, reused so the two cannot drift (check_answer).
        for token in unknown_suffixed_numbers(unit.text):
            result.problems.append(f"{unit.label}: number {token} has an unknown suffix")
        for quote in unverified_quotes(unit.text, texts):
            result.problems.append(
                f"{unit.label}: quote {quote} is not verbatim in its cited passages {cites}")
        for quote in quotes_without_evidence(unit.text):
            result.problems.append(f"{unit.label}: quote {quote} has no piece of "
                                   f"{MIN_QUOTE_FRAGMENT_WORDS} or more words between its edits")
        for phrase in unsupported_identity_claims(unit.text, texts):
            result.problems.append(f"{unit.label}: identity/absence claim \"{phrase}\" is not "
                                   f"stated in its cited passages {cites}")
        _, spans = number_spans(unit.text)
        for sp in spans:
            if sp.is_year:
                continue
            hits = []
            for p in cited:
                at = _find_number(p["context"]["match"], sp)
                if at:
                    hits.append((at[2] != VERIFIED, p["cite"],
                                 _snippet(p["context"]["match"], at[0], at[1]), at[2]))
            hits.sort(key=lambda h: h[0])  # verified passages first, then cite order
            status = (VERIFIED if hits and hits[0][3] == VERIFIED
                      else SCALE_UNVERIFIED if hits else UNSUPPORTED)
            result.numbers.append(NumberItem(
                unit, _display_number(unit.text, sp.start, sp.end), sp.value,
                tuple((c, snip) for _, c, snip, _ in hits), status))
        years = {(p["ticker"], p["fiscal_year"]) for p in cited}
        for m in _FY_LABEL.finditer(unit.text):
            digits = m.group(1) or m.group(2)
            year = int(digits) + (2000 if len(digits) == 2 else 0)
            result.fiscal_years.append(FiscalLabel(unit, m.group(0), year, years))
        flag = _flag(unit)
        if flag:
            result.claims_to_judge.append(flag)
    return result


# --- network ---------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkResult:
    """Distinct URLs requested and those that did not return 200 (status None: no response)."""

    checked: int
    failures: list[tuple[str, int | None]]


def _http_status(url: str, agent: str) -> int | None:
    """Status of a GET (the body is not read); None when there was no HTTP response."""
    request = urllib.request.Request(url, headers={"User-Agent": agent})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def check_urls(
    urls: list[str], agent: str, *, fetch: Callable[[str, str], int | None] | None = None,
    sleep: Callable[[float], None] | None = None, clock: Callable[[], float] | None = None,
) -> NetworkResult:
    """Request each distinct URL once, at most ``REQUESTS_PER_SECOND`` per second."""
    fetch = fetch or _http_status
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    gap = 1 / REQUESTS_PER_SECOND
    failures: list[tuple[str, int | None]] = []
    distinct = list(dict.fromkeys(urls))
    last: float | None = None
    for url in distinct:
        if last is not None:
            wait = gap - (clock() - last)
            if wait > 0:
                sleep(wait)
        last = clock()
        status = fetch(url, agent)
        if status != 200:
            failures.append((url, status))
    return NetworkResult(len(distinct), failures)


# --- audit list ------------------------------------------------------------------


def _filing_label(p: dict[str, Any]) -> str:
    return f"{p['ticker']} FY{p['fiscal_year']} {p['form']}"


def _source(p: dict[str, Any]) -> str:
    return f"{_filing_label(p)} · {p['section']} · [sec.gov]({p['sec_url']})"


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines())


def _cites(unit: Unit) -> str:
    return "".join(f"[{n}]" for n in unit.cites)


def _heading(i: int, result: TraceResult) -> str:
    return f"### {i}. {result.title} (`{result.slug}`)"


def _numbers_to_check(numbered: list[tuple[int, TraceResult]],
                      passages: dict[str, dict[int, dict[str, Any]]]) -> list[str]:
    """Numbers that are not verified: not found first, then scale unverified, by trace."""
    items = [(i, r, n) for i, r in numbered for n in r.numbers if n.status != VERIFIED]
    lines = [f"## Numbers to check ({len(items)})", ""]
    if not items:
        return lines + ["None: every number is verified.", ""]
    lines += ["Each figure appears in a cited passage only without its scale (usually a "
              "table cell under an \"in millions\" header). Check the scale in the filing.", ""]
    for i, result in numbered:
        mine = sorted((n for _, r, n in items if r is result),
                      key=lambda n: n.status != UNSUPPORTED)
        if not mine:
            continue
        lines += [_heading(i, result), ""]
        last: Unit | None = None
        for item in mine:
            if item.unit != last:
                lines += [f"**{item.unit.label}** {item.unit.text} {_cites(item.unit)}", ""]
                last = item.unit
            if item.status == UNSUPPORTED:
                lines += [f"- [ ] **NOT FOUND** `{item.display}` — cited {_cites(item.unit)}", ""]
                continue
            cite, snippet = item.found[0]
            more = "".join(f"[{c}]" for c, _ in item.found[1:])
            lines += [f"- [ ] `{item.display}` — {item.status} — [{cite}] "
                      f"{_source(passages[result.slug][cite])}"
                      f"{f' (also in {more})' if more else ''}", f"  {_quote(snippet)}", ""]
    return lines


def _labels_to_check(numbered: list[tuple[int, TraceResult]]) -> list[str]:
    items = [(i, r, fy) for i, r in numbered for fy in r.fiscal_years if not fy.matches_cited]
    lines = [f"## Fiscal-year labels to check ({len(items)})", ""]
    if not items:
        return lines + ["None: every label matches a cited filing's fiscal year.", ""]
    for _, result, fy in items:
        cited = ", ".join(f"{t} FY{y}" for t, y in sorted(fy.cited_years)) or "nothing"
        lines.append(f"- [ ] `{fy.label}` in `{result.slug}` {fy.unit.label} — cites {cited} — "
                     "**no cited filing has this fiscal year**")
    return lines + [""]


def _claims_to_judge(numbered: list[tuple[int, TraceResult]],
                     passages: dict[str, dict[int, dict[str, Any]]]) -> list[str]:
    total = sum(len(r.claims_to_judge) for _, r in numbered)
    lines = [f"## Claims to judge ({total})", ""]
    for i, result in numbered:
        if not result.claims_to_judge:
            continue
        lines += [_heading(i, result), ""]
        for flag in result.claims_to_judge:
            cites = _cites(flag.unit)
            triggers = ", ".join(f"`{t}`" for t in flag.triggers)
            lines += [f"- [ ] **{flag.unit.label}** ({triggers}) {flag.marked} {cites}", "",
                      f"  <details><summary>cited passages {cites}</summary>", ""]
            for n in flag.cites:
                if n in passages[result.slug]:
                    p = passages[result.slug][n]
                    lines += [f"  **[{n}]** {_source(p)}", "",
                              *(f"  {x}" for x in _quote(p["context"]["match"]).splitlines()),
                              ""]
            lines += ["  </details>", ""]
    return lines


def _count(results: list[TraceResult], status: str) -> int:
    return sum(1 for r in results for n in r.numbers if n.status == status)


def write_audit_list(traces: list[dict[str, Any]], out: Path,
                     network: NetworkResult | None) -> list[TraceResult]:
    """Write the audit checklist for ``traces`` to ``out``; return each trace's result.

    Verified numbers and fiscal-year labels that match a cited filing are counted, not
    listed. Listed in full: numbers that are not verified (scale unverified, or not found
    when the automatic check failed), labels no cited filing has, and claims to judge.
    """
    results = [check_trace(t) for t in traces]
    numbered = list(enumerate(results, start=1))
    passages = {r.slug: {p["cite"]: p for p in t["passages"]
                         if p.get("included") and p.get("cite") is not None}
                for t, r in zip(traces, results, strict=True)}
    n_numbers = sum(len(r.numbers) for r in results)
    n_labels = sum(len(r.fiscal_years) for r in results)
    n_match = sum(1 for r in results for fy in r.fiscal_years if fy.matches_cited)
    n_judge = sum(len(r.claims_to_judge) for r in results)
    lines = [
        "# Audit list", "",
        f"{len(results)} traces. {n_numbers} numbers: {_count(results, VERIFIED)} verified, "
        f"{_count(results, SCALE_UNVERIFIED)} scale unverified, "
        f"{_count(results, UNSUPPORTED)} not found. {n_labels} fiscal-year labels: {n_match} "
        f"match a cited filing, {n_labels - n_match} to check. {n_judge} claims to judge.", "",
        "How to read it. Tick a box when you have checked it against the passage.",
        "- **Verified numbers** (same quantity and scale in a cited passage: `$1.2 billion` "
        "matches `$1,200 million`, `12%` does not match `12 million`) are counted, not listed.",
        "- **Numbers to check:** figures a cited passage holds only without a scale. They pass "
        "the automatic check; the scale is yours to confirm.",
        "- **Fiscal-year labels** that match the fiscal year of a cited filing are counted. "
        "NVDA's fiscal year ends in January, so `FY2025` is the year ended Jan 2025.",
        "- **Claims to judge:** sentences that say what a filing does not say, says again or "
        "says of all of something (`no specific`, `absent`, `the same two`, `all three`, ...), "
        "found with a word list. No number check can see these. They are not failures: read "
        "the cited passages and decide whether the sentence is supported.",
    ]
    if network is not None:
        lines.append(f"- **SEC links:** {network.checked - len(network.failures)} of "
                     f"{network.checked} distinct `sec_url` values returned 200.")
    lines.append("")
    failing = [(i, r) for i, r in numbered if r.problems]
    if failing:
        lines += ["## Automatic check failed", ""]
        for i, r in failing:
            lines += [_heading(i, r), "", *(f"- {p}" for p in r.problems), ""]
    lines += _numbers_to_check(numbered, passages)
    lines += _labels_to_check(numbered)
    lines += _claims_to_judge(numbered, passages)
    lines += ["## Traces", ""]
    for i, r in numbered:
        verified = sum(1 for n in r.numbers if n.status == VERIFIED)
        match = sum(1 for fy in r.fiscal_years if fy.matches_cited)
        lines.append(f"{i}. **{r.title}** (`{r.slug}`): {len(r.numbers)} numbers ({verified} "
                     f"verified), {len(r.fiscal_years)} fiscal-year labels ({match} match), "
                     f"{len(r.claims_to_judge)} claims to judge, {len(r.problems)} problems.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    """Check every trace in a directory and write the audit list.

    Returns:
        1 when a citation does not resolve, a number is not in its cited passages, or
        (with ``--network``) a ``sec_url`` does not return 200; else 0. Claims to judge
        never change the exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", type=Path, help="directory of exported trace JSON files")
    ap.add_argument("--network", action="store_true", help="also request every sec_url")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: exports/audit_list.md with --network, else the gitignored "
                         "data/audit_list.local.md (only a --network run may replace the committed list)")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = REPO_ROOT / ("exports/audit_list.md" if args.network
                                else "data/audit_list.local.md")

    files = sorted(args.traces.glob("*.json"))
    if not files:
        print(f"no traces in {args.traces}")
        return 1
    traces = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    network: NetworkResult | None = None
    if args.network:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")  # EDGAR_IDENTITY lives there; never printed
        agent = os.environ.get("EDGAR_IDENTITY", DEFAULT_EDGAR_IDENTITY)
        urls = [p["sec_url"] for t in traces for p in t["passages"]]
        network = check_urls(urls, agent, fetch=_http_status)

    results = write_audit_list(traces, args.out, network)
    bad = 0
    for r in results:
        bad += len(r.problems)
        print(f"{r.slug}: {len(r.numbers)} numbers, {len(r.fiscal_years)} fiscal-year labels, "
              f"{len(r.claims_to_judge)} claims to judge, {len(r.problems)} problems")
        for problem in r.problems:
            print(f"  PROBLEM {problem}")
    if network is not None:
        print(f"sec_url: {network.checked - len(network.failures)} of {network.checked} "
              "returned 200")
        for url, status in network.failures:
            print(f"  FAILED {status} {url}")
        bad += len(network.failures)
    print(f"wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
