"""Section detection tests. Fixtures under tests/fixtures/sections are
hand-made SEC-style 10-K excerpts (no real filing text)."""

import hashlib
from datetime import date
from pathlib import Path

import pytest

from filings_kb.config import load_sample
from filings_kb.sections import (
    ITEM_TOKENS,
    Section,
    detect_sections,
    is_toc_stub,
    _ITEM_MAX_CHARS,
    _clean_body,
    _strip_html_if_markup,
    section_text_sha256,
)
from filings_kb.types import FilingRef

FIXTURES = Path(__file__).parent / "fixtures" / "sections"
ROOT = Path(__file__).parent.parent


def _ref(name: str) -> FilingRef:
    return FilingRef(
        ticker="TEST", cik="0000000001", company="Example Corp",
        accession="0000000001-25-000001", form="10-K", filed=date(2025, 2, 1),
        fiscal_year=2024, fiscal_year_end=date(2024, 12, 31),
        primary_doc="x.htm", html_path=FIXTURES / name,
    )


def _by_item(sections: list[Section]) -> dict[str, Section]:
    return {s.item: s for s in sections}


def test_item_tokens_match_sample_config_allowlist() -> None:
    assert list(ITEM_TOKENS) == load_sample(ROOT / "config" / "sample.yaml").extract_items


def test_detects_item_1a_and_7_from_fixture() -> None:
    sections = detect_sections(_ref("tenk_with_1c.html"))
    by_item = _by_item(sections)
    assert by_item["Item_1A"].text.startswith("Item 1A. Risk Factors")
    assert "Section 1A paragraph 0" in by_item["Item_1A"].text
    assert by_item["Item_7_MDA"].text.startswith("Item 7.")
    assert "Section 7 paragraph 3" in by_item["Item_7_MDA"].text
    assert "Section 7A" not in by_item["Item_7_MDA"].text
    first = by_item["Item_1A"]
    assert (first.accession, first.ticker, first.fiscal_year) == (
        "0000000001-25-000001", "TEST", 2024)
    assert first.title.startswith("Item 1A")


def test_item_1c_detected_when_present() -> None:
    assert "Item_1C" in _by_item(detect_sections(_ref("tenk_plain_1c.html")))


def test_missing_item_omitted_not_empty() -> None:
    sections = detect_sections(_ref("tenk_no_1c.html"))
    assert "Item_1C" not in _by_item(sections)
    assert all(s.text.strip() for s in sections)


def test_toc_stub_rejected() -> None:
    stub = "Item 1A. Risk Factors ........ 12\nItem 1B. Unresolved Staff Comments 20"
    assert is_toc_stub(stub) is True
    sections = detect_sections(_ref("tenk_toc_only.html"))
    assert "Item_1A" not in _by_item(sections)
    assert "Item_7_MDA" in _by_item(sections)


def test_real_section_not_stub() -> None:
    text = " ".join(["The company faces competition in many markets."] * 50)
    assert len(text.split()) > 300
    assert is_toc_stub(text) is False


def test_short_text_with_page_number_or_leaders_is_stub() -> None:
    assert is_toc_stub("Item 1B. Unresolved Staff Comments 20") is True
    assert is_toc_stub("Item 1B. Unresolved Staff Comments ........ see index") is True


def test_short_real_item_none_is_kept() -> None:
    assert is_toc_stub("Item 1B. Unresolved Staff Comments\nNone.") is False
    assert is_toc_stub("Item 6. [Reserved]") is False


def test_short_incorporated_by_reference_item_is_kept() -> None:
    text = ("Item 10. Directors, Executive Officers and Corporate Governance\n"
            "The information required by this item is incorporated by reference "
            "to our 2025 Proxy Statement.")
    assert is_toc_stub(text) is False


def test_page_number_lines_over_30_percent_is_stub() -> None:
    long_lines = ["Risk factor sentence number %d about markets and supply chains 12" % i
                  for i in range(40)]
    assert len(" ".join(long_lines).split()) >= 200
    assert is_toc_stub("\n".join(long_lines)) is True


def test_page_number_lines_under_30_percent_is_not_stub() -> None:
    lines = ["Prose sentence number %d about markets and supply chains today." % i
             for i in range(40)]
    lines[:3] = [ln + " 12" for ln in lines[:3]]
    assert is_toc_stub("\n".join(lines)) is False


def test_regex_fallback_when_edgartools_finds_nothing() -> None:
    sections = detect_sections(_ref("tenk_plain.html"))
    by_item = _by_item(sections)
    assert "Section 1A paragraph 0" in by_item["Item_1A"].text
    assert "Section 7 paragraph 0" in by_item["Item_7_MDA"].text
    assert by_item["Item_1A"].text.count("Risk Factors ........ 12") == 0


def test_section_text_sha256_is_plain_utf8_sha256() -> None:
    text = "Item 1A. Risk Factors — café"
    assert section_text_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_regex_last_heading_is_dropped_not_run_to_end_of_document() -> None:
    # tenk_plain has no heading after Item 8, so Item 8 has no known end.
    by_item = _by_item(detect_sections(_ref("tenk_plain.html")))
    assert "Item_8" not in by_item


def test_regex_ignores_cross_reference_lines_as_headings() -> None:
    by_item = _by_item(detect_sections(_ref("tenk_crossref.html")))
    assert "Section 1A paragraph 3" in by_item["Item_1A"].text
    assert "Item 7 of this report discusses" in by_item["Item_1A"].text


def test_markup_body_is_stripped_to_text() -> None:
    raw = "<div>" + "<span>word</span> " * 300 + "</div>"
    assert _strip_html_if_markup(raw).split() == ["word"] * 300
    assert _strip_html_if_markup("plain text with a < b") == "plain text with a < b"


def test_over_cap_body_is_dropped() -> None:
    assert _clean_body("x" * (_ITEM_MAX_CHARS + 1), "Item_1", "acc") is None
    assert _clean_body("  ok  ", "Item_1", "acc") == "ok"


def test_edgartools_construction_failure_falls_back_to_regex(monkeypatch) -> None:
    def boom(_filing):
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr("filings_kb.sections.TenK", boom)
    assert "Item_7_MDA" in _by_item(detect_sections(_ref("tenk_plain.html")))


def test_missing_html_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        detect_sections(_ref("does_not_exist.html"))
