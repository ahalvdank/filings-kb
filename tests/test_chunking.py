"""Chunking tests. Token counting is a plain word counter; no model is loaded."""

import re

import pytest

from filings_kb.chunking import Chunk, chunk_section, context_window
from filings_kb.sections import Section


def words(text: str) -> int:
    return len(text.split())


def _section(text: str, item: str = "Item_7_MDA") -> Section:
    return Section(
        accession="0001045810-25-000023", ticker="NVDA", fiscal_year=2025,
        item=item, title="MD&A", text=text,
    )


def _sentence(i: int, n_words: int = 10) -> str:
    return f"Sentence {i} " + " ".join(f"w{i}x{j}" for j in range(n_words - 2)) + "."


def _big_section(n_sentences: int = 2000, per_paragraph: int = 8) -> Section:
    sentences = [_sentence(i) for i in range(n_sentences)]  # 20K words
    paragraphs = [
        " ".join(sentences[i:i + per_paragraph])
        for i in range(0, n_sentences, per_paragraph)
    ]
    return _section("\n\n".join(paragraphs))


def test_no_text_lost_concatenation_covers_section():
    s = _big_section()
    chunks = chunk_section(s, words, max_tokens=100)
    assert len(chunks) > 1
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    for m in re.finditer(r"Sentence \d+ [^.]*\.", s.text):
        assert set(range(m.start(), m.end())) <= covered, m.group(0)[:30]
    assert chunks[0].start == 0
    assert chunks[-1].end == len(s.text)


def test_chunk_text_equals_section_slice_and_ids_are_sequential():
    s = _big_section(200)
    chunks = chunk_section(s, words, max_tokens=50)
    for i, c in enumerate(chunks):
        assert c.text == s.text[c.start:c.end]
        assert c.chunk_id == f"0001045810-25-000023:Item_7_MDA:{i:04d}"
        assert (c.accession, c.ticker, c.fiscal_year, c.item) == (
            s.accession, s.ticker, s.fiscal_year, s.item)
    assert [c.start for c in chunks] == sorted(c.start for c in chunks)


def test_chunks_end_on_sentence_boundary():
    s = _big_section(200)
    for c in chunk_section(s, words, max_tokens=50):
        assert c.text.endswith(".")
        nxt = s.text[c.end:]
        assert nxt == "" or nxt[0].isspace()
        assert c.text[0].isupper()


def test_each_chunk_under_max_tokens():
    s = _big_section(500)
    for c in chunk_section(s, words, max_tokens=57):
        assert words(c.text) <= 57


def test_chunks_pack_greedily():
    s = _section(" ".join(_sentence(i) for i in range(10)))  # 10 words each
    chunks = chunk_section(s, words, max_tokens=35)
    assert [words(c.text) for c in chunks] == [30, 30, 30, 10]


def test_oversized_unit_is_split_so_every_chunk_fits():
    long = "Long " + " ".join(f"word{i}" for i in range(100)) + "."
    s = _section(f"{_sentence(0)} {long} {_sentence(1)}")
    chunks = chunk_section(s, words, max_tokens=30)
    assert all(words(c.text) <= 30 for c in chunks)
    assert all(c.text == s.text[c.start:c.end] for c in chunks)
    assert " ".join(c.text for c in chunks) == s.text


def test_oversized_unit_splits_on_newlines_before_whitespace():
    lines = [" ".join(f"l{i}w{j}" for j in range(10)) for i in range(6)]
    s = _section("\n".join(lines))  # one unit, 60 words, no punctuation
    chunks = chunk_section(s, words, max_tokens=25)
    assert [c.text for c in chunks] == ["\n".join(lines[0:2]), "\n".join(lines[2:4]),
                                        "\n".join(lines[4:6])]


def test_paragraph_break_is_a_boundary():
    heading = "Results of Operations Overview"  # no punctuation
    s = _section(f"{heading}\n\n{_sentence(0)}")
    chunks = chunk_section(s, words, max_tokens=12)
    assert [c.text for c in chunks] == [heading, _sentence(0)]


def test_force_split_logs_warning(caplog):
    s = _section("Long " + " ".join(["word"] * 100) + ".")
    with caplog.at_level("WARNING", logger="filings_kb.chunking"):
        chunks = chunk_section(s, words, max_tokens=30)
    records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert records
    assert all(c.chunk_id in " ".join(r.getMessage() for r in records) for c in chunks)


def test_no_warning_when_nothing_is_force_split(caplog):
    with caplog.at_level("WARNING", logger="filings_kb.chunking"):
        chunk_section(_big_section(50), words, max_tokens=50)
    assert not caplog.records


def test_first_chunk_starts_on_text_not_whitespace():
    s = _section("  \n Hello world. Second one.")
    chunks = chunk_section(s, words)
    assert chunks[0].start == s.text.index("Hello")
    assert chunks[0].text == "Hello world. Second one."


def test_negative_paragraphs_raises():
    s = _paragraph_section()
    c = chunk_section(s, words, max_tokens=10)[0]
    with pytest.raises(ValueError):
        context_window(s, c, paragraphs=-1)


def test_empty_section_yields_no_chunks():
    assert chunk_section(_section(""), words) == []


def test_single_sentence_without_terminal_punctuation():
    s = _section("Net revenue rose")
    chunks = chunk_section(s, words)
    assert [c.text for c in chunks] == ["Net revenue rose"]


def test_abbreviation_followed_by_lowercase_is_not_split():
    s = _section("Revenue was $5 million vs. last year. Costs fell.")
    chunks = chunk_section(s, words, max_tokens=8)
    assert [c.text for c in chunks] == ["Revenue was $5 million vs. last year.", "Costs fell."]


def test_invalid_max_tokens_raises():
    with pytest.raises(ValueError):
        chunk_section(_section("A b."), words, max_tokens=0)


def _paragraph_section() -> Section:
    # Seven paragraphs of one 10-word sentence each.
    return _section("\n\n".join(_sentence(i) for i in range(7)))


def test_context_window_match_equals_chunk_text_and_has_two_paragraphs_each_side():
    s = _paragraph_section()
    chunks = chunk_section(s, words, max_tokens=10)
    assert len(chunks) == 7
    c = chunks[3]
    w = context_window(s, c)
    assert w["match"] == c.text == _sentence(3)
    assert w["before"] == _sentence(1) + "\n\n" + _sentence(2) + "\n\n"
    assert w["after"] == "\n\n" + _sentence(4) + "\n\n" + _sentence(5)
    assert w["before"] + w["match"] + w["after"] in s.text


def test_context_window_is_contiguous_slice_with_partial_paragraph_chunks():
    s = _big_section(120, per_paragraph=8)
    chunks = chunk_section(s, words, max_tokens=25)  # chunks split paragraphs
    for c in chunks[::7]:
        w = context_window(s, c)
        joined = w["before"] + w["match"] + w["after"]
        i = s.text.find(joined)
        assert i != -1
        assert i + len(w["before"]) == c.start
        assert s.text[i:i + len(joined)] == joined


def test_context_window_at_section_start_has_empty_before():
    s = _paragraph_section()
    c = chunk_section(s, words, max_tokens=10)[0]
    w = context_window(s, c)
    assert w["before"] == ""
    assert w["after"].startswith("\n\n" + _sentence(1))


def test_context_window_at_section_end_has_empty_after():
    s = _paragraph_section()
    c = chunk_section(s, words, max_tokens=10)[-1]
    w = context_window(s, c)
    assert w["after"] == ""
    assert w["before"].endswith(_sentence(5) + "\n\n")


def test_context_window_includes_own_paragraph_remainder():
    para = " ".join(_sentence(i) for i in range(4))
    s = _section(f"{_sentence(90)}\n\n{para}\n\n{_sentence(91)}")
    chunks = chunk_section(s, words, max_tokens=10)
    c = chunks[2]  # second sentence of the middle paragraph
    w = context_window(s, c)
    assert w["before"] == f"{_sentence(90)}\n\n{_sentence(0)} "
    assert w["after"] == f" {_sentence(2)} {_sentence(3)}\n\n{_sentence(91)}"


def test_context_window_paragraphs_param_and_chunk_is_a_dataclass():
    s = _paragraph_section()
    c = chunk_section(s, words, max_tokens=10)[3]
    w = context_window(s, c, paragraphs=1)
    assert w["before"] == _sentence(2) + "\n\n"
    assert w["after"] == "\n\n" + _sentence(4)
    assert isinstance(c, Chunk)
