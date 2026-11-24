"""Tests for entity linking: identifiers first, then fuzzy name matching."""

from pathlib import Path

import pytest

from filings_kb.link import LinkResult, link_entity
from filings_kb.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "kb.sqlite")


def test_identifier_match_beats_fuzzy(store: Store) -> None:
    existing = store.put_entity("Company", "NVIDIA Corporation", {}, {"ticker": "NVDA"})
    res = link_entity(store, "Company", "Totally Different Name Inc", {"ticker": "NVDA"})
    assert res == LinkResult(entity_id=existing, needs_review=False)


def test_identifier_match_ignores_case_and_whitespace(store: Store) -> None:
    existing = store.put_entity("Company", "NVIDIA Corporation", {}, {"ticker": "NVDA"})
    res = link_entity(store, "Company", "Other", {"ticker": " nvda "})
    assert res.entity_id == existing


def test_identifier_match_requires_same_entity_type(store: Store) -> None:
    store.put_entity("Company", "NVIDIA Corporation", {}, {"ticker": "NVDA"})
    res = link_entity(store, "Security", "NVIDIA common stock", {"ticker": "NVDA"})
    assert res.entity_id is not None
    assert len(store.entities_of_type("Security")) == 1


def test_weak_identifier_keys_do_not_match(store: Store) -> None:
    store.put_entity("Company", "Alpha Holdings", {}, {"exchange": "NASDAQ"})
    res = link_entity(store, "Company", "Zeta Industries", {"exchange": "NASDAQ"})
    assert len(store.entities_of_type("Company")) == 2
    assert res.needs_review is False


def test_fuzzy_high_score_links(store: Store) -> None:
    existing = store.put_entity("Person", "Jensen Huang", {}, {})
    res = link_entity(store, "Person", "Jensen  Huang.", {})
    assert res == LinkResult(entity_id=existing, needs_review=False)


def test_ambiguous_score_goes_to_manual_review_not_merged(store: Store) -> None:
    existing = store.put_entity("Person", "Colette Kress", {}, {})
    res = link_entity(store, "Person", "Colette M. Kress Jr", {}, link_auto=96, link_review=90)
    assert res.needs_review is True
    assert res.entity_id is not None and res.entity_id != existing
    assert len(store.entities_of_type("Person")) == 2
    reviews = store.manual_reviews()
    assert [r.entity_id for r in reviews] == [res.entity_id]


def test_low_score_creates_new_entity_without_review(store: Store) -> None:
    store.put_entity("Person", "Jensen Huang", {}, {})
    res = link_entity(store, "Person", "Lisa Su", {})
    assert res.needs_review is False
    assert len(store.entities_of_type("Person")) == 2
    assert store.manual_reviews() == []


def test_conflicting_strong_identifier_blocks_fuzzy_merge(store: Store) -> None:
    store.put_entity("Filing", "NVDA 10-K FY2025", {}, {"accession": "A-1"})
    res = link_entity(store, "Filing", "NVDA 10-K FY2026", {"accession": "A-2"})
    assert len(store.entities_of_type("Filing")) == 2
    assert res.needs_review is False


def test_linking_same_input_twice_is_idempotent(store: Store) -> None:
    a = link_entity(store, "Company", "NVIDIA Corporation", {"ticker": "NVDA"})
    b = link_entity(store, "Company", "NVIDIA Corporation", {"ticker": "NVDA"})
    assert a.entity_id == b.entity_id
    assert len(store.entities_of_type("Company")) == 1


def test_attributes_are_stored_on_new_entities(store: Store) -> None:
    res = link_entity(store, "Company", "Acme", {}, attributes={"cik": "0000000001"})
    ent = store.entities_of_type("Company")[0]
    assert ent.id == res.entity_id and ent.attributes == {"cik": "0000000001"}


def test_default_thresholds_come_from_config() -> None:
    from filings_kb.link import load_link_thresholds
    auto, review = load_link_thresholds()
    assert (auto, review) == (90, 75)


def test_review_threshold_above_auto_is_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="link_review"):
        link_entity(store, "Person", "X", {}, link_auto=70, link_review=80)
