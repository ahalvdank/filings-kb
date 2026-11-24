"""Entity linking: strong identifiers first, then fuzzy name matching.

A new mention of an entity is linked to an existing entity of the same type when
a strong identifier (ticker, CIK, accession ...) is equal. Otherwise its name is
compared to every existing entity of that type with rapidfuzz ``WRatio``:

- score >= ``link_auto``: link to the best match;
- ``link_review`` <= score < ``link_auto``: create a *new* entity and queue it for
  manual review, so an uncertain match is never silently merged;
- score < ``link_review``: create a new entity.

Two entities whose strong identifiers disagree (same key, different value) are
never fuzzy-matched, so "NVDA 10-K FY2025" and "NVDA 10-K FY2026" stay apart.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, utils

from filings_kb.extract import MODELS_PATH
from filings_kb.store import EntityRow, Store

# Identifier keys that pin down one real-world entity. Weak keys such as
# ``exchange`` or ``segment_name`` are shared by many entities and are ignored.
STRONG_IDENTIFIER_KEYS: frozenset[str] = frozenset(
    {"ticker", "cik", "accession", "cusip", "isin", "lei"}
)


@dataclass(frozen=True)
class LinkResult:
    """Outcome of linking one mention.

    ``entity_id`` is the entity the mention now belongs to (an existing one, or a
    newly created one); ``needs_review`` is True when the new entity was created
    next to a similar-looking existing one and queued for manual review.
    """

    entity_id: str | None
    needs_review: bool


def load_link_thresholds(path: Path = MODELS_PATH) -> tuple[float, float]:
    """Return ``(link_auto, link_review)`` from the models config.

    Raises:
        ValueError: if either key is missing or ``link_review > link_auto``.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("link_auto", "link_review"):
        if key not in data:
            raise ValueError(f"{path}: missing required key '{key}'")
    return float(data["link_auto"]), float(data["link_review"])


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _strong(identifiers: dict[str, Any]) -> dict[str, str]:
    """Strong identifiers with non-empty values, normalised."""
    return {
        k: _norm(v) for k, v in identifiers.items()
        if k in STRONG_IDENTIFIER_KEYS and _norm(v)
    }


def _identifier_match(existing: list[EntityRow], strong: dict[str, str]) -> EntityRow | None:
    for ent in existing:
        theirs = _strong(ent.identifiers)
        if any(theirs.get(k) == v for k, v in strong.items()):
            return ent
    return None


def _conflicts(ent: EntityRow, strong: dict[str, str]) -> bool:
    theirs = _strong(ent.identifiers)
    return any(k in theirs and theirs[k] != v for k, v in strong.items())


def link_entity(
    store: Store,
    type: str,
    name: str,
    identifiers: dict[str, Any],
    *,
    attributes: dict[str, Any] | None = None,
    link_auto: float | None = None,
    link_review: float | None = None,
) -> LinkResult:
    """Link a mention to an existing entity of ``type`` or create a new one.

    Args:
        store: The knowledge store.
        type: Entity type; only entities of this type are candidates.
        name: Mention text.
        identifiers: Identifiers from the mention (only strong keys are matched).
        attributes: Stored on a newly created entity.
        link_auto: Override of the auto-link threshold (default: models.yaml).
        link_review: Override of the manual-review threshold (default: models.yaml).

    Returns:
        The linked or new entity id and whether it needs manual review.

    Raises:
        ValueError: if ``link_review > link_auto``.
    """
    if link_auto is None or link_review is None:
        default_auto, default_review = load_link_thresholds()
        link_auto = default_auto if link_auto is None else link_auto
        link_review = default_review if link_review is None else link_review
    if link_review > link_auto:
        raise ValueError(f"link_review ({link_review}) must not exceed link_auto ({link_auto})")

    existing = store.entities_of_type(type)
    strong = _strong(identifiers)
    by_id = _identifier_match(existing, strong)
    if by_id is not None:
        return LinkResult(by_id.id, needs_review=False)

    best: EntityRow | None = None
    best_score = -1.0
    for ent in existing:
        if _conflicts(ent, strong):
            continue
        score = fuzz.WRatio(name, ent.name, processor=utils.default_process)
        if score > best_score:
            best, best_score = ent, score

    if best is not None and best_score >= link_auto:
        return LinkResult(best.id, needs_review=False)

    new_id = store.put_entity(type, name, attributes or {}, identifiers)
    if best is not None and best_score >= link_review:
        store.add_manual_review(
            new_id, f"possible duplicate of {best.id} ({best.name!r}), score {best_score:.1f}"
        )
        return LinkResult(new_id, needs_review=True)
    return LinkResult(new_id, needs_review=False)
