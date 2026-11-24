"""Tests for scripts/import_private_cache.py (no network, no LLM)."""

import importlib.util
import json
from pathlib import Path

import pytest

from filings_kb.schema import ExtractionPayload

_SPEC = importlib.util.spec_from_file_location(
    "import_private_cache",
    Path(__file__).resolve().parent.parent / "scripts" / "import_private_cache.py",
)
imp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(imp)

ACC = "0001045810-25-000023"
OTHER_ACC = "0000000000-00-000000"
MODEL = "claude-sonnet-4-6"

PAYLOAD = {
    "source_file": "filings/NVDA/x.json", "source_type": "sec_filing",
    "entities": [{"type": "Risk", "name": "R", "aliases": [], "identifiers": {},
                  "attributes": {"category": "regulatory"}}],
    "relationships": [],
    "chunks": [{"entity_name": "R", "entity_type_entity": "Risk", "content": "q",
                "chunk_type": "risk_factor", "valid_from": None, "valid_until": None,
                "attributes": {}}],
}

SAMPLE = """\
rows:
  - {ticker: NVDA, fiscal_year: 2025, accession: "%s"}
extract_items:
  - Item_1A
  - Item_3
""" % ACC


def _write(cache: Path, name: str, *, acc=ACC, item="Item_1A", prompt="v3",
           model=MODEL, sha="a" * 64, payload=None, at="2026-01-01T00:00:00Z") -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{name}.json").write_text(json.dumps({
        "backend": "subprocess", "extracted_at": at, "model": model,
        "prompt_version": prompt, "wallclock_seconds": 1.0,
        "input": {"accession_number": acc, "section_type": item,
                  "text_sha256": sha, "ticker": "NVDA"},
        "output_payload": PAYLOAD if payload is None else payload,
    }))


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    p = tmp_path / "sample.yaml"
    p.write_text(SAMPLE)
    return p


def test_import_filters_by_accession_prompt_and_model(tmp_path, sample):
    cache, out = tmp_path / "cache", tmp_path / "out"
    _write(cache, "keep")
    _write(cache, "other_acc", acc=OTHER_ACC)
    _write(cache, "old_prompt", prompt="v2")
    _write(cache, "other_model", model="claude-sonnet-5")
    _write(cache, "not_allowlisted", item="Item_7_MDA")

    manifest = imp.import_cache(cache, sample, out)

    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json"))
    assert files == [f"{ACC}/Item_1A.json", "manifest.json"]
    assert manifest["files_imported"] == 1
    assert manifest["per_accession"][ACC] == {"sections": 1, "parts": 1}
    envelope = json.loads((out / ACC / "Item_1A.json").read_text())
    assert envelope["origin"] == "imported"
    assert envelope["prompt_version"] == "v3" and envelope["model"] == MODEL
    assert envelope["parts"][0]["payload"] == PAYLOAD  # raw payload, byte-faithful
    ExtractionPayload.model_validate(envelope["parts"][0]["payload"])
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest


def test_import_merges_split_sections_into_one_file(tmp_path, sample):
    cache, out = tmp_path / "cache", tmp_path / "out"
    _write(cache, "p2", sha="b" * 64, at="2026-01-02T00:00:00Z")
    _write(cache, "p1", sha="a" * 64, at="2026-01-01T00:00:00Z")

    manifest = imp.import_cache(cache, sample, out)

    envelope = json.loads((out / ACC / "Item_1A.json").read_text())
    assert [p["text_sha256"] for p in envelope["parts"]] == ["a" * 64, "b" * 64]
    assert manifest["per_accession"][ACC] == {"sections": 1, "parts": 2}


def test_import_rejects_invalid_payload(tmp_path, sample):
    cache, out = tmp_path / "cache", tmp_path / "out"
    bad = json.loads(json.dumps(PAYLOAD))
    bad["entities"][0]["type"] = "NotAType"
    _write(cache, "bad", payload=bad)

    with pytest.raises(imp.ImportValidationError, match="bad.json"):
        imp.import_cache(cache, sample, out)
    assert not out.exists() or not list(out.rglob("*.json"))


def test_import_excludes_payload_with_private_strings_and_reports_it(tmp_path, sample):
    cache, out = tmp_path / "cache", tmp_path / "out"
    leaky = json.loads(json.dumps(PAYLOAD))
    leaky["chunks"][0]["attributes"] = {"path": "/Users/someone/x"}
    _write(cache, "leaky", item="Item_3", payload=leaky)
    _write(cache, "fine")

    manifest = imp.import_cache(cache, sample, out)

    assert not (out / ACC / "Item_3.json").exists()
    assert (out / ACC / "Item_1A.json").exists()
    assert [e["file"] for e in manifest["excluded"]] == ["leaky.json"]
    assert "/Users/" not in (out / "manifest.json").read_text()
