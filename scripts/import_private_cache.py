"""Import the author's private v3 extraction cache for the pinned sample.

Usage:
    python scripts/import_private_cache.py --cache DIR --sample config/sample.yaml \
        [--out data/committed/extractions]

Keeps only files for the sample's accessions with ``prompt_version == "v3"``,
``model == "claude-sonnet-4-6"`` and a section in ``extract_items``. Every
``output_payload`` must validate against ``ExtractionPayload``; one that does
not aborts the import. A payload containing a private marker string is excluded
and listed in the manifest. Files are grouped into one envelope per
``<accession>/<section_type>.json`` (the private pipeline split long sections,
so several source files can share a section).
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filings_kb.config import load_sample  # noqa: E402
from filings_kb.extract import PROMPT_VERSION, cache_path, write_envelope  # noqa: E402
from filings_kb.schema import ExtractionPayload  # noqa: E402

MODEL = "claude-sonnet-4-6"
PRIVATE_MARKERS = ("ThesisPage", "own_research", "/Users/")


class ImportValidationError(ValueError):
    """A cached payload failed schema validation; the import is refused."""


def import_cache(cache: Path, sample_path: Path, out_dir: Path) -> dict[str, Any]:
    """Import matching cache files and write the manifest.

    Args:
        cache: Directory of private ``*.json`` extraction files.
        sample_path: ``config/sample.yaml``.
        out_dir: Destination, normally ``data/committed/extractions``.

    Returns:
        The manifest dict that was written to ``out_dir/manifest.json``.

    Raises:
        ImportValidationError: if any matching payload fails validation.
    """
    sample = load_sample(sample_path)
    accessions = {r.accession for r in sample.rows}
    items = set(sample.extract_items)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, str]] = []
    scanned = 0
    for f in sorted(cache.glob("*.json")):
        scanned += 1
        raw = f.read_bytes()
        d = json.loads(raw)
        inp = d["input"]
        if (inp["accession_number"] not in accessions
                or d["prompt_version"] != PROMPT_VERSION or d["model"] != MODEL
                or inp["section_type"] not in items):
            continue
        try:
            ExtractionPayload.model_validate(d["output_payload"])
        except ValidationError as exc:
            raise ImportValidationError(f"{f.name}: {exc}") from exc
        text = json.dumps(d["output_payload"], ensure_ascii=False)
        if any(m in text for m in PRIVATE_MARKERS):
            # The reason stays generic so the committed manifest holds no marker.
            excluded.append({"file": f.name, "reason": "contains a private marker string"})
            continue
        groups[(inp["accession_number"], inp["section_type"])].append({
            "text_sha256": inp["text_sha256"],
            "extracted_at": d["extracted_at"],
            "payload": d["output_payload"],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        })

    per_accession: dict[str, dict[str, int]] = {}
    source_hashes: list[str] = []
    for (acc, item), parts in sorted(groups.items()):
        parts.sort(key=lambda p: (p["extracted_at"], p["text_sha256"]))
        source_hashes += [p.pop("source_sha256") for p in parts]
        write_envelope(cache_path(out_dir, acc, item), {
            "accession": acc, "item": item, "prompt_version": PROMPT_VERSION,
            "model": MODEL, "origin": "imported", "parts": parts,
        })
        counts = per_accession.setdefault(acc, {"sections": 0, "parts": 0})
        counts["sections"] += 1
        counts["parts"] += len(parts)

    manifest = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "files_scanned": scanned,
        "files_imported": len(source_hashes),
        "sections_imported": sum(c["sections"] for c in per_accession.values()),
        "per_accession": per_accession,
        "accessions_without_imports": sorted(accessions - set(per_accession)),
        "excluded": excluded,
        "source_file_sha256": sorted(source_hashes),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/committed/extractions"))
    args = ap.parse_args()
    m = import_cache(args.cache, args.sample, args.out)
    print(f"imported {m['files_imported']} files into {m['sections_imported']} sections")
    for acc, c in m["per_accession"].items():
        print(f"  {acc}: {c['sections']} sections, {c['parts']} parts")
    for acc in m["accessions_without_imports"]:
        print(f"  WARNING no imports for {acc}")
    for e in m["excluded"]:
        print(f"  EXCLUDED {e['file']}: {e['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
