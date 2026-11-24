"""Write exports/score_distribution.txt from the committed answers' passage snapshots.

No model call and no store: every number comes from ``data/committed/answers/*.json``.
Run it on answers made with ``display_cutoff: 0.0`` to see the whole retrieved range.
``--pool`` adds a second section over the full top-k pool of every question (the
candidates before the per-filing quota and ``max_passages`` cut them). That section
queries the local embedding index and ``data/kb.sqlite`` read-only; it makes no LLM call.
"""

import argparse

import json
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUCKET = 0.05


def _pool_section() -> list[str]:
    import yaml

    from filings_kb.answer import load_questions, retrieve
    from filings_kb.embed import EmbeddingIndex
    from filings_kb.store import Store

    models = yaml.safe_load((REPO / "config/models.yaml").read_text(encoding="utf-8"))
    cfg = load_questions()
    store = Store(REPO / "data/kb.sqlite")
    index = EmbeddingIndex(REPO / "data/chroma", str(models["embedding"]), max_tokens=512)
    pool = {q.slug: [c.score for c in retrieve(q, index, store, cfg.top_k, 10**6, 0.0, 0)]
            for q in cfg.questions}
    everything = [s for v in pool.values() for s in v]
    buckets = Counter(int(s / BUCKET + 1e-9) for s in everything)
    lines = ["", f"Full top-k pool before the quota and max_passages cut: {len(everything)} "
             "candidates", "bucket        count"]
    for b in range(min(buckets), max(buckets) + 1):
        lines.append(f"{b * BUCKET:.2f}-{(b + 1) * BUCKET:.2f}  {buckets[b]:5d}  "
                     + "#" * round(buckets[b] / 2))
    cutoff = cfg.display_cutoff
    lines += ["", f"Pool per trace; 'below' counts candidates under display_cutoff {cutoff}",
              f"{'slug':56s} {'n':>3s} {'min':>6s} {'median':>6s} {'max':>6s} {'below':>5s}"]
    for slug, v in pool.items():
        lines.append(f"{slug:56s} {len(v):3d} {min(v):6.3f} {statistics.median(v):6.3f} "
                     f"{max(v):6.3f} {sum(1 for x in v if x < cutoff):5d}")
    return lines


def main(pool: bool = False) -> None:
    rows = []
    for path in sorted((REPO / "data/committed/answers").glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        cited = {c for para in d["paragraphs"] for u in para for c in u["cites"]}
        scores = [p["score"] for p in d["passages"]]
        cited_scores = [p["score"] for p in d["passages"] if p["cite"] in cited]
        rows.append((path.stem, scores, min(cited_scores)))
    everything = [s for _, scores, _ in rows for s in scores]
    buckets = Counter(int(s / BUCKET + 1e-9) for s in everything)
    lines = [f"Score distribution over {len(rows)} traces, {len(everything)} retrieved passages",
             "(cosine similarity; from the committed answer snapshots)", "",
             "Overall histogram, 0.05 buckets", "bucket        count"]
    for b in range(min(buckets), max(buckets) + 1):
        lines.append(f"{b * BUCKET:.2f}-{(b + 1) * BUCKET:.2f}  {buckets[b]:5d}  "
                     + "#" * round(buckets[b] / 2))
    lines += ["", "Per trace", f"{'slug':56s} {'n':>3s} {'min':>6s} {'median':>6s} {'max':>6s}"
              f" {'lowest cited':>12s}"]
    for slug, scores, low in rows:
        lines.append(f"{slug:56s} {len(scores):3d} {min(scores):6.3f} "
                     f"{statistics.median(scores):6.3f} {max(scores):6.3f} {low:12.3f}")
    lines += ["", f"Lowest-scoring passage any current answer cites: "
              f"{min(low for _, _, low in rows):.3f}"]
    if pool:
        lines += _pool_section()
    out = REPO / "exports/score_distribution.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="also report the full top-k pool")
    main(ap.parse_args().pool)
