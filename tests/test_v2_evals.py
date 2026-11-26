"""Tests for the v2 eval skeletons (audit fixes 1.10 and 2.11 applied)."""

import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

from evals.v2 import eval_claim_audit, eval_faithfulness, eval_ingest_cost, eval_recall


def _passage(score: float, *, ticker="NVDA", fy=2025, section="Item 1A Risk Factors",
             before="", match="", after="") -> dict:
    return {"score": score, "ticker": ticker, "fiscal_year": fy, "section": section,
            "context": {"before": before, "match": match, "after": after}}


def _golden(tmp_path: Path, phrase: str = "export control") -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"questions": [{"slug": "q", "expected_targets": [
        {"ticker": "NVDA", "fiscal_year": 2025, "section": "item 1a",
         "must_contain_phrase": phrase}]}]}))
    return path


def _traces(tmp_path: Path, passages: list[dict]) -> Path:
    folder = tmp_path / "traces"
    folder.mkdir()
    (folder / "q.json").write_text(json.dumps({"slug": "q", "passages": passages}))
    return folder


def test_eval_recall_uses_must_contain_phrase_sorted_by_score_and_full_context(tmp_path) -> None:
    # The hit sits in `after` (not `match`) and ranks 6th by score although it is listed first.
    passages = [_passage(0.30, after="New Export Control rules")] + [
        _passage(0.9 - i / 100, ticker="AMD", match="filler") for i in range(5)]
    result = eval_recall.evaluate_recall(_golden(tmp_path), _traces(tmp_path, passages))
    assert result["recall@5"] == 0.0   # sorted by score, the hit is 6th
    assert result["recall@20"] == 1.0  # found through before + match + after, case-insensitive


def test_eval_recall_counts_missing_trace_targets_as_misses(tmp_path) -> None:
    empty = tmp_path / "traces"
    empty.mkdir()
    result = eval_recall.evaluate_recall(_golden(tmp_path), empty)
    assert result == {"recall@5": 0.0, "recall@20": 0.0, "targets": 1, "missing_traces": ["q"]}


def test_eval_recall_requires_ticker_year_and_section(tmp_path) -> None:
    passages = [_passage(0.9, fy=2024, match="export control"),
                _passage(0.8, section="Item 7", match="export control")]
    result = eval_recall.evaluate_recall(_golden(tmp_path), _traces(tmp_path, passages))
    assert result["recall@20"] == 0.0


def test_kappa_is_none_not_one_when_chance_agreement_is_certain() -> None:
    assert eval_faithfulness.cohens_kappa([1, 1, 1], [1, 1, 1]) == (None, 1.0)
    assert eval_faithfulness.cohens_kappa([], []) == (None, None)
    kappa, po = eval_faithfulness.cohens_kappa([1, 0, 1, 0], [1, 0, 0, 0])
    assert po == 0.75 and kappa == pytest.approx(0.5)


def test_faithfulness_parses_booleans_strictly(tmp_path) -> None:
    path = tmp_path / "judge.jsonl"
    path.write_text('{"judge_supported": "false"}\n{"judge_supported": true}\n')
    assert eval_faithfulness.evaluate_faithfulness(path)["judge_faithfulness"] == 0.5
    path.write_text('{"judge_supported": "yes"}\n')
    with pytest.raises(ValueError, match="judge_supported"):
        eval_faithfulness.evaluate_faithfulness(path)


def test_faithfulness_without_spot_check_reports_no_kappa(tmp_path) -> None:
    path = tmp_path / "judge.jsonl"
    path.write_text('{"judge_supported": true}\n{"judge_supported": false, "human_supported": null}\n')
    result = eval_faithfulness.evaluate_faithfulness(path)
    assert result["human_spot_check_n"] == 0 and result["cohens_kappa"] is None


def _worksheet(tmp_path: Path, labels: list[str]) -> Path:
    path = tmp_path / "audit.csv"
    rows = ["claim_id,ticker,claim_type,statement,source_quote,label"]
    rows += [f"c{i},NVDA,Risk,s,q,{label}" for i, label in enumerate(labels)]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_claim_audit_scores_precision_and_failure_breakdown(tmp_path) -> None:
    result = eval_claim_audit.score_claim_audit(
        _worksheet(tmp_path, ["TP", "TP", "TP", "hallucinated"]))
    assert result["precision"] == 0.75 and result["false_positive_rate"] == 0.25
    assert result["failure_breakdown"]["hallucinated"] == 1


@pytest.mark.parametrize("label", ["", "halucinated", "tp"])
def test_claim_audit_rejects_blank_or_misspelled_labels(tmp_path, label) -> None:
    with pytest.raises(ValueError, match="c1"):
        eval_claim_audit.score_claim_audit(_worksheet(tmp_path, ["TP", label]))


def _log(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _rec(accession: str, stage: str, usd: str, seconds: float = 1.0, **extra) -> dict:
    return {"accession": accession, "ticker": "NVDA", "fiscal_year": 2025, "stage": stage,
            "model": "m", "input_tokens": 10, "output_tokens": 5, "wall_seconds": seconds,
            "usd_cost": usd, **extra}


def test_ingest_cost_keys_by_accession_and_reports_p90_and_stages(tmp_path) -> None:
    records = [_rec("a", "embed", "0.1"), _rec("a", "extract_claims", "0.2"),
               _rec("b", "extract_claims", "0.4")]  # a and b share a ticker and fiscal year
    result = eval_ingest_cost.summarize_ingest_costs(_log(tmp_path, records))
    assert result["filings_measured"] == 2
    assert Decimal(result["total_usd"]) == Decimal("0.7")  # exact: no float 0.7000000000000001
    assert result["p90_usd"] == "0.4" and result["median_usd"] == "0.35"
    assert result["per_stage"]["extract_claims"] == {"usd": "0.6", "seconds": 2.0, "tokens": 30}


def test_ingest_cost_rejects_an_unknown_stage(tmp_path) -> None:
    with pytest.raises(ValueError, match="stage"):
        eval_ingest_cost.summarize_ingest_costs(_log(tmp_path, [_rec("a", "rerank", "0")]))


class TestPooledRecall:
    @staticmethod
    def _setup(tmp_path: Path, questions: dict[str, dict], labels: dict[tuple[str, str], str]):
        pool = {"meta": {}, "questions": {
            qid: {"accession": "a", "item": "1A", "question": "q",
                  "dense_ranking": q["dense"], "bm25_top": q.get("bm25", []),
                  "pooled": sorted(set(q["dense"][:5]) | set(q.get("bm25", [])))}
            for qid, q in questions.items()}}
        pool_json = tmp_path / "pool.json"
        pool_json.write_text(json.dumps(pool), encoding="utf-8")
        labels_csv = tmp_path / "labels.csv"
        with labels_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["qid", "question", "row", "chunk_id", "passage", "relevant"])
            for i, ((qid, cid), lab) in enumerate(labels.items(), start=1):
                writer.writerow([qid, "q", i, cid, "text", lab])
        return pool_json, labels_csv

    @staticmethod
    def _dense(prefix: str) -> list[str]:
        return [f"{prefix}{i}" for i in range(1, 21)]

    def _standard(self, tmp_path: Path):
        q1, q2 = self._dense("a"), self._dense("b")
        questions = {"q1": {"dense": q1}, "q2": {"dense": q2}}
        labels = {("q1", c): "n" for c in q1[:5]}
        labels[("q1", "a1")] = "y"
        labels.update({("q2", c): "n" for c in q2[:5]})
        labels[("q2", "b3")] = "y"
        return questions, labels

    def test_pooled_recall_micro_and_macro(self, tmp_path) -> None:
        # q1: 2 relevant at dense ranks 1 and 8 (rank 8 is outside the pooled top 5, so it
        # is unlabelled by construction; the second relevant id is pooled via BM25).
        q1, q2 = self._dense("a"), self._dense("b")
        questions = {"q1": {"dense": q1, "bm25": ["a8"]}, "q2": {"dense": q2}}
        labels = {("q1", c): "n" for c in q1[:5] + ["a8"]}
        labels[("q1", "a1")] = "y"
        labels[("q1", "a8")] = "y"
        labels.update({("q2", c): "n" for c in q2[:5]})
        labels[("q2", "b3")] = "y"
        pool_json, labels_csv = self._setup(tmp_path, questions, labels)

        result = eval_recall.evaluate_pooled_recall(pool_json, labels_csv)

        assert result["recall@5"] == {"micro": 2 / 3, "macro": (0.5 + 1) / 2}
        assert result["recall@10"] == {"micro": 1.0, "macro": 1.0}
        assert result["recall@20"] == {"micro": 1.0, "macro": 1.0}
        assert result["relevant_passages"] == 3
        assert result["questions_scored"] == 2
        assert result["questions_excluded_no_relevant"] == []
        assert result["labelled_rows"] == 11

    def test_pooled_recall_relevant_outside_dense_top20_counts_as_miss(self, tmp_path) -> None:
        q1 = self._dense("a")
        questions = {"q1": {"dense": q1, "bm25": ["only_bm25"]}}
        labels = {("q1", c): "n" for c in q1[:5]}
        labels[("q1", "only_bm25")] = "y"
        labels[("q1", "a2")] = "y"
        pool_json, labels_csv = self._setup(tmp_path, questions, labels)

        result = eval_recall.evaluate_pooled_recall(pool_json, labels_csv)

        for k in (5, 10, 20):
            assert result[f"recall@{k}"] == {"micro": 0.5, "macro": 0.5}

    def test_pooled_recall_excludes_question_with_no_relevant(self, tmp_path) -> None:
        questions, labels = self._standard(tmp_path)
        labels[("q1", "a1")] = "n"
        pool_json, labels_csv = self._setup(tmp_path, questions, labels)

        result = eval_recall.evaluate_pooled_recall(pool_json, labels_csv)

        assert result["questions_excluded_no_relevant"] == ["q1"]
        assert result["questions_scored"] == 1
        assert result["relevant_passages"] == 1
        assert result["recall@5"] == {"micro": 1.0, "macro": 1.0}

    def test_pooled_recall_rejects_label_not_y_or_n(self, tmp_path) -> None:
        questions, labels = self._standard(tmp_path)
        labels[("q1", "a1")] = "Y "
        pool_json, labels_csv = self._setup(tmp_path, questions, labels)
        assert eval_recall.evaluate_pooled_recall(pool_json, labels_csv)["relevant_passages"] == 2

        for bad in ("yes", ""):
            labels[("q1", "a2")] = bad
            pool_json, labels_csv = self._setup(tmp_path, questions, labels)
            with pytest.raises(ValueError, match="q1.*a2"):
                eval_recall.evaluate_pooled_recall(pool_json, labels_csv)

    def test_pooled_recall_rejects_rows_not_in_pool_and_missing_pool_rows(self, tmp_path) -> None:
        questions, labels = self._standard(tmp_path)
        del labels[("q2", "b4")]
        labels[("q2", "stranger")] = "n"
        pool_json, labels_csv = self._setup(tmp_path, questions, labels)

        with pytest.raises(ValueError) as exc:
            eval_recall.evaluate_pooled_recall(pool_json, labels_csv)

        message = str(exc.value)
        assert "stranger" in message and "q2" in message
        assert "b4" in message
