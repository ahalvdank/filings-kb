# filings-kb

A pipeline over 61 SEC filings (12 companies, five fiscal years each, plus one 10-K/A). It fetches the filings, splits them into sections and chunks, embeds them, links the claims extracted from them into a small knowledge store, and answers questions with sentence-level citations back to the source passages. The "Query Traces" demo on my portfolio site renders the 12 answers exported in `exports/traces/`, 8 of them featured.

I was the architect, product manager and engineering team lead; AI coding agents worked like my staff engineers and wrote most of the code.

This is a public rewrite of my private pipeline: the code is rewritten, not copied. Two things come over unchanged, the v3 extraction prompt and its extraction outputs, described below.

The year-over-year signals (tone shift, FinBERT sentiment shift, claim diffs) are research only. They run in shadow mode, nothing reads them, and nothing here drives a trade. The idea comes from the "Lazy Prices" paper (Cohen, Malloy and Nguyen), which studied year-over-year changes in 10-K language. That paper is the motivation and nothing more: I did not replicate it and make no claim about returns.

## Quickstart

```
git clone https://github.com/ahalvdank/filings-kb && cd filings-kb
make install
make traces-from-cache
```

The repository URL goes live when the portfolio site launches.

`make traces-from-cache` needs no API key, no database and no network. It rebuilds `exports/` (traces, audit notes, `manifest.json`, `failed.json`) from the committed answers in `data/committed/answers/`, and the output is byte-identical to what is committed. The install is the slow part (torch and chromadb).

You only need `make ingest` to re-run retrieval. It fetches the 61 filings from EDGAR, embeds them locally and builds `data/kb.sqlite` and the vector index. Put `EDGAR_IDENTITY="Your Name your@email"` in `.env` first, because the SEC wants contact details in the User-Agent. After that, `make traces` re-answers every question with the `claude` CLI. It needs a clean git tree, because the commit is recorded in each trace.

Sample extractions were produced by this v3 prompt and model in my private pipeline and are committed here; `make sample-refresh` re-runs them with your own key. On a clean clone, `make sample` re-uses the committed extractions: it calls the extraction model only for a section with no cached envelope, and every section has one, so it makes no extraction call. It builds the API client only on the first cache miss, so it needs neither `ANTHROPIC_API_KEY` nor the optional `anthropic` package. Its signals step needs the `claude` CLI installed, and it re-uses the claim diffs committed in `data/committed/claim_diffs/` whenever their inputs are unchanged.

`make ingest`, `make sample` and `make signals` never write a tracked file, so `make traces` can run straight after them. Their outputs go to the gitignored `data/local/`: the ingest report, any new extraction or claim diff, and `signals.json`, with every difference from `data/committed/signals.json` printed. Run times go to the gitignored `data/run_log.json`. Only `make sample-refresh` passes `--write-committed` and rewrites `data/committed/`. `make sample-refresh` (set `ANTHROPIC_API_KEY` and `.venv/bin/pip install -e '.[anthropic]'`) ignores the cache, extracts every detected section again, overwrites `data/committed/extractions/` and then computes the signals. It needs a fresh store, so move `data/kb.sqlite` aside first. A section the detector does not find has no text to send, so its committed extraction is kept. The ingest report records the cost in `cost_usd`. The committed report says `"0"` because every section came from the cache. I have not run the live path over the whole sample, so I have no live cost to quote.

## Pipeline

```
config/sample.yaml  (pinned: ticker, fiscal year, accession)
        |
   fetch (edgartools, by accession) -> raw HTML in data/raw (gitignored)
        |
   sections (Item 1, 1A, 1C, 7, 7A, 8 ...); table-of-contents stubs are rejected
        |
   chunks (<= 512 tokens) -> embeddings (Qwen3-Embedding-0.6B) -> data/chroma
        |
   claims: committed v3 extractions (data/committed/extractions), or live extraction
        |  one claim per filing section, linked to the filer; a 10-K/A closes out
        |  the claims of the sections it supersedes
        v
   data/kb.sqlite  (entities, relationships, runs)
        |
   signals: YoY tone shift, FinBERT sentiment shift, claim diff   (shadow only)

questions (config/questions.yaml)
        |
   retrieve -> answer -> check -> export    =>  exports/traces/<slug>.json
```

## How a trace is produced

1. **Retrieve.** The top 20 chunks per (filing, section) come from the vector index. Passages scoring under the display cutoff (0.35, picked from the score distribution in `exports/score_distribution.txt`) are not shown to the model. The rest are capped at 40 and shared out evenly per (filing, item) group, so one filing or section can't crowd out the others; slots a group can't fill go to the best remaining passages. The best 10 below-cutoff candidates are exported as rejected, so a reader can see what was left out.
2. **Answer.** `claude-sonnet-4-6`, through the `claude` CLI. The model returns JSON: paragraphs of one-sentence units, each with the numbers of the passages that support it. The citation markers are rendered from that structure, not written by the model.
3. **Check, per sentence, against the passages that sentence cites.** Every sentence must cite a passage. Every number must appear in a cited passage, and a number glued to a lowercase suffix the check can't read (`10x`, `$5tn`) fails; product codes such as `H20` or `14A` don't. Every quoted span must be verbatim in one cited passage. In a quote split by `[edits]` or an ellipsis, pieces under four words are ignored, every longer piece must be in the passage, and at least one piece must be that long. Absence and identity phrases ("unchanged", "all other", "no mention") must be used by a cited passage. A failing answer is retried once. A second failure drops the question and records it in `exports/failed.json`.
4. **Export.** The trace holds the answer, every passage with its score and context, the claims behind it, the signals, and `meta` with the four models, the cutoff and the build commit. A claim points at a cited passage of its own section that holds its name or shares eight consecutive words with its verbatim extraction excerpt: 118 of the 626 exported claims have one. A live run refuses to export from a dirty git tree. `make traces-from-cache` refuses (exit 2) when code, schemas, prompts or config changed since the build commit, because the commit would no longer be true.

13 questions were run and 12 are exported. `unh-cyber-regulatory-risk-fy23-fy25` passed the automatic checks but failed my human audit on three weak sentences, so it is not exported and `exports/failed.json` lists it as "failed human audit". Eight traces are featured:

- NVIDIA export-control and customer-concentration risk, FY2022 to FY2026
- What AMD changed in its amended FY2025 10-K
- Meta AI investment ambition against monetization risk, FY2025
- Boeing liquidity, debt and equity actions, FY2024 to FY2025
- JPMorgan capital returns and regulatory-capital posture, FY2024 to FY2025
- NVIDIA, AMD and Intel on foundry and packaging dependency
- The AI value chain, chip suppliers against hyperscale buyers
- ExxonMobil against Chevron on energy-transition risk

## Evals

`exports/parity.json` and `exports/audit_list.md` come from the two evals below. Neither calls a model. `make evals` runs the trace check, and parity too when `PRIVATE_KB` is set.

### Parity with my private knowledge base (`make parity PRIVATE_KB=...`)

Parity needs my private KB, which it opens read-only, so it can't run from a clean clone. `parity.json` holds counts only, no claim text.

**Ingest reproduces the committed cache (per-type counts).** For every claim type, the committed extraction files hold exactly as many claims as the public store, counting the parts of a split section the way ingest merges them (`ingest_vs_cache` in `parity.json`). The only open-versus-total difference is the 17 claims closed out by the AMD 10-K/A. The eval exits 1 if that equality breaks.

**Section-text fidelity.** Of the sections the private pipeline extracted, 681 of the 682 that could be compared have the same text as the public section detector's (99.9%). Read that next to the coverage: only 682 of 824 sections were comparable (82.8%). Of the rest, 103 were split into several extraction calls by the private pipeline, so there is no single hash to compare, and 39 were not detected by the public detector. The one mismatch is Amazon's FY2023 Item 8.

**Matched claim recall.** Raw claim counts can't be compared. This repo keeps one claim per filing section, so a segment named in Items 1, 7 and 15 is three claims, while the private KB merges claims into entities. So I compare per filing and per claim type, on clusters of names. Each side's claim names are clustered with rapidfuzz `WRatio` at 90, the linker's auto-link threshold. Each private claim also has recorded aliases. A cluster counts as matched when one of its names scores 90 or more (`WRatio`) against a name on the other side for the same filing, or when a private alias scores 90 or more by `token_sort_ratio` against a public name (an alias is never matched by `WRatio`, so a short alias can't match every longer name that contains it). A private claim belongs to a filing when an open relationship links it from the filing's accession document or from its accession-less manifest, which holds claims of its own. Recall is matched clusters over clusters, summed over the 61 filings, in each direction; the counts below are clusters, not raw claims. A type fails when recall in either direction is under 80% and the type has at least 20 private clusters.

| claim type | private clusters matched here | clusters here matched in private | clusters pub/priv |
|---|---:|---:|---:|
| Risk | 87.3% (names only 84.8%) | 82.0% (79.2%) | 2,051 / 1,911 |
| MaterialEvent | 95.6% (94.5%) | 86.4% (84.1%) | 748 / 655 |
| LegalProceeding | 92.6% (89.7%) | **77.8%** (73.6%) | 594 / 486 |
| RevenueSegment | 96.4% (93.5%) | 82.4% (77.1%) | 380 / 306 |
| CapitalReturn | 96.7% (95.9%) | 81.4% (79.3%) | 295 / 244 |
| CapitalStructureEvent | 95.9% (94.6%) | **79.5%** (79.1%) | 268 / 222 |
| GuidanceUpdate | 97.1% (95.9%) | 83.8% (82.8%) | 198 / 171 |
| ESGCommitment | 93.8% (93.8%) | 84.9% (84.1%) | 126 / 113 |
| InsiderTransaction | not comparable | not comparable | 77 / 0 |

With aliases, private-to-public recall is 87.3% to 97.1% and public-to-private recall is 77.8% to 86.4%; on names alone they are 84.8% to 95.9% and 73.6% to 84.1%. The gate fails today: LegalProceeding (77.8%) and CapitalStructureEvent (79.5%) are under 80% public-to-private. The private KB links no InsiderTransaction claim to these filings, so that type has nothing to compare. A name match is only a proxy: two names for the same claim can score under 90, and two different claims can score over it, and the clustering is greedy, so it isn't transitive. The raw counts behind the clusters are in `parity.json` as diagnostics.

### Trace check (`make trace-check`, or `ARGS=--network`)

This re-checks every exported trace from the JSON alone. Every `[n]` must resolve to an included passage, every sentence must cite something, and every number in a sentence must be supported by a passage that sentence cites. With `--network` it requests each distinct `sec_url` once, at most two a second, and requires 200. It also runs the answer-time quote, identity and suffix checks, with the same code. The last run covered 12 traces with 0 problems, and 19 of 19 distinct URLs returned 200.

The number match compares quantity and scale, and the answer-time check uses the same code. `12%` against `12 million` fails. `$12 billion` against `$12 million` fails. `$1.2 billion` matches `$1,200 million`, and `40bps` matches `40 basis points`. Basis points, percentage points and percent are different kinds, so one never matches another. Digits at the same scale must be equal, so `5.5` doesn't match `5.50`. A figure that a cited passage holds only without a scale, such as a table cell under an "in millions" header, passes as "scale unverified" and goes on the audit list. Re-checking the 12 exported answers under these rules dropped none. 8 of their 144 numbers are scale unverified.

Without `--network` the list goes to the gitignored `data/audit_list.local.md`; only a `--network` run rewrites the committed `exports/audit_list.md`, so the committed copy always has the link check in it. `exports/audit_list.md` lists what a person still has to check: the 8 scale-unverified numbers, each with its source snippet and SEC link; the one fiscal-year label that no cited filing has; and 5 "claims to judge". These are sentences that say what a filing doesn't say, says again or says of all of something ("no specific incident", "absent from", "the same two", "all three", "entirely new"). A number check can't see them, and they are not failures. The 136 verified numbers and the 83 labels that match a cited filing are counted, not listed. The word list is narrow on purpose, so it misses claims phrased in other ways. The list helps with reading the traces. It is not a count of what is wrong.

### v2 evals (designed, not run)

The four v2 evals are in `evals/v2/`, with the input formats in each module's docstring and tests in `tests/test_v2_evals.py`. They need labelling time I have not spent, so there are no results.

| eval | metric | sample | proposed pass |
|---|---|---|---|
| retrieval recall | Recall@5 and @20 over `(ticker, fiscal_year, section, must_contain_phrase)` targets | 30 questions, 1 to 3 targets each | @20 >= 0.85, @5 >= 0.65 |
| citation faithfulness | share of cited sentences supported by their passages; an LLM judge against a 25-sentence human check, with Cohen's kappa | every cited sentence in the 12 published traces | faithfulness >= 0.95, kappa >= 0.80 |
| 50-claim extraction audit | precision and a five-way failure breakdown | 50 claims across the 9 claim types | precision >= 0.88, hallucinated < 0.02 |
| ingest cost | median and p90 cost and seconds per filing, by stage | the 61 filings | none, it is a measurement |

## Limits

- **Fidelity coverage.** The 99.9% above covers 682 of 824 sections. Sections the private pipeline split, and sections the detector did not find, are not in it.
- **Sentiment truncation.** FinBERT reads 512 tokens, and 747 of the 2,898 chunks scored (25.8%) were longer and were truncated, so their sentiment reflects the first 512 tokens only. 4 of the 96 signal rows have no sentiment or tone value.
- **Signals are shadow only.** No question, answer or trace depends on them.
- **Same-day amendment.** AMD's FY2025 10-K and 10-K/A were filed the same day (2026-02-04), so a date-granular point-in-time query can't show the superseded claims. The trace shows them from their close-out record instead. 17 claims were closed out.
- **Intel has no detected MD&A.** No Item 7 section is found in any of Intel's five 10-Ks. Its MD&A is folded into other items, and the questions avoid Intel's Item 7.
- **CVX, JPM and XOM MD&A is incorporated by reference.** Their Item 7 is a pointer of roughly 220 to 400 characters (XOM FY2021 is longer), not analysis. The questions avoid Item 7 for those three.
- **No Item 1C before FY2023.** The cybersecurity item only exists for fiscal years ending on or after 15 December 2023, so the FY2021 and FY2022 filings (and NVDA's and MSFT's FY2023) have none. The detector also finds no Item 1C for Intel or JPM in any year.
- **Dropped traces.** 13 questions, 12 exported. The UNH trace failed the human audit and is listed in `exports/failed.json`. Rejected model attempts are kept in `data/rejected/`, which is not committed.
- **Sample size.** 12 companies, 61 filings, 12 traces, one run and one model each. They show how it works; they don't evaluate it.
- **Extractions are not free to reproduce.** The committed extractions come from my private pipeline. `make sample-refresh` re-extracts them with your key and at your cost, and a new model run will not reproduce them exactly.
- **Scale of table figures.** A number that a cited passage holds only as a bare table cell is checked by a person, not by code. The audit list names each one.
