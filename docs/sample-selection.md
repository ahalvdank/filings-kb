# Sample selection

Pinned 2026-11-22 (widened from 8 companies x 2 years to 12 companies x 5 years) from the private KB. Rule: for each ticker, the five newest fiscal years that have a 10-K manifest (`filings/<TICKER>/*_10-K_<accession>.md`; accession-less duplicates ignored) and cached v3 / `claude-sonnet-4-6` extractions. Fiscal years come from the private knowledge graph (`entities` of type `Filing`, `attributes.fiscal_year`). NVDA and MSFT have non-calendar fiscal years and span FY2022 to FY2026; the other ten span FY2021 to FY2025. Every ticker matched this expectation, and no filing was skipped for a missing manifest or cache.

## The 60 filings

| Ticker | Fiscal year | Accession | v3 / claude-sonnet-4-6 cached files | Excluded cache files |
|---|---|---|---|---|
| NVDA | FY2026 | 0001045810-26-000021 | 24 | v3/claude-sonnet-5 x2 |
| NVDA | FY2025 | 0001045810-25-000023 | 24 | v1/claude-sonnet-4-6 x3 |
| NVDA | FY2024 | 0001045810-24-000029 | 24 | none |
| NVDA | FY2023 | 0001045810-23-000017 | 23 | v3/claude-sonnet-5 x1 |
| NVDA | FY2022 | 0001045810-22-000036 | 22 | v3/claude-sonnet-5 x1 |
| AMD | FY2025 | 0000002488-26-000018 | 16 | v2/claude-sonnet-4-6 x4, v3/claude-sonnet-5 x1 |
| AMD | FY2024 | 0000002488-25-000012 | 16 | v2/claude-sonnet-4-6 x4 |
| AMD | FY2023 | 0000002488-24-000012 | 16 | v2/claude-sonnet-4-6 x4 |
| AMD | FY2022 | 0000002488-23-000047 | 16 | v2/claude-sonnet-4-6 x8 |
| AMD | FY2021 | 0000002488-22-000016 | 15 | v2/claude-sonnet-4-6 x7 |
| INTC | FY2025 | 0000050863-26-000011 | 20 | none |
| INTC | FY2024 | 0000050863-25-000009 | 20 | none |
| INTC | FY2023 | 0000050863-24-000010 | 21 | none |
| INTC | FY2022 | 0000050863-23-000006 | 17 | none |
| INTC | FY2021 | 0000050863-22-000007 | 14 | none |
| BA | FY2025 | 0001628280-26-004357 | 18 | none |
| BA | FY2024 | 0000012927-25-000015 | 18 | none |
| BA | FY2023 | 0000012927-24-000010 | 17 | none |
| BA | FY2022 | 0000012927-23-000007 | 17 | v3/claude-sonnet-5 x2 |
| BA | FY2021 | 0000012927-22-000010 | 17 | v3/claude-sonnet-5 x2 |
| UNH | FY2025 | 0000731766-26-000062 | 16 | v3/claude-sonnet-5 x2 |
| UNH | FY2024 | 0000731766-25-000063 | 15 | v1/claude-sonnet-4-6 x2, v3/claude-sonnet-5 x1 |
| UNH | FY2023 | 0000731766-24-000081 | 15 | none |
| UNH | FY2022 | 0000731766-23-000008 | 14 | none |
| UNH | FY2021 | 0000731766-22-000008 | 14 | v2/claude-sonnet-4-6 x6 |
| META | FY2025 | 0001628280-26-003942 | 18 | none |
| META | FY2024 | 0001326801-25-000017 | 18 | v1/claude-sonnet-4-6 x2 |
| META | FY2023 | 0001326801-24-000012 | 18 | none |
| META | FY2022 | 0001326801-23-000013 | 17 | none |
| META | FY2021 | 0001326801-22-000018 | 17 | none |
| JPM | FY2025 | 0001628280-26-008131 | 32 | none |
| JPM | FY2024 | 0000019617-25-000270 | 33 | v1/claude-sonnet-4-6 x2 |
| JPM | FY2023 | 0000019617-24-000225 | 33 | none |
| JPM | FY2022 | 0000019617-23-000231 | 31 | none |
| JPM | FY2021 | 0000019617-22-000272 | 31 | none |
| XOM | FY2025 | 0000034088-26-000045 | 14 | none |
| XOM | FY2024 | 0000034088-25-000010 | 14 | none |
| XOM | FY2023 | 0000034088-24-000018 | 15 | none |
| XOM | FY2022 | 0000034088-23-000020 | 14 | none |
| XOM | FY2021 | 0000034088-22-000011 | 14 | none |
| MSFT | FY2026 | 0001193125-26-323660 | 16 | none |
| MSFT | FY2025 | 0000950170-25-100235 | 16 | v1/claude-sonnet-4-6 x2 |
| MSFT | FY2024 | 0000950170-24-087843 | 16 | none |
| MSFT | FY2023 | 0000950170-23-035122 | 15 | v3/claude-sonnet-5 x2 |
| MSFT | FY2022 | 0001564590-22-026876 | 15 | none |
| GOOGL | FY2025 | 0001652044-26-000018 | 17 | v3/claude-sonnet-5 x1 |
| GOOGL | FY2024 | 0001652044-25-000014 | 16 | v1/claude-sonnet-4-6 x1 |
| GOOGL | FY2023 | 0001652044-24-000022 | 16 | none |
| GOOGL | FY2022 | 0001652044-23-000016 | 15 | v3/claude-sonnet-5 x1 |
| GOOGL | FY2021 | 0001652044-22-000019 | 15 | none |
| AMZN | FY2025 | 0001018724-26-000004 | 16 | v3/claude-sonnet-5 x1 |
| AMZN | FY2024 | 0001018724-25-000004 | 16 | v1/claude-sonnet-4-6 x1, v3/claude-sonnet-5 x1 |
| AMZN | FY2023 | 0001018724-24-000008 | 14 | v3/claude-sonnet-5 x1 |
| AMZN | FY2022 | 0001018724-23-000004 | 15 | v2/claude-sonnet-4-6 x5 |
| AMZN | FY2021 | 0001018724-22-000005 | 14 | v2/claude-sonnet-4-6 x5, v3/claude-sonnet-5 x1 |
| CVX | FY2025 | 0000093410-26-000078 | 15 | v2/claude-sonnet-4-6 x2 |
| CVX | FY2024 | 0000093410-25-000009 | 15 | v2/claude-sonnet-4-6 x2 |
| CVX | FY2023 | 0000093410-24-000013 | 15 | v2/claude-sonnet-4-6 x2 |
| CVX | FY2022 | 0000093410-23-000009 | 14 | none |
| CVX | FY2021 | 0000093410-22-000019 | 14 | none |

## The 10-K/A (row 61)

| Ticker | Fiscal year | Accession | Form | Amends | Committed extractions |
|---|---|---|---|---|---|
| AMD | FY2025 | 0000002488-26-000021 | 10-K/A | 0000002488-26-000018 | `Item_7_MDA`, `Item_15` |

Its `Item_7_MDA` and `Item_15` supersede the original FY2025 10-K's: ingest closes out the original's claims for those two items (bitemporal close-out). Both filings are dated 2026-02-04.

Every accession has a v3 / claude-sonnet-4-6 cache. Files with any other prompt version or model are excluded, never substituted.

## Extraction allowlist (`extract_items`)

The distinct `input.section_type` values across the cached v3 / claude-sonnet-4-6 extractions for these 60 accessions (22 values, private token names). Widening the sample added none, so it is the same list as before:

`Item_1`, `Item_1A`, `Item_1B`, `Item_1C`, `Item_2`, `Item_3`, `Item_4`, `Item_5`, `Item_6`, `Item_7_MDA`, `Item_7A`, `Item_8`, `Item_9`, `Item_9A`, `Item_9B`, `Item_9C`, `Item_10`, `Item_11`, `Item_12`, `Item_13`, `Item_14`, `Item_15`

## Coverage gaps

Parity should compare per accession rather than assume every section exists everywhere.

- `Item_1C` (cybersecurity) has no extraction for 27 of the 60 filings: the five INTC filings (see below) and 22 others, namely the FY2021 and FY2022 filings of AMD, AMZN, BA, CVX, GOOGL, JPM, META, UNH and XOM, NVDA FY2022 and FY2023, and MSFT FY2022 and FY2023. The item only exists in 10-Ks for fiscal years ending on or after 2023-12-15, so this is expected. Every other non-INTC filing has 14 or more sections.
- INTC has no `Item_7_MDA` extraction in any of its five filings, and none for `Item_1C`, `Item_3`, `Item_9`, `Item_9A`, `Item_9B` or `Item_13` either (6 to 7 distinct sections per filing; FY2025 also lacks `Item_5`). Its cached files carry several text hashes per section, so file count is not section count.
- `Item_1B`, `Item_4`, `Item_6`, `Item_9C` and `Item_10` to `Item_12`, `Item_14` appear only for NVDA. Items 10 to 14 are usually incorporated by reference to the proxy statement, so expect short stubs.
