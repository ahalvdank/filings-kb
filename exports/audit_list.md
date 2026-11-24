# Audit list

12 traces. 144 numbers: 136 verified, 8 scale unverified, 0 not found. 84 fiscal-year labels: 83 match a cited filing, 1 to check. 5 claims to judge.

How to read it. Tick a box when you have checked it against the passage.
- **Verified numbers** (same quantity and scale in a cited passage: `$1.2 billion` matches `$1,200 million`, `12%` does not match `12 million`) are counted, not listed.
- **Numbers to check:** figures a cited passage holds only without a scale. They pass the automatic check; the scale is yours to confirm.
- **Fiscal-year labels** that match the fiscal year of a cited filing are counted. NVDA's fiscal year ends in January, so `FY2025` is the year ended Jan 2025.
- **Claims to judge:** sentences that say what a filing does not say, says again or says of all of something (`no specific`, `absent`, `the same two`, `all three`, ...), found with a word list. No number check can see these. They are not failures: read the cited passages and decide whether the sentence is supported.
- **SEC links:** 19 of 19 distinct `sec_url` values returned 200.

## Numbers to check (8)

Each figure appears in a cited passage only without its scale (usually a table cell under an "in millions" header). Check the scale in the filing.

### 6. Intel capital returns and external funding, FY2024 to FY2025 (`intc-capital-return-and-external-funding-fy24-fy25`)

**p2u2** However, the FY2024 filing frames these arrangements explicitly under a "Smart Capital approach to capital spending," language that the FY2025 filing drops, and the balance sheet shows partner contributions to non-controlling interests surging from $1,702 million in 2024 to $5,108 million in 2025. [4][5]

- [ ] `$1,702 million` — scale unverified — [5] INTC FY2025 10-K · Item 8 Financial Statements and Supplementary Data · [sec.gov](https://www.sec.gov/Archives/edgar/data/50863/000005086326000011/intc-20251227.htm)
  > …— 2,359 1,838 178 4,375 Partner contributions— 1,702 — — 1,702 Partner distributions(43)— — — (43)…

- [ ] `$5,108 million` — scale unverified — [5] INTC FY2025 10-K · Item 8 Financial Statements and Supplementary Data · [sec.gov](https://www.sec.gov/Archives/edgar/data/50863/000005086326000011/intc-20251227.htm)
  > …61 3,888 1,672 141 5,762 Partner contributions— 5,108 — — 5,108 Partner distributions(217)— — — (217)…

### 8. JPMorgan capital returns and regulatory-capital posture, FY2024 to FY2025 (`jpm-capital-return-and-regulatory-capital-fy24-fy25`)

**p1u2** During 2024, the Firm repurchased 91,719,226 shares at an average price of $205.43 for a total of $18,841 million, leaving $19,326 million remaining at year-end. [2]

- [ ] `$18,841 million` — scale unverified — [2] JPM FY2024 10-K · Item 5 Market for Registrant's Common Equity · [sec.gov](https://www.sec.gov/Archives/edgar/data/19617/000001961725000270/jpm-20241231.htm)
  > ….37 4,313 19,326 Full year 91,719,226 $205.43 $18,841 $19,326 (a)Excludes excise tax and commissions.…

- [ ] `$19,326 million` — scale unverified — [2] JPM FY2024 10-K · Item 5 Market for Registrant's Common Equity · [sec.gov](https://www.sec.gov/Archives/edgar/data/19617/000001961725000270/jpm-20241231.htm)
  > ….03 1,240 21,054 December7,170,130 241.10 1,728 19,326 Fourth quarter18,485,627 233.37 4,313 19,326…

**p1u4** During 2025, the Firm repurchased shares at an average price of $276.55 for a total of $31,640 million, with $33,423 million remaining under the repurchase program at year-end. [4]

- [ ] `$31,640 million` — scale unverified — [4] JPM FY2025 10-K · Item 5 Market for Registrant's Common Equity · [sec.gov](https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/jpm-20251231.htm)
  > …1 8,262 33,423 (c) Full year114,408,296 $276.55 $31,640 $33,423 (c) (a)Excludes excise tax and commissio…

- [ ] `$33,423 million` — scale unverified — [4] JPM FY2025 10-K · Item 5 Market for Registrant's Common Equity · [sec.gov](https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/jpm-20251231.htm)
  > ….82 2,439 36,256 December8,928,561 317.28 2,833 33,423 (c) Fourth quarter26,666,700 309.81 8,262 33,423…

### 9. Meta AI investment ambition against monetization risk, FY2025 (`meta-ai-capex-vs-monetization-risk-fy25`)

**p4u3** Free cash flow in 2025 was $43,585 million, compared to $52,103 million in 2024, as the surge in property and equipment purchases offset the increase in operating cash flows, illustrating how the scale of infrastructure investment is already constraining the residual cash available even before the larger 2026 program begins. [11]

- [ ] `$43,585 million` — scale unverified — [11] META FY2025 10-K · Item 7 Management's Discussion and Analysis · [sec.gov](https://www.sec.gov/Archives/edgar/data/1326801/000162828026003942/meta-20251231.htm)
  > …nance leases(2,524)(1,969)(1,058) Free cash flow$43,585 $52,103 $43,010 78 Material Cash Requirements…

- [ ] `$52,103 million` — scale unverified — [11] META FY2025 10-K · Item 7 Management's Discussion and Analysis · [sec.gov](https://www.sec.gov/Archives/edgar/data/1326801/000162828026003942/meta-20251231.htm)
  > …ases(2,524)(1,969)(1,058) Free cash flow$43,585 $52,103 $43,010 78 Material Cash Requirements We cur…

## Fiscal-year labels to check (1)

- [ ] `FY2022` in `nvda-export-control-risk-fy22-fy26` p2u3 — cites NVDA FY2026 — **no cited filing has this fiscal year**

## Claims to judge (5)

### 4. Boeing liquidity, debt and equity actions, FY2024 to FY2025 (`ba-liquidity-and-debt-fy24-fy25`)

- [ ] **p4u2** (`all/every + count`) By the end of 2025, **all three** major rating agencies had revised Boeing's outlooks to stable from negative: Fitch affirmed the BBB- credit rating in June 2025, S&P affirmed the BBB- credit rating in October 2025, and Moody's affirmed the Baa3 credit rating in December 2025. [4]

  <details><summary>cited passages [4]</summary>

  **[4]** BA FY2025 10-K · Item 7 Management's Discussion and Analysis · [sec.gov](https://www.sec.gov/Archives/edgar/data/12927/000162828026004357/ba-20251231.htm)

  > At December 31, 2025, we had $10.9 billion of cash, $18.5 billion of short-term investments, and $10.0 billion of unused borrowing capacity on revolving credit line agreements. We expect to pay $8.5 billion in short-term and long-term debt due within the next 12 months from our available cash balance. In August 2025, we entered into a $3.0 billion, 364-day revolving credit agreement expiring in August 2026. This facility replaced the $3.0 billion, three-year revolving credit agreement which was scheduled to terminate in August 2025. The 364-day credit facility has a one-year term out option which allows us to extend the maturity of any borrowings until August 2027. Our legacy $3.0 billion, five-year revolving credit agreement expiring in August 2028 and $4.0 billion, five-year revolving credit agreement expiring in May 2029 each remain in effect. We anticipate that these credit lines will primarily serve as back-up liquidity to support our general corporate borrowing needs.
  >
  > For discussion related to the Spirit Acquisition and Digital Aviation Solutions Divestiture, see Note 2 and Note 3 to our Consolidated Financial Statements.
  >
  > We currently maintain investment grade credit ratings across all three credit rating agencies. In June 2025, Fitch affirmed the BBB- credit rating and revised the outlook to stable from negative. In October 2025, S&P affirmed the BBB- credit rating and revised the outlook to stable from negative. In December 2025, Moody’s affirmed the Baa3 credit rating and revised the outlook to stable from negative.
  >
  > We may, from time to time, purchase, redeem or retire any of our outstanding debt securities in open market or privately negotiated transactions, by tender offer or otherwise, after consideration of market conditions, our liquidity needs and other factors.

  </details>

### 6. Intel capital returns and external funding, FY2024 to FY2025 (`intc-capital-return-and-external-funding-fy24-fy25`)

- [ ] **p2u1** (`same + count`) On partner co-investments, both the FY2024 and FY2025 filings disclose the **same two** Semiconductor Co-Investment Program arrangements—the 2022 joint investment with Brookfield in Arizona manufacturing and the 2024 joint investment with Apollo related to Fab 34 in Ireland—and both note the $755 million charge recognized in the fourth quarter of 2024 for penalties related to Ireland SCIP construction delays. [3][4]

  <details><summary>cited passages [3][4]</summary>

  **[3]** INTC FY2025 10-K · Item 1A Risk Factors · [sec.gov](https://www.sec.gov/Archives/edgar/data/50863/000005086326000011/intc-20251227.htm)

  > Our alternative financing arrangements and pursuit of government grants involve risks and may not be successful.
  >
  > To support our capital investments, we have pursued alternative financing arrangements, such as our 2022 joint investment with Brookfield in the manufacturing expansion of our Arizona campus, and our 2024 joint investment with Apollo related to Fab 34 in Ireland, and may enter into similar arrangements in the future. These transactions may fail to advance our business strategy, may include unfavorable pricing or other terms such as penalties should key metrics not be attained as prescribed by our agreements, and may fail to achieve their anticipated benefits. Both arrangements include commitments we may not be able to satisfy, including commitments relating to construction and/or wafer demand or purchase, in which case we may be required to make additional payments to our partners. For example, in the fourth quarter of 2024, we recognized a $755 million charge related to penalties we expect to pay in connection with Ireland SCIP for construction delays we decided to make as we reduced our near-term capacity requirements. Further, both arrangements are expected to significantly and increasingly impact our net income (loss) attributable to Intel and earnings (loss) per share attributable to Intel in future periods as wafer production volumes increase at our expanded Arizona campus and at Fab 34 in Ireland. Our partners may also fail to satisfy financial or other obligations on which we rely and we may fail to resolve any potential disputes. Any of these risks, including our ability to effectuate any additional transactions at all, could have a material adverse effect on our business, results of operations, financial condition or cash flows, which may limit our ability to raise sufficient capital for our required investments.
  >
  > In addition, we have applied for, received and expect to receive additional grants and incentives from domestic and foreign local, regional and national governments. Legislation in the U.S. and EU has been adopted to provide government funding for semiconductor manufacturing expansions in those regions. However, any amounts we may receive under any agreements enabled by such legislation may not be sufficient in amount or timeliness to support our capital investment plans and offset the higher costs of operations in many of the locations of our facilities as compared to those of many of our competitors, we may be unable to comply with the requirements and limitations of any such grants and incentives, or such agreements may contain restrictions that limit our flexibility to pursue changes in business strategy or transactions that may enhance stockholder value.

  **[4]** INTC FY2024 10-K · Item 1A Risk Factors · [sec.gov](https://www.sec.gov/Archives/edgar/data/50863/000005086325000009/intc-20241228.htm)

  > We may also not be successful in developing a competitive foundry business for external customers with respect to either leading-edge or mature process nodes, which would make it difficult for us to realize a favorable return on our investments in process technology and manufacturing capacity investments. To be successful, we need to cultivate relationships with customers and partners in these market segments and continue to improve our offerings. Despite our ongoing efforts, there is no guarantee that we will achieve or maintain market demand or acceptance for our products and services in these various market segments or realize an adequate return on our investments, which could lead to impairment of assets and restructuring charges, as well as opportunity costs. 
  >
  > Our Smart Capital approach to capital spending, alternative financing arrangements, and pursuit of government grants involves risks and may not be successful.
  >
  > As we pursue our strategy, we have utilized our Smart Capital approach to capital spending in an effort to appropriately time and scale our capital investments. To support our capital investments, we have pursued alternative financing arrangements, such as our 2022 joint investment with Brookfield in the manufacturing expansion of our Arizona campus, and our 2024 joint investment with Apollo related to Fab 34 in Ireland, and may enter into similar arrangements in the future. These transactions may fail to advance our business strategy, may include unfavorable pricing or other terms such as penalties should key metrics not be attained as prescribed by our agreements, and may fail to achieve their anticipated benefits. Both arrangements include commitments we may not be able to satisfy, including commitments relating to construction and/or wafer demand or purchase, in which case we may be required to make additional payments to our partners. For example, in the fourth quarter of 2024, we recognized a $755 million charge related to penalties we expect to pay in connection with Ireland SCIP for construction delays we decided to make as we reduced our near-term capacity requirements. Further, both arrangements are expected to significantly and increasingly impact our net income (loss) attributable to Intel and earnings (loss) per share attributable to Intel in future periods as wafer production volumes increase at our expanded Arizona campus and at Fab 34 in Ireland. Our partners may also fail to satisfy financial or other obligations on which we rely and we may fail to resolve any potential disputes. Any of these risks, including our ability to effectuate any additional transactions at all, could have a material adverse effect on our business, results of operations, financial condition, or cash flows, which may limit our ability to raise sufficient capital for our required investments.

  </details>

- [ ] **p4u1** (`entirely new`) The FY2025 10-K adds an **entirely new** category of risk around the U.S. government becoming one of Intel's largest stockholders, warning that the equity issuances are dilutive to existing stockholders, that the U.S. government's equity position reduces the voting and other governance rights of other stockholders, and that the transactions may limit Intel's ability to pursue strategic transactions beneficial to stockholders. [9]

  <details><summary>cited passages [9]</summary>

  **[9]** INTC FY2025 10-K · Item 1A Risk Factors · [sec.gov](https://www.sec.gov/Archives/edgar/data/50863/000005086326000011/intc-20251227.htm)

  > It is uncertain whether the transactions may cause other government entities to seek to convert their existing grant arrangements with us into equity investments or be unwilling to support us with future grants, either of which could limit our access to capital, increase our cost of capital or increase our future operating costs.
  >
  > ▪The transactions are dilutive to existing stockholders. The issuance of shares of common stock to the U.S. government at a discount to the current market price is dilutive to existing stockholders, and stockholders may suffer significant additional dilution if the conditions to the warrant are triggered and the warrant is exercised. 
  >
  > ▪The U.S. government’s equity position reduces the voting and other governance rights of our other stockholders and may limit potential future transactions that may be beneficial to such stockholders. The transactions resulted in the U.S. government becoming one of our largest stockholders. The U.S. government’s interests in us may not be the same as those of other stockholders. The U.S. government has agreed to vote its shares of common stock as recommended by our board of directors, subject to applicable law and exceptions to protect the U.S. government’s interests. This will reduce the voting influence of other stockholders with respect to the selection of our directors and proposals voted on by our stockholders. The existence of a significant U.S. government equity interest, the voting of such shares either as directed by our board of directors or the U.S. government, and the U.S. government’s substantial additional powers with respect to the laws and regulations impacting us, may substantially limit our ability to pursue potential future strategic transactions that may be beneficial to stockholders, including by potentially limiting the willingness of other third parties to engage in such potential strategic transactions with us.
  >
  > Risk Factors
  >
  > 41
  >
  > ▪Our non-U.S. business may be adversely impacted by the U.S. government being a significant stockholder. Sales outside the U.S. accounted for 70% of our revenue for 2025. Having the U.S. government as a significant stockholder could subject us to additional regulations, obligations or restrictions, such as foreign subsidy laws or otherwise, in other countries.
  >
  > Any of the foregoing could have a material adverse effect on our revenue, operations, financial position, cash flows, access to financing, cost structure, competitiveness, reputation, profitability and prospects and could exacerbate other risks discussed in this Form 10-K.
  >
  > Changes in product demand and margins can adversely affect our financial results.

  </details>

### 11. NVIDIA export-control and customer-concentration risk, FY2022 to FY2026 (`nvda-export-control-risk-fy22-fy26`)

- [ ] **p2u3** (`entirely new`, `absent`) The FY2026 disclosures also added **entirely new** concerns **absent** from FY2022, including China's government publicly questioning whether H20 products "have built-in vulnerabilities, discouraging customers from purchasing our products," and risks from customers pivoting to open-source foundation models originating in China such as "DeepSeek, Qwen, or KIMMI." [3]

  <details><summary>cited passages [3]</summary>

  **[3]** NVDA FY2026 10-K · Item 1A Risk Factors · [sec.gov](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)

  > Even if not enacted into binding legislation, draft bills have impacted and may in the future negatively impact our business. For example, following U.S. legislative proposals calling for mandatory features in our chips, China’s government publicly questioned whether our H20 products have built-in vulnerabilities, discouraging customers from purchasing our products. We provided a public response explaining that our GPUs, including H20, do not include such built-in vulnerabilities, and will respond to any follow-up questions we receive.
  >
  > Open-source foundation models are rapidly growing in popularity with developers worldwide. Any regulatory control or other restriction that limits our ability to provide products and services that support third-party applications and models, including applications built on foundation models originating in China such as DeepSeek, Qwen, or KIMMI, could have a material impact on our business, operating results, and financial condition.
  >
  > The USG already imposed export controls restricting certain gaming GPUs, and if the USG expands such controls to restrict additional gaming products, it may disrupt a significant portion of our supply and distribution chain and 
  >
  > 27
  >
  >
  > negatively impact sales of such products to markets outside China, including the U.S. and Europe. In addition, as the performance of the gaming GPUs increases over time, export controls may have a greater impact on our ability to compete in markets subject to those controls. Export controls may disrupt our supply and distribution chain for a substantial portion of our products, which are warehoused in and distributed from Hong Kong.
  >
  > Export controls restricting our ability to sell data center GPUs may also negatively impact demand for our networking products used in servers containing our GPUs. The USG may also impose export controls on our networking products, such as high-speed network interconnects, to limit the ability of downstream parties to create large clusters for frontier model training.
  >
  > Export controls have and are likely in the future to have a disproportionate impact on NVIDIA and may disadvantage us against certain of our competitors that sell chips that are outside the scope of such control. Export controls have already and may in the future encourage customers outside China and other impacted regions to “design-out” certain U.S. semiconductors from their products to reduce the compliance burden and risk, and to ensure that they are able to serve markets worldwide. Export controls have already encouraged and may in the future encourage overseas governments to request that our customers purchase from our competitors rather than NVIDIA or other U.S. firms, harming our business, market position, and financial results.

  </details>

- [ ] **p4u2** (`entirely new`) The FY2026 filing also identified **entirely new** legislative risks — such as the GAIN AI Act, which the FY2026 10-K states "would restrict the Trump Administration's ability to adapt the Biden Administration's export control rules" — illustrating how the regulatory landscape had grown considerably more complex over the intervening years. [6]

  <details><summary>cited passages [6]</summary>

  **[6]** NVDA FY2026 10-K · Item 1A Risk Factors · [sec.gov](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)

  > The GAIN AI Act would restrict the Trump Administration’s ability to adapt the Biden Administration’s export control rules, and could also allow private U.S. persons to review and overturn licensing and foreign policy decisions made by the Trump Administration.
  >
  > Our competitive position has been harmed by export controls, and our competitive position and future results will be further harmed, over the long term, if the restrictions remain in place or are expanded in geographic, customer, or product scope, if customers purchase product from competitors, if customers develop their own internal solution, if we are unable to provide contractual warranty or other extended service obligations, if the USG does not grant licenses in a timely manner or denies licenses to significant customers or if we incur significant transition costs. The licensing process may not be resolved before significant business opportunities evaporate. Even if the USG grants any requested licenses, the licenses have already and may in the future be temporary, impose burdensome conditions regarding the installation, maintenance, and use of such products, or include financial or economic requirements that we or our customers or end users cannot or choose not to fulfill. The licensing requirements have already and may in the future benefit certain of our competitors, as the licensing process will make our pre-sale and post-sale technical support efforts more cumbersome and less certain and encourage customers in China, the Middle East, and other regions to pursue alternatives to our products, including semiconductor suppliers based in China, Europe, and Israel.
  >
  > Given the increasing strategic importance of AI and rising geopolitical tensions, the USG has changed and may again change the export control rules at any time and further subject a wider range of our products to export restrictions and licensing requirements, negatively impacting our business and financial results. In the event of such change, we may be unable to sell our inventory of such products and may be unable to develop replacement products not subject to the licensing requirements.
  >
  > For example, the USG already imposed license conditions that limit the ability of foreign firms to create and offer as a service large-scale GPU clusters, such as imposing license conditions on the use of products to be exported to certain countries, and may impose additional conditions such as requiring chip tracking and throttling mechanisms that could disable or impair GPUs if certain events, including unauthorized system configuration, use, or location, are detected. Such government mandates in chip designs could introduce system vulnerabilities and expose us to significant risk and potential liability, negatively impact demand for our products, and could have a material impact on our business, operating results, and financial condition.

  </details>

## Traces

1. **The AI value chain, chip suppliers against hyperscale buyers** (`ai-value-chain-suppliers-vs-buyers`): 3 numbers (3 verified), 11 fiscal-year labels (11 match), 0 claims to judge, 0 problems.
2. **What AMD changed in its amended FY2025 10-K** (`amd-fy2025-10ka-amendment`): 18 numbers (18 verified), 2 fiscal-year labels (2 match), 0 claims to judge, 0 problems.
3. **Boeing delivery recovery against regulatory and supply constraints, FY2025** (`ba-delivery-outlook-vs-constraints-fy25`): 32 numbers (32 verified), 1 fiscal-year labels (1 match), 0 claims to judge, 0 problems.
4. **Boeing liquidity, debt and equity actions, FY2024 to FY2025** (`ba-liquidity-and-debt-fy24-fy25`): 32 numbers (32 verified), 2 fiscal-year labels (2 match), 1 claims to judge, 0 problems.
5. **Chevron lower-carbon commitments against hydrocarbon investment, FY2025** (`cvx-lower-carbon-vs-upstream-investment-fy25`): 4 numbers (4 verified), 1 fiscal-year labels (1 match), 0 claims to judge, 0 problems.
6. **Intel capital returns and external funding, FY2024 to FY2025** (`intc-capital-return-and-external-funding-fy24-fy25`): 8 numbers (6 verified), 10 fiscal-year labels (10 match), 2 claims to judge, 0 problems.
7. **Intel foundry execution and capital-intensity risk, FY2024 to FY2025** (`intc-foundry-execution-risk-fy24-fy25`): 7 numbers (7 verified), 12 fiscal-year labels (12 match), 0 claims to judge, 0 problems.
8. **JPMorgan capital returns and regulatory-capital posture, FY2024 to FY2025** (`jpm-capital-return-and-regulatory-capital-fy24-fy25`): 24 numbers (20 verified), 14 fiscal-year labels (14 match), 0 claims to judge, 0 problems.
9. **Meta AI investment ambition against monetization risk, FY2025** (`meta-ai-capex-vs-monetization-risk-fy25`): 8 numbers (6 verified), 1 fiscal-year labels (1 match), 0 claims to judge, 0 problems.
10. **NVIDIA, AMD and Intel on foundry and packaging dependency** (`nvda-amd-intc-foundry-dependency-fy25`): 3 numbers (3 verified), 5 fiscal-year labels (5 match), 0 claims to judge, 0 problems.
11. **NVIDIA export-control and customer-concentration risk, FY2022 to FY2026** (`nvda-export-control-risk-fy22-fy26`): 3 numbers (3 verified), 16 fiscal-year labels (15 match), 2 claims to judge, 0 problems.
12. **ExxonMobil against Chevron on energy-transition risk** (`xom-vs-cvx-energy-transition-risk-fy25`): 2 numbers (2 verified), 9 fiscal-year labels (9 match), 0 claims to judge, 0 problems.
