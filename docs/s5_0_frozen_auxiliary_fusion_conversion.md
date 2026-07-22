# S5-0 Frozen CF+SASRec Auxiliary Fusion Conversion Gate

## Scope

CPU-only analysis. No GPU, SFT, generation, SID reconstruction, Text-SID fusion, p3 expansion, or test split evaluation was run. The script reads frozen S4 formal valid artifacts and existing Stage 7 P2 split/ranker evidence only.

## Compatibility Matrix

| component | direct compatible | source hardcoded | allowed use | reason |
|---|---:|---:|---|---|
| P2-4 frozen history-aware ranker | False | True | explicit alias or source-independent projection only | feature schema contains text_* and cf_* fields trained for Text+CF, not CF+SASRec |
| sasrec_as_text_auxiliary alias | True | True | diagnostic frozen-ranker conversion evaluation | alias preserves item/provenance semantics but changes source-feature interpretation; report must not call this direct compatibility |
| source_independent_projection | True | False | diagnostic frozen-ranker conversion evaluation | removes source identity dependence but is a projection of a frozen model, not a retrained fair gate |

## Selection Protocol

- valid_select source: `valid_select`
- valid_report source: `valid_fit`
- valid_select rows: `1360`
- valid_report rows: `3172`
- No new random split was created. There is no separate historical valid_report artifact; S5 uses the existing non-select `valid_fit` subset as the report split and records this limitation.

## Selected Config

`source_aware_rrf_lam0.75_bonus0.01` selected on valid_select by HR@20, then NDCG@20, then lower CF hit loss and smaller candidate pool.

## Valid Report Metrics

| strategy | HR@10 | HR@20 | HR@50 | NDCG@20 | MRR | avg size | SASRec-only recoveries@20 | CF lost@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only | 0.081337 | 0.102144 | 0.140921 | 0.066432 | 0.057585 | 50.00 | 0 | 0 |
| selected fixed/RRF | 0.085120 | 0.105927 | 0.139975 | 0.066808 | 0.057321 | 95.74 | 19 | 10 |
| best frozen ranker diagnostic | 0.093947 | 0.117907 | 0.150063 | 0.072088 | 0.060310 | 95.74 | 31 | 8 |

## Gate

Decision: **FROZEN_RANKER_CONVERTS**

The source-independent frozen-ranker projection improves valid_report HR@20/NDCG@20 over CF-only, preserves at least 97% of CF@20 hits, and recovers part of the SASRec-only oracle space.

The frozen ranker cannot be treated as directly schema-compatible with CF+SASRec because it was trained with Text+CF source features. Its S5 result is diagnostic under an explicit alias/projection manifest.
