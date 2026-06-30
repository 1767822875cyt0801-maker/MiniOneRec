# Stage 8 Completion Audit

Audit date: 2026-06-25  
Workspace: `/home/dell/projects/MiniOneRec`

## Scope

This audit checks the current local workspace for the completed CS-SID pipeline:

1. SID/mapping infrastructure
2. SID diagnostics and evaluation
3. CS embedding
4. New SID generation and CSV rewrite
5. SID-only SFT and evaluation artifacts
6. Candidate expansion and lightweight rerank

Important note: the local workspace does not currently contain the heavy AutoDL result directories:

```text
results/
data/Amazon/sid_versions/
```

Therefore code and CS embedding artifacts can be verified locally, while final experiment result artifacts should be verified on AutoDL by running:

```bash
python summarize_final_results.py --output-dir results/final_summary
```

## 1. SID / Mapping Infrastructure

Status: mostly complete locally.

Evidence files:

```text
utils_sid.py
build_sid_mapping.py
check_sid_stage0.py
data/Amazon/sid_maps/experiment_manifest.json
data/Amazon/sid_maps/Industrial_and_Scientific/item2sid_text.json
data/Amazon/sid_maps/Industrial_and_Scientific/sid2items_text.json
data/Amazon/sid_maps/Industrial_and_Scientific/valid_sid_set_text.json
data/Amazon/sid_maps/Industrial_and_Scientific/item_mapping_text.json
data/Amazon/sid_maps/Office_Products/item2sid_text.json
data/Amazon/sid_maps/Office_Products/sid2items_text.json
data/Amazon/sid_maps/Office_Products/valid_sid_set_text.json
data/Amazon/sid_maps/Office_Products/item_mapping_text.json
```

Missing locally:

```text
data/Amazon/sid_versions/<sid_version>/<category>/item2sid.json
data/Amazon/sid_versions/<sid_version>/<category>/sid2items.json
data/Amazon/sid_versions/<sid_version>/<category>/valid_sid_set.json
data/Amazon/sid_versions/<sid_version>/<category>/item_mapping.json
```

These are expected to exist on AutoDL after Stage 3/4 generation.

Risk notes:

- Text baseline mapping exists locally.
- Versioned CS-SID mapping directories are absent locally, so local-only checks cannot verify final CS-SID mapping artifacts.

## 2. SID Diagnostics and Evaluation

Status: complete locally.

Evidence files:

```text
analyze_sid.py
calc_plus.py
compare_sft_rl_outputs.py
scripts/diagnose_sid_smoke_by_groups.py
scripts/summarize_sidonly_smoke_grid.py
data/Amazon/sid_maps/analysis/text/sid_quality_overall_summary.csv
```

Missing locally:

```text
results/calc_plus_sidonly_*/eval_report_*.json
results/calc_plus_sidonly_*/per_sample_eval.csv
results/compare_*/summary.json
```

These are AutoDL result artifacts.

## 3. CS Embedding

Status: complete locally.

Industrial evidence:

```text
data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.cf_emb.npy
data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.cs_emb_alpha0.2.npy
data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.cs_emb_alpha0.5.npy
data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.cs_emb_alpha0.7.npy
data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.item_order.json
data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.row_index.json
data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.cs_embedding_report.json
```

Office evidence:

```text
data/Amazon/cs_embeddings/Office_Products/Office_Products.cf_emb.npy
data/Amazon/cs_embeddings/Office_Products/Office_Products.cs_emb_alpha0.2.npy
data/Amazon/cs_embeddings/Office_Products/Office_Products.cs_emb_alpha0.5.npy
data/Amazon/cs_embeddings/Office_Products/Office_Products.cs_emb_alpha0.7.npy
data/Amazon/cs_embeddings/Office_Products/Office_Products.item_order.json
data/Amazon/cs_embeddings/Office_Products/Office_Products.row_index.json
data/Amazon/cs_embeddings/Office_Products/Office_Products.cs_embedding_report.json
```

Risk notes:

- `item_order.json` and `row_index.json` are present for both categories.
- This mitigates embedding row / item_id misalignment risk.

## 4. New SID Generation and CSV Rewrite

Status: code complete locally; final generated artifacts are not present locally.

Evidence scripts:

```text
run_rqkmeans_with_emb.py
rewrite_sid_csv.py
update_sid_manifest.py
scripts/generate_sid_versions_and_rewrite_csv.sh
```

Expected final SID versions:

```text
text_mbk_k512_dedup
Industrial_and_Scientific/cs_alpha0.2_k512_dedup
Office_Products/cs_alpha0.7_k512_dedup
```

Expected AutoDL files:

```text
data/Amazon/sid_versions/text_mbk_k512_dedup/<category>/train.csv
data/Amazon/sid_versions/text_mbk_k512_dedup/<category>/valid.csv
data/Amazon/sid_versions/text_mbk_k512_dedup/<category>/test.csv
data/Amazon/sid_versions/cs_alpha0.2_k512_dedup/Industrial_and_Scientific/train.csv
data/Amazon/sid_versions/cs_alpha0.7_k512_dedup/Office_Products/train.csv
data/Amazon/sid_versions/<sid_version>/<category>/reports/generation_report.json
data/Amazon/sid_versions/<sid_version>/<category>/reports/rewrite_train_report.json
data/Amazon/sid_versions/<sid_version>/<category>/reports/stage0_check_report.json
```

Missing locally:

```text
data/Amazon/sid_versions/
```

Risk notes:

- Local absence is expected if heavy generated artifacts only live on AutoDL.
- `scripts/tmp_run_sidonly_10k_compare.sh`, `scripts/run_sid_30k_validation.sh`, and `scripts/run_sid_k_ablation_10k.sh` were updated to prevent Office `cs_alpha0.7` results from being written under `Industrial_...` paths.

## 5. SID-only SFT and Evaluation

Status: code complete locally; result artifacts absent locally.

Expected AutoDL results:

Industrial:

```text
results/calc_plus_sidonly_Industrial_and_Scientific_text_mbk_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_text_mbk_k512_dedup.json
results/calc_plus_sidonly_Industrial_and_Scientific_cs_alpha0.2_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.2_k512_dedup.json
```

Office:

```text
results/calc_plus_sidonly_Office_Products_text_mbk_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_text_mbk_k512_dedup.json
results/calc_plus_sidonly_Office_Products_cs_alpha0.7_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.7_k512_dedup.json
```

Missing locally:

```text
results/
```

Risk notes:

- Final scripts use `SAVE_DURING_TRAINING=False` for disk-light 30k validation.
- This avoids saving large intermediate `optimizer.pt` checkpoint files.

## 6. Candidate Expansion and Rerank

Status: code complete locally; result artifacts absent locally.

Evidence files:

```text
evaluate_candidates.py
rerank.py
scripts/run_stage6_candidate_rerank.sh
```

Expected AutoDL final reports:

Industrial:

```text
results/candidates_text_k512_30k_exact_v2/report.json
results/rerank_text_k512_30k_exact_v2/rerank_report.json
results/candidates_cs_alpha02_k512_30k_p3_c500_v2/report.json
results/rerank_cs_alpha02_k512_30k_p3_c500_v2_sourcew4/rerank_report.json
```

Office:

```text
results/candidates_Office_Products_text_k512_30k_exact_c500_v2/report.json
results/rerank_Office_Products_text_k512_30k_exact_c500_sourcew4.0_v2/rerank_report.json
results/candidates_Office_Products_cs_alpha0_7_k512_dedup_30k_p3_c500_v2/report.json
results/rerank_Office_Products_cs_alpha0_7_k512_dedup_30k_p3_c500_sourcew4.0_v2/rerank_report.json
```

Missing locally:

```text
results/
```

## 7. Risk Audit

### Text / CS mixing risk

Current mitigation:

- Versioned SID directories are used by manifest.
- Rewritten CSVs are version-specific.
- SFT smoke scripts read train/valid/index paths from manifest.

Remaining local limitation:

- Final versioned CSVs are not present locally, so this must be verified on AutoDL.

### Office / Industrial hardcoded path risk

Risk found:

- Legacy scripts had special paths such as `results/calc_plus_sidonly_Industrial_cs_alpha0.7...`.

Mitigation:

- The scripts were updated so these legacy paths only apply when `CATEGORY=Industrial_and_Scientific`.

Need AutoDL confirmation:

```bash
find results -path "*Industrial*cs_alpha0.7*k512*30000*" -print
find results -path "*Office_Products*cs_alpha0.7*k512*30000*" -print
```

### Target leakage risk

Current mitigation:

- `build_cs_embeddings.py` uses train CSV only.
- `rerank.py` uses train-only popularity.
- `rerank.py` reads `history_item_id` from candidate JSONL and states target item is used only for metrics.

Need AutoDL consistency check:

```bash
python summarize_final_results.py --output-dir results/final_summary
cat results/final_summary/final_consistency_report.md
```

## 8. Overall Completion Status

| Stage | Local status | AutoDL result status |
|---|---|---|
| SID/mapping infrastructure | Complete | Expected complete |
| SID diagnostics/evaluation | Complete | Expected complete |
| CS embedding | Complete | Expected complete |
| New SID generation + CSV rewrite | Code complete, artifacts missing locally | Expected complete |
| SID-only SFT/evaluation | Code complete, results missing locally | Expected complete |
| Candidate expansion/rerank | Code complete, results missing locally | Expected complete |
| Final summary/consistency tooling | Complete after Stage 8 script addition | Run on AutoDL |

## 9. Required Final AutoDL Command

Run this on AutoDL to verify final result artifacts:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python summarize_final_results.py --output-dir results/final_summary
cat results/final_summary/final_results_summary.md
cat results/final_summary/final_consistency_report.md
```
