# S5-1 Frozen Release and Final Test Protocol

## Scope

This release freezes the S5-0 valid-only auxiliary conversion result. Local work prepares manifests and dry-run commands only; it does not read or evaluate test labels.

## Frozen Selected Config

- candidate policy: `source_aware_rrf_lam0.75_bonus0.01`
- lambda_sasrec: `0.75`
- source_bonus: `0.01`
- ranker compatibility: `source_independent_projection`
- candidate mode: `exact_full_sid_only`
- num_beams: `50`
- max_new_tokens: `6`
- seed: `42`

## Projection Contract

Zeroing occurs **after normalization**. A zero value therefore means the frozen training-distribution mean for that feature, not raw missing/false.

- zeroed normalized source-specific features: `['text_present', 'cf_present', 'both_sources', 'text_only', 'cf_only', 'source_count', 'text_rank_filled', 'cf_rank_filled', 'reciprocal_text_rank', 'reciprocal_cf_rank']`
- retained normalized features: `['min_source_rank', 'fusion_rank', 'reciprocal_min_source_rank', 'reciprocal_fusion_rank', 'fusion_score', 'sid_rank_score', 'source_score', 'popularity_score', 'history_cosine', 'recent_cosine', 'bucket_penalty', 'bucket_size_log', 'expansion_level_score', 'exact_source', 'prefix_source', 'heuristic_score']`
- ordered feature schema hash: `dd6f852e0f3d9098de8b02a24a012fe218d0e76e7a88be2237711064d34a7078`
- normalization metadata hash: `5600cb14e47f23abdb32be6d7292d49ff3eb0c3d446b0fef97fb231f2ddef215`
- ranker model hash: `6a3bca7cc714d295572ce4a3953d60719a1c72757895835554a83d747dca65ce`

## AutoDL Dry-Run Commands

```bash
export PROJECT_ROOT="$(pwd)"
export BASE_MODEL="<AutoDL base model path>"
DRY_RUN=1 bash scripts/run_s5_final_test_confirmation.sh --action dry-run
python3 scripts/s5_final_test_confirmation.py --action audit
```

## AutoDL Final Test Command

Run exactly once after dry-run and parity pass:

```bash
export PROJECT_ROOT="$(pwd)"
export BASE_MODEL="<AutoDL base model path>"
CONFIRM_FINAL_TEST=1 DRY_RUN=0 bash scripts/run_s5_final_test_confirmation.sh --action run-final-test
```

The final test result must not be used to change lambda, source bonus, projection, ranker checkpoint, or candidate policy.

## Command Plan

```text
env_preflight: PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}" python3 scripts/s4_sasrec_sid_valid_pipeline.py --action env-preflight --category Industrial_and_Scientific --sid-root data/Amazon/sid_versions --baseline-version cf_k512_dedup --treatment-version sasrec_v3_k512_dedup --base-model "${BASE_MODEL:?set BASE_MODEL}" --results-root results/s4_sasrec_sid_valid --outputs-root outputs/s4_sasrec_sid_valid --seed 42 --config-mode formal --num-beams 50 --max-new-tokens 6 --max-pred-sids 50 --max-candidates 1000
parity: PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}" python3 scripts/s4_sasrec_sid_valid_pipeline.py --action parity-audit --category Industrial_and_Scientific --sid-root data/Amazon/sid_versions --baseline-version cf_k512_dedup --treatment-version sasrec_v3_k512_dedup --base-model "${BASE_MODEL:?set BASE_MODEL}" --results-root results/s4_sasrec_sid_valid --outputs-root outputs/s4_sasrec_sid_valid --seed 42 --config-mode formal --num-beams 50 --max-new-tokens 6 --max-pred-sids 50 --max-candidates 1000
cf_test_generation: PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}" python3 evaluate.py --base_model 'outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/sft/final_checkpoint' --train_file 'data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/train.csv' --info_file 'data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/info.txt' --category Industrial_and_Scientific --test_data_path 'data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/test.csv' --result_json_data 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup/generation/predictions.json' --batch_size 4 --seed 42 --length_penalty 0.0 --max_new_tokens 6 --num_beams 50
cf_test_candidate_eval: PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}" python3 evaluate_candidates.py --prediction-file 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup/generation/predictions.json' --eval-csv 'data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/test.csv' --eval-split test --item2sid 'data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/item2sid.json' --sid2items 'data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/sid2items.json' --valid-sid-set 'data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/valid_sid_set.json' --output-jsonl 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup/candidates/candidates.jsonl' --output-report 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup/candidates/candidate_report.json' --topk 1 5 10 20 50 --max-pred-sids 50 --max-candidates 1000
sasrec_test_generation: PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}" python3 evaluate.py --base_model 'outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/sft/final_checkpoint' --train_file 'data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/train.csv' --info_file 'data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/info.txt' --category Industrial_and_Scientific --test_data_path 'data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/test.csv' --result_json_data 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/sasrec_v3_k512_dedup/generation/predictions.json' --batch_size 4 --seed 42 --length_penalty 0.0 --max_new_tokens 6 --num_beams 50
sasrec_test_candidate_eval: PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}" python3 evaluate_candidates.py --prediction-file 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/sasrec_v3_k512_dedup/generation/predictions.json' --eval-csv 'data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/test.csv' --eval-split test --item2sid 'data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/item2sid.json' --sid2items 'data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/sid2items.json' --valid-sid-set 'data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/valid_sid_set.json' --output-jsonl 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/sasrec_v3_k512_dedup/candidates/candidates.jsonl' --output-report 'results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/sasrec_v3_k512_dedup/candidates/candidate_report.json' --topk 1 5 10 20 50 --max-pred-sids 50 --max-candidates 1000
finalize: PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}" CONFIRM_FINAL_TEST=1 DRY_RUN=0 python3 scripts/s5_final_test_confirmation.py --action finalize-test --config 'configs/s5_auxiliary_fusion/frozen_release_config.json'
```

## Return Files

```text
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_metrics.json
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_complete_manifest.json
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/candidates/candidate_report.json
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/summary/*.json
```

## Safe Cleanup

Only after the return bundle is verified:

```bash
rm -rf results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/generation
rm -f results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/candidates/candidates.jsonl
```

Do not delete formal final checkpoints unless the final test is complete and checkpoint backups are verified.
