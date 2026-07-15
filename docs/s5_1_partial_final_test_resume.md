# S5-1 Partial Final-Test Resume

## Root Cause

1. `sasrec_v3_k512_dedup` test.csv was missing when SASRec test generation was about to start.
2. The S5 dry-run/preflight did not check both baseline and treatment test.csv files before GPU generation.

The LogitProcessor step-5 warning is retained as a warning but is not the crash root cause.

## Allowed Partial State

- CF generation and CF candidate evaluation complete.
- SASRec generation/candidates/report absent.
- final metrics and complete manifest absent.
- CF artifacts are never deleted, quarantined, or overwritten by the resume path.

## Resume Sequence

```text
input_inventory_preflight
frozen_hash_verification
cf_completed_artifact_verification
sasrec_test_generation
sasrec_row_validation
sasrec_test_candidate_eval
deterministic_cf_sasrec_provenance_merge
frozen_source_aware_rrf
frozen_source_independent_projection
frozen_ranker_inference
final_metrics
final_artifact_manifest
protocol_closeout
```

## AutoDL Audit

```bash
export PROJECT_ROOT="$(pwd)"
export BASE_MODEL="<AutoDL base model path>"
DRY_RUN=1 bash scripts/run_s5_final_test_confirmation.sh --action audit-partial-final-test
```

## AutoDL Resume Dry-Run

```bash
DRY_RUN=1 bash scripts/run_s5_final_test_confirmation.sh --action resume-final-test-after-cf
```

## Unique Resume Command

Run only after the partial-state audit, rebuilt SASRec test.csv validation, and concrete CF artifact hashes match the resume manifest:

```bash
CONFIRM_FINAL_TEST_RESUME=1 DRY_RUN=0 bash scripts/run_s5_final_test_confirmation.sh --action resume-final-test-after-cf
```

## Return Files

```text
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_metrics.json
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_complete_manifest.json
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/partial_resume_closeout_manifest.json
results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/candidates/candidate_report.json
```

No lambda, source bonus, projection, ranker, beam, token budget, seed, checkpoint, or SID parameter may be changed during resume.
