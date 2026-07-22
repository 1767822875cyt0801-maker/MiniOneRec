# S6 Cost Evidence Closeout

## Verdict

`S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT`

This is cost-only validation evidence. It does not reopen ranking selection, K selection, thresholds, models, policies, or any test-set decision.

## Integrity

- Bundle: `/home/dell/projects/MiniOneRec/incoming/s6_qwen_cost_profile/cost_profile_qwen_sasrec_sid_v2_bundle.tar.gz`
- Bundle SHA256: `d8227aee32c0b9179c95ee34e6a422e6d62fc71d142f75d0c8db31cf499a9c3d`
- Internal file hashes: `21` checked, all OK
- Exit code: `0`
- v1 excluded: earlier v1 attempt exited with code 127 before model inference because GNU time was missing

## Qwen Configuration

- stream: `treatment`
- SID version: `sasrec_v3_k512_dedup`
- split: `valid`
- rows: `4532`
- checkpoint tree SHA256: `ed25347e7985ef5aed255fe92688fbc05335b6465524f8fff6faac7cc368309c`
- beams: `50`
- max_new_tokens: `6`
- candidate mode: `exact_full_sid_only`
- test_read: `False`

## Hardware

- GPU: `NVIDIA A100-PCIE-40GB`
- CUDA available: `True`
- Python: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- executable: `/root/miniconda3/envs/minionerec/bin/python3`

## Runtime

| Runtime surface | Seconds | Seconds/sample | Samples/sec |
|---|---:|---:|---:|
| Direct SASRec formal wall | 34.233973 | 0.007553833 | 132.383115 |
| SASRec-SID Qwen full candidate pipeline wall | 662.680000 | 0.146222418 | 6.838897 |

Direct/Qwen wall-time ratio: `0.051660`.

Qwen/Direct wall-time factor: `19.357x`.

Pure model-time ratio is not claimed because Qwen model generation time was not isolated from pipeline overhead.

## Candidate Validity

- predictions: `4532`
- candidate rows: `4532`
- invalid SID count: `0`
- invalid SID rate: `0.0`
- LogitProcessor warnings: `1848`

The warning count does not invalidate the cost artifact because all 4532 validation rows completed, candidate rows were written, and invalid SID count is zero.

## Conclusion

The v2 validation-side evidence supports a wall-clock comparison: Direct SASRec uses about `5.17%` of the SASRec-SID Qwen full candidate-pipeline wall time on the audited 4532-row validation population. This strengthens S6's cost argument while preserving the existing scientific verdict: direct retrieval GO, ranking partial closeout.
