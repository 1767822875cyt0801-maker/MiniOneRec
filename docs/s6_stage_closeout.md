# S6 Stage Closeout

## Verdict

`S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT`

S6 closes with direct retrieval and candidate-level complementarity validated, while ranking conversion remains partial. S6-5 frozen projected ranker is the strongest validated ranking result, but it does not pass the strict 97% CF-preservation gate on `valid_select`. S6-6 and S6-6R are retained as negative evidence, and no further S6 ranker tuning is allowed.

## Motivation

S6 tested whether a cheap direct item-level SASRec retrieval stream can replace or augment the expensive SASRec-SID Qwen generation stream. The stage separated retrieval/candidate evidence from ranking evidence so that candidate success would not be mistaken for production-ready ranking success.

## Protocol

- Validation-only; no test data.
- K selected from validation candidate evidence and frozen at K20.
- `valid_fit` trains/calibrates lightweight revisions.
- `valid_select` selects one final configuration when allowed.
- `valid_gate` is opened only after a configuration passes fit/select.
- S6-6 and S6-6R did not pass fit/select, so their `valid_gate` remained unopened.

## Stage Timeline

- S6-0 checkpoint and protocol audit: `GO_CHECKPOINT_COMPATIBLE`.
- S6-1 direct exporter implementation.
- S6-2 smoke and deterministic reproduction: `GO_SMOKE_AUDIT`.
- S6-3 formal direct-SASRec validation: `GO_UNION_VALIDATION`.
- S6-4 CF + direct union validation: `GO_FROZEN_RANKER_VALIDATION`.
- S6-5 frozen projected-ranker validation: `GO_LIGHTWEIGHT_RANKER_VALIDATION`.
- S6-6 residual ranker revision: `REVISE_LIGHTWEIGHT_RANKER`.
- S6-6R direct-promotion revision: `REVISE_STAGE_CLOSEOUT`.

## Final Metric Table

| Method | Gate status | fit HR@20 | fit NDCG@20 | fit CF preserve | fit direct recovery | select HR@20 | select NDCG@20 | select CF preserve | select direct recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only | baseline | 0.107895 | 0.071681 | 1.000000 | 0 | 0.098650 | 0.061860 | 1.000000 | 0 |
| Direct SASRec direct order | baseline | 0.134586 | 0.104068 | 0.682927 | 162 | 0.127726 | 0.094707 | 0.684211 | 58 |
| Fixed RRF | baseline | 0.114286 | 0.082575 | 0.982578 | 22 | 0.100727 | 0.073357 | 0.978947 | 4 |
| CF + direct K20 candidate oracle | candidate_layer_only | 0.197744 | N/A | 1.000000 | 162 | 0.188993 | N/A | 1.000000 | 58 |
| S6-5 frozen projected ranker | best_validated_ranking_but_valid_select_cf_preservation_failed | 0.149248 | 0.093252 | 0.972125 | 102 | 0.130841 | 0.082337 | 0.957895 | 34 |
| S6-6 aggressive residual | not_selected_valid_gate_unopened | 0.155263 | 0.095161 | 0.972125 | 112 | 0.137072 | 0.084272 | 0.947368 | 37 |
| S6-6 preservation residual | not_selected_valid_gate_unopened | 0.133459 | 0.085736 | 0.982578 | 66 | 0.116303 | 0.076331 | 0.978947 | 19 |
| S6-6R best-HR promotion policy | not_selected_valid_gate_unopened | 0.129699 | 0.076776 | 0.944251 | 74 | 0.119418 | 0.066684 | 0.947368 | 25 |
| S6-6R best-preservation policy | not_selected_valid_gate_unopened | 0.125188 | 0.075633 | 0.996516 | 47 | 0.115265 | 0.065666 | 1.000000 | 16 |

## Runtime and Cost Evidence

Direct SASRec validation runtime:

- model inference: `24.967018` seconds / 4532 samples
- wall: `34.233973` seconds
- peak CUDA allocated: `13118464` bytes

SASRec-SID Qwen validation-side v2 cost profile:

- full candidate-pipeline wall: `662.68` seconds / 4532 samples
- per-sample wall: `0.14622241835834068` seconds
- samples/sec: `6.8388966016780355`
- GPU: `NVIDIA A100-PCIE-40GB`
- beams: `50`
- max_new_tokens: `6`
- invalid SID count: `0`

Comparable wall-time ratio:

- Direct/Qwen wall-time ratio: `0.051659885615983586`
- Qwen/Direct wall-time factor: `19.3573792910335`

Pure model-time ratio is not claimed because Qwen generation time was not isolated from the full candidate pipeline. Cost evidence is complete for wall-clock candidate-pipeline comparison and does not change ranking selection.

## Candidate Conclusions

Direct SASRec retrieval is validated as a candidate source. K20 was selected as the smallest candidate budget satisfying validation candidate uplift gates. Direct retrieval adds target recall beyond CF while preserving CF candidate pool membership in the union.

## Ranking Conclusions

Fixed RRF is insufficient. S6-5 frozen projected ranker converts auxiliary recall into ranking gains, but its `valid_select` CF preservation is `91/95 = 95.79%`, below the required 97%. Therefore S6 ranking is partial, not a production GO.

## Failed Alternatives

S6-6 residual ranking showed a sharp preservation-versus-recovery trade-off. S6-6R CF-anchored promotion preserved CF under conservative policies but lost too much direct gain; aggressive promotion recovered more direct targets but violated CF preservation. Both are retained as negative evidence.

## Limitations

- Comparable SASRec-SID Qwen validation-side runtime evidence is pending.
- No S6 method is authorized to use test data for tuning.
- No additional thresholds, model families, or ranker configurations may be added under S6.

## Deployment Recommendation

Retain direct SASRec as a validated auxiliary candidate generator. Do not deploy a replacement ranking policy from S6 as production-ready under the strict CF-preservation requirement. If deployed experimentally, S6-5 should be labeled as a partial ranking result with known CF-preservation limitation.

## Future Research

- Cost-only profiling for frozen SASRec-SID Qwen valid generation.
- New validation-only ranker research with explicit CF-preservation regularization, not S6 continuation.
- Independent future test policy after a new validation protocol is frozen.

## Boundary

S6 is closed without test access and without further ranker tuning. S6 negative experiments remain first-class evidence.
