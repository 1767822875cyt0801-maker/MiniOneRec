# S6 Validation-Only Research Plan

S5 is closed. S5 final-test results are frozen evidence and must not be used to tune parameters. S6 work must return to validation-only experimentation until a new independent final test policy is approved.

## Scope

S6 may explore candidate-generation and ranking ideas on validation splits only. It must not use S5 final-test outcomes to select lambda, source bonus, ranker features, projection behavior, SID construction, checkpoints, generation budgets, or decoding policies.

## Hypotheses

1. Auxiliary behavior candidates can improve recall if the candidate pool is made less noisy.
2. Ranker conversion can improve when features better distinguish useful auxiliary candidates from high-volume noise.
3. Candidate-generation changes and ranking changes should be evaluated separately before being combined.
4. Robustness requires improvements across validation slices, not only aggregate HR@20.

## Validation Protocol

S6 experiments should use validation-only splits with fixed predeclared protocols:

- train or fit only on approved training/validation-fit partitions;
- select configurations only on validation-select partitions;
- report locked validation-report results separately;
- preserve all failed runs and validation errors;
- keep raw predictions, candidate files, reports, and derived metrics separate;
- record full provenance for each candidate source, ranker, SID mapping, checkpoint, and generation budget.

No S6 experiment may inspect or optimize against S5 final-test metrics.

## Candidate-Generation Changes

Candidate-generation work may investigate:

- reducing invalid or terminal-prefix generation warnings;
- candidate pool size controls;
- source-specific candidate caps;
- exact-SID-only versus bounded prefix-expansion policies;
- candidate deduplication order;
- SID bucket diagnostics;
- behavior representation alternatives evaluated only on validation data.

Every candidate-generation change must report:

- candidate count distribution;
- target-in-pool count and rate;
- invalid SID count and rate;
- source contribution;
- candidate noise ratio;
- oracle upper bounds;
- deterministic reproducibility hashes.

## Ranking Changes

Ranking work may investigate:

- source-independent feature refinements;
- source-aware features, if explicitly separated from the S5 frozen projection;
- calibration of candidate scores;
- history-aware ranking variants;
- pairwise versus listwise validation objectives;
- robustness across user history length, item popularity, bucket size, and source-overlap slices.

Ranking changes must keep feature schema, normalization, and training split boundaries explicit. Any source-aware feature must be declared before validation evaluation and must not be inferred from final-test behavior.

## Ablations

Required ablations for any S6 proposal:

- CF-only baseline;
- auxiliary-only baseline;
- fixed fusion without learned ranker;
- learned or fitted ranker without auxiliary candidates where applicable;
- candidate generation change without ranker change;
- ranker change without candidate generation change;
- slice-level metrics for short history, long history, low popularity, high popularity, exact-only hits, and auxiliary-only recoveries.

## Acceptance Gates

A candidate S6 direction may proceed only if it satisfies validation-only gates:

- improves HR@20 and NDCG@20 on validation-report compared with the relevant frozen validation baseline;
- does not degrade key robustness slices beyond a predeclared tolerance;
- preserves deterministic reproducibility;
- has complete provenance and artifact hashes;
- has no test-read evidence;
- documents failed runs and warning counts;
- includes a no-test-tuning declaration.

Validation gains alone do not authorize final-test execution. A separate final-test protocol must be approved before any new test run.

## Future Independent Test Policy

Future test execution must be independent of S5 final-test tuning. Before any future test:

1. freeze candidate-generation code and parameters;
2. freeze ranking code, ranker checkpoint, feature schema, and normalization;
3. freeze SID mappings and checkpoints;
4. generate a release manifest with content hashes;
5. run dry-run and environment preflight;
6. document one-shot execution and recovery policy;
7. require explicit approval for test execution.

If a future test fails operationally, any resume must follow the S5 standard: accept only a precisely audited partial state, forbid overwriting completed artifacts, and resume only from the first unstarted stage.

## Non-Goals

S6 must not:

- tune on S5 final-test metrics;
- rerun S5 final test;
- change S5 closed conclusions;
- treat S5 final-test results as validation data;
- use LLM judgment to select experiment winners;
- hide failed validation experiments or warnings.

S6 is a validation-only research phase until a new independent final-test gate is explicitly defined.
