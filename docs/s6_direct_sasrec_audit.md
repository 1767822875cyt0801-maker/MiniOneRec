# S6 Direct SASRec Auxiliary Audit

Status: `GO_CHECKPOINT_COMPATIBLE`

This audit is validation-only. It does not read test data, does not rerun S5, and does not run GPU inference.

## Goal

S6 asks whether the second SASRec-SID Qwen generation stream can be replaced by a cheaper direct item-level SASRec retrieval stream:

```text
CF-SID Qwen candidates
+ direct SASRec item top-K / SASRec scores
+ history-aware candidate ranker
```

The S5 final test remains frozen and is used only as background motivation. No S5 final-test artifact or parameter is modified.

## Confirmed Repository Facts

### SASRec Training Implementation

The current frozen SASRec v3 artifact was produced by `build_sasrec_embeddings.py`, not by the older `sasrec.py` training path.

Observed `MiniSASRec` contract in `build_sasrec_embeddings.py`:

- model: `MiniSASRec`
- item embedding: `nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)`
- positional embedding: `nn.Embedding(max_seq_len, embedding_dim)`
- encoder: `num_layers` `nn.TransformerEncoderLayer` blocks
- attention: causal upper-triangular mask
- padding convention: right padding with `padding_id = 0`
- item-index convention: `internal_id = canonical_row_index[item_id] + 1`
- valid sequence representation: gather hidden state at `length - 1`, not `hidden[:, -1]`
- scoring: `last_hidden @ item_embedding.weight[1:].T`
- training objective: full-softmax cross entropy over all canonical items for non-padding shifted targets
- finite checks: input embedding, positional embedding, every transformer layer output, last hidden, logits, loss, gradients, and parameters

The formal v3 config records:

- category: `Industrial_and_Scientific`
- train rows: `36259`
- valid rows: `4532`
- item universe: `3686`
- embedding dimension: `128`
- max sequence length: `10`
- num layers: `2`
- num heads: `2`
- dropout: `0.2`
- seed: `42`
- checkpoint metric: `ndcg@20`
- valid input includes target: `false`
- valid candidate count: `3686`

### SASRec Artifacts

Observed local formal v3 directory:

```text
data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/
```

Present:

- `Industrial_and_Scientific.sasrec_item_emb.npy`
- `Industrial_and_Scientific.item_order.json`
- `Industrial_and_Scientific.row_index.json`
- `Industrial_and_Scientific.train_item_count.json`
- `artifact_manifest.json`
- `config.json`
- `metrics_summary.json`

The manifest declares the full checkpoint path, and the recovered file is now present locally:

```text
data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/checkpoint/best_model.pt
```

The companion checksum file verifies this exact artifact.

Recorded hashes from the formal v3 manifest:

| artifact | SHA256 |
|---|---|
| config | `24a4b00b86d7e08d6773fb90ed6b6280496fd809b705644b78facff9644aee04` |
| exported item embedding | `e8de748590a46522a15b8f73765b17081c277053f0e4881d29863357a2369d75` |
| item order | `b1527009791b626b05067a5c9ce0f610459993d1ee5cf6e0ebe55a29bbf0e53e` |
| row index | `553116f1d1df1e4b51d91b441d1a618600e034e312768d80dc8296a2f1245467` |
| train item count | `a319651d727e50d505c10648a762765d055203df2b35f3db74120d2241d46983` |
| metrics summary | `94a23967771ef4d848a61a06292ddb849a3ccb274b81b41a74354bdae38d0b1f` |

### SASRec Valid Evaluation Behavior

The formal v3 `metrics_summary.json` confirms:

- full item universe scoring: `candidate_count_expected = 3686`
- valid candidate count min/max: `3686 / 3686`
- finite scored samples: `4532`
- nonfinite scored samples: `0`
- unranked samples: `0`
- target leakage guard: debug valid samples have `target_in_input = false`
- random and popularity baselines were computed over the same full item universe

Final valid metrics for the sequence model:

- HR@20: `0.1290820829655781`
- NDCG@20: `0.09907201177843943`

### CF Valid Candidate Artifacts

Existing S4 formal validation candidate artifacts are present under:

```text
incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/
```

CF candidate JSONL:

```text
results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl
```

Observed properties:

- rows: `4532`
- row key: `row_index`
- target field: `target_item_id`
- history field: `history_item_id`
- candidate list field: `candidate_item_ids`
- detail field: `candidate_details`
- candidate details include `item_id`, `source_sid`, `matched_prefix`, `source_type`, `sid_rank_0_based`, `bucket_size`, and `expansion_level`
- candidate report records `invalid_sid_count = 0`
- candidate report records item-level HR@20 `0.10238305383936452`

This schema is reusable as the nearest existing item-level candidate contract.

### S5 Fusion and Ranker Compatibility

S5 reused CF/SASRec-SID candidate JSONL rows, merged item-level candidates, and applied a source-independent projection for the frozen P2 history-aware ranker.

The frozen S5 release config records:

- `lambda_sasrec = 0.75`
- `source_bonus = 0.01`
- `ranker_compatibility_mode = source_independent_projection`
- frozen ranker model path: `results/stage7_validation_protocol/valid/Industrial_and_Scientific/p2_history_ranker/model.json`
- ordered feature schema hash: `dd6f852e0f3d9098de8b02a24a012fe218d0e76e7a88be2237711064d34a7078`

The source-independent projection zeroes source identity features after normalization and retains source-independent rank, score, popularity, history similarity, bucket, expansion, and heuristic features.

## Reasonable Inferences

1. Direct SASRec retrieval must load the full sequence checkpoint and reproduce `MiniSASRec.last_valid_logits`.
2. The exported item embedding matrix alone is not enough because it lacks:
   - transformer layer weights;
   - positional embeddings;
   - layer norm weights;
   - the user-history-conditioned hidden state `h_u`.
3. With the full checkpoint available and compatible, direct retrieval should compute:

   ```text
   input_ids = right_padded([row_index[item] + 1 for item in history_item_id], max_seq_len=10)
   h_u = MiniSASRec.encode(input_ids).gather(length - 1)
   scores = h_u dot item_embedding.weight[1:].T
   topK = stable top-K over canonical item_order rows
   ```

4. The existing S4/S5 candidate JSONL contract can represent direct SASRec candidates if direct output keeps:
   - `row_index`
   - `target_item_id`
   - `history_item_id`
   - `candidate_item_ids`
   - `candidate_details`

5. Direct SASRec-specific score features should be written to a sidecar JSONL unless and until a new S6 ranker schema is frozen.

## Unresolved Issues

1. The local repository does not contain `checkpoint/best_model.pt`, so checkpoint restoration cannot be tested.
2. The formal v3 manifest declares the checkpoint path but does not include a checkpoint hash in the observed manifest.
3. Direct SASRec score features are not semantically part of the frozen S5 projected ranker schema.
4. S5 candidate details depend on SID concepts such as bucket size and prefix expansion. Direct item retrieval does not naturally produce SID bucket or prefix features.
5. Historical timing for SASRec-SID Qwen generation is incomplete in candidate reports; any cost ratio must use measured future direct timing and available audited Qwen timing, or record null with a missing reason.

## Checkpoint Compatibility Verdict

The recovered formal-v3 checkpoint is compatible.

Checkpoint:

```text
data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/checkpoint/best_model.pt
```

Actual SHA256:

```text
577f51a9a539b9cd6307eea16c04db303691e349c0f32e245e44b8710cee94bd
```

Compatibility report:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json
```

Evidence:

- SHA256 matches the expected formal-v3 checkpoint hash.
- Top-level checkpoint keys are `model_state_dict`, `config`, and `train_summary`.
- State dict tensor count is `28`.
- The state dict contains item embedding, position embedding, two self-attention blocks, two feed-forward blocks, per-layer norms, and final `layer_norm`.
- Observed item embedding shape is `[3687, 128]`, confirming one padding row plus 3686 item rows.
- Observed positional embedding shape is `[10, 128]`.
- Observed layer count is `2`.
- Checkpoint config matches repository formal-v3 config.
- `train_summary.best_epoch = 20`.
- `train_summary.best_metric = 0.09907201177843943`.
- Reproducing the exported item embedding with the original `prepare_export_matrix` semantics gives exact agreement:
  - selected slice: `item_embedding.weight[1:]`
  - cold-item fallback: `mean_train_positive_item_embedding`
  - cold items: `39`
  - L2 normalization: applied
  - max absolute error: `0.0`
  - mean absolute error: `0.0`
  - cosine similarity mean/min: `1.0 / 1.0`
  - mismatched rows under tolerance `1e-6`: `0`

Local limitation:

- The WSL Python used for this audit does not have PyTorch installed, so the audit used a restricted PyTorch-zip parser plus raw FloatStorage comparison.
- Real direct candidate inference must run in an environment with PyTorch, such as AutoDL.

## Data-Leakage Audit

The S6 development split uses only:

```text
data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv
```

It is a validation-only development gate, not an independent final test. It must not be used to retune any S5 final-test parameter. The final test remains closed and frozen.

## Files Reused

- `build_sasrec_embeddings.py`
- `data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/artifact_manifest.json`
- `data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/config.json`
- `data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/metrics_summary.json`
- `incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/.../cf_k512_dedup/.../candidates.jsonl`
- `configs/s5_auxiliary_fusion/frozen_release_config.json`
- `scripts/s5_auxiliary_fusion_conversion.py`
- `scripts/apply_stage7_p2_history_ranker.py`

## Direct Candidate Implementation Status

Implemented validation-only entry points:

```text
export_sasrec_direct_candidates.py
merge_direct_sasrec_candidates.py
```

The exporter refuses to run unless the checkpoint compatibility report verdict is `GO_CHECKPOINT_COMPATIBLE`. It restores the exact `MiniSASRec` model through `build_sasrec_embeddings.py` when PyTorch is available, uses the formal-v3 config and mappings, scores the full frozen item universe, and writes validation-only item-level candidates plus a score sidecar.
