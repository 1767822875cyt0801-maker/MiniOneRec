# S6 Direct SASRec 阶段技术总结

## 1. 阶段背景

MiniOneRec 当前主线是生成式推荐：把物品表示成结构化 SID，让 Qwen 根据用户历史生成目标物品对应的 SID，再通过 SID 到 item 的映射还原候选物品。S5 的正式方案是双流结构：CF-SID Qwen 作为主流，SASRec-SID Qwen 作为行为辅助流，然后对两个候选池做融合和重排。

这个设计有一个清晰成本问题：第二个 Qwen 流需要再次执行 beam search、受约束解码、候选展开和 exact candidate evaluation。SASRec-SID Qwen 生成的是由 SASRec embedding 聚类得到的 SID，并不等同于直接使用 SASRec 推荐器本身。静态 item embedding 只描述物品位置；完整 SASRec 推荐需要先根据用户历史得到序列表示，再对全物品打分。

## 2. S5遗留问题

S5 验证了 SASRec-SID 候选与 CF 候选存在互补性，但也留下三个问题。第一，双 Qwen 推理成本高。第二，SASRec-SID Qwen 的辅助候选召回有限，最终 test 中 SASRec-only 低于 CF-only。第三，候选 union 的 oracle recall 增长并不会自动转化为 top-20 ranking 增长，排序器必须同时保住 CF 强项并利用辅助候选。

因此 S6 的问题是：能否绕过第二个生成模型，直接用完整 SASRec 模型做用户条件化 item retrieval，以更低成本提供更强行为候选。

## 3. S6核心研究假设

假设：完整 SASRec 模型可以基于用户历史直接给全量 item 打分，它可能比“把 SASRec embedding 聚类成 SID，再让第二个 Qwen 生成 SID”的路径更强、更便宜。

关键区别如下：

- SASRec embedding export：只导出 item embedding，是静态物品表示。
- complete SASRec forward：恢复 item embedding、position embedding、attention blocks、FFN、norm 等完整参数。
- sequence representation：对用户历史序列前向计算得到用户条件化表示 `z_h`。
- full-item scoring：对每个候选物品 embedding `e_i` 计算 `score(i | h) = z_h^T e_i`。
- top-K item retrieval：按分数稳定排序，得到直接 item 候选。

公式中，`h` 表示用户历史，`z_h` 是 SASRec 对历史编码后的最终序列表示，`e_i` 是物品 `i` 的 embedding。静态 `e_i` 不是推荐器；推荐器需要 `z_h` 与全物品 `e_i` 的匹配。

## 4. Validation-only实验协议

S6 只使用 4532 行 validation 数据，并按 user-disjoint 规则拆成：

| Split | Rows | Users | 用途 |
|---|---:|---:|---|
| valid_fit | 2660 | 1188 | 训练或校准允许的 lightweight ranking 机制 |
| valid_select | 963 | 413 | 一次性选择候选机制或 ranker 配置 |
| valid_gate | 909 | 410 | 仅在 fit/select 通过后做最终确认 |

`valid_gate` 不是调参集。S6-6 和 S6-6R 没有任何配置通过 fit/select，所以它们的 `valid_gate` 保持未打开。S6 全程 `test_read=false`，不使用 test 结果选择 K、阈值、ranker、模型或策略。

数据来源：`results/s6_cost_aware_aux/Industrial_and_Scientific/s6_validation_split_manifest.json`。

## 5. SASRec checkpoint恢复和兼容性

S6-0 审计 formal-v3 SASRec checkpoint，确认不是只读 embedding，而是恢复完整 SASRec 结构。兼容性检查覆盖：

- formal-v3 checkpoint SHA；
- item embedding offset；
- position embedding；
- attention blocks；
- FFN；
- normalization layers；
- cold-item fallback；
- L2 normalization where applicable；
- checkpoint 与 exported embedding 的一致性。

最终 verdict 是 `GO_CHECKPOINT_COMPATIBLE`。

数据来源：`results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json`。

## 6. Direct SASRec检索实现

Direct retrieval 的数据流：

```text
history item IDs
→ sequence padding/truncation
→ SASRec forward
→ final user-sequence representation
→ dot product with all item embeddings
→ masking invalid/padding items
→ stable top-K
→ candidates.jsonl
→ candidate_features.jsonl
→ candidate_report.json
```

实现中保留 `row_index`、`target_item_id`、`history_item_id`、item ID mapping、legal item set 和候选 provenance。排序按 SASRec score 降序，稳定 tie break 使用 item id。正式运行一次 K100 model pass，再派生 K20/K50/K100 prefix views，避免重复推理。

主要代码：`export_sasrec_direct_candidates.py`。

## 7. Smoke、确定性和正式验证

Smoke 和 deterministic rerun 显示 direct retrieval 输出可复现。三次 K20 candidates hash 一致：

`6d757d9fe825e2c330bed7b5a3fd8403566356625dfc6e6bde81f1a57f85e644`

三次 candidate_features hash 一致：

`1f321b86036541f04341ad1a56e5cdfae67e27599cea0f3afe042ab873877669`

正式 direct aggregate：

| K | HR@20 | NDCG@20 | target-in-pool |
|---:|---:|---:|---:|
| 20 | 0.129082 | 0.099072 | 585 / 4532 |
| 50 | 0.129082 | 0.099072 | 717 / 4532 |
| 100 | 0.129082 | 0.099072 | 872 / 4532 |

K50/K100 增加 pool recall，但 HR@20/NDCG@20 视图仍以前 20 排名计算。prefix consistency、invalid、duplicate、non-finite safety 均通过。`test_read=false`。

数据来源：`results/s6_cost_aware_aux/Industrial_and_Scientific/s6_2_smoke_audit_report.json`，`results/s6_cost_aware_aux/Industrial_and_Scientific/s6_3_formal_direct_sasrec_audit_report.json`。

## 8. CF + Direct候选融合

S6-4 使用冻结 CF full-valid candidate artifacts，通过 validation split manifest 过滤出 CF split views，再与 direct K20/K50/K100 candidates 合并。合并检查 `row_index`、target、history 对齐，保留 `from_cf`、`from_sasrec_direct`、overlap、source rank 等 provenance。去重是确定性的：CF candidates 保持原始顺序，direct-only candidates 按 direct rank 追加。

K20 aggregate：

| 指标 | 数值 |
|---|---:|
| CF target-in-pool | 629 |
| union target-in-pool | 865 |
| added targets beyond CF | 236 |

候选 uplift 公式：

```text
union_added_targets = union_target_in_pool - cf_target_in_pool
```

因此 `865 - 629 = 236`。K20 是满足 candidate uplift gate 的最小 K；K50/K100 保留为 ablation，不作为后续 ranking 主预算。

数据来源：`results/s6_cost_aware_aux/Industrial_and_Scientific/s6_4_union_validation_report.json`。

## 9. Frozen ranker验证

S6-5 将 S5 frozen projected ranker 应用到 CF + Direct union。它使用 26-feature contract，并采用 `source_independent_projection`：保留源无关历史、流行度、bucket 和 ranking 特征，排除 direct raw-score features，例如 `sasrec_direct_score`、`sasrec_direct_zscore`、`sasrec_direct_top1_margin`、`sasrec_direct_score_minus_topk_mean`。

S6-5 aggregate：

| 指标 | 数值 |
|---|---:|
| HR@20 | 0.141659 |
| NDCG@20 | 0.088344 |
| hits@20 | 642 |
| CF preserved/lost | 451 / 13 |
| direct-only recovered | 171 / 268 |

CF preservation 公式：

```text
CF_preservation = preserved_CF_hits / original_CF_hits
```

valid_select 上 S6-5 为 `91 / 95 = 95.79%`，低于 97% gate，因此 ranking verdict 是 partial。direct recovery 公式：

```text
Direct_recovery = recovered_non_CF_top20_targets / available_non_CF_top20_targets
```

数据来源：`results/s6_cost_aware_aux/Industrial_and_Scientific/s6_5_frozen_ranker_validation_report.json`。

## 10. S6-6 residual ranker

S6-6 尝试一个轻量 residual ranker：linear pairwise logistic residual，使用 direct score features、residual scale、clipping 和 CF head quota。64 个配置全部在读取 valid_select 之前预声明，训练只用 valid_fit。

代表性结果：

| Config | Split | HR@20 | NDCG@20 | CF preserved/lost | direct recovery |
|---|---|---:|---:|---:|---:|
| aggressive `alpha1_l20.001_clip0.5_cfq0` | valid_fit | 0.155263 | 0.095161 | 279 / 8 | 112 |
| aggressive `alpha1_l20.001_clip0.5_cfq0` | valid_select | 0.137072 | 0.084272 | 90 / 5 | 37 |
| preservation `alpha0.1_l20.001_clip0.25_cfq18` | valid_fit | 0.133459 | 0.085736 | 282 / 5 | 66 |
| preservation `alpha0.1_l20.001_clip0.25_cfq18` | valid_select | 0.116303 | 0.076331 | 93 / 2 | 19 |

aggressive 配置提升强但 valid_select CF preservation 不足；preservation 配置保住部分 CF，但 uplift 和 direct recovery retention 不足。最终 `REVISE_LIGHTWEIGHT_RANKER`，无新模型冻结，valid_gate 未打开。

## 11. S6-6R promotion gate

S6-6R 是最后允许的 lightweight revision：CF-anchored direct-candidate promotion gating。它从 CF order 出发，只允许 direct-source-exclusive candidates 替换 top20 尾部 CF candidates，不做全局 rerank。训练 pairs 只在 valid_fit 内构造，threshold 只用 valid_fit 校准，policy grid 为 12 个预声明配置。

代表性 valid_select：

| Policy | HR@20 | NDCG@20 | CF preserved/lost | direct recovery |
|---|---:|---:|---:|---:|
| best-HR `mp3_thr0.800224` | 0.119418 | 0.066684 | 90 / 5 | 25 |
| best-preservation `mp2_thr0.958302` | 0.115265 | 0.065666 | 95 / 0 | 16 |

best-HR 仍低于 97% CF preservation；best-preservation 牺牲太多 direct recovery 和 ranking uplift。最终 `REVISE_STAGE_CLOSEOUT`，valid_gate 未打开。

## 12. 成本分析

| Runtime surface | Seconds | Seconds/sample | Samples/sec | Memory evidence |
|---|---:|---:|---:|---:|
| Direct SASRec model inference | 24.967018 | N/A | N/A | peak CUDA 13,118,464 bytes |
| Direct SASRec candidate pipeline wall | 34.233973 | 0.007553833 | 132.383115 | peak CUDA 13,118,464 bytes |
| SASRec-SID Qwen beam-50 candidate pipeline wall | 662.68 | 0.146222418 | 6.838897 | max RSS 1,836,448 KB |

Wall-time factor 公式：

```text
Qwen_wall / Direct_wall
```

即 `662.68 / 34.233973 = 19.36x`。Direct wall time 是 Qwen pipeline wall time 的 `5.17%`。这不是 pure model-kernel inference ratio，因为 Qwen profile 没有分离纯 generation model time。

数据来源：`results/s6_cost_aware_aux/Industrial_and_Scientific/s6_cost_evidence_report.json`。

## 13. 最终阶段结论

### Candidate-layer verdict

GO。Direct SASRec 正式验证为强行为候选源，提供 236 个 CF candidate pool 之外的 target。

### Ranking-layer verdict

PARTIAL。S6-5 projected ranker 有 top20 增益，但 valid_select CF preservation 为 95.79%，未达到 97%。

### Lightweight-ranker verdict

STOP FURTHER VALIDATION TUNING。S6-6 和 S6-6R 都没有产生 gate-complete 新模型。

### Overall verdict

`S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT`

## 14. 局限性

- 严格 CF preservation gate 未完全满足。
- S6-5 ranker 来自旧 auxiliary semantics，并非为 direct score 特征重新训练。
- validation population 有限，且 test 保持未打开。
- 成本比较是 candidate-pipeline wall-clock，不是 pure model-time。
- 没有生产在线延迟测量。
- 没有端到端 serving integration。

## 15. 后续研究方向

后续工作可以研究 listwise ranker、calibrated uncertainty、teacher-student ranking、更大的独立训练 split、多任务 CF preservation loss、direct retrieval serving 集成、更强 SASRec/LightGCN behavior encoder，以及 semantic-behavior joint representation。这些都不是 S6 已完成结果。
