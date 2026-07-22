# MiniOneRec Optimization Phase 4 Architecture Audit

> 主题：ChronoSID + UniSGR Inspired Architecture Audit
> 日期：2026-07-22
> 仓库：`/home/dell/projects/MiniOneRec`
> 阶段边界：本报告只做代码与架构审计；未修改代码或配置，未启动训练、推理、GPU 实验，也未连接 AutoDL。

## Executive Summary

本次审计的核心结论如下。

1. **ChronoSID 的 gap-token 思想可以低风险迁移。** 当前数据预处理曾使用 timestamp 做用户内排序、时间范围过滤和全局时间切分，但 timestamp 在写入 `.inter` 和最终 MiniOneRec CSV 时被丢弃，Qwen、SASRec-SID generator 和所有 reranker 都只看到顺序、不看到时间间隔。恢复 timestamp 并把历史 gap token 插入 decoder-only Qwen 的 prompt 即可；输出 SID、SID lookup 和 constrained decoding 均可保持不变。
2. **仅增加 gap token 不需要重新训练 SID。** Text-SID、CF-SID、SASRec-SID 都可被当成不透明的 item code tuple；需要重新进行的是 Qwen SFT（至少要学习新增 gap-token embedding，并让 decoder 学会使用时间条件），不是 item embedding、RQ/KMeans 或 SID mapping。
3. **ChronoSID 的 TA-FAMAE 在当前仓库没有直接对应实现。** 它处于“item representation → SID quantization”之间，概念上对应 `rq/text2emb/amazon_text2emb.py`、`build_cs_embeddings.py`、`build_sasrec_embeddings.py` 的上游表示学习位置，而不是 `sft.py`。实现 TA-FAMAE 会产生新 item embedding，进而要求重做 SID、CSV、tokenizer 扩展、SFT 和所有下游候选，因此应后置。
4. **当前 rerank 完全是 generator 外部后处理。** `rerank.py`、Stage 7 pairwise linear ranker 和 S6 frozen/residual ranker均读取落盘 JSONL/手工特征，不读取 Qwen hidden state，也不与 Qwen optimizer 共享参数。任何 ranking loss 都不能回传 generator。
5. **UniSGR 的共享表示思想可迁移，但不能原样复制。** UniSGR 是 encoder-decoder、multi-objective、工业多行为架构；当前 MiniOneRec 是 Qwen decoder-only causal LM，而且本地 Amazon CSV 只有单一 next-item positive，没有 click/cart/purchase、曝光负样本或 rating。适合先实现单目标版本：同一个 Qwen backbone 同时承担 SID generation 和 candidate scoring，使用 `L = L_gen + alpha * L_rank`，再视结果决定是否增加 distribution alignment。
6. **Qwen hidden state 可以作为 ranking representation。** 对候选 SID 做 teacher forcing，把候选 SID token 位置的最后层 hidden state做 last-token pooling 或 masked mean pooling，即可得到 history-conditioned、candidate-aware 表示。主要风险不是可行性，而是候选前向成本、3/4-token SID 的 mask 正确性、false negatives 和 joint loss 对生成能力的干扰。
7. **推荐顺序：Gap Token → hidden-state probe → 单流 joint training → 多源候选 joint training → rank-to-generation alignment → TA-FAMAE/GAOQ。** 不建议第一轮引入 MoE、VA-PMTP、Task-Aware Token、FACL 或 STARK；当前数据与工程条件不足以支持这些组件的公平验证。

---

# 1. Current MiniOneRec Architecture

## 1.1 审计范围与当前 source of truth

本报告重点读取了以下主线。

- 数据与原始顺序：`data/amazon18_data_process.py`、`data/amazon23_data_process.py`、`data/process.py`、`convert_dataset.py`。
- SID 表示与量化：`rq/text2emb/amazon_text2emb.py`、`rq/rqvae.py`、`rq/generate_indices.py`、`rq/rqkmeans_constrained.py`、`rq/rqkmeans_plus.py`、`run_rqkmeans_with_emb.py`。
- CF/CS/SASRec 表示：`build_cs_embeddings.py`、`build_sasrec_embeddings.py`、`scripts/build_sasrec_v3_behavior_sid.py`。
- SID 映射：`build_sid_mapping.py`、`rewrite_sid_csv.py`、`utils_sid.py`、`update_sid_manifest.py`。
- SFT：`sft.py`、`data.py`、`scripts/run_sft_ddp.sh`、`scripts/run_sft_smoke_sid_version.sh`。
- 生成与指标：`evaluate.py`、`LogitProcessor.py`、`calc.py`、`calc_plus.py`。
- item candidate、融合与排序：`evaluate_candidates.py`、`fuse_dual_sid_candidates.py`、`rerank.py`、`scripts/train_stage7_p2_history_ranker.py`、S6 candidate/ranker scripts。
- 当前阶段证据：`docs/s6_final/`、`results/s6_cost_aware_aux/`、`results/stage7_validation_protocol/`。

需要特别说明：当前工作区包含大量既有未提交/未跟踪文件；本报告没有改动它们。仓库内 `data/Amazon/sid_maps/experiment_manifest.json` 当前只登记 `text` 版本，而本地实际还有 `cf_k512_dedup` 和 `sasrec_v3_k512_dedup`。因此，“当前架构”以代码和实际 artifact 共同判断，不能只依赖该 manifest。

## 1.2 总体架构映射

```text
Amazon raw review
  ├─ timestamp: 过滤、用户内排序、全局时间切分
  └─ item metadata: title / description
             │
             ├──────── Text embedding (frozen Qwen mean pooling)
             │                  │
             ├──────── CF embedding (train-only co-occurrence → PPMI → SVD)
             │                  │
             └──────── SASRec item embedding (train fit, valid checkpoint select)
                                │
              item embedding → residual quantization / RQ family → SID index
                                │
               item_id → item_sid / history_item_sid CSV rewrite
                                │
                    Qwen decoder-only SFT on SID generation
                                │
                  constrained beam generation of SID tuples
                                │
                 SID bucket / prefix → item candidates
                                │
              Text/CF/SASRec stream fusion or CF+Direct union
                                │
                  external heuristic / learned reranker
                                │
                       HR / NDCG / MRR
```

“Dual-SID”在当前仓库中不是一个 item 同时输出两组 SID 的单模型，也不是一个联合 tokenizer。它是两个独立 SID namespace、通常配两个独立 Qwen checkpoint，分别生成候选后再做 item-level union/RRF/rerank。

## 1.3 当前 SID pipeline 完整流程

### 1.3.1 Text-SID

1. `data/amazon18_data_process.py` 或 `data/amazon23_data_process.py` 读取 review 与 metadata，做时间过滤、K-core、item/user remap、用户内 timestamp 排序和 sliding-window next-item 样本。
2. `rq/text2emb/amazon_text2emb.py` 把 item 的 `title + description` 输入冻结 Qwen `AutoModel`，对 non-padding last hidden states 做 masked mean pooling，保存 `*.emb-qwen-td.npy`。
3. 仓库支持多种 SID quantizer：RQ-VAE、RQ-KMeans、constrained RQ-KMeans、RQ-KMeans+；最终输出 `item_id -> [<a_i>, <b_j>, <c_k>, ...]` 的 `index.json`。
4. `convert_dataset.py` 读取 `*.index.json`，把 token list直接连接成 SID 字符串，写入 `history_item_sid` 和 `item_sid`。
5. `build_sid_mapping.py` 可从 index/info/CSV 交叉核验并生成 `item2sid`、`sid2items`、`valid_sid_set`、`item_mapping` 及 manifest entry。

当前本地基础 Text artifact 位于 `data/Amazon/index/`，是三层 `<a_*>/<b_*>/<c_*>` SID。结果文档还引用 `text_mbk_k512_dedup`，但该版本的完整本地 SID 目录、generation report 和 manifest entry 当前缺失，不能仅凭文件名反推其全部构造参数。

### 1.3.2 CF-SID / CS-SID

1. `build_cs_embeddings.py` 只读取 train CSV。
2. 每个训练样本把 history 与 target item 组成上下文，统计 item-item 共现。
3. 共现矩阵转换为 PPMI，再用 `TruncatedSVD` 得到 CF embedding；没有 train signal 的 item 保持零向量。
4. CF-SID 直接量化 CF embedding；CS-SID 则先做

   ```text
   z_cs = normalize(alpha * normalize(z_text) + (1-alpha) * normalize(z_cf))
   ```

   CF 不可用的 item 回退到 text embedding。
5. `run_rqkmeans_with_emb.py` 对归一化 embedding 做多层 global residual MiniBatchKMeans：每层拟合当前 residual，减去 assigned centroid，再进入下一层。
6. 三层 prefix发生 collision 时，`append_collision_suffixes()` 按 bucket 内 item 顺序附加 `<d_1>...<d_n>`，形成 3/4-token 混合 SID。
7. `rewrite_sid_csv.py` 根据 item ID 重写 train/valid/test CSV 的两列 SID，并在 item 缺失或 mapping 冲突时失败。

本地 `cf_k512_dedup` 的 report显示：三层 prefix 在 dedup 前 collision 很高，2182/3686 item需 `<d_*>` suffix；完整 SID dedup 后一一映射。这个 `<d_*>` 是 collision disambiguation token，不是第四层推荐语义。

### 1.3.3 SASRec-SID

1. `build_sasrec_embeddings.py` 用 train CSV 构造 shifted sequence examples，训练一个小型 causal Transformer SASRec；valid 只用于 checkpoint selection，CLI 不接受 test。
2. 模型的 item embedding table被导出为 `*.sasrec_item_emb.npy`；train cold item使用 train-positive item embedding均值回退，不读 valid/test 生成回退。
3. `scripts/build_sasrec_v3_behavior_sid.py` 检查 artifact manifest、SHA、item order、finite 和 test-read policy，再调用 `run_rqkmeans_with_emb.py`。
4. 后续量化、`<d_*>` collision suffix、mapping 与 CSV rewrite与 CF-SID 相同。

本地 `sasrec_v3_k512_dedup` 是 128 维、三层 residual MiniBatchKMeans。dedup 前 199/3686 item需 suffix，显著少于本地 CF-SID；dedup 后完整 SID 唯一。

### 1.3.4 SID mapping / manifest contract

一个完整 SID version应至少有：

```text
index.json             item_id -> SID token list
info.txt               full_sid<TAB>title<TAB>item_id
item2sid.json          item_id -> full SID
sid2items.json         full SID -> item id list
valid_sid_set.json     合法完整 SID 集合
item_mapping.json      item metadata + SID token list
train/valid/test.csv   version-matched SID columns
generation_report.json
manifest entry
```

`utils_sid.py` 的 SID parser要求 token匹配 `<[A-Za-z]_\d+>`，因此 `<a_*>/<b_*>/<c_*>/<d_*>` 都可解析。`build_sid_mapping.py` 以 index、info、CSV 为多源证据并可在冲突时严格失败。`update_sid_manifest.py` 负责把版本路径登记进 manifest。

当前一致性问题：manifest只登记基础 `text`；CF/SASRec artifact存在但未登记。部分 runner依靠 `data/Amazon/sid_versions/<version>/<category>` fallback继续工作，这降低了 manifest作为唯一 source of truth 的可靠性。下一阶段实现前应先补齐而不是继续扩大 fallback。

## 1.4 SID token 如何进入 SFT 数据

`convert_dataset.py` 与 `rewrite_sid_csv.py` 把 SID 作为普通字符串写入 CSV：

```text
history_item_sid = ['<a_...><b_...><c_...>', ...]
item_sid         = '<a_...><b_...><c_...>'
```

`data.py::SidSFTDataset.get_history()` 使用 `eval()` 解析历史 list，将各 item SID 以逗号连接到自然语言 prompt；target SID 保持无空格连接，并追加换行。

推荐 SFT 的实际格式为：

```text
Below is an instruction ...

### Instruction:
Can you predict the next possible item that the user may expect?

### User Input:
The user has interacted with items <a_i><b_j><c_k>, <a_x><b_y><c_z>
in chronological order. Can you predict the next possible item that the user may expect?

### Response:
<a_t><b_u><c_v>\n<EOS>
```

训练 labels 对整个 instruction + prompt置 `-100`，只监督 response SID、换行及 EOS。

当 `sft_task_mode=all/mixed` 时，`sft.py` 还会拼接：

- `SidItemFeatDataset`：title → SID 与 SID → title；
- `FusionSeqRecDataset`：history SID → target title。

当 `sft_task_mode=sid_only/rec_only` 时，只训练 history SID → target SID。这意味着“当前 generation loss”在 mixed 模式并不纯粹是 SID recommendation loss，还包含语言对齐任务。

## 1.5 item_sid、history_item_sid 如何生成

有两条路径。

1. **首次转换：** `convert_dataset.py` 从原始 `*.inter` 的 history item IDs和 target item ID查 `index.json`；每个 token list直接 `''.join(tokens)`。target mapping缺失时整行被跳过；history mapping缺失时该 history item会被跳过，因此应依赖后续一致性检查确保 ID/SID 等长。
2. **SID 版本切换：** `rewrite_sid_csv.py` 从 CSV 的 `history_item_id`、`item_id` 重新查 version-specific `item2sid.json`。它核对原 history 长度、统计 missing，并在任何 item不在 mapping时写报告后失败。这是 CF/SASRec/新 SID 版本更安全的主路径。

## 1.6 tokenizer / vocabulary 如何处理 SID token

`sft.py::TokenExtender` 遍历 `sid_index_path` 中所有 token，去重排序后执行：

```python
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```

因此每个 `<a_i>` 等是一个新增原子 token，而不是由 Qwen原词表拆分。训练结束后 model与 tokenizer一起保存到 `final_checkpoint`。评估时 `evaluate.py` 从 checkpoint加载 tokenizer，确保生成约束使用同一 token ID。

这些 SID token不是 Hugging Face `special_tokens`；它们只是新增普通 vocabulary row。不同层通过 token字符串中的 `a/b/c/d` 天然占据不相交 token集合，因此当前架构已经实现“level-disjoint vocabulary”，不需要再做 ChronoSID 的数值区间 remapping。

## 1.7 SFT / Generation pipeline

### 模型类型

当前 SFT 和 generation模型是 **decoder-only causal language model**：`AutoModelForCausalLM`。仓库主线 runner使用 Qwen2.5 0.5B/1.5B 等 checkpoint。它不是 ChronoSID 的 T5 encoder-decoder。

### Model loading

- 默认从 pretrained checkpoint加载 `AutoModelForCausalLM`；也支持从 config随机初始化。
- SID token加入后 resize embedding。
- `freeze_LLM=True` 时冻结全部参数，只允许新增 token embedding rows通过 gradient hook更新；常规主线使用 `freeze_LLM=False`。
- DDP由 `torchrun` + Transformers `Trainer` 驱动。

### Data collator

当前使用 `transformers.DataCollatorForSeq2Seq` 对 `input_ids`、`attention_mask`、`labels` 动态 padding；tokenizer使用 left padding，label padding按 collator规则变成 ignore index。它能处理单序列 causal-LM SFT，但不能直接处理未来 joint ranking需要的 `[batch, candidates, candidate_length]` 张量。

### Generation loss

`sft.py` 没有自定义 `compute_loss`；Transformers `Trainer` 把 labels传给 Qwen causal LM，由模型内部完成 shifted cross entropy：

```text
L_gen = - sum_{t in response positions} log p(y_t | prompt, y_<t)
```

prompt positions因 label为 `-100` 不参与 loss。mixed 模式中相同机制也监督 title等辅助 response。

### 输出 SID 如何生成

`evaluate.py`：

1. 从 `info.txt` 收集所有合法 full SID；
2. 把 `### Response:\n<SID>` tokenization构造成 prefix hash/trie等价结构；
3. `ConstrainedLogitsProcessor` 在每个 beam step把非合法后继 token logit置为 `-inf`；无合法后继时强制 EOS；
4. 调用 `model.generate(num_beams=N, num_return_sequences=N)`；
5. decode并截取 `Response:` 后内容，保存每个样本的 SID list。

虽然 `generate()` 请求了 `output_scores=True`，当前代码没有保存 `sequences_scores` 或逐 token log probability。这也是后续 rank-to-generation alignment与 score-aware fusion需要补的观测缺口。

## 1.8 Evaluation pipeline 代码映射

| 阶段 | 主要代码 | 当前逻辑 | 主要输出 |
|---|---|---|---|
| generation | `evaluate.py`, `LogitProcessor.py` | Qwen constrained beam search生成合法 SID tuple | legacy prediction JSON，`predict=[sid...]` |
| SID metric | `calc.py`, `calc_plus.py` | SID exact hit、prefix hit、invalid/duplicate、分组指标 | eval report / per-sample CSV |
| candidate extraction | `evaluate_candidates.py` | exact SID bucket先展开为 item；可选 prefix@3/@2继续扩展；按首次出现去重 | `candidates.jsonl` |
| candidate expansion | `evaluate_candidates.py::expand_candidates` | 保存 source SID、prefix、beam rank、bucket size、source type | candidate details + pool recall |
| dual fusion | `fuse_dual_sid_candidates.py` | 对 Text与 behavior/CF stream做 union、min-rank或 weighted RRF | dual fused candidate JSONL |
| current S6 union | `scripts/s6_cf_direct_union_validation.py` | frozen CF候选与 direct SASRec候选 union，保留 provenance | K20/K50/K100 union views |
| heuristic rerank | `rerank.py` | SID rank、source、train popularity、history/recent embedding cosine、bucket penalty线性加权 | reranked JSONL + HR/NDCG |
| learned rerank | `scripts/train_stage7_p2_*ranker.py` | valid_fit上以 target-vs-negative pairwise logistic训练轻量线性模型 | frozen model JSON + reranked candidates |
| S6 ranker | `scripts/s6_frozen_ranker_validation.py`, `scripts/s6_lightweight_ranker_validation.py`, `scripts/s6_direct_promotion_gate_validation.py` | frozen projection、residual pairwise、promotion gate；均与 Qwen分离 | ranked candidates + gate reports |
| final metric | 上述脚本内 `rank_metrics`，以及 S6 audit scripts | item-level HR/NDCG/MRR、pool recall、CF preservation、direct recovery | JSON/CSV/Markdown evidence |

当前 S6证据说明了为什么值得做 joint training：K20 CF+Direct union在 valid各 split中明显提高 target-in-pool，但已有轻量 ranker不能稳定地兼顾新增命中与 CF hit preservation。这个问题不是“再做一次无状态后处理”能够自然解决的；它正对应 generator representation与最终 item ranking目标之间的断层。

## 1.9 当前本地缺口

- processed train/valid/test CSV没有 `history_timestamp`、`timestamp`、rating或 action type。
- raw review/intermediate timestamp数据不在本地；现有审计明确标记 raw unavailable。
- 本地没有 Qwen SFT `outputs/` checkpoint/tokenizer目录。
- manifest只登记基础 Text-SID；CF、CS、SASRec及文档引用的 `text_mbk_k512_dedup`未完整登记。
- `sasrec_v3_k512_dedup` 本地有 train/valid rewrite，但没有 test CSV；本阶段不需要 test，但下一阶段最终确认前需要补齐。
- legacy Qwen候选只有部分 Text文件；没有一套完整的 train/valid Text+CF+SASRec beam artifact可直接用于 joint ranking数据构造。
- generation beam score没有落盘。

---

# 2. ChronoSID Mapping

## 2.1 ChronoSID 方法拆解

ChronoSID包含三个层次，不应只等价成“加时间 token”。

1. **TA-FAMAE：** 用历史上下文和结构化 item fields做 field-aware masked reconstruction，并用 target item representation预测与前一交互的 log time gap；训练结束后为每个 item导出确定性 frozen embedding。
2. **GAOQ：** balanced Level-1 clustering；Level-1内部做 Level-2 local clustering，再用全局 orthogonal anchors与 assignment对齐 local label；Level-3用 representation-aware orthogonal assignment解决同 prefix collision，得到唯一三层 SID。
3. **Temporal Seq2Seq：** 把历史 inter-interaction gap离散成固定 log-scale token，在每个历史 SID tuple前插入一个 gap token；gap只做输入条件，target仍只生成 SID。

论文消融表明 gap-token injection的增益大于单独 TA-FAMAE，5个时间桶优于过粗或过细的桶数。因此项目计划把 gap token作为最低风险入口是合理的。

## 2.2 TA-FAMAE 对应 MiniOneRec 哪个模块

当前没有一对一实现。其架构槽位是：

```text
raw item fields + timestamped history
                ↓
item representation learner       ← TA-FAMAE 应在这里
                ↓
RQ / residual KMeans
                ↓
SID mapping
```

对应代码边界为：

- Text路径：替代或增强 `rq/text2emb/amazon_text2emb.py` 的 frozen Qwen mean pooling；
- CF/CS路径：替代或融合 `build_cs_embeddings.py` 产生的 PPMI/SVD representation；
- SASRec路径：与 `build_sasrec_embeddings.py` 同属 behavior-aware item representation，但目标不同。SASRec优化 next-item sequence CE；TA-FAMAE优化 masked field reconstruction + time-gap regression。

TA-FAMAE **不对应** `data.py`、`sft.py` 或当前 reranker。把它放到 Qwen SFT中只加一个时间 loss，不能复现论文的“先学 item embedding，再量化 SID”含义。

## 2.3 当前 MiniOneRec 是否利用 timestamp

结论是“**数据切分利用，模型表示不利用**”。

已经利用的地方：

- review按时间范围过滤；
- 每个用户内部按 `unixReviewTime` / Amazon23 timestamp排序；
- sliding-window history按时间顺序构造；
- 所有 interaction samples按 target timestamp全局排序后做 8:1:1 split。

没有利用的地方：

- `.train/.valid/.test.inter` 只写 user、history item IDs、target item ID；
- `convert_dataset.py` 最终 CSV只有七列，无 timestamp；
- `SidSFTDataset` prompt只写“in chronological order”；
- SASRec builder只读 item顺序；
- fusion/rerank特征没有 recency或 gap；
- target-side time只在原始预处理内存在，本地已不可恢复。

因此两个 item顺序完全相同但间隔不同的历史，在当前 generator看来完全相同。

## 2.4 Gap token 的推荐适配

ChronoSID把 gap放在 encoder侧；MiniOneRec没有 encoder。适配方式是在 Qwen causal prompt中把 gap token放在对应历史 SID tuple之前：

```text
<g_start><a_1><b_2><c_3>,
<g_lt_1d><a_4><b_5><c_6>,
<g_1d_1w><a_7><b_8><c_9>
```

Qwen response仍然是：

```text
<a_target><b_target><c_target>[<d_target>]\n<EOS>
```

建议第一版遵循项目计划和论文的 5-bin结论：

```text
<g_lt_1h>
<g_1h_1d>
<g_1d_1w>
<g_1w_1mo>
<g_ge_1mo>
```

再加 `<g_start>`。所有边界换算必须在一个纯函数中完成，统一使用秒，并记录 timezone无关的 Unix timestamp。对于相同 timestamp，gap=0应进入 `<g_lt_1h>`；负 gap、长度不一致、非单调历史必须直接报错，不能静默 clip。

只允许历史内部 gap作为输入。`last_history_timestamp → target_timestamp` 的 gap不得输入模型，否则会在 next-item任务中引入未来信息；它只能用于 long-gap slice诊断。

## 2.5 增加 gap token 需要修改/新生成哪些数据文件

下一阶段应新建 versioned temporal dataset，不覆盖冻结基线。

需要恢复或新生成：

- timestamp-preserving atomic data：每行至少包含 `history_item_id`、`item_id`、`history_timestamp`、`timestamp`；
- Text-SID temporal train/valid/test CSV；
- CF-SID temporal train/valid/test CSV；
- SASRec-SID temporal train/valid/test CSV；
- 每个 temporal version的 alignment report、bucket distribution、SHA/manifest。

未来代码改动边界：

- `data/amazon18_data_process.py`、`data/amazon23_data_process.py`：写出 timestamp，而不是只保留在内存；
- `data/process.py`：若继续支持 legacy路径，同步 timestamp contract；
- `convert_dataset.py`：解析 timestamp-bearing input并写 `history_timestamp` / `timestamp`；
- `convert_dataset_gpr.py`：只有继续维护 GPR变体时才同步；
- `rewrite_sid_csv.py`：当前会保留未知额外列，原则上无需改变重写逻辑，但必须加测试保证 timestamp byte/value不变；
- `data.py`：新增统一 gap bucketing与 gap-aware `SidSFTDataset` / `EvalSidDataset`；
- `sft.py`：显式注册 gap tokens，并保存 temporal schema；
- `evaluate.py`：构造与训练完全一致的 gap-aware prompt，并验证 checkpoint tokenizer含全部 gap tokens；
- SID manifest/runners：记录 temporal CSV、bin边界和 tokenizer token列表。

## 2.6 是否需要重新训练 SID

分两种情况。

| 改造 | item embedding重做 | SID重做 | Qwen SFT重做 |
|---|---:|---:|---:|
| 仅 gap token | 否 | 否 | 是 |
| TA-FAMAE | 是 | 是 | 是 |
| GAOQ替换当前 quantizer | 否或是，取决于输入 embedding | 是 | 是 |
| 仅把现有 code level做字符串 remap | 否 | mapping改变，等价于新 SID version | 是 |

已有 Qwen checkpoint可以作为 gap版本的 warm start，但由于 tokenizer新增 row且 prompt分布改变，不能把它当成无需训练的兼容功能。公平实验建议 baseline与 gap版本采用同一初始化、相同样本和训练预算。

## 2.7 是否直接兼容 Text/CF/SASRec SID

**gap token兼容，TA-FAMAE不直接兼容。**

- Gap token按“每个历史 item一个 gap + 一个完整 SID tuple”插入，不关心 tuple由 Text、CF或SASRec embedding生成。
- 3-token与带 `<d_*>` 的4-token SID都兼容；一个 gap对应一个 item tuple，而不是对应每个 SID level。
- Dual-SID的两个 generator必须各自用 version-matched history SID CSV训练；gap token本身可以共享同一组字符串与 bin定义。
- TA-FAMAE会改变 item embedding，因此不能保留原 SID作为严格意义上的 TA-FAMAE结果；若只把 TA loss加到 generator hidden state，那是另一个方法，不应命名为 ChronoSID TA-FAMAE。

## 2.8 ChronoSID SID construction 与当前 SID 的区别

| 维度 | ChronoSID / GAOQ | 当前 Text-SID | 当前 CF-SID / SASRec-SID |
|---|---|---|---|
| item representation | structured fields + history context；masked reconstruction + time regression | frozen Qwen title+description mean pooling（本地基础主线） | PPMI/SVD CF或 train-only SASRec item embedding |
| Level-1 | balanced KMeans | 仓库支持 RQ-VAE/多种 KMeans；本地 Text provenance不完整 | global residual MiniBatchKMeans |
| Level-2 | 每个 L1簇内 local KMeans | 取决于具体 Text版本 | 全局 residual codebook，不是 nested local cluster |
| label alignment | local clusters对齐全局 orthogonal anchors | 无该步骤 | 不需要处理 local label permutation，因为每层 codebook本来就是全局的 |
| collision resolution | representation-aware orthogonal anchors + assignment，固定三层唯一 SID | 取决于版本 | prefix collision按 item顺序追加 `<d_i>`，形成3/4-token混合长度 |
| token remap | code level remap到全局不冲突词表 | `<a>/<b>/<c>`前缀天然区分 level | `<a>/<b>/<c>/<d>`天然区分 level |
| temporal input | 5-bin gap token + start token | 无 | 无 |

## 2.9 可直接借鉴与不适合当前架构的部分

### 可直接借鉴

- 5-bin固定 gap token与 start token；
- 历史 gap输入、target-side gap只做诊断的防泄漏原则；
- gap-bin sensitivity（3/5/7）和 long-gap slice评估；
- balanced assignment作为未来 SID quality ablation；
- representation-aware collision resolution，用于替代顺序敏感的 `<d_i>` suffix；
- 对 timestamp、SID、CSV与manifest做三方对齐检查。

### 不应第一阶段直接搬用

- T5 encoder-decoder：当前 Qwen decoder-only无需为 gap token更换 backbone；
- 完整 TA-FAMAE：需要结构化 field vocabulary、timestamped samples和新的 representation training pipeline；
- GAOQ Level-2 orthogonal alignment：当前 residual quantizer使用全局 codebook，不存在论文要解决的 local-index permutation问题；只有改成 nested clustering后才有意义；
- ChronoSID的 token数值 remapping：当前字母前缀已解决 level collision；
- 一次同时改 representation、quantizer与generator：无法定位收益来源，且会让现有 Text/CF/SASRec baseline失去可比性。

---

# 3. UniSGR Mapping

## 3.1 不能把 UniSGR 简化为 reranker 后处理

UniSGR的核心是：generation与ranking共享 SID representation、user/history representation及 decoder states；ranking BCE梯度通过共享 decoder回传，使生成表示本身朝最终业务目标移动。其完整系统还包括：

- multi-scenario pretraining；
- multi-behavior click/cart/purchase alignment；
- Value-Aware Parallel Multi-Token Prediction；
- Target Attention + PLE multi-objective ranker；
- Task-Aware Tokens与 Funnel-Aware Contrastive Learning；
- STARK tree-attention inference。

当前 MiniOneRec最多只能合理迁移其中“共享 decoder representation + joint loss”这一核心。其余组件要么缺数据，要么与本阶段研究问题无关。

## 3.2 当前 MiniOneRec rerank 是否完全独立

是。

证据包括：

- `evaluate.py` 先把 SID预测写成 JSON；
- `evaluate_candidates.py` 再离线展开 item；
- `fuse_dual_sid_candidates.py` 只使用 item ID、source rank与provenance；
- `rerank.py` 只使用手工特征与外部 embedding；
- Stage 7 learned ranker是单独的纯 Python pairwise linear模型，模型写入 `model.json`；
- S6 frozen/residual/promotion ranker读取 frozen candidate artifact，不加载 Qwen checkpoint；
- 当前 Qwen `forward()` 从未接收 ranking label或 candidate set。

因此当前结构是：

```text
Qwen generator --serialize--> candidate JSONL --CPU features--> ranker
```

序列化边界同时也是梯度边界。

## 3.3 ranking loss 是否可以影响当前 generator

当前不可以。要使其可以，至少必须满足：

1. ranking head与 Qwen backbone处于同一 PyTorch module / autograd graph；
2. candidate representation由当次 Qwen forward产生，不能从 detached JSON特征产生；
3. `compute_loss()` 同时返回 `L_gen` 和 `L_rank`；
4. optimizer包含共享 decoder参数与 rank head参数；
5. 训练日志证明 `L_rank` 对共享 decoder有非零梯度。

仅把 external ranker score加入 generation后的融合、用 ranker重新排序 beam、或把 ranker结果作为额外 CSV label但冻结 Qwen，都不构成 UniSGR式 joint optimization。

## 3.4 Qwen decoder hidden state 是否可作为 ranking representation

可以，推荐从最简单、可审计的 teacher-forcing方案开始。

对用户历史 prompt `P_u` 和候选 item的当前 SID `S_i=(s_1,...,s_m)`，构造：

```text
[P_u][### Response:\n][s_1]...[s_m][EOS]
```

以 `output_hidden_states=True` 前向。候选表示可选：

```text
z_ui = mean(h_t for t in candidate SID token positions)
```

或：

```text
z_ui = h_last_SID_token
```

第一轮建议同时实现但只选择一个为主配置；last SID token天然看到 prompt与所有此前 SID token，masked mean对3/4-token长度更稳健。不要把 newline、EOS、padding或 prompt token混入 pooling。

当前 token设计已经用 `<a>/<b>/<c>/<d>`区分 SID level，第一版不必再增加 UniSGR Token Type Embedding。若后续发现 `<d_*>`主导 representation，再单独做 level embedding ablation。

decoder-only适配与 UniSGR encoder-decoder的差异：

- 没有独立 encoder memory可复用；user/history表示就在 causal prompt hidden/KV中；
- 每个候选 teacher forcing都要共享同一 prompt，可通过 flatten batching先实现，后续再做 prefix KV cache复用；
- ranking hidden state可合法看到候选 SID本身，因为线上 rerank时候选已知；但训练与推理必须使用同一构造，不能训练看完整 SID、推理只看 beam前缀。

## 3.5 如何构造 candidate ranking training data

### 当前可用标签

本地 processed Amazon CSV只有一个 next-item target。没有 impression、click、add-to-cart、purchase、rating字段，raw数据也不在本地。因此第一版只能做 **single-positive listwise ranking**，不能声称复现 UniSGR multi-objective ranking或 VA-PMTP。

`convert_dataset_gpr.py` / `sft_gpr.py` 中的模拟 value不应被当成真实业务 funnel label。除非拿到原始行为语义并完成审计，否则不要把 rating人工重命名成 click/cart/purchase。

### 每个 query的候选集

```text
C_u = {ground-truth positive}
    ∪ Text beam hard negatives
    ∪ CF beam hard negatives
    ∪ SASRec-SID beam hard negatives
    ∪ direct SASRec top items
    ∪ current external ranker top false positives
    ∪ SID collision/prefix-neighbor negatives
    ∪ optional in-batch/random popularity negatives
```

规范：

- 先按 item ID去重，而不是按 SID去重；
- 所有候选 item统一映射到“当前 joint generator的 SID namespace”再 teacher-force。候选来源可以跨流，但 representation必须同 namespace；
- positive始终注入以保证 listwise loss可计算，同时写 `positive_forced`；自然 candidate recall必须在注入前单独统计；
- hard negative是“未观察为正”，不是可信负反馈，必须承认 false-negative风险；
- 保留 source、source rank、generation score、prefix level、bucket size供诊断，但第一版 rank head的主输入应是 Qwen hidden representation，避免退化回手工 ranker；
- 训练候选应来自 train的 out-of-fold/held-out generation或冻结 checkpoint。直接用已在该 target上训练过的 generator为同一 train row产候选，会放大记忆与候选偏差；
- valid_fit用于 rank训练/校准时，valid_select和 valid_gate必须隔离；test只能在全部超参冻结后一次性确认。

推荐的 ranking dataset schema：

```text
query_id
split
history_item_id
history_item_sid
history_timestamp / gap_tokens (Phase A后可选)
positive_item_id
candidate_item_ids[]
candidate_sids[]
candidate_labels[]          # one positive, remaining unlabeled-as-negative
candidate_mask[]
candidate_sources[][]
source_ranks{}
positive_forced
candidate_pool_hit_before_force
sid_version
generator_checkpoint_sha
candidate_artifact_sha
```

## 3.6 目标 joint architecture

当前：

```text
generator → SID beams → item candidates → external reranker
```

建议 Phase B：

```text
                         ┌─ LM / generation head → SID logits → L_gen
history + target SID ─── Qwen shared decoder
                         │
history + candidate SID ─┴─ pooled decoder state → ranking MLP → L_rank
```

共享参数：

- Qwen input/SID token embeddings；
- Qwen全部或选定上层 decoder blocks；
- RMSNorm等 backbone参数；
- 若 Qwen LM head与 token embedding tied，则 ranking梯度也会间接影响共享 embedding。

独有参数：

- generation head / LM head继续服务 next-token generation；
- ranking head建议 `LayerNorm → Linear → GELU → Dropout → Linear(1)`；
- candidate pooling本身无参数，便于首轮审计。

参数更新原则：

- `L_gen` 更新 backbone、SID embeddings和 LM head；
- `L_rank` 必须更新 rank head并回传到至少部分 shared decoder/embeddings；
- hidden-state probe阶段可以冻结 backbone只训 rank head，但该阶段只回答“表示是否可排序”，不属于最终 joint model；
- joint初期可只解冻 top-N decoder blocks + SID embeddings，确认稳定后再比较 full-backbone joint；若全部冻结，就失去 UniSGR核心。

## 3.7 Loss 设计

当前单 positive场景推荐 listwise softmax：

```text
L_rank = -log exp(score_positive / tau)
               / sum_{j in valid candidates} exp(score_j / tau)
```

总损失：

```text
L = L_gen + alpha * L_rank
```

实现时必须分别对 token数和 query数归一化后再相加，避免候选数量或 SID长度隐式改变 alpha含义。日志至少包含：`L_gen`、`L_rank`、总 loss、generation token accuracy、rank positive position、shared-gradient norm和两个 loss梯度夹角/余弦的抽样诊断。

若后续拿到多行为 labels，才升级到 UniSGR风格多任务 BCE/PLE；当前不应提前增加 click/cart/purchase heads。

---

# 4. Required Code Changes

本节是下一阶段改造方案，不是本阶段已执行的修改。

## 4.1 Phase A — ChronoSID Gap Token（最低风险）

### 目标

在不改变任何 SID mapping的前提下，让 Qwen看到历史交互间隔，验证 temporal signal本身是否提升 generator与最终 item指标。

### 建议文件清单

| 文件 | 未来修改逻辑 |
|---|---|
| `data/amazon18_data_process.py` | timestamp-preserving `.inter`/中间文件；保留 history与target timestamp |
| `data/amazon23_data_process.py` | 同上，并明确毫秒转秒只做一次 |
| `data/process.py` | legacy path同步 timestamp schema（若仍保留） |
| `convert_dataset.py` | 读取并写出 `history_timestamp`、`timestamp`；验证与 item list等长 |
| `data.py` | 统一 `bucket_time_gap()`；gap-aware train/eval prompt；禁止 target gap入模 |
| `sft.py` | 新增 temporal mode与显式 gap token注册；保存 temporal metadata |
| `evaluate.py` | 使用同一 gap formatter；checkpoint tokenizer preflight；输出 temporal mode |
| `scripts/run_sft_ddp.sh` | 传入 temporal CSV、bin spec、mode；拒绝 baseline/temporal路径混用 |
| `scripts/run_sft_smoke_sid_version.sh` | manifest解析 temporal keys与 preflight |
| `scripts/eval_smoke_sid_version.sh` | temporal checkpoint/CSV一致性检查 |
| `update_sid_manifest.py` / 新的 data manifest helper | 记录 timestamp source SHA、bin边界、temporal CSV、token列表 |
| `tests/` 新测试 | bucket边界、非单调时间、长度错位、prompt parity、target-gap leakage、3/4-token SID兼容 |

`rewrite_sid_csv.py` 理论上会原样保留额外 timestamp列，只需要补回归测试；除非测试失败，不必修改主逻辑。

### 修改后的数据流

```text
history_item_id + history_timestamp
              ↓
adjacent gap seconds
              ↓
5-bin tokens + <g_start>
              ↓
interleave(gap_token, full_item_sid)
              ↓
Qwen prompt → unchanged target SID
```

### 是否重新训练

- SID quantizer：否；
- item embedding：否；
- SID CSV：需要生成带 timestamp/gap schema的新版本，但 SID列值不变；
- tokenizer：增加6个 atomic tokens；
- Qwen：需要重新 SFT；
- external ranker：Phase A单独测 generator时可保持冻结；若其输入候选发生变化，应重新报告而不是悄悄沿用旧结果。

### 风险

- 本地没有 timestamp source，无法从七列 CSV可靠恢复；
- decoder-only prompt长度增加约“每个历史 item一个 token”，但还包含自然语言分隔，实际 token增量需测；
- tokenizer若把 gap注册为 special token，`skip_special_tokens=True` 的行为可能掩盖调试输出；第一版建议和 SID一样作为普通 atomic token；
- 历史 timestamp等长/单调问题会直接污染 gap；
- 时间桶边界和时区处理若训练/评估不一致，会造成静默分布漂移；
- target-side gap泄漏是最高优先级测试项。

## 4.2 Phase B — Generation-Ranking Joint Training

### 建议新增/修改文件

| 文件 | 未来职责 |
|---|---|
| 新 `build_joint_ranking_dataset.py` | 从冻结 multi-source candidate artifact构造 query list、positive injection、hard negatives、provenance与SHA |
| `data.py` 或新 `joint_data.py` | 返回 generation sample + variable candidate list + SID-position mask |
| 新 `joint_collator.py` | padding `[B,L]` generation与 `[B,C,Lc]` candidate tensor；candidate mask/label/mapping |
| 新 `modeling_minionerec_joint.py` | 包装 `AutoModelForCausalLM`，暴露 generation logits、hidden pooling、ranking head与 `generate()` delegation |
| `sft.py` 或新 `train_joint.py` | joint Trainer、loss权重、冻结/解冻策略、参数组、日志、checkpoint contract |
| `evaluate.py` | 兼容 joint checkpoint，仅负责 generator beam与 beam score落盘 |
| 新 `evaluate_joint_ranker.py` | 对固定 candidate pool做 teacher-forced shared-head scoring，输出 item ranking |
| `evaluate_candidates.py` | 保存 generation `sequences_scores`/token logprobs或接受 score sidecar |
| `fuse_dual_sid_candidates.py` | 保留作为 candidate source与 external baseline，不再冒充 joint模块 |
| tests | pooling mask、candidate padding、loss手算、rank梯度回传、save/load/generate parity、3/4-token SID |

建议用新 `train_joint.py`，而不是把大量候选逻辑继续塞进 `sft.py`。保留原 SFT入口有利于 baseline parity与回滚。

### Model forward contract

建议输入：

```text
input_ids                 [B, Lg]
attention_mask            [B, Lg]
labels                    [B, Lg]
candidate_input_ids       [B, C, Lc]
candidate_attention_mask  [B, C, Lc]
candidate_sid_mask        [B, C, Lc]
candidate_mask            [B, C]
positive_index            [B]
```

候选维度先 flatten到 `[B*C, Lc]` 前向，再 reshape。输出：

```text
gen_loss
rank_loss
loss
logits
candidate_scores [B,C]
candidate_repr   [B,C,H]   # debug时可选，默认不落盘
```

### 训练阶段建议

1. **B0 representation probe：** 冻结 Qwen，固定候选集，只训练 rank head。目的仅是验证 Qwen hidden state是否优于现有手工 ranker。
2. **B1 partial joint：** 解冻 SID embedding + top-N decoder blocks + rank head，`L_gen + alpha L_rank`。
3. **B2 full joint：** 仅当 B1稳定且有收益，再比较 full-backbone更新。
4. **B3 candidate source ablation：** Text-only、CF-only、SASRec-only、multi-source union；候选池必须固定才能隔离 ranking能力。
5. **B4 refresh ablation：** 比较 frozen candidate与每若干 epoch刷新 hard negatives。首轮不要把 refresh引入主实验，以免训练不可复现。

### 推理路径

```text
shared Qwen generate SID beams
          ↓
SID → item candidate extraction / multi-source union
          ↓
same Qwen teacher-force each candidate SID
          ↓
ranking head score
          ↓
Top-K item
```

这仍是“两步执行”，但不是独立 cascade：生成与排序共享参数，ranking loss已改变 generator representation。它满足本阶段的 UniSGR核心定义。

### 计算优化边界

第一版可把 `B*C`候选直接 batch化，先证明有效性。若有效，再做：

- 同 query共享 prompt prefill / KV cache；
- 只重排 top-C而非整个 expanded pool；
- mixed precision与gradient checkpointing；
- offline teacher-forced candidate representation cache只能用于 frozen probe，不能用于 joint training，否则切断梯度；
- STARK不属于第一轮，因为它改 beam-search kernel，和 joint objective可独立验证。

## 4.3 Phase C — 高级方向

### Rank-to-Generation Distribution Alignment

在同一候选集上定义 generation item score：

```text
g_i = sum_t log p_theta(sid_i,t | history, sid_i,<t>) / length_i^gamma
p_G(i|u) = softmax(g_i / tau_G)
p_R(i|u) = softmax(rank_score_i / tau_R)
```

加入：

```text
L_align = KL(stopgrad(p_R) || p_G)
L_total = L_gen + alpha L_rank + beta L_align
```

首轮建议对 `p_R` stop-gradient，避免 rank head和 generator共同向一个退化分布移动。候选必须含 positive，3/4-token SID要做长度归一化 ablation，且 generation score必须来自 teacher forcing或完整 beam score，不能用当前缺失的 JSON字段假装存在。

价值判断：

- **值得做的前提：** Phase B证明 rank head排序更好，但 generator beam recall/beam order没有同步改善；
- **创新性：** 单纯 KL alignment不是全新范式，创新性中等；若结合 MiniOneRec dual-SID、多源候选、prefix-level score和 collision suffix处理，可形成更有项目辨识度的问题；
- **实现难度：** 高。需要可比的 item-level generation score、温度/长度校准、stable candidate support和额外显存；
- **推荐级别：** Phase C，而不是 Phase B默认组件。

### Prefix-aware ranking alignment

对于 `<a><b><c>[<d>]`，可进一步监督 positive prefix分数高于 hard-negative prefix。但 `<d_*>` 是去重 suffix，不能当作与 a/b/c同等语义层。该方向与当前 constrained trie及 prefix expansion天然相连，创新空间高于简单 KL，但实现和评估都更复杂。

### TA-FAMAE + GAOQ

只有在 gap-only收益得到确认、且能拿到完整 structured fields/timestamp后再做。推荐分别验证：

1. frozen current embedding + gap；
2. TA-FAMAE embedding + current residual quantizer；
3. current embedding + GAOQ；
4. TA-FAMAE + GAOQ + gap。

否则无法区分表示、量化和 temporal prompt三类收益。

### 暂不推荐的 UniSGR组件

- VA-PMTP / Task-Aware Tokens / PLE multi-objective：缺真实多行为 label；
- Funnel-Aware Contrastive Learning：缺曝光、点击未转化等层级负样本；
- Sparse MoE：显存、训练和归因成本过高；
- STARK：属于推理 kernel优化，不回答 ranking supervision是否改善 generator的核心问题。

---

# 5. Risks

| 风险 | 严重度 | 具体表现 | 缓解/门禁 |
|---|---|---|---|
| timestamp不可恢复或错位 | 高 | 当前 CSV无时间；错误 join会把别人的时间赋给样本 | 必须从原始/时间保留中间件重建；按 user/history/target全键对齐并输出SHA与失败样本 |
| target-time leakage | 高 | 把 last-history→target gap放入 prompt，指标虚高 | 训练/推理 schema明确只含历史内部 gap；单元测试扫描 prompt |
| manifest漂移 | 高 | 当前 manifest只有 text，runner fallback掩盖版本错配 | 实现前补齐所有 version entry；checkpoint/CSV/index/info SHA绑定 |
| SID namespace错配 | 高 | Text candidate用 CF SID teacher-force或 history与candidate不在同一空间 | 每个 joint experiment只选择一个 canonical SID namespace；跨流只作为 item candidate source |
| false negative | 高 | 未交互 item不等于不喜欢，尤其相似/重复消费 | hard-negative source分层；label smoothing/采样 ablation；不把 ranking概率解释成真实CTR |
| train target记忆泄漏 | 高 | 用已拟合该 train row的 generator产生过易候选 | out-of-fold候选或冻结更早 checkpoint；记录 generator SHA与 candidate build split |
| valid/test调参泄漏 | 高 | 当前已有大量 validation机制，joint模型容易反复看 select/gate | valid_fit训练、valid_select一次选择、valid_gate最终门禁、test一次性；候选artifact分split隔离 |
| generation-ranking梯度冲突 | 高 | HR排序升但 beam recall/invalid恶化 | alpha warmup、partial unfreeze、分别记录 loss/gradient、generation preservation gate |
| 候选计算成本 | 高 | 每 query×C次 teacher-forced Qwen，远高于线性 rerank | 首轮限制C；flatten batch；成功后再做 prompt KV复用；报告完整latency/memory |
| 3/4-token SID pooling错误 | 中高 | `<d_*>`、newline、EOS或padding进入表示；长度偏置 | 显式 `candidate_sid_mask`；手算测试；按 SID length分组报告 |
| candidate staleness | 中高 | joint更新 generator后，训练 hard negatives来自旧 generator | 主实验固定候选保证可复现；后续单独做 refresh ablation |
| tokenizer/约束树不一致 | 中高 | gap token意外进入输出合法集合；旧 tokenizer拆词 | gap token独立注册，不加入 SID `index.json`；save/load atomicity与 trie测试 |
| 不安全/脆弱的 list解析 | 中 | `data.py` 多处直接 `eval()` CSV字段；新增 timestamp列会扩大失败与输入风险 | 新 temporal/joint路径统一使用 `ast.literal_eval` + schema validator，不继续复制 `eval()` |
| mixed SFT任务干扰 | 中 | `sft_task_mode=all`还输出 title，自变量增加 | Phase A/B主实验先用 sid_only；mixed作为独立 ablation |
| 指标混淆 | 中 | union pool recall被误报为 ranker HR；forced positive污染 recall | 明确区分 natural candidate recall、forced-positive training coverage、final HR/NDCG |
| 本地无GPU | 约束 | 无法验证模型forward显存、速度或数值稳定性 | 本地仅代码/静态测试/脚本；所有训练推理在 AutoDL执行并回传 evidence |
| 完整 UniSGR数据缺失 | 高 | 无 click/cart/purchase/exposure，无法做 BCE/VA-PMTP/FACL | 第一版明确命名 single-objective joint ranking；拿到真实多行为数据前不宣称完整 UniSGR复现 |

关键停止条件：如果无法获得与当前 item/user remap严格对齐的 timestamp，就停止 Phase A，不得用随机时间、均匀间隔或从行号推断 gap。如果无法获得可信的 train/valid候选和 checkpoint provenance，就停止 Phase B的数据构造，不得用 test候选训练。

---

# 6. Recommended Implementation Order

## 6.1 推荐顺序

1. **P0 — 基线与数据契约冻结**
   - 补齐 Text/CF/SASRec manifest；
   - 固定 Industrial validation主线、checkpoint SHA、tokenizer SHA、candidate artifact SHA；
   - 重放当前 baseline只在 AutoDL执行，本地不推理。
2. **A0 — Timestamp恢复审计**
   - 回传 timestamp-aligned data；
   - 只做行数、user/item序列、单调性、gap分布与split边界审计；
   - 不训练。
3. **A1 — 5-bin gap-only**
   - 不重做 SID；
   - 先 Industrial + 单一冻结 SID主线；
   - 与同初始化、同预算 sid-only baseline比较。
4. **A2 — temporal ablation**
   - 3/5/7 bins；
   - shuffled-gap control；
   - long-gap与history-length slice。
5. **B0 — Frozen hidden-state rank probe**
   - 固定 Qwen和 candidate pool；
   - 只训小 ranking head；
   - 判断 hidden state是否提供超过 external feature ranker的信息。
6. **B1 — Partial joint training**
   - `L_gen + alpha L_rank`；
   - 解冻 SID embedding + top decoder layers；
   - 先 single SID namespace、single positive listwise。
7. **B2 — Full joint / multi-source negatives**
   - 仅在 B1通过 generation preservation与 final metric门禁后执行；
   - 候选来源扩展到 Text/CF/SASRec/direct，但统一映射为当前 backbone SID。
8. **AB — Gap + joint组合**
   - 单独验证两者是否互补，不能从 A/B各自结果直接推断叠加有效。
9. **C1 — Rank-to-generation alignment**
   - 只在 rank head好而 beam order不改善时加入；
   - beta、temperature、length normalization做小规模预声明网格。
10. **C2 — TA-FAMAE / GAOQ / prefix alignment**
    - 作为独立高级研究，不和第一版 joint pipeline同时落地。

## 6.2 为什么这样排序

- ChronoSID论文自身表明 gap token是主要收益来源，而且它不破坏现有 SID资产；
- hidden-state probe能在低风险条件下判断 Qwen representation是否适合排序；
- 只有 `L_rank` 真正回传 decoder后才能回答项目核心研究问题；
- distribution alignment、TA-FAMAE和GAOQ会同时改变更多变量，必须在核心 insight成立后做；
- 当前 S6已经证明 candidate pool有额外 headroom，但轻量 ranker存在 preservation问题，因此 joint representation比继续堆 external hand-crafted rule更符合下一阶段目标。

---

# 7. Future Experiment Plan

## 7.1 数据与 protocol

- 首个研究域：`Industrial_and_Scientific`；Office作为跨域确认，不同时调参。
- 保留现有 `valid_fit / valid_select / valid_gate` user-disjoint protocol。
- `valid_fit`：ranking训练、alpha warmup诊断；
- `valid_select`：一次性选择 bins/alpha/beta/候选预算；
- `valid_gate`：只在 fit/select门禁通过后打开；
- test：代码、配置、checkpoint和所有超参冻结后一次性确认。
- 每个实验保存 git SHA、环境、输入/输出 SHA、SID version、tokenizer、checkpoint、candidate source与test-read flag。

## 7.2 Phase A 实验矩阵

| ID | 模型 | SID | 时间输入 | 目的 |
|---|---|---|---|---|
| A0 | 当前 Qwen SFT | frozen current SID | 无 | 同预算可复现 baseline |
| A1 | Qwen SFT | 同 A0 | 5-bin gap | 主实验 |
| A2 | Qwen SFT | 同 A0 | 3-bin | 粗粒度 ablation |
| A3 | Qwen SFT | 同 A0 | 7-bin | 稀疏度 ablation |
| A4 | Qwen SFT | 同 A0 | shuffled 5-bin | 检验真实时间信号而非额外 token/位置效应 |
| A5 | Qwen SFT | CF或SASRec SID | 5-bin | 验证 SID-agnostic迁移性 |

主指标：item HR@20、NDCG@20。辅助指标：SID HR@K、candidate recall@20/50/100、pool recall、invalid SID rate、duplicate rate、MRR、history-length slice、last-history→target gap slice、token长度、latency与peak memory。

## 7.3 Phase B 实验矩阵

| ID | Backbone更新 | Ranking head | 候选 | Loss | 研究问题 |
|---|---|---|---|---|---|
| B0 | 无 | external current ranker | fixed union | external | 当前基线 |
| B1 | frozen | Qwen hidden MLP | fixed union | `L_rank` | hidden state是否可排序 |
| B2 | top layers + SID emb | 同 B1 | fixed union | `L_gen + alpha L_rank` | ranking gradient是否改善共享表示 |
| B3 | full backbone | 同 B1 | fixed union | 同上 | full joint是否必要 |
| B4 | B2最佳 | 同 B1 | Text only / CF only / SASRec only / union | 同上 | hard-negative source贡献 |
| B5 | B2最佳 | 同 B1 | fixed vs refreshed | 同上 | candidate staleness影响 |
| B6 | B2最佳 + A1 | 同 B1 | fixed union | 同上 | gap与joint是否互补 |

建议 `alpha` 首轮只做小而预声明的网格，例如 `{0.01, 0.05, 0.1}`，再加 `alpha=0` parity。不要在 test上扩大搜索。

必须同时报告：

- frozen candidate pool recall（排序上界）；
- generator独立 beam recall与 beam NDCG；
- joint ranker final HR/NDCG；
- external ranker在同一 candidate pool上的结果；
- positive注入前 natural recall；
- generation loss/perplexity、invalid/duplicate；
- rank loss、rank MRR、positive mean rank；
- CF hit preservation与新 source recovery；
- training step时间、inference分阶段 latency、显存。

## 7.4 Phase C 实验矩阵

| ID | Loss | 目的 |
|---|---|---|
| C0 | `L_gen + alpha L_rank` | Phase B最佳 |
| C1 | `+ beta KL(stopgrad(p_R)||p_G)` | rank-to-generation alignment |
| C2 | C1 + generation length normalization ablation | 处理3/4-token SID偏差 |
| C3 | prefix-level alignment | 检查 coarse-to-fine beam质量 |

`beta`、`tau_G`、`tau_R`必须预声明小网格。若 C1只改善 rerank而不改善 generation beam，不能宣称 generator alignment成功。

## 7.5 建议验收门禁

### 数据门禁

- history item、SID、timestamp长度100%一致；
- timestamp单调性100%，负 gap为0；
- baseline与 temporal版本 item ID/SID列逐行一致；
- train/valid/test行数与当前 frozen split一致；
- target-side gap未进入任何 model input；
- manifest、CSV、index、checkpoint SHA可追溯。

### Phase A门禁

- tokenizer中每个 gap marker都是单 token；
- constrained output合法集合不含 gap token；
- baseline parity先通过；
- A1在 valid_select至少改善一个 primary metric且另一个不显著退化；
- invalid SID、candidate recall和 long-gap slice完整报告。

### Phase B门禁

- 手工单 batch测试证明 `L_rank.backward()` 后 shared decoder gradient非零；
- `alpha=0` 与原 SFT在相同条件下 parity；
- fixed candidate pool完全一致，避免把召回变化误记为 ranker变化；
- final HR/NDCG相对 external ranker有可复现提升；
- generator candidate recall/invalid rate不过度退化；
- valid_select通过后才打开 valid_gate。

### Phase C门禁

- generation item score可复现、长度校准透明；
- KL support与candidate mask正确；
- beam order或candidate recall实际改善，而不只是 ranking head继续变强；
- 推理成本增量有独立报告。

## 7.6 需要从 AutoDL 回传的文件

### Phase A 必需

1. **timestamp-aligned interaction data**，优先回传最小、已对齐版本：
   - 当前 train/valid/test每行对应的 `history_timestamp` 与 `timestamp`；
   - 同时包含 user ID、history item IDs、target item ID用于全键核验；
   - 生成脚本、参数、原始文件 SHA和split统计。
2. 如果 AutoDL没有上述派生文件，则回传可重建它的 raw Amazon review数据或 timestamp-preserving intermediate；只回传现有三列 `.inter` 无法恢复 gap。
3. 当前 baseline Qwen checkpoint完整目录：model weights、`config.json`、tokenizer files、training args、run log、best-checkpoint说明和SHA。

### Phase B 必需

4. 完整 SID version资产及 manifest：
   - `text_mbk_k512_dedup`；
   - 实际使用的 CF/CS版本；
   - `sasrec_v3_k512_dedup`；
   - 每个版本的 index/info/item2sid/sid2items/valid set、rewrite CSV、generation report。
5. train/valid的 Text、CF、SASRec-SID Qwen candidate artifacts；最好包含 beam sequence score/token logprob、row index、history、target、checkpoint SHA。若没有 train候选，需要在 AutoDL按 out-of-fold方案生成。
6. direct SASRec candidate artifacts与 checkpoint/config（本地已有部分 validation/S6 bundle，但 joint训练需要明确的 train或valid_fit来源）。
7. 当前 strongest external ranker模型、feature contract、candidate provenance和对应 validation split manifest；本地已有部分 Stage 7/S6 artifact，但需确认与准备使用的 generator/SID版本完全一致。

### 复现与环境必需

8. AutoDL Python/Transformers/Torch/CUDA版本、`pip freeze`或environment lock、base Qwen snapshot路径/commit、GPU型号。
9. 所有回传 bundle的 SHA256清单和生成命令；不要只回传散落结果文件。

### 当前阶段不需要回传

- 新训练的 Chrono/UniRank模型；
- test预测或新 test指标；
- GPU生成的新实验结果。

这些属于下一阶段，必须在实现与实验协议冻结后再产生。

## 7.7 AutoDL 执行边界

本地 WSL只负责代码、静态/CPU小单测、git和实验脚本。任何 Qwen/SASRec训练、candidate generation、joint teacher-forcing推理或大规模评估统一在：

```text
ssh -p 14779 root@region-9.autodl.pro
```

执行。本阶段没有发起该连接。

---

## Appendix A — 论文与计划依据

- 本地计划：`docs/优化四：MiniOneRec + ChronoSID + UniSGR/7.22 MiniOneRec_Chrono_UniRank_project_plan.md`
- 本地 ChronoSID PDF：`docs/优化四：MiniOneRec + ChronoSID + UniSGR/BeyondItemOrderTemporalGapTokenizationforGenerative.pdf`
- 本地 UniSGR PDF：`docs/优化四：MiniOneRec + ChronoSID + UniSGR/UniSGR Unified Framework for Semantic ID Generation and.pdf`
- ChronoSID：<https://arxiv.org/abs/2607.03918>
- UniSGR：<https://arxiv.org/abs/2607.04068>

## Appendix B — 最终架构判断

```text
可立即进入下一阶段实现：
  Phase A gap token
  Phase B single-objective shared-decoder joint ranking

需先补数据/证据：
  timestamp-aligned CSV
  complete SID manifests
  Qwen checkpoints/tokenizers
  train/valid multi-source candidates

应后置：
  TA-FAMAE
  GAOQ replacement
  multi-objective VA-PMTP / TAT / FACL
  rank-to-generation KL
  STARK / MoE
```

最终建议：先用 gap-only建立一个干净、可归因的 temporal baseline；随后用固定候选池证明 Qwen hidden state具备排序价值，再让 `L_rank` 回传 shared decoder。只有这两个核心假设分别成立，才值得进入 full alignment与新 SID construction。
