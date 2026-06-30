# MiniOneRec 单卡 A100 复现实验与面试材料

> 目标：把 MiniOneRec 从官方 SID/SFT/RL 链路复现到可解释、可展示、可被面试追问的程度。  
> 本文重点服务于面试表达：不仅列结果，还说明为什么可信、哪里有 caveat、被追问时怎么回答。

## 1. 一句话项目介绍

我在单卡 A100 资源下复现并增强了 MiniOneRec 的生成式推荐链路：先基于官方 Text-SID 完成 SFT，再用 mixed HEPO RL 做 GRPO 风格强化学习，并补充了 SID mapping、静态 SID 诊断、增强评估和 SFT/RL 对比工具。最终在 `Industrial_and_Scientific` 上实现 SFT 到 RL 的全 K 提升，在 `Office_Products` 上实现 NDCG 全 K 提升和 HR@1-20 提升，同时分析了 HR@50 轻微下降的排序-召回 tradeoff。

## 2. 最终主结果

评估口径：

- 数据集：Amazon `Industrial_and_Scientific`、`Office_Products`
- SID：官方 Text-SID
- 评估：constrained decoding，`num_beams=50`
- 指标：SID-level HR/NDCG@1/3/5/10/20/50
- 合法性：`invalid_sid_count=0`
- 资源：单卡 A100

### 2.1 Industrial_and_Scientific

最佳 RL checkpoint：

```text
outputs/rl_Industrial_and_Scientific_mixed_hepo_beam_full_lr5e7_beta008_bs48/final_checkpoint
```

| Model | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT beam50 | 0.045665 | 0.061549 | 0.076550 | 0.095742 | 0.119568 | 0.176925 | 0.045665 | 0.054387 | 0.060499 | 0.066630 | 0.072617 | 0.083893 |
| RL mixed HEPO beta0.08 | 0.051180 | 0.071035 | 0.083388 | 0.107655 | 0.138540 | 0.199426 | 0.051180 | 0.062667 | 0.067727 | 0.075466 | 0.083264 | 0.095176 |
| Delta | +0.005515 | +0.009486 | +0.006838 | +0.011913 | +0.018972 | +0.022501 | +0.005515 | +0.008280 | +0.007228 | +0.008836 | +0.010646 | +0.011283 |

可讲结论：

- Industrial 上 RL 是全面成功：HR 和 NDCG 的所有 K 都提升。
- HR@20 相对提升约 `15.9%`，HR@50 相对提升约 `12.7%`。
- `rl_rank_better=541`，`rl_rank_worse=206`，说明不是几个样本的偶然波动。

### 2.2 Office_Products

最佳 RL checkpoint：

```text
outputs/rl_Office_Products_mixed_hepo_beam_full_lr5e7_beta012_bs48/final_checkpoint
```

| Model | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT beam50 | 0.069873 | 0.101932 | 0.114879 | 0.133169 | 0.158035 | 0.205508 | 0.069873 | 0.088539 | 0.093836 | 0.099786 | 0.106049 | 0.115396 |
| RL mixed HEPO beta0.12 | 0.075832 | 0.103576 | 0.119400 | 0.138923 | 0.158241 | 0.202425 | 0.075832 | 0.092180 | 0.098670 | 0.104904 | 0.109762 | 0.118580 |
| Delta | +0.005959 | +0.001644 | +0.004521 | +0.005754 | +0.000206 | -0.003083 | +0.005959 | +0.003641 | +0.004834 | +0.005118 | +0.003713 | +0.003184 |

可讲结论：

- Office 上 RL 主要提升排序质量：NDCG@1-50 全部提升，HR@1-20 提升。
- HR@50 小幅下降，说明 RL 更倾向于把高置信候选排到前面，而不是扩大 beam50 内覆盖。
- `beta=0.12` 比 `beta=0.08` 更适合作为 Office 主结果：所有 NDCG 更高，HR@50 下降幅度更小；但不要说 HR 全面更好，因为 HR@3、HR@20 略低于 beta0.08。

## 3. 实验链路怎么讲

### 3.1 复现主链路

```text
官方 Text-SID / index / info / train-valid-test
        ↓
SFT：让 LLM 学会根据用户历史生成合法 SID
        ↓
Constrained decoding：约束生成只走合法 SID token trie
        ↓
RL：mixed data + HEPO hierarchical reward + GRPO-style optimization
        ↓
evaluate.py / calc.py / calc_plus.py / compare_sft_rl_outputs.py
```

### 3.2 我补的工程基础设施

- `utils_sid.py`：统一 SID parse、normalize、prefix、mapping IO。
- `build_sid_mapping.py`：生成 item2sid、sid2items、valid_sid_set、manifest。
- `analyze_sid.py`：静态 SID 质量诊断，包括 collision、prefix、token distribution、head/tail 分桶。
- `calc_plus.py`：在原始 `calc.py` 之外，补充 invalid、duplicate、prefix hit、head/mid/tail、cold/warm、history length 分桶。
- `compare_sft_rl_outputs.py`：逐样本比较 SFT 和 RL，输出 sft_only、rl_only、rank better/worse、prefix 变化。
- `build_cs_embeddings.py` / `run_rqkmeans_with_emb.py`：为后续 CS-SID 做了 embedding 和 SID 生成基础设施。

面试表达重点：

> 我不是只跑脚本，而是先把 SID 契约、版本管理、质量诊断和评估体系补齐。这样后续 SFT/RL/CS-SID 的结果不会混在一起，能定位收益来自哪里，也能解释失败样本。

## 4. 关键 caveat：当前 RL 不是纯 sid_rec-only

这是面试里很可能被追问的点，必须主动讲清楚。

当前 `rl_gpr.py` 中即使命令传：

```bash
--rl_task_mode sid_rec
--reward_type hepo
```

训练数据仍然是 mixed：

```text
SidDataset
RLTitle2SidDataset
RLSeqTitle2SidDataset
```

`rl_task_mode=sid_rec` 当前主要作用是触发 HEPO/SID reward 路径，并没有筛选为纯 `SidDataset`。

因此严谨说法是：

```text
我当前主结果证明的是：mixed HEPO RL 能提升生成式推荐的 SID-level ranking/retrieval。
不能直接声称纯 sid_rec-only RL 已经被证明有效。
```

如果被问“这是不是数据泄漏”：

- 不是 valid/test 泄漏。RL 训练数据来自 train 和 item/index 元数据。
- 但它是多任务 mixed RL，不是纯推荐序列 RL，所以归因要谨慎。
- 最严谨的后续消融是补 `rl_data_mode={mixed,sid_rec_only,title2sid_only,seq_title2sid_only}`。

## 5. HEPO Reward 怎么解释

HEPO 是层级 SID 奖励，用 SID token prefix 的匹配程度缓解 exact match reward 稀疏。

当前 Text-SID 形如：

```text
<a_236><b_231><c_226>
```

reward 逻辑：

```text
level-1 prefix 命中：0.2
level-2 prefix 命中：0.5
完整三层 SID 命中：1.0
否则：0.0
```

面试官可能问：为什么不用 exact match？

回答：

> SID 生成是离散 token 序列，完全命中的奖励非常稀疏。层级 SID 本身带有 coarse-to-fine 结构，prefix 命中说明模型已经捕捉到一部分 item cluster 信息。HEPO 把这种层级相似性转成 dense-ish reward，让 RL 不至于大部分样本都是 0 奖励。

面试官可能问：这会不会奖励错 item？

回答：

> 会有这个风险，所以最终仍然用 exact SID-level HR/NDCG 评估。HEPO 只是训练信号，不是最终指标。我的 `calc_plus.py` 还额外输出 prefix hit，是为了观察 HEPO 是否只是提升粗粒度 prefix，而没有提升完整 SID。Industrial 的结果显示 exact HR/NDCG 也全面提升，所以不只是 prefix 变好。

## 6. 为什么 Office HR@50 会下降但 NDCG 上升

这是最重要的“结果解释题”。

现象：

- Office beta0.12 相比 SFT：NDCG 全 K 提升。
- HR@1/3/5/10/20 提升。
- HR@50 从 `0.205508` 下降到 `0.202425`。

解释：

```text
RL 在 Office 上更像排序优化器，而不是深召回扩展器。
它把一部分正确候选从较靠后位置推到前面，所以 NDCG 和 shallow HR 提升。
但也可能把少数原本在 20-50 的正确 SID 挤出 beam50，导致 HR@50 轻微下降。
```

为什么 `beta=0.12` 比 `beta=0.08` 更好：

- `beta` 是 KL 约束强度，更大意味着离 SFT 策略更近。
- Office 上 `beta=0.08` 的 HR@50 是 `0.201603`，`beta=0.12` 是 `0.202425`。
- 更保守的 KL 减少了深召回损失，同时保持 NDCG 全 K 提升。

面试表达：

> 这说明 RL 不是免费午餐，它会改变排序分布。Industrial 上收益是 recall + ranking 都提升；Office 上更偏 ranking，深召回有轻微代价。我通过 beta0.12 对照验证了更强 KL 可以缓解这个代价。

## 7. 为什么结果可信

### 7.1 指标双重验证

- `calc_plus.py` 输出增强指标。
- 原始 `calc.py` 复核 HR/NDCG/invalid count。
- 两者在 beam50 上一致。

### 7.2 合法性验证

两类数据最终预测均满足：

```text
invalid_sid_rate = 0
duplicate_generation_rate = 0
calc.py invalid count = 0
```

说明提升不是来自非法 SID 或重复候选的评估 bug。

### 7.3 公平口径

- SFT 和 RL 都用同一 `evaluate.py`。
- 同一 test csv。
- 同一 `info_file` constrained decoding。
- 同一 `beam50`。
- `compare_sft_rl_outputs.py` 用 `row_index` 对齐逐样本比较。

### 7.4 训练规模趋势

Industrial 上从 mid512 到 full 的趋势：

- 小规模 RL 不稳定，mid512 基本持平或略降。
- 2048、4096、8192、16384 逐步变好。
- full mixed HEPO 最终全 K 显著提升。

这说明 RL 对训练规模敏感，小 smoke 不能代表最终效果。

## 8. 面试官可能拷问的问题与回答

### Q1：你到底复现了论文哪部分？

回答：

> 我重点复现了官方 SID 之后的生成式推荐训练链路：SFT、RL、constrained decoding、HR/NDCG 评估。SID 生成阶段我没有作为主复现目标重新训练，但我补了 SID mapping、诊断和 CS-SID 基础设施，为后续 SID 改进做准备。

### Q2：为什么先不用重新生成 SID？

回答：

> 因为 SFT/RL 本身依赖的是离散好的 SID index/info/csv，不依赖 RQ-VAE checkpoint。为了控制变量，我先使用官方 Text-SID 复现后半段训练链路。这样能先确认 tokenizer 扩展、constrained decoding、RL reward 和评估是稳定的。

### Q3：你的 RL 为什么有效？

回答：

> SFT 已经让模型学会合法 SID 格式，但它优化的是 teacher forcing token loss，不直接优化 top-K 推荐指标。RL 用 HEPO reward 直接鼓励目标 SID 或 prefix 靠近，在 constrained decoding 下把推荐目标和生成过程对齐。Industrial 上 HR/NDCG 全 K 提升，说明 RL 不只是格式学习，而是改善了排序和召回。

### Q4：为什么 Office 不如 Industrial？

回答：

> 两个数据集的 item 分布、history 行为和 SID collision/cluster 结构不同。Office 的 SFT baseline 本身更高，RL 的边际空间更小；结果上 RL 更偏向把正确候选前移，所以 NDCG 提升明显，但 HR@50 有轻微损失。这反映了 ranking quality 和 deep recall 的 tradeoff。

### Q5：`invalid=0` 是不是因为 constrained decoding 太强，所以指标没意义？

回答：

> constrained decoding 只保证输出是合法 SID，不保证输出等于目标 SID。它解决的是生成空间合法性问题，HR/NDCG 仍然由目标 SID 是否在 top-K 决定。SFT 和 RL 都在同一约束下评估，所以对比是公平的。

### Q6：SID collision 会不会影响结果？

回答：

> 当前评估是 SID-level，不是 item-level。Text-SID 存在约 0.43% collision，所以严格 item-level 指标需要候选 item 展开和 tie-breaking。我的 `analyze_sid.py` 已经诊断了 collision，`calc_plus.py` 明确把 item-level 指标标为不可可靠计算。这是我没有夸大 item-level 效果的原因。

### Q7：为什么不用 valid/test 定义 head/tail？

回答：

> 为避免数据泄漏，popularity/head-tail 只能由 train_csv 统计。`calc_plus.py` 的分桶评估也是基于 train interaction count 定义 popularity group。

### Q8：为什么 Office SFT 没跑满 10 epoch 还能用？

回答：

> `sft.py` 设置了 `EarlyStoppingCallback` 和 `load_best_model_at_end=True`。Office 训练在后续 eval loss 变差后提前停止，final checkpoint 保存的是 best checkpoint 权重，而不是坏模型。它能离线加载，beam50 评估合法率为 100%，并且 `calc.py/calc_plus.py` 一致，所以可以作为 SFT baseline。

### Q9：你怎么避免实验结果串台？

回答：

> 我使用明确的输出目录命名，比如 `category + method + beta + bs + beam`；同时新增 `experiment_manifest.json`、SID mapping 文件和 compare 输出。之前也遇到过 prediction 路径变量为空导致目录名和输入文件不一致的问题，后来通过固定命名和 `ls` 检查避免。

### Q10：如果继续改进，你会做什么？

回答：

> 第一，补 `rl_data_mode` 做 mixed vs sid_rec_only 消融，解决归因问题。第二，把 SID-level 评估扩展到 candidate item-level，处理 SID collision。第三，用 CS embedding 重新生成 SID，比较 Text-SID 和 CS-SID 的 collision、prefix distribution 以及最终 HR/NDCG。

## 9. 简历 bullet 写法

可选简历版本：

```text
- Reproduced MiniOneRec generative recommendation pipeline on a single A100, including Text-SID SFT, constrained decoding, and mixed HEPO RL; achieved +15.9% HR@20 and +12.7% HR@50 relative improvements on Industrial_and_Scientific under beam50.
- Built a unified SID infrastructure with item2sid/sid2items/versioned manifest, SID static diagnostics, and enhanced evaluation tools covering invalid SID rate, duplicate generation, prefix hit, head/tail buckets, and per-sample SFT-vs-RL transitions.
- Diagnosed dataset-specific RL behavior: Industrial showed consistent HR/NDCG gains across all K, while Office improved all NDCG and HR@1-20 but exposed an HR@50 recall tradeoff mitigated by stronger KL regularization.
```

中文版本：

```text
- 在单卡 A100 上复现 MiniOneRec 的 Text-SID SFT → constrained decoding → mixed HEPO RL 流程，Industrial 数据集 beam50 下 HR@20 相对提升约 15.9%，HR@50 相对提升约 12.7%。
- 构建统一 SID/mapping 与诊断评估体系，支持 SID collision、prefix distribution、invalid/duplicate generation、head/mid/tail 分桶和逐样本 SFT/RL 对比分析。
- 分析不同数据集上的 RL 收益形态：Industrial 呈现全 K HR/NDCG 提升，Office 呈现排序质量提升但 HR@50 轻微下降，并通过 beta0.12 验证更强 KL 可缓解深召回损失。
```

## 10. 面试 30 秒 / 2 分钟 / 5 分钟讲法

### 30 秒版

> 我复现了 MiniOneRec 的生成式推荐训练链路，用官方 Text-SID 先做 SFT，再用 mixed HEPO RL 优化 SID 生成。为了保证实验可信，我补了 SID mapping、静态诊断、增强评估和逐样本 SFT/RL 对比工具。最终 Industrial 在 beam50 下 HR/NDCG 全 K 提升，Office 则 NDCG 全 K 提升但 HR@50 有轻微 tradeoff，我进一步用更强 KL 做了验证。

### 2 分钟版

> 这个项目的核心是把推荐问题转成 LLM 生成 SID 的问题。SFT 学的是给定用户历史生成下一个 item 的 semantic ID，但 SFT 只优化 token loss，不直接优化 top-K 推荐指标。所以我在 SFT 后接了 RL，用 HEPO 层级 SID reward：预测对第一层 prefix 给 0.2，第二层给 0.5，完整 SID 给 1.0。这样比 exact reward 更稠密，也符合 SID 的层级结构。
>
> 我没有直接相信结果，而是补了评估和诊断工具。比如 valid SID set、collision、prefix hit、invalid rate、duplicate rate、head/tail 分桶、逐样本比较。最终 Industrial 上 RL 相比 SFT 在 HR/NDCG@1-50 全部提升；Office 上 NDCG 全部提升、HR@1-20 提升，但 HR@50 小幅下降。这个现象说明 RL 在 Office 上更像排序增强器，而不是深召回增强器。
>
> 这也让我形成一个更严谨的结论：mixed HEPO RL 对生成式推荐有效，但收益形态和数据集有关，后续需要做 sid_rec_only vs mixed 消融进一步归因。

### 5 分钟版结构

1. 背景：OneRec/MiniOneRec 把 item 映射为 SID，LLM 生成 SID 做推荐。
2. 工程复现：官方 Text-SID、SFT、RL、constrained decoding、beam50 评估。
3. 自己补的东西：SID versioning、诊断、calc_plus、compare 工具。
4. 结果：Industrial 全 K 提升，Office NDCG 提升但 HR@50 tradeoff。
5. caveat：当前 RL 是 mixed HEPO，不是纯 sid_rec-only。
6. 下一步：rl_data_mode 消融、item-level candidate expansion、CS-SID。

## 11. 后续最值得做的三件事

### 11.1 `rl_data_mode` 消融

目的：回答“收益来自 mixed data 还是 sid_rec data”。

建议参数：

```text
--rl_data_mode mixed
--rl_data_mode sid_rec_only
--rl_data_mode title2sid_only
--rl_data_mode seq_title2sid_only
```

最小实验：

- Office：`sid_rec_only + beta0.12 + mid16384`
- Industrial：`sid_rec_only + beta0.08 + mid16384`

### 11.2 item-level evaluation

当前是 SID-level。由于 SID collision 存在，item-level 需要：

- `sid2items`
- candidate expansion
- tie-breaking/ranking score
- target item_id 对齐

这适合作为下一阶段，不应在当前结果里强行声称 item-level HR。

### 11.3 CS-SID

已有 `build_cs_embeddings.py`，可以把 collaborative signal 融入 text embedding 后重新生成 SID。

面试价值：

> 当前结果优化的是训练和 RL；CS-SID 优化的是 item representation 和 SID construction。二者是两个不同层次的改进。

## 12. 文件与结果归档建议

重要 checkpoint 不要删：

```text
outputs/sft_Industrial_and_Scientific_full/final_checkpoint
outputs/rl_Industrial_and_Scientific_mixed_hepo_beam_full_lr5e7_beta008_bs48/final_checkpoint
outputs/sft_Office_Products_full/final_checkpoint
outputs/rl_Office_Products_mixed_hepo_beam_full_lr5e7_beta012_bs48/final_checkpoint
```

重要结果目录：

```text
results/calc_plus_sft_full_beam50
results/calc_plus_rl_full_mixed_hepo_bs48_beam50
results/compare_sft_rl_full_mixed_hepo_bs48_beam50_top50

results/calc_plus_sft_Office_Products_full_beam50
results/calc_plus_rl_Office_Products_full_mixed_hepo_beta012_bs48_beam50
results/compare_sft_rl_Office_Products_full_mixed_hepo_beta012_bs48_beam50
```

如果要清空间，优先删中间 smoke/mid checkpoint，但保留最终 prediction、calc、compare 输出。

## 13. 最终结论

这个项目目前可以定位为：

```text
一个在有限单卡资源下，对 MiniOneRec 生成式推荐链路进行可复现、可诊断、可解释增强的工程复现项目。
```

最有价值的点不是单个 HR 数字，而是：

- 复现了 SFT/RL 主链路；
- 建立了 SID 质量和预测结果诊断体系；
- 找到了 RL 在两个数据集上的不同收益形态；
- 能解释为什么 Industrial 全面提升，而 Office 出现 ranking vs recall tradeoff；
- 明确知道当前实验 caveat 和下一步消融方向。
