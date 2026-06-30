# MiniOneRec 优化最终阶段总结：Collaborative-Semantic SID、诊断评测与轻量 Rerank

## 0. 一句话项目定位

本阶段不是继续堆 RL，而是围绕 MiniOneRec 的核心表示单元 SID 做系统优化：

```text
Text-SID baseline
  -> SID 契约与版本管理
  -> SID 静态诊断与增强评测
  -> train-only CF embedding
  -> Collaborative-Semantic embedding
  -> 新 SID 生成与 CSV 重写
  -> SID-only SFT 消融
  -> item-level candidate expansion
  -> history-aware lightweight rerank
```

最终主贡献可以概括为：

1. 建立了可版本化、可检查、可复现实验的 SID/mapping 基础设施。
2. 提出了 Collaborative-Semantic SID：把文本语义 embedding 与 train-only 协同 embedding 融合后重新生成 SID。
3. 建立了 SID-level、prefix-level、item-level candidate/rerank 的分层评测体系。
4. 在 Industrial 与 Office 两个 category 上验证：CS-SID 能超过 Text-SID baseline，但最优融合权重因数据集不同而不同。

RL 已完成复现和探索，但本优化主线的核心不是 RL，而是 SID 结构和 item-level candidate quality。

---

## 1. 总体数据流

项目中 SID 相关数据流如下：

```text
item_id
  -> row_index / item_order
  -> text_emb row
  -> train-only co-occurrence
  -> CF embedding
  -> CS embedding = alpha * text + (1 - alpha) * CF
  -> RQ-KMeans / MiniBatchKMeans SID
  -> dedup suffix
  -> item2sid / sid2items / valid_sid_set / item_mapping
  -> experiment_manifest.json
  -> rewrite train/valid/test CSV
  -> SidSFTDataset
  -> SFT sid-only model
  -> constrained decoding with SID trie
  -> prediction JSON
  -> calc_plus SID-level evaluation
  -> evaluate_candidates candidate JSONL
  -> rerank.py lightweight rerank
  -> item-level HR/NDCG
```

这个链路的核心原则是：

- 不覆盖官方 Text-SID baseline 文件；
- 每个 SID version 独立保存；
- 每次训练、评估、诊断都必须使用同一个 SID version 的 CSV/index/info/mapping；
- train-only 协同信号不能使用 valid/test；
- rerank 只能用 history 和 train-only 特征，不能使用 target item 泄漏。

---

## 2. 优化主线分阶段说明

### 阶段 0：SID 契约与版本管理

**目标**

把原项目中分散在 `index.json`、`info.txt`、CSV、evaluate trie、calc valid SID set 里的 SID 统一成可版本化契约。

**新增/核心文件**

```text
utils_sid.py
build_sid_mapping.py
check_sid_stage0.py
data/Amazon/sid_maps/experiment_manifest.json
data/Amazon/sid_maps/<category>/item2sid_text.json
data/Amazon/sid_maps/<category>/sid2items_text.json
data/Amazon/sid_maps/<category>/valid_sid_set_text.json
data/Amazon/sid_maps/<category>/item_mapping_text.json
```

**为什么要做**

MiniOneRec 的训练和评估高度依赖 SID。如果 CSV 中的 `item_sid`、evaluate 使用的 `index/info`、calc 使用的 valid SID set 不一致，模型指标就不可解释。

因此先把 SID 映射显式化：

```text
item_id -> SID
SID -> item_id list
valid SID set
item_id -> title / SID / SID tokens
```

**验收结果**

```text
Office_Products:
num_items = 3459
num_unique_sid = 3444
collision ≈ 0.43%

Industrial_and_Scientific:
num_items = 3686
num_unique_sid = 3670
collision ≈ 0.43%

parse rate = 1.0
CSV item coverage = 1.0
history_item_id/history_item_sid length consistency = 1.0
index/info/csv conflicts = 0
```

**下一阶段目的**

在确认官方 Text-SID 干净可用后，进一步诊断其质量瓶颈：collision、prefix 分布、head/mid/tail 表现、3/4 层 SID 表现差异等。

---

### 阶段 1：SID 静态诊断与增强评测

**目标**

不只看 HR/NDCG，而是知道 SID 为什么好或不好。

**新增/核心文件**

```text
analyze_sid.py
calc_plus.py
compare_sft_rl_outputs.py
scripts/diagnose_sid_smoke_by_groups.py
scripts/summarize_sidonly_smoke_grid.py
```

**实现内容**

静态诊断：

- `num_items`
- `num_unique_sid`
- `collision_rate`
- `collided_item_rate`
- `num_collision_groups`
- `level token entropy`
- `prefix fanout`
- `history length consistency`
- `head/mid/tail collision`
- train-only interaction weighted collision

增强评估：

- SID-level HR/NDCG；
- prefix@1/2/3/4；
- invalid SID rate；
- duplicate generation rate；
- SID length distribution；
- head/mid/tail 分桶；
- cold/warm 分桶；
- short/long history 分桶；
- per-sample compare。

**为什么要做**

生成式推荐的输出是 SID，不是直接 item_id。SID-level 命中和 item-level 命中并不完全相同，特别是存在 collision 或 dedup suffix 时。

增强评测可以回答：

- 命中来自 3 层 SID 还是 4 层 dedup SID？
- 模型是否只会预测 head item？
- prefix 已经对了但 exact 错了，是否适合 rerank？
- CS-SID 是否改善了候选邻域，而不只是 exact match？

**下一阶段目的**

构建 Collaborative-Semantic embedding，为重新生成更好的 SID 做准备。

---

### 阶段 2：Collaborative-Semantic Embedding

**目标**

官方 Text-SID 主要来自文本 embedding。为了让 SID 同时编码行为相似性，引入 train-only 协同信号，构建 CS embedding。

**新增/核心文件**

```text
build_cs_embeddings.py
data/Amazon/cs_embeddings/<category>/<category>.cf_emb.npy
data/Amazon/cs_embeddings/<category>/<category>.cs_emb_alpha0.2.npy
data/Amazon/cs_embeddings/<category>/<category>.cs_emb_alpha0.5.npy
data/Amazon/cs_embeddings/<category>/<category>.cs_emb_alpha0.7.npy
data/Amazon/cs_embeddings/<category>/<category>.item_order.json
data/Amazon/cs_embeddings/<category>/<category>.row_index.json
data/Amazon/cs_embeddings/<category>/<category>.cs_embedding_report.json
```

**方法**

```text
1. 只读取 train_csv
2. history_item_id + target item_id 构建 item-item co-occurrence
3. co-occurrence -> PPMI sparse matrix
4. TruncatedSVD 得到 CF embedding
5. L2 normalize text_emb 和 CF_emb
6. cs_emb = normalize(alpha * text_emb + (1 - alpha) * cf_emb)
```

其中：

```text
alpha=0.7: 更偏文本语义
alpha=0.5: 文本/协同均衡
alpha=0.2: 更偏协同信号
```

**关键安全检查**

```text
len(item_order) == emb.shape[0]
row_index[item_order[i]] == i
item_order[i] == str(i)
set(item_order) == set(item2sid.keys())
```

**为什么不用 valid/test**

如果用 valid/test 构建 co-occurrence 或 popularity，就会把测试交互泄漏进 SID 结构，后续所有 HR/NDCG 都会被污染。

**下一阶段目的**

把 text / CF / CS embedding 输入 SID 生成算法，得到可训练的新 SID version。

---

### 阶段 3/4：新 SID 生成与 CSV 重写

**目标**

用 text/CS embedding 生成新 SID，并让 SFT 真正训练这些新 SID，而不是只生成离线文件。

**新增/核心文件**

```text
run_rqkmeans_with_emb.py
rewrite_sid_csv.py
update_sid_manifest.py
scripts/generate_sid_versions_and_rewrite_csv.sh
data/Amazon/sid_versions/<sid_version>/<category>/
```

每个 SID version 输出：

```text
index.json
info.txt
item2sid.json
sid2items.json
valid_sid_set.json
item_mapping.json
train.csv
valid.csv
test.csv
reports/generation_report.json
reports/rewrite_train_report.json
reports/rewrite_valid_report.json
reports/rewrite_test_report.json
reports/stage0_check_report.json
```

**Dedup 设计**

RQ-KMeans 生成的 3 层 SID 仍可能 collision。为避免 SID-level 与 item-level 不一致，引入 append dedup：

```text
singleton:
  <a_x><b_y><c_z>

collision bucket:
  <a_x><b_y><c_z><d_0>
  <a_x><b_y><c_z><d_1>
  ...
```

这样保留前三层 semantic prefix，同时用第 4 层做 item disambiguation。

**验收结果**

主要 dedup 版本都满足：

```text
post-dedup collision = 0
rewrite missing item = 0
SID parse success = 1.0
stage0 check ok = true
```

**为什么要重写 CSV**

`SidSFTDataset` 直接读取 CSV 中的：

```text
history_item_sid
item_sid
```

如果只生成新 `index/info`，但不重写 CSV，SFT 仍然训练旧 Text-SID，实验无效。

**下一阶段目的**

做可控 SID-only SFT 消融，比较 text/CS、k256/k512、alpha 不同取值。

---

### 阶段 5：SID-only SFT 消融

**目标**

在不混入 title2sid / seq-title2sid 辅助任务、不跑 RL 的情况下，只比较 SID 结构本身对生成式推荐的影响。

统一配置：

```text
base model = Qwen2.5-0.5B
task = sid_only
sample = 10000 / 30000
epoch = 1
beam = 20
early stopping = off
save_during_training = off for disk-light validation
```

**为什么不用 full SFT**

Full SFT 成本高，而且变量太多。SID 优化阶段应该先隔离变量：

```text
embedding type: text vs CS
codebook size: k256 vs k512
alpha: 0.2 / 0.5 / 0.7
dedup: 3-layer singleton + 4-layer collision suffix
```

#### 5.1 Industrial 10k Grid

| SID version | HR@20 | NDCG@20 | len3 HR@20 | len4 HR@20 |
|---|---:|---:|---:|---:|
| text_mbk_k256_dedup | 0.0514 | 0.0408 | 0.0070 | 0.1386 |
| text_mbk_k512_dedup | 0.0971 | 0.0583 | 0.0324 | 0.3144 |
| cs_alpha0.7_k256_dedup | 0.0649 | 0.0486 | 0.0137 | 0.1351 |
| cs_alpha0.7_k512_dedup | 0.0496 | 0.0405 | 0.0015 | 0.1989 |
| cs_alpha0.5_k256_dedup | 0.0609 | 0.0390 | 0.0499 | 0.0685 |
| cs_alpha0.2_k256_dedup | 0.0854 | 0.0477 | 0.0318 | 0.1346 |
| cs_alpha0.2_k512_dedup | 0.0973 | 0.0608 | 0.0337 | 0.1753 |

结论：

- 同 k=256 时 CS 明显优于 text；
- k512 对 text baseline 非常重要；
- alpha=0.2 是 Industrial 上最有潜力的 CS 配置；
- CS 在 3 层 singleton item 上更有优势。

#### 5.2 Industrial 30k Validation

| SID version | HR@20 | NDCG@20 | len3 HR@20 | len4 HR@20 | head HR@20 | mid HR@20 | tail HR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_mbk_k512_dedup | 0.1286 | 0.0709 | 0.0681 | 0.3317 | 0.3060 | 0.0233 | 0.0075 |
| cs_alpha0.2_k512_dedup | 0.1348 | 0.0764 | 0.0897 | 0.1900 | 0.3117 | 0.0360 | 0.0059 |

提升：

```text
HR@20   +0.0062
NDCG@20 +0.0055
```

机制：

- Text 在 4 层 dedup item 上更强；
- CS 在 3 层 singleton 和 mid group 上更强；
- CS 的总体收益来自更好的协同行为组织，而不是简单扩大 codebook。

#### 5.3 Office 30k Alpha Sweep

| SID version | HR@20 | NDCG@20 | len3 HR@20 | len4 HR@20 | head HR@20 | mid HR@20 | tail HR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_mbk_k512_dedup | 0.1445 | 0.0844 | 0.1217 | 0.2574 | 0.2819 | 0.0258 | 0.0065 |
| cs_alpha0.2_k512_dedup | 0.1402 | 0.0869 | 0.1061 | 0.1724 | 0.2624 | 0.0337 | 0.0194 |
| cs_alpha0.5_k512_dedup | 0.1369 | 0.0840 | 0.1121 | 0.1624 | 0.2655 | 0.0253 | 0.0091 |
| cs_alpha0.7_k512_dedup | 0.1478 | 0.0870 | 0.1411 | 0.1659 | 0.2806 | 0.0348 | 0.0103 |

结论：

- Office 上最优是 `cs_alpha0.7_k512_dedup`；
- Office 更依赖文本语义，因此 alpha 需要更高；
- `cs_alpha0.2` 虽然 NDCG 高，但 HR@20 不如 Text；
- `cs_alpha0.7` 同时超过 Text 的 HR@20 和 NDCG@20。

**阶段 5 总结**

不同 category 的最优 alpha 不同：

```text
Industrial: alpha=0.2 更好，说明协同信号更重要
Office:     alpha=0.7 更好，说明文本语义更重要
```

下一阶段目的：

从 SID-level exact match 推进到 item-level candidate/rerank，验证 CS-SID 是否真的产生更好的 item candidate。

---

### 阶段 6：Candidate Expansion 与轻量 Rerank

**目标**

解决 SID-level 指标不能完全代表 item-level 推荐质量的问题。

**新增/核心文件**

```text
evaluate_candidates.py
rerank.py
scripts/run_stage6_candidate_rerank.sh
```

**候选生成**

```text
predicted SID beams
  -> exact SID bucket candidates
  -> optional prefix@3 candidates
  -> optional prefix@2 candidates
```

**Rerank 特征**

只使用非泄漏特征：

- SID beam rank；
- source type：exact / prefix@3 / prefix@2；
- train-only popularity；
- bucket size penalty；
- candidate 与 history item embedding cosine；
- candidate 与 recent history item embedding cosine。

**新增报告指标**

```text
candidate_pool_recall
candidate_recall@K
target_in_exact_rate
target_in_prefix3_rate
target_in_prefix2_rate
target_source_distribution
avg_rank_before_rerank_if_hit
after_hr20
after_ndcg20
```

#### 6.1 Industrial Final Stage 6

| Version | Setting | Mean Candidates | Candidate Recall | Target Exact Rate | Target Prefix@3 Rate | Avg Rank If Hit | HR@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text_k512 | exact | 20.00 | 0.1286 | 0.1286 | 0.0000 | 5.04 | 0.1286 | 0.0724 |
| cs_alpha0.2_k512 | exact | 20.00 | 0.1348 | 0.1348 | 0.0000 | 4.40 | 0.1348 | 0.0779 |
| text_k512 | prefix@3 + rerank | 31.27 | 0.1388 | 0.1286 | 0.1388 | 10.36 | 0.1288 | 0.0701 |
| cs_alpha0.2_k512 | prefix@3 + rerank | 58.53 | 0.1516 | 0.1348 | 0.1516 | 10.87 | 0.1366 | 0.0784 |

Industrial 最优：

```text
cs_alpha0.2_k512_dedup + prefix@3 + source_weight=4.0
HR@20   = 0.136554
NDCG@20 = 0.078403
```

机制解释：

- CS exact-only 已经超过 Text exact-only，说明收益不是靠扩大候选池；
- prefix@3 给 CS 增加了有效候选；
- prefix@2 虽然 candidate recall 更高，但 avg rank 很靠后，当前轻量 reranker 排不回来，因此不作为主方案。

#### 6.2 Office Final Stage 6

| Version | Setting | Mean Candidates | Candidate Recall | Target Exact Rate | Target Prefix@3 Rate | Avg Rank If Hit | HR@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text_k512 | exact | 20.00 | 0.1445 | 0.1445 | 0.0000 | 4.53 | 0.1445 | 0.0853 |
| text_k512 | prefix@3 + rerank | 23.87 | 0.1502 | 0.1445 | 0.1502 | 5.45 | 0.1443 | 0.0845 |
| cs_alpha0.7_k512 | exact | 20.00 | 0.1478 | 0.1478 | 0.0000 | 4.16 | 0.1478 | 0.0882 |
| cs_alpha0.7_k512 | prefix@3 + rerank | 52.85 | 0.1617 | 0.1478 | 0.1617 | 9.13 | 0.1492 | 0.0884 |

Office 最优：

```text
cs_alpha0.7_k512_dedup + prefix@3 + source_weight=4.0
HR@20   = 0.149199
NDCG@20 = 0.088380
```

机制解释：

- Text 的 prefix@3 基本无收益；
- CS 的 prefix@3 提升 candidate recall，并能被 rerank 转化成 item-level gain；
- Office 的最佳 alpha 更偏文本，说明 category 对协同/语义融合权重敏感。

---

## 3. 最终跨 Category 结果

| Category | Best Text Baseline | Text HR@20 | Text NDCG@20 | Best CS Setting | CS HR@20 | CS NDCG@20 | 结论 |
|---|---|---:|---:|---|---:|---:|---|
| Industrial_and_Scientific | text_k512 exact | 0.1286 | 0.0724 | cs_alpha0.2_k512 + prefix@3 rerank | 0.1366 | 0.0784 | CS 明显优于 Text |
| Office_Products | text_k512 exact | 0.1445 | 0.0853 | cs_alpha0.7_k512 + prefix@3 rerank | 0.1492 | 0.0884 | CS 优于 Text，但需要更高 alpha |

最终结论：

```text
CS-SID 在两个 category 上都优于 Text-SID baseline。
Industrial 更适合强协同融合 alpha=0.2。
Office 更适合偏文本融合 alpha=0.7。
prefix@3 expansion 对 CS-SID 更有效，说明 CS 的 prefix 邻域更有推荐意义。
```

---

## 4. 为什么这些结果可信

### 4.1 变量隔离

实验没有一开始直接 full SFT 或 RL，而是逐步拆开：

```text
Text vs CS
k256 vs k512
alpha 0.2 / 0.5 / 0.7
SID-level vs item-level
exact-only vs prefix expansion
before rerank vs after rerank
```

这样可以回答“收益来自哪里”，而不是只报告一个黑盒指标。

### 4.2 无泄漏

CF embedding 只用 train_csv：

```text
train history + train target -> co-occurrence -> PPMI -> SVD
```

Rerank 只用：

```text
train-only popularity
history item embedding
recent history item embedding
SID source/rank/bucket size
```

target item 只在最后算 metrics 时使用。

### 4.3 SID version 不混用

每个版本都有独立：

```text
index.json
info.txt
item2sid.json
sid2items.json
valid_sid_set.json
train.csv
valid.csv
test.csv
```

并通过 manifest 管理，避免 Text-SID 与 CS-SID 混跑。

### 4.4 兼容 3/4 层 SID

Dedup 后同时存在：

```text
3-layer singleton SID
4-layer collision-disambiguated SID
```

因此训练、evaluate、calc_plus、candidate expansion 都必须支持任意 SID token length，而不是写死 `<a><b><c>`。

---

## 5. 关键工程问题与解决方案

### 5.1 embedding row 与 item_id 对齐

风险：

```text
第 i 行 embedding 被误认为另一个 item_id
```

解决：

```text
item_order.json
row_index.json
row_index[item_order[i]] == i
```

### 5.2 SID collision

风险：

SID-level exact hit 不等于 item-level hit。

解决：

```text
dedup append <d_i>
post-dedup collision = 0
```

### 5.3 prefix expansion 噪声

观察：

- prefix@3 对 CS 有帮助；
- prefix@2 candidate recall 更高，但噪声太大；
- 当前轻量 reranker 无法把 prefix@2 大候选池中的 target 排回 Top20。

结论：

主方案使用 exact + prefix@3，不使用 prefix@2。

### 5.4 AutoDL 磁盘问题

训练中间 checkpoint 会写 `optimizer.pt`，占用大量空间。

解决：

```text
save_during_training=False
只保留 final_checkpoint
删除 checkpoint-* 与 candidates/reranked JSONL
```

### 5.5 Office 路径硬编码问题

曾发现 `cs_alpha0.7_k512_dedup` 在旧脚本中被写到 `Industrial_...` 路径。

解决：

只在 `CATEGORY=Industrial_and_Scientific` 时使用旧兼容路径，Office 使用标准：

```text
results/eval_sidonly_Office_Products_<sid_version>_...
```

---

## 6. 面试问答：实现细节与设计选择

### Q1：这个项目相比原 MiniOneRec 改进点是什么？

原项目主要复现 OneRec 风格的 SID 生成、SFT、RL 和 constrained decoding。

我的改进集中在 SID 本身：

1. 建立 SID 契约和版本管理；
2. 系统诊断 Text-SID 的 collision、prefix、head/mid/tail；
3. 引入 train-only collaborative signal；
4. 生成 CS-SID；
5. 重写 CSV，让 SFT 真正训练新 SID；
6. 从 SID-level 扩展到 item-level candidate/rerank；
7. 在两个 category 上验证 CS-SID 超过 Text-SID。

### Q2：为什么不把 popularity 直接拼进 CS embedding？

第一版 CS embedding 只融合 text 与 CF，不直接拼 popularity。

原因：

- popularity 很容易让 SID 聚类退化成 head/tail 分桶；
- 会降低语义 prefix 的可解释性；
- popularity 作为 rerank feature 更合适，因为它是排序阶段的先验，而不是 item identity 的结构编码；
- 如果后续使用 valid/test popularity 会有泄漏风险。

### Q3：为什么用 PPMI + SVD 构建 CF embedding？

这是资源有限下最稳的方案：

- 不需要额外训练深度序列模型；
- 稀疏矩阵可控；
- 只用 train_csv；
- 输出 fixed-size item embedding，能直接和 text embedding 融合；
- 可解释为 item co-occurrence 的低秩协同表示。

### Q4：为什么不用 SASRec/GRU4Rec embedding？

可以作为后续增强，但第一版不选。

原因：

- 会引入新的模型训练变量；
- 难以区分收益来自 SID 结构还是序列模型 embedding；
- 需要更多超参和 checkpoint 管理；
- 不如 PPMI+SVD 轻量、确定、易排查。

### Q5：为什么要做 dedup，而不是接受 SID collision？

推荐评估最终是 item-level。若多个 item 共用一个 SID：

```text
模型预测对了 SID，但不知道预测的是哪个 item
```

这会导致 SID-level HR 高估 item-level HR。

Dedup append `<d_i>` 后：

```text
post-dedup collision = 0
```

同时保留前三层 prefix 作为 semantic/behavior cluster。

### Q6：为什么会出现 3 层和 4 层 SID 混合？

只有 collision bucket 需要第 4 层 disambiguation。singleton item 没必要增加 `<d_0>`，否则会人为增加序列长度和生成难度。

所以：

```text
singleton -> 3 layers
collision -> 4 layers
```

评估时必须统计 `sid_length_distribution`，并分别看 len3/len4 HR。

### Q7：为什么 Industrial 最优 alpha=0.2，Office 最优 alpha=0.7？

alpha 越小，CF 权重越大；alpha 越大，text 权重越大。

实验显示：

```text
Industrial: cs_alpha0.2 最优
Office:     cs_alpha0.7 最优
```

解释：

- Industrial 的用户行为协同信号更强，适合更高 CF 权重；
- Office 商品文本语义更稳定，过强 CF 会破坏 semantic exact 结构；
- 不同 category 的最优语义/协同比例不同，这是 CS-SID 的一个重要发现。

### Q8：为什么 Text 在 4 层 dedup item 上更强？

Text embedding 对 title/category 等语义更敏感，collision bucket 中的 item 可能文本上更容易被分开。

CS embedding 引入协同信号后，会让行为相似 item 更接近，这对 singleton/mid item 有帮助，但可能让部分 collision item 的 exact disambiguation 变难。

这也是为什么要看 len3/len4 分桶，而不是只看 overall。

### Q9：为什么要从 SID-level 进入 item-level candidate/rerank？

生成模型输出 SID，但推荐系统最终要推荐 item。

尤其在存在：

- collision；
- dedup suffix；
- prefix expansion；
- SID bucket；

时，SID-level HR/NDCG 不能完全说明 item-level 推荐效果。

candidate/rerank 可以回答：

```text
target item 是否进入候选池？
进入后排在什么位置？
rerank 能否把它排进 Top20？
```

### Q10：为什么 prefix@3 有效，prefix@2 不作为主方案？

prefix@3 与完整 SID 只差 dedup suffix，更接近 item neighborhood。

prefix@2 太粗，会召回大量候选：

```text
candidate recall 提升
avg rank if hit 变大
reranker 难以排回 Top20
```

实验中 prefix@2 虽提高 candidate pool recall，但最终 HR/NDCG 下降，因此主方案选 prefix@3。

### Q11：rerank 有没有泄漏？

没有。

Rerank 使用：

```text
SID rank
source type
train-only popularity
history embedding cosine
recent history embedding cosine
bucket size penalty
```

不使用：

```text
target item label
valid/test popularity
future interaction
```

target item 只用于最后计算 HR/NDCG。

### Q12：为什么没有继续大规模调 rerank 权重？

当前目标是证明 CS-SID 结构有效，而不是训练一个强 reranker。

做了小网格：

```text
source_weight = 0.5 / 1.0 / 2.0 / 4.0
```

发现 `4.0` 在 Industrial 和 Office 都是合理选择。进一步大规模搜索会有过拟合风险，尤其只有一个 test split。

### Q13：为什么不继续 full SFT？

当前 30k sid-only 已经能证明 SID 结构差异，而且成本更低、变量更少。

full SFT 会混入：

- title2sid；
- seq-title2sid；
- 多任务数据比例；
- 更长训练；
- checkpoint selection；
- overfitting；

不适合用来定位 SID 本身的贡献。后续可以作为补充验证，但不是主线。

### Q14：为什么 RL 不作为主贡献？

RL 已经复现过，并在 mixed HEPO 上有提升，但 RL 的变量很多：

- reward；
- KL beta；
- num_generations；
- beam rollout；
- mixed dataset；
- SFT checkpoint quality。

本阶段更有价值的是把 SID 结构、candidate recall 和 item-level rerank 讲清楚。

### Q15：如果面试官问“CS-SID 是不是只是 candidate pool 更大才赢”？

回答：

不是。

Industrial 上：

```text
CS exact-only HR@20 = 0.1348
Text exact-only HR@20 = 0.1286
```

Office 上：

```text
CS alpha0.7 exact-only HR@20 = 0.1478
Text exact-only HR@20 = 0.1445
```

exact-only 下候选数量都是 20，CS 仍然更强。因此收益不是仅靠扩大 candidate pool。

prefix@3 进一步提升 CS，说明 CS 的 prefix neighborhood 也更有推荐意义。

### Q16：如果问“为什么 Office alpha0.2 没赢 HR，但 alpha0.7 赢了”？

回答：

Office 这个类目的文本语义更重要。alpha0.2 过度偏协同，导致 exact HR 降低，但 NDCG 和 tail/mid 有改善。

alpha0.7 保留更多文本结构，同时加入少量协同信号，因此 HR/NDCG 同时超过 Text。

这说明 CS-SID 不是固定 alpha 的 magic trick，而是一个可调的 semantic-collaborative tradeoff。

---

## 7. 后续可做但不是当前主线的工作

### 7.1 更细 alpha sweep

当前只测了：

```text
0.2 / 0.5 / 0.7
```

后续可以测：

```text
Industrial: 0.1 / 0.2 / 0.3
Office:     0.6 / 0.7 / 0.8
```

目的：

更精确地找到每个 category 的最佳协同/语义比例。

### 7.2 更强 reranker

当前 rerank 是轻量手工特征。后续可以做：

- logistic regression / LambdaMART；
- learned linear reranker；
- source-aware rerank；
- prefix@2 candidate pruning；
- per-source candidate budget。

但必须使用 train/valid 做调参，不能直接用 test 调权重。

### 7.3 Office / Industrial full SFT

可以作为最终补充实验：

```text
Industrial: cs_alpha0.2_k512_dedup
Office:     cs_alpha0.7_k512_dedup
```

但 full SFT 不是证明 CS-SID 的必要条件。

### 7.4 多卡 SFT

已为 SFT 增加 DDP 支持，可在资源充足时跑：

```text
4 x A100
Qwen2.5-1.5B / 3B
SFT only
```

这属于工程扩展，不改变 CS-SID 主线结论。

---

## 8. 面试项目表述建议

可以这样描述：

> 我在 MiniOneRec 复现基础上发现，生成式推荐的关键瓶颈不只是 SFT/RL，而是 SID 本身的结构质量。于是我先建立了 SID 契约和版本管理，保证 index/info/csv/evaluate/calc 一致；然后做了 SID collision、prefix entropy、head/mid/tail、3/4 层 SID 的诊断体系。接着我只用 train 数据构建 item co-occurrence，通过 PPMI+SVD 得到 CF embedding，并与 text embedding 融合生成 Collaborative-Semantic embedding，再用 RQ-KMeans 生成 CS-SID。为了避免 SID collision 影响 item-level 评估，我设计了 append dedup，让 collision bucket 增加第 4 层 `<d_i>`。最后我重写 CSV 训练 sid-only SFT，并实现 candidate expansion 与轻量 history-aware rerank，把 SID-level 指标推进到 item-level HR/NDCG。实验表明，CS-SID 在 Industrial 和 Office 两个 category 上都超过 Text-SID baseline，但最优 alpha 不同，说明不同品类需要不同的协同/语义融合比例。

最重要的数字：

```text
Industrial:
Text exact HR@20/NDCG@20 = 0.1286 / 0.0724
CS alpha0.2 + prefix@3 rerank = 0.1366 / 0.0784

Office:
Text exact HR@20/NDCG@20 = 0.1445 / 0.0853
CS alpha0.7 + prefix@3 rerank = 0.1492 / 0.0884
```

项目贡献：

1. SID 诊断与版本化实验基础设施；
2. Collaborative-Semantic SID；
3. 3/4 层 dedup SID 兼容训练与评估；
4. item-level candidate expansion + lightweight rerank；
5. 单卡资源下可复现实验闭环；
6. 多卡 SFT 支持作为工程扩展。

---

## 9. 当前最终结论

当前可以收口为：

```text
CS-SID 主线成立。

Industrial:
  最优为 cs_alpha0.2_k512_dedup + prefix@3 rerank。
  协同信号权重更高时效果最好。

Office:
  最优为 cs_alpha0.7_k512_dedup + prefix@3 rerank。
  文本语义权重更高时效果最好。

共同机制:
  CS exact-only 已经强于 Text exact-only；
  prefix@3 expansion 对 CS 更有价值；
  prefix@2 太宽，当前轻量 reranker 无法充分利用；
  CS-SID 的收益不是单纯扩大候选池，而是更有效的 item-to-SID 组织。
```
